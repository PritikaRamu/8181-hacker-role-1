"""Code correctness (syntax) and code smell (semgrep) checks.

Syntax validity is checked in-process per snippet:
  - Python via `ast.parse`.
  - JavaScript / Rust / Ruby via `tree_sitter_language_pack` grammars
    (checking for ERROR nodes), with a subprocess fallback when the package
    is not installed: `node --check` (JS), `ruby -c` (Ruby),
    `rustfmt --edition 2021` (Rust).  Returns (None, None) only when neither
    tree-sitter nor the runtime binary is available.

Code smell detection shells out to `semgrep`. Since semgrep's fixed
per-invocation startup cost (~2-3s) dwarfs its per-file cost, findings for
an entire result file (up to ~1000 samples) are collected with a *single*
semgrep invocation over a temp directory of one file per sample, instead of
one subprocess per sample.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LANG_TO_SEMGREP_CONFIG = {
    "Python": "p/python",
    "JavaScript": "p/javascript",
    "Rust": "p/rust",
    "Ruby": "p/ruby",
}

LANG_TO_EXTENSION = {
    "Python": ".py",
    "JavaScript": ".js",
    "Rust": ".rs",
    "Ruby": ".rb",
}

_TREE_SITTER_LANG = {
    "JavaScript": "javascript",
    "Rust": "rust",
    "Ruby": "ruby",
}

# Subprocess fallback when tree_sitter_language_pack is not installed.
# Maps language -> (runtime_binary, build_command_fn, file_extension).
# node --check: parses JS without executing it.
# ruby -c: syntax-check only, no execution.
# rustfmt: re-formats using rustc's parser; exits non-zero on parse errors.
_SUBPROCESS_CHECKER: Dict[str, Tuple[str, object, str]] = {
    "JavaScript": ("node",    lambda f: ["node", "--check", f],               ".js"),
    "Ruby":       ("ruby",    lambda f: ["ruby", "-c", f],                    ".rb"),
    "Rust":       ("rustfmt", lambda f: ["rustfmt", "--edition", "2021", f],  ".rs"),
}

_thread_local = threading.local()


def _check_syntax_subprocess(code: str, language: str) -> Tuple[Optional[bool], Optional[str]]:
    """Fallback syntax check via installed language runtimes (no Python deps)."""
    entry = _SUBPROCESS_CHECKER.get(language)
    if entry is None:
        return None, None
    binary, make_cmd, ext = entry
    if not shutil.which(binary):
        return None, None

    with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False,
                                     encoding="utf-8") as fh:
        fh.write(code)
        tmp = fh.name
    try:
        proc = subprocess.run(
            make_cmd(tmp), capture_output=True, text=True, timeout=15, check=False
        )
        if proc.returncode == 0:
            return True, None
        raw = (proc.stderr or proc.stdout or "").strip()
        lines = [l for l in raw.splitlines() if l.strip()]
        # node prints the file path first; the actual error is on the
        # "SyntaxError:" line.  For ruby/rustfmt the first line is already
        # the human-readable message.
        error_line = next(
            (l for l in lines if "Error" in l or "error" in l),
            lines[0] if lines else "syntax error",
        )
        return False, error_line.strip()
    except subprocess.TimeoutExpired:
        return None, None
    except Exception:
        return None, None
    finally:
        Path(tmp).unlink(missing_ok=True)


def _get_parser(language: str):
    # Tree-sitter's native Parser is not Send: it must be created and dropped
    # on the same thread, so each worker thread gets its own cache.
    cache = getattr(_thread_local, "parsers", None)
    if cache is None:
        cache = {}
        _thread_local.parsers = cache
    if language not in cache:
        from tree_sitter_language_pack import get_parser  # local import: optional dependency

        cache[language] = get_parser(_TREE_SITTER_LANG[language])
    return cache[language]


def _root_node(tree):
    root = tree.root_node
    return root() if callable(root) else root


def _has_error(node) -> bool:
    value = node.has_error
    return bool(value() if callable(value) else value)


def is_semgrep_available() -> bool:
    return shutil.which("semgrep") is not None


def _first_error_position(node) -> Optional[str]:
    """Return a human-readable position string for the first ERROR node, or None."""
    value = node.has_error
    if not (value() if callable(value) else value):
        return None
    # BFS to find the first node that is itself an error (not just has a descendant error).
    queue = [node]
    while queue:
        current = queue.pop(0)
        is_err = getattr(current, "is_error", False)
        if callable(is_err):
            is_err = is_err()
        if is_err:
            start = getattr(current, "start_point", None)
            if start:
                row, col = (start[0] + 1, start[1] + 1) if isinstance(start, (list, tuple)) else (start.row + 1, start.column + 1)
                return f"line {row}, col {col}"
            return "unknown position"
        queue.extend(getattr(current, "children", []))
    return "unknown position"


def check_syntax(code: str, language: str) -> Tuple[Optional[bool], Optional[str]]:
    """Check whether `code` is syntactically valid for `language`.

    Returns a ``(valid, error_msg)`` tuple:
      (True,  None)    — valid
      (False, str)     — invalid; ``error_msg`` describes the first error
      (None,  None)    — not checkable (empty/whitespace input)
    """
    if not code or not code.strip():
        return None, None

    if language == "Python":
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            msg = f"line {e.lineno}: {e.msg}"
            if e.text and e.text.strip():
                msg += f" — {e.text.strip()!r}"
            return False, msg
        except Exception as e:
            return False, str(e)

    if language in _TREE_SITTER_LANG:
        try:
            parser = _get_parser(language)
        except ImportError:
            # tree_sitter_language_pack not installed — try subprocess fallback
            # (node / ruby / rustfmt) before giving up.
            return _check_syntax_subprocess(code, language)
        try:
            try:
                tree = parser.parse(code)
            except TypeError:
                tree = parser.parse(code.encode("utf-8"))
            root = _root_node(tree)
            if _has_error(root):
                pos = _first_error_position(root)
                return False, f"parse error at {pos}"
            return True, None
        except Exception as e:
            return False, str(e)

    return None, None


def run_semgrep_batch(
    samples: List[Tuple[int, str]],
    language: str,
    *,
    timeout_seconds: int = 300,
) -> Optional[Dict[int, List[dict]]]:
    """Run one semgrep scan over all `samples` for a single language.

    `samples` is a list of (index, code) pairs; empty code strings are
    skipped. Returns a mapping from index -> list of findings for every
    index that had non-empty code, or None if semgrep failed/timed out for
    the whole batch (caller should treat smell data as unavailable, not
    zero).
    """
    config = LANG_TO_SEMGREP_CONFIG.get(language)
    extension = LANG_TO_EXTENSION.get(language)
    if config is None or extension is None:
        return None

    non_empty = [(idx, code) for idx, code in samples if code and code.strip()]
    results: Dict[int, List[dict]] = {idx: [] for idx, _ in samples}
    if not non_empty:
        return results

    with tempfile.TemporaryDirectory(prefix="semgrep_batch_") as tmpdir:
        tmp_path = Path(tmpdir)
        for idx, code in non_empty:
            (tmp_path / f"{idx}{extension}").write_text(code, encoding="utf-8")

        cmd = ["semgrep", "--json", "--quiet", "--metrics=off", "--config", config, str(tmp_path)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired:
            return None

        if proc.returncode not in (0, 1):
            return None

        try:
            payload = json.loads(proc.stdout) if proc.stdout.strip() else {"results": []}
        except json.JSONDecodeError:
            return None

        for item in payload.get("results", []):
            path = Path(item.get("path", ""))
            try:
                idx = int(path.stem)
            except ValueError:
                continue
            if idx not in results:
                continue
            results[idx].append(
                {
                    "rule": item.get("check_id"),
                    "severity": item.get("extra", {}).get("severity"),
                    "message": item.get("extra", {}).get("message"),
                    "line": item.get("start", {}).get("line"),
                }
            )

    return results
