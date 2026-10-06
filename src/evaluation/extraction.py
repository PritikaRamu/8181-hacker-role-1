"""Extract candidate package names from raw model responses.

The `reformat_results/` corpus is heterogeneous:

- Some rows store the *entire* rendered chat prompt plus completion
  (`"system: ...user: ...assistant: <reply>user: <hallucinated follow-up>..."`),
  because the smaller instruct models happily keep inventing extra turns
  once they run out of real content.
- Other rows (e.g. self_refine, nudging) already store just the final,
  cleaned reply.
- Byte-level BPE artifacts (`Ġ` for space, `Ċ`/`ģ` for newline) leak into
  some decoded strings and must be normalised before any parsing.
- The "packages" task asks for a bracketed list like `[requests, flask]`,
  but weaker models sometimes ignore the format and answer in prose with a
  code block instead - in which case falling back to import-parsing that
  code block recovers a usable package list.

Everything here is self-contained (no dependency on the legacy `Compare`
project) and works on the model's raw text.
"""

from __future__ import annotations

import ast
import re
from typing import List, Optional

LANGUAGES = ["Python", "JavaScript", "Rust", "Ruby"]

# Aliases a fenced code block might use for a given language.
_LANGUAGE_FENCE_ALIASES = {
    "Python": ["python", "py"],
    "JavaScript": ["javascript", "js", "jsx", "typescript", "ts"],
    "Rust": ["rust", "rs"],
    "Ruby": ["ruby", "rb"],
}

# Tokens that show up inside brackets but are not package names: markdown
# fence tags the model echoed literally (e.g. `[code]...[/code]`), or the
# placeholder example from the system prompt itself.
_TOKEN_STOPLIST = {
    "code", "/code", "block", "output", "result", "results", "text",
    "plaintext", "bash", "shell", "sh", "json", "xml", "html", "css", "sql",
    "yaml", "yml", "markdown", "example", "examples", "pkg", "module",
    "library", "libraries", "package", "packages",
    "package1", "package2", "package3", "packagename",
    "python", "javascript", "js", "typescript", "ts", "rust", "rs", "ruby", "rb",
}

# A plausible package/module/crate/gem name: no whitespace, reasonable length,
# allows the punctuation real registry names use (@scope/name, dashes, dots).
_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9@][A-Za-z0-9_.\-/@]{0,98}$")

_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")

# Matches an opening code fence (``` optionally followed by a language tag and newline).
_OPEN_FENCE_RE = re.compile(r"```+[A-Za-z0-9_+-]*\s*\n")

# Matches empty fences like ```python``` or ```javascript``` with no content
# between the opening and closing backticks.  These are instruction-echo
# artefacts ("respond with code inside a ```python``` block") that appear in
# BPE-stripped answers and confuse the fence-extraction regexes.
_EMPTY_FENCE_RE = re.compile(r"```+[A-Za-z0-9_+-]*```+")


# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------

def clean_text(text: Optional[str]) -> str:
    """Normalise BPE artifacts and a few known decoding glitches."""
    if not isinstance(text, str) or not text:
        return ""
    cleaned = text.replace("Ġ", " ").replace("Ċ", "\n").replace("ģ", "\n")
    cleaned = _EMPTY_FENCE_RE.sub("", cleaned)
    cleaned = re.sub(r"(?m)^##\s*\[\d{2}:\d{2}:\d{2}]\s*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def isolate_reply(text: str) -> str:
    """Isolate the model's first real reply from a possibly full chat transcript.

    Two shapes show up in the data:
    1. The full rendered prompt is echoed (`system: ...user: ...assistant: <reply>`):
       take the text after the first "assistant:" marker.
    2. Only the reply (and possibly a self-invented follow-up turn) is stored:
       there is no leading "system:" prefix, so the reply starts at position 0.

    In both cases, cut off anything from the first subsequent "user:" marker
    onward, since that marks the start of a follow-up turn the model
    hallucinated on its own.
    """
    if not text:
        return ""

    if text.lstrip().lower().startswith("system:"):
        # Find "assistant:" that opens a line, not one embedded in text/code.
        asst = re.search(r"(?m)^\s*assistant\s*:", text, re.IGNORECASE)
        if asst:
            text = text[asst.end():]
        else:
            # No line-starting assistant: marker found.  This happens with
            # BPE-space-stripped answers where all whitespace was removed —
            # the entire string is one long line and we can't identify where
            # the assistant reply begins.  Returning "" signals callers to
            # treat the sample as non-evaluable (syntax_valid=None).
            return ""

    # Only cut at "user:" that opens a line (turn marker), not inside code —
    # e.g. type annotations `user: str`, dict keys `{"user": …}`, or URLs
    # `http://user:pass@host` would all be false positives with a plain find().
    turn_marker = re.search(r"(?m)^\s*user\s*:", text, re.IGNORECASE)
    if turn_marker:
        text = text[: turn_marker.start()]

    return text.strip()


# ---------------------------------------------------------------------------
# Bracket-list extraction (packages task)
# ---------------------------------------------------------------------------

def _split_bracket_content(content: str) -> List[str]:
    candidate = "[" + content + "]"
    try:
        parsed = ast.literal_eval(candidate)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [p.strip().strip("'\"") for p in content.split(",") if p.strip()]


def extract_bracket_list(text: str) -> List[str]:
    """Collect plausible package names out of every `[...]` group in `text`."""
    tokens: List[str] = []
    seen = set()
    for match in _BRACKET_RE.finditer(text):
        for token in _split_bracket_content(match.group(1)):
            key = token.lower()
            if key in _TOKEN_STOPLIST:
                continue
            if not _TOKEN_PATTERN.match(token):
                continue
            if key in seen:
                continue
            seen.add(key)
            tokens.append(token)
    return tokens


def _extract_plain_list(text: str) -> List[str]:
    """Last-resort parser for comma- or newline-separated bare package names.

    Handles RAG-style output such as "requests, httpx, http3" or numbered
    lists like "1. requests\\n2. httpx\\n3. aiohttp".  Only tokens that match
    the strict package-name pattern and are not in the stop-list are kept, so
    long prose sentences are naturally filtered out.
    """
    out: List[str] = []
    seen: set = set()
    for tok in re.split(r"[,\n]+", text):
        tok = re.sub(r"^\s*\d+[.)]\s*", "", tok)   # strip leading "1." / "1)"
        tok = tok.strip().strip("`").strip("*").strip("'\"").strip()
        if not tok:
            continue
        key = tok.lower()
        if key in _TOKEN_STOPLIST:
            continue
        if not _TOKEN_PATTERN.match(tok):
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Code-fence extraction (code task, and packages-task fallback)
# ---------------------------------------------------------------------------

def extract_code_fence(text: str, language: str) -> Optional[str]:
    """Pull code out of a fenced block, tolerant of the glued ```lang``` style
    seen in this corpus (opener and closing backticks with no newline).

    Handles truncated responses (no closing ```) by extracting everything after
    the opening fence line, so that code cut off at max_new_tokens is not
    misclassified as a syntax error due to leftover fence markers.
    """
    aliases = _LANGUAGE_FENCE_ALIASES.get(language, [language.lower()])

    # Closed fence — language-specific aliases.
    for alias in aliases:
        pattern = re.compile(
            r"```+\s*" + re.escape(alias) + r"\s*```*(.*?)```", re.IGNORECASE | re.DOTALL
        )
        match = pattern.search(text)
        if match:
            return match.group(1)

    # Closed fence — generic (any language tag or none).
    generic = re.search(r"```[A-Za-z0-9_+-]*\s*(.*?)```", text, re.DOTALL)
    if generic:
        return generic.group(1)

    # Unclosed fence — model was cut off before the closing ```.
    # Try language-specific opener first, then any opener.
    for alias in aliases:
        open_pat = re.compile(
            r"```+\s*" + re.escape(alias) + r"[^\n]*\n(.*)", re.IGNORECASE | re.DOTALL
        )
        match = open_pat.search(text)
        if match:
            return match.group(1).rstrip()

    open_generic = re.search(r"```[A-Za-z0-9_+-]*\n(.*)", text, re.DOTALL)
    if open_generic:
        return open_generic.group(1).rstrip()

    # A handful of responses use a non-markdown `[code]...[/code]` fence instead.
    bracket_fence = re.search(r"\[code\](.*?)\[/code\]", text, re.IGNORECASE | re.DOTALL)
    if bracket_fence:
        return bracket_fence.group(1)

    return None


# ---------------------------------------------------------------------------
# Per-language import/require parsing
# ---------------------------------------------------------------------------

def _strip_magic_lines(code: str) -> str:
    return "\n".join(
        ln for ln in code.splitlines() if not ln.lstrip().startswith(("!", "%", "%%"))
    )


def get_python_packages(code: str) -> List[str]:
    cleaned = _strip_magic_lines(code)
    try:
        tree = ast.parse(cleaned)
    except Exception:
        out: List[str] = []
        for m in re.finditer(r"^\s*import\s+([A-Za-z_][\w.]*)", cleaned, flags=re.M):
            out.append(m.group(1).split(".")[0])
        for m in re.finditer(r"^\s*from\s+([A-Za-z_][\w.]*)\s+import", cleaned, flags=re.M):
            out.append(m.group(1).split(".")[0])
        return out

    packages: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                packages.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                packages.append(node.module.split(".")[0])
    return packages


def get_javascript_packages(code: str) -> List[str]:
    out: List[str] = []
    for m in re.finditer(r"import\s+[^;]*?\s+from\s+['\"]([^'\"]+)['\"]", code):
        out.append(m.group(1))
    for m in re.finditer(r"^\s*import\s+['\"]([^'\"]+)['\"]", code, flags=re.M):
        out.append(m.group(1))
    for m in re.finditer(r"require\(\s*['\"]([^'\"]+)['\"]\s*\)", code):
        out.append(m.group(1))
    for m in re.finditer(r"\bimport\(\s*['\"]([^'\"]+)['\"]\s*\)", code):
        out.append(m.group(1))
    return [pkg.split("/")[0] if not pkg.startswith("@") else "/".join(pkg.split("/")[:2]) for pkg in out]


def get_rust_packages(code: str) -> List[str]:
    out: List[str] = []
    for m in re.finditer(r"\bextern\s+crate\s+([A-Za-z_]\w*)", code):
        out.append(m.group(1))
    for m in re.finditer(r"^\s*use\s+([A-Za-z_]\w*)\s*(?:::|;)", code, flags=re.M):
        out.append(m.group(1))
    return out


def get_ruby_packages(code: str) -> List[str]:
    out: List[str] = []
    for m in re.finditer(r"require\s*(?:_relative)?\s*['\"]([^'\"]+)['\"]", code):
        out.append(m.group(1).split("/")[0])
    return out


_PACKAGE_PARSERS = {
    "Python": get_python_packages,
    "JavaScript": get_javascript_packages,
    "Rust": get_rust_packages,
    "Ruby": get_ruby_packages,
}


def get_code_packages(code: str, language: str) -> List[str]:
    parser = _PACKAGE_PARSERS.get(language)
    if parser is None:
        return []
    seen = set()
    out = []
    for pkg in parser(code):
        pkg = pkg.strip()
        key = pkg.lower()
        if pkg and key not in seen:
            seen.add(key)
            out.append(pkg)
    return out


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def extract_packages(raw_answer: str, *, language: str, task: str) -> List[str]:
    """Extract the candidate package/import list for one sample.

    `task` is "packages" (model asked to answer with a bracketed list) or
    "code" (model asked to write a code snippet, from which we parse imports).
    """
    text = clean_text(raw_answer)
    if not text:
        return []
    reply = isolate_reply(text)
    if not reply:
        return []

    if task == "packages":
        pkgs = extract_bracket_list(reply)
        if pkgs:
            return pkgs
        # The model ignored the requested format and answered with code instead.
        code = extract_code_fence(reply, language)
        if code:
            return get_code_packages(code, language)
        # Last resort: bare comma/newline-separated list (RAG output style).
        return _extract_plain_list(reply)

    # task == "code"
    code = extract_code_fence(reply, language)
    if code is None:
        code = reply
    return get_code_packages(code, language)


def is_truncated(text: str) -> bool:
    """Return True if `text` contains an opening code fence with no closing fence.

    This indicates the model was cut off before finishing the code block
    (typically at max_new_tokens).  Such responses are excluded from syntax
    checking (syntax_valid → None) rather than being misclassified as errors.
    """
    opens = list(_OPEN_FENCE_RE.finditer(text))
    if not opens:
        return False
    # Everything after the last opening fence — if no ``` appears there, it's unclosed.
    return "```" not in text[opens[-1].end():]


def extract_code_for_quality(raw_answer: str, *, language: str) -> str:
    """Return the best-effort code string to run syntax/smell checks on.

    Returns an empty string — which check_syntax() maps to (None, None) and
    excludes from analysis — in any of these cases:
      • the reply cannot be isolated (BPE-stripped or malformed transcript)
      • the response contains an unclosed code fence (truncated at max_new_tokens)
      • no fenced code block is present (model generated prose instead of code)
    """
    text = clean_text(raw_answer)
    reply = isolate_reply(text)
    if not reply or is_truncated(reply):
        return ""
    code = extract_code_fence(reply, language)
    return code if code is not None else ""
