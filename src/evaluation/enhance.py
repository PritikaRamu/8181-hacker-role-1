"""Shared implementation for enhancing generation-result corpora in place.

Both `output/` (produced by `scripts/generate.py`) and `reformat_results/`
(the pre-generated legacy corpus) share the same
`<strategy>/<task>/<model>/<Language>.json` layout and
`{"packages": [{"answer": ..., ...}, ...]}` schema, so `scripts/evaluate.py`
and `scripts/enhance_reformat_results.py` are both thin CLI wrappers around
this module.

Each sample dict gains:
    extracted_packages, extracted_count,
    registry_valid_count, stdlib_count, hallucinated_count, hallucination_rate
    (code task only) syntax_valid, smell_issues, smell_count

Each file gains a top-level "stats" dict with the same metrics aggregated
across all of its samples, plus its own strategy/task/model/language.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evaluation import code_quality, extraction
from evaluation.registries import DEFAULT_PACKAGE_LIST_DIR, PackageRegistry

KNOWN_TASKS = {"packages", "code"}


def discover_files(
    root: Path,
    *,
    strategies: Optional[List[str]],
    tasks: Optional[List[str]],
    models: Optional[List[str]],
    languages: Optional[List[str]],
) -> List[Tuple[str, str, str, str, Path]]:
    """Return (strategy, task, model, language, path) tuples matching the filters."""
    found: List[Tuple[str, str, str, str, Path]] = []
    for strategy_dir in sorted(root.iterdir()):
        if not strategy_dir.is_dir():
            continue
        if strategies and strategy_dir.name not in strategies:
            continue
        for task_dir in sorted(strategy_dir.iterdir()):
            if not task_dir.is_dir() or task_dir.name not in KNOWN_TASKS:
                continue
            if tasks and task_dir.name not in tasks:
                continue
            for model_dir in sorted(task_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                if models and model_dir.name not in models:
                    continue
                for path in sorted(model_dir.glob("*.json")):
                    language = path.stem
                    if languages and language not in languages:
                        continue
                    found.append((strategy_dir.name, task_dir.name, model_dir.name, language, path))
    return found


def _classify_counts(packages: List[str], language: str, registry: PackageRegistry) -> Dict[str, int]:
    counts = {"registry_valid_count": 0, "stdlib_count": 0, "hallucinated_count": 0}
    for pkg in packages:
        cls = registry.classify(pkg, language)
        if cls == "registry":
            counts["registry_valid_count"] += 1
        elif cls == "stdlib":
            counts["stdlib_count"] += 1
        else:
            counts["hallucinated_count"] += 1
    return counts


def process_file(
    strategy: str,
    task: str,
    model: str,
    language: str,
    path: Path,
    *,
    registry: PackageRegistry,
    skip_syntax: bool,
    skip_smell: bool,
    semgrep_timeout: int,
    dry_run: bool,
) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("packages", [])
    is_code = task == "code" and not skip_syntax

    quality_targets: List[Tuple[int, str]] = []

    for idx, sample in enumerate(samples):
        answer = sample.get("answer", sample.get("response", ""))
        packages = extraction.extract_packages(answer, language=language, task=task)
        sample["extracted_packages"] = packages
        sample["extracted_count"] = len(packages)
        counts = _classify_counts(packages, language, registry)
        sample.update(counts)
        sample["hallucination_rate"] = (
            counts["hallucinated_count"] / len(packages) if packages else 0.0
        )

        if is_code:
            code = extraction.extract_code_for_quality(answer, language=language)
            valid, error_msg = code_quality.check_syntax(code, language)
            sample["syntax_valid"] = valid
            sample["syntax_error_msg"] = error_msg
            if not skip_smell and code and code.strip():
                quality_targets.append((idx, code))

    if is_code and not skip_smell and quality_targets:
        findings_map = code_quality.run_semgrep_batch(
            quality_targets, language, timeout_seconds=semgrep_timeout
        )
        for idx, _code in quality_targets:
            if findings_map is None:
                samples[idx]["smell_issues"] = None
                samples[idx]["smell_count"] = None
            else:
                findings = findings_map.get(idx, [])
                samples[idx]["smell_issues"] = findings
                samples[idx]["smell_count"] = len(findings)

    stats = compute_file_stats(samples, code_task=is_code)
    stats.update({"strategy": strategy, "task": task, "model": model, "language": language})
    data["stats"] = stats

    if not dry_run:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return stats


def compute_file_stats(samples: List[Dict[str, Any]], *, code_task: bool) -> Dict[str, Any]:
    n_samples = len(samples)
    n_extracted = sum(s.get("extracted_count", 0) for s in samples)
    n_registry = sum(s.get("registry_valid_count", 0) for s in samples)
    n_stdlib = sum(s.get("stdlib_count", 0) for s in samples)
    n_hallucinated = sum(s.get("hallucinated_count", 0) for s in samples)

    n_valid_with_stdlib = n_registry + n_stdlib
    n_valid_without_stdlib = n_registry

    def pct_hallucinated(n_valid: int) -> float:
        if n_extracted == 0:
            return 0.0
        return round(100.0 * (n_extracted - n_valid) / n_extracted, 2)

    phr_with_stdlib = pct_hallucinated(n_valid_with_stdlib)
    phr_without_stdlib = pct_hallucinated(n_valid_without_stdlib)

    # Mean per-sample PHR: average hallucination_rate over all samples.
    # Samples that produced no packages contribute 0 (hallucination_rate=0).
    sample_phrs = [s["hallucination_rate"] * 100 for s in samples]
    mean_phr_per_sample = round(sum(sample_phrs) / len(sample_phrs), 2) if sample_phrs else 0.0

    stats: Dict[str, Any] = {
        "n_samples": n_samples,
        "n_extracted_packages": n_extracted,
        "n_registry_valid": n_registry,
        "n_stdlib": n_stdlib,
        "n_hallucinated": n_hallucinated,
        "phr_with_stdlib": phr_with_stdlib,
        "phr_without_stdlib": phr_without_stdlib,
        "mean_phr_per_sample": mean_phr_per_sample,
        "delta": round(phr_without_stdlib - phr_with_stdlib, 2),
    }

    if code_task:
        # Exclude None (truncated / no code produced) — only score samples where
        # a syntax verdict could actually be reached.
        syntax_flags = [s["syntax_valid"] for s in samples
                        if s.get("syntax_valid") is not None]
        if syntax_flags:
            stats["syntax_valid_rate"] = round(sum(syntax_flags) / len(syntax_flags), 4)
        stats["syntax_excluded"] = sum(
            1 for s in samples if s.get("syntax_valid") is None
        )

        # Collect and rank error messages from invalid samples for diagnostics.
        error_counts: Dict[str, int] = {}
        for s in samples:
            if s.get("syntax_valid") is False and s.get("syntax_error_msg"):
                msg = s["syntax_error_msg"]
                error_counts[msg] = error_counts.get(msg, 0) + 1
        if error_counts:
            stats["top_syntax_errors"] = [
                {"msg": msg, "count": cnt}
                for msg, cnt in sorted(error_counts.items(), key=lambda x: -x[1])[:5]
            ]

        smell_counts = [s["smell_count"] for s in samples if s.get("smell_count") is not None]
        if smell_counts:
            stats["mean_smell_count"] = round(sum(smell_counts) / len(smell_counts), 4)
            stats["total_smell_issues"] = sum(smell_counts)

    return stats


def build_arg_parser(*, default_root: Path, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(default_root), help="Root of the result tree to enhance.")
    parser.add_argument("--package-list-dir", default=str(DEFAULT_PACKAGE_LIST_DIR))
    parser.add_argument("--strategies", nargs="+", default=None, help="Only process these strategy directories.")
    parser.add_argument("--tasks", nargs="+", choices=sorted(KNOWN_TASKS), default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=4, help="Parallel files processed at once.")
    parser.add_argument("--semgrep-timeout", type=int, default=300, help="Per-file semgrep timeout (seconds).")
    parser.add_argument("--skip-syntax", action="store_true", help="Skip syntax validity checks (code task).")
    parser.add_argument("--skip-smell", action="store_true", help="Skip semgrep smell detection (code task).")
    parser.add_argument("--dry-run", action="store_true", help="Compute stats but do not write files.")
    return parser


def run(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    if not args.skip_smell and not code_quality.is_semgrep_available():
        print("Warning: `semgrep` not found on PATH; smell findings will be null. "
              "Install it (pip install semgrep) or pass --skip-smell.", file=sys.stderr)

    registry = PackageRegistry(args.package_list_dir)

    files = discover_files(
        root,
        strategies=args.strategies,
        tasks=args.tasks,
        models=args.models,
        languages=args.languages,
    )
    if not files:
        raise SystemExit("No matching result files found.")

    print(f"Processing {len(files)} file(s) from {root} ...")

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **_kwargs):
            return it

    all_stats: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_file,
                strategy, task, model, language, path,
                registry=registry,
                skip_syntax=args.skip_syntax,
                skip_smell=args.skip_smell,
                semgrep_timeout=args.semgrep_timeout,
                dry_run=args.dry_run,
            ): (strategy, task, model, language)
            for strategy, task, model, language, path in files
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            strategy, task, model, language = futures[future]
            try:
                stats = future.result()
            except Exception as exc:
                print(f"[ERROR] {strategy}/{task}/{model}/{language}: {exc}", file=sys.stderr)
                continue
            all_stats.append(stats)
            line = (
                f"  {strategy:<20} {task:<10} {model:<16} {language:<12} "
                f"n={stats['n_samples']:<5} phr_with_stdlib={stats['phr_with_stdlib']:>6.2f}% "
                f"phr_without_stdlib={stats['phr_without_stdlib']:>6.2f}%"
            )
            if task == "code":
                if "syntax_valid_rate" in stats:
                    line += f" syntax_valid={stats['syntax_valid_rate'] * 100:>6.2f}%"
                if "mean_smell_count" in stats:
                    line += f" mean_smell={stats['mean_smell_count']:.3f}"
            print(line)

    total_samples = sum(s["n_samples"] for s in all_stats)
    total_extracted = sum(s["n_extracted_packages"] for s in all_stats)
    total_hallucinated = sum(s["n_hallucinated"] for s in all_stats)
    print(
        f"\nDone. {len(all_stats)}/{len(files)} files processed, "
        f"{total_samples} samples, {total_extracted} packages extracted, "
        f"{total_hallucinated} true hallucinations."
    )
    if args.dry_run:
        print("(--dry-run: no files were modified)")
    return 0
