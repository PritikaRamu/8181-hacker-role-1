#!/usr/bin/env python3
"""
Analyze evaluation results and produce plots and a statistical report.

Reads the per-file "stats" dicts written into output/*.json by
scripts/evaluate.py (no separate CSV) and writes:
  results/plots/phr_by_model_language.pdf
  results/plots/phr_by_strategy.pdf
  results/plots/stdlib_delta.pdf
  results/plots/valid_vs_hallucinated_packages_by_language_and_model_all_packages.pdf
  results/plots/valid_vs_hallucinated_packages_by_language_and_model.pdf
  results/plots/valid_vs_hallucinated_packages_by_language_and_method_code.pdf
  results/plots/valid_vs_hallucinated_packages_by_model_and_method_code.pdf
  results/plots/syntax_error_by_model_and_strategy.pdf
  results/plots/code_smell_by_model_and_strategy.pdf
  results/plots/syntax_error_heatmap_by_model_strategy_language.pdf
  results/plots/smell_heatmap_by_model_strategy_language.pdf

Usage
-----
    python scripts/analyze.py
    python scripts/analyze.py --results-dir output
    python scripts/analyze.py --no-plots       # report only
    python scripts/analyze.py --print-report   # print statistical report to stdout
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.enhance import discover_files  # noqa: E402

LANGUAGES = ["Python", "JavaScript", "Ruby", "Rust"]
STRATEGY_DISPLAY = {
    "baseline": "Vanilla",
    "greedy_decoding": "Greedy",
    "self_refine": "Self-Refine",
    "dola": "DoLa",
    "rag": "RAG",
    "nudging": "Nudging",
    "contrastive_decoding": "Contrastive",
    "actlcd": "ActLCD",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _backfill_syntax_errors(root: Path) -> int:
    """Walk every code-task JSON file under ``root`` and:

    1. Fill in ``syntax_error_msg`` for samples with ``syntax_valid=False``
       but a null/absent message (newly added field from a prior run).
    2. Correct samples that were incorrectly marked ``syntax_valid=False``
       solely because ``tree_sitter_language_pack`` was not installed: reset
       them to ``syntax_valid=None, syntax_error_msg=None`` (not checkable).

    Rewrites files in-place and refreshes ``stats.top_syntax_errors``.
    Returns the total number of samples updated.
    """
    from evaluation import code_quality, extraction  # local import: heavy deps

    files = discover_files(root, strategies=None, tasks=["code"],
                           models=None, languages=None)
    total_updated = 0

    for _strategy, _task, _model, language, path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        samples = data.get("packages", [])

        updated = 0
        for sample in samples:
            if sample.get("syntax_valid") is not False:
                continue

            # Re-run extraction + check with the current pipeline so that both
            # syntax_valid and syntax_error_msg are always consistent.
            # This fixes: (1) tree_sitter missing-dep artefacts, (2) null
            # messages from older runs, (3) BPE-space-stripped answers that
            # previously produced garbage error messages.
            answer = sample.get("answer", "")
            code = extraction.extract_code_for_quality(answer, language=language)
            new_valid, new_msg = code_quality.check_syntax(code, language)
            sample["syntax_valid"] = new_valid
            sample["syntax_error_msg"] = new_msg
            updated += 1

        # Always recompute syntax stats — a prior run may have fixed sample-level
        # syntax_valid fields but left a stale syntax_valid_rate in the stats
        # block (e.g. 0.0 from when all samples were False, which renders as
        # 100% error in the heatmap even though all samples are now excluded).
        stats_changed = False
        if "stats" in data:
            syntax_flags = [s["syntax_valid"] for s in samples
                            if s.get("syntax_valid") is not None]
            new_rate = (round(sum(syntax_flags) / len(syntax_flags), 4)
                        if syntax_flags else None)
            if new_rate != data["stats"].get("syntax_valid_rate"):
                stats_changed = True
                if new_rate is not None:
                    data["stats"]["syntax_valid_rate"] = new_rate
                else:
                    data["stats"].pop("syntax_valid_rate", None)

            new_excluded = sum(1 for s in samples if s.get("syntax_valid") is None)
            if new_excluded != data["stats"].get("syntax_excluded", 0):
                stats_changed = True
                data["stats"]["syntax_excluded"] = new_excluded

            error_counts: Dict[str, int] = {}
            for s in samples:
                if s.get("syntax_valid") is False and s.get("syntax_error_msg"):
                    msg = s["syntax_error_msg"]
                    error_counts[msg] = error_counts.get(msg, 0) + 1
            new_top = ([{"msg": m, "count": c}
                        for m, c in sorted(error_counts.items(), key=lambda x: -x[1])[:5]]
                       if error_counts else [])
            if new_top != data["stats"].get("top_syntax_errors", []):
                stats_changed = True
                data["stats"]["top_syntax_errors"] = new_top

        if updated or stats_changed:
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            total_updated += updated

    return total_updated


def load_result_dir(root: Path) -> List[Dict[str, Any]]:
    """Collect the "stats" dict written by scripts/evaluate.py into each
    output/<strategy>/<task>/<model>/<Language>.json, remapped onto the
    field names the report/table/plot functions below expect.
    """
    if not root.is_dir():
        raise SystemExit(f"Results directory not found: {root}\nRun scripts/evaluate.py first.")

    files = discover_files(root, strategies=None, tasks=None, models=None, languages=None)
    if not files:
        raise SystemExit(f"No result files found under {root}.")

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for strategy, task, model, language, path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        stats = data.get("stats")
        if not stats:
            skipped += 1
            continue
        rows.append({
            "task": task,
            "strategy": strategy,
            "model": model,
            "language": language,
            "n_generated": stats["n_extracted_packages"],
            "n_valid_with_stdlib": stats["n_registry_valid"] + stats["n_stdlib"],
            "n_valid_without_stdlib": stats["n_registry_valid"],
            "phr_with_stdlib": stats["phr_with_stdlib"],
            "phr_without_stdlib": stats["phr_without_stdlib"],
            "mean_phr_per_sample": stats.get("mean_phr_per_sample"),
            "delta": stats["delta"],
            "n_records": stats["n_samples"],
            "syntax_valid_rate": stats.get("syntax_valid_rate"),
            "syntax_excluded": stats.get("syntax_excluded"),
            "top_syntax_errors": stats.get("top_syntax_errors", []),
            "mean_smell_count": stats.get("mean_smell_count"),
            "total_smell_issues": stats.get("total_smell_issues"),
        })

    if skipped:
        print(f"Warning: {skipped} file(s) have no \"stats\" yet — run scripts/evaluate.py to fill them in.")
    if not rows:
        raise SystemExit(f"No enhanced result files found under {root}.\nRun scripts/evaluate.py first.")
    return rows


# ---------------------------------------------------------------------------
# Statistical report
# ---------------------------------------------------------------------------

def print_report(data: List[Dict]) -> None:

    def section(title: str) -> None:
        print(f"\n{'='*80}\n{title}\n{'='*80}")

    def subsection(title: str) -> None:
        print(f"\n{title}\n{'-'*60}")

    # --- Baseline performance ---
    section("BASELINE PERFORMANCE")
    baseline = [r for r in data if r["strategy"] == "baseline"]

    by_lang: Dict[str, List[float]] = defaultdict(list)
    by_model: Dict[str, List[float]] = defaultdict(list)
    for r in baseline:
        by_lang[r["language"]].append(r["phr_with_stdlib"])
        by_model[r["model"]].append(r["phr_with_stdlib"])

    subsection("PHR by Language")
    for lang in LANGUAGES:
        phrs = by_lang.get(lang, [])
        if not phrs:
            continue
        print(f"  {lang:<15} mean={statistics.mean(phrs):.1f}%  "
              f"min={min(phrs):.1f}%  max={max(phrs):.1f}%"
              + (f"  std={statistics.stdev(phrs):.1f}%" if len(phrs) > 1 else ""))

    subsection("PHR by Model")
    for model in sorted(by_model):
        phrs = by_model[model]
        print(f"  {model:<35} mean={statistics.mean(phrs):.1f}%  "
              f"range=[{min(phrs):.1f}%, {max(phrs):.1f}%]")

    # --- Strategy effectiveness ---
    section("STRATEGY EFFECTIVENESS")
    by_strat: Dict[str, Dict] = defaultdict(lambda: {"gen": 0, "valid": 0})
    for r in data:
        s = r["strategy"]
        by_strat[s]["gen"] += r["n_generated"]
        by_strat[s]["valid"] += r["n_valid_with_stdlib"]

    baseline_phr = None
    results_strat = []
    for s, acc in by_strat.items():
        phr = 100.0 * (acc["gen"] - acc["valid"]) / acc["gen"] if acc["gen"] else 0
        results_strat.append((s, acc["gen"], acc["valid"], phr))
        if s == "baseline":
            baseline_phr = phr

    print(f"\n{'Strategy':<25} {'N_gen':>10} {'N_valid':>10} {'PHR':>8} {'vs baseline':>12}")
    print("-" * 70)
    for s, n_gen, n_valid, phr in sorted(results_strat, key=lambda x: x[3]):
        vs = f"{baseline_phr - phr:+.1f}pp" if baseline_phr is not None and s != "baseline" else ""
        print(f"{s:<25} {n_gen:>10,} {n_valid:>10,} {phr:>8.1f}% {vs:>12}")

    # --- Standard library impact ---
    section("STANDARD LIBRARY CORRECTION IMPACT")
    by_lang_delta: Dict[str, List[float]] = defaultdict(list)
    for r in data:
        by_lang_delta[r["language"]].append(r["delta"])

    print(f"\n{'Language':<15} {'Mean Δ':>12} {'Max Δ':>12}")
    print("-" * 42)
    for lang in LANGUAGES:
        deltas = by_lang_delta.get(lang, [])
        if not deltas:
            continue
        print(f"{lang:<15} {statistics.mean(deltas):>11.2f}pp {max(deltas):>11.2f}pp")

    total_gen = sum(r["n_generated"] for r in data)
    if total_gen:
        global_delta = sum(r["delta"] * r["n_generated"] for r in data) / total_gen
        print(f"\nGlobal overestimation without stdlib correction: {global_delta:.2f}pp")

    # --- Language comparison ---
    section("LANGUAGE COMPARISON (baseline only)")
    for lang in LANGUAGES:
        phrs = by_lang.get(lang, [])
        if phrs:
            print(f"  {lang:<15} mean={statistics.mean(phrs):.1f}%")

    large = [statistics.mean(by_lang[l]) for l in ["Python", "JavaScript"] if by_lang.get(l)]
    small = [statistics.mean(by_lang[l]) for l in ["Ruby", "Rust"] if by_lang.get(l)]
    if large and small:
        print(f"\n  Large ecosystems (Py, JS):      {statistics.mean(large):.1f}% avg PHR")
        print(f"  Smaller ecosystems (Ruby, Rust): {statistics.mean(small):.1f}% avg PHR")
        print(f"  Difference: {statistics.mean(small) - statistics.mean(large):.1f}pp")

    # --- Code quality: syntax errors and smells by model and strategy ---
    section("CODE QUALITY (SYNTAX ERRORS & SMELLS) BY MODEL AND STRATEGY")
    code_rows = [r for r in data if r["task"] == "code" and r["syntax_valid_rate"] is not None]
    if not code_rows:
        print("  No code-task syntax/smell data found. Run scripts/evaluate.py without --skip-syntax.")
    else:
        by_model_strat: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        for r in code_rows:
            by_model_strat[(r["model"], r["strategy"])].append(r)

        subsection("Syntax valid rate & mean smell count per (model, strategy)")
        print(f"{'Model':<16} {'Strategy':<15} {'SyntaxErr%':>11} {'Excluded':>9} {'MeanSmells':>11} {'TotalSmells':>12}")
        print("-" * 80)
        for model, strat in sorted(by_model_strat):
            rows = by_model_strat[(model, strat)]
            syntax_err = 100.0 * (1 - statistics.mean(r["syntax_valid_rate"] for r in rows))
            excluded = sum(r.get("syntax_excluded") or 0 for r in rows)
            smell_rows = [r["mean_smell_count"] for r in rows if r["mean_smell_count"] is not None]
            mean_smell = statistics.mean(smell_rows) if smell_rows else float("nan")
            total_smell = sum(r["total_smell_issues"] or 0 for r in rows)
            print(f"{model:<16} {STRATEGY_DISPLAY.get(strat, strat):<15} "
                  f"{syntax_err:>10.1f}% {excluded:>9,} {mean_smell:>11.3f} {total_smell:>12,}")

        subsection("Top syntax error messages per (model, strategy, language)")
        for model, strat in sorted(by_model_strat):
            rows = by_model_strat[(model, strat)]
            errors_by_lang = {
                r["language"]: r.get("top_syntax_errors", [])
                for r in rows if r.get("top_syntax_errors")
            }
            if not errors_by_lang:
                continue
            print(f"\n  {model} / {STRATEGY_DISPLAY.get(strat, strat)}")
            for lang, errors in sorted(errors_by_lang.items()):
                print(f"    [{lang}]")
                for e in errors:
                    print(f"      {e['count']:>3}×  {e['msg']}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

LANG_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
LANG_HATCHES = ["", "////", "xx", "..."]
STRAT_HATCHES = ["", "////", "xx", "oo", "...", "++", "\\\\"]
VALID_COLOR = "steelblue"
HALLU_COLOR = "salmon"


def _style_axes(ax) -> None:
    ax.grid(True, axis="y", alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _draw_grouped_stacked_bars(ax, categories, series, x, width, colors=None, hatches=None):
    """Draw a stacked (valid + hallucinated) bar for each series at each x position.

    ``series`` maps series name -> (valid_values, hallu_values) aligned with ``x``.
    Returns the list of series names in draw order.
    """
    names = list(series.keys())
    for i, name in enumerate(names):
        valid_vals, hallu_vals = series[name]
        hatch = (hatches or STRAT_HATCHES)[i % len(hatches or STRAT_HATCHES)]
        ax.bar(
            x + i * width, valid_vals, width=width,
            label=f"{name} (valid)", color=VALID_COLOR, edgecolor="black", hatch=hatch,
        )
        ax.bar(
            x + i * width, hallu_vals, width=width, bottom=valid_vals,
            label=f"{name} (hallucinated)", color=HALLU_COLOR, edgecolor="black", hatch=hatch,
        )
    return names


def make_plots(data: List[Dict], plots_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib/numpy not available — skipping plots.")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 18})

    # --- Plot 1: baseline PHR by model × language ---
    baseline = [r for r in data if r["strategy"] == "baseline"]
    by_model: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in baseline:
        by_model[r["model"]][r["language"]] = r["phr_with_stdlib"]

    models = sorted(by_model)
    x = np.arange(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 2.5), 7))
    for i, (lang, color, hatch) in enumerate(zip(LANGUAGES, LANG_COLORS, LANG_HATCHES)):
        vals = [by_model[m].get(lang, 0) for m in models]
        ax.bar(x + i * width, vals, width, label=lang, color=color,
               edgecolor="black", hatch=hatch)
    ax.set_xlabel("Model", fontsize=22)
    ax.set_ylabel("PHR (%) with stdlib correction", fontsize=22)
    ax.set_title("Baseline Package Hallucination Rate\nby Model and Language", fontsize=26, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.split("__")[0] for m in models], rotation=25, ha="right", fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.legend(title="Language", fontsize=16, title_fontsize=18)
    ax.set_ylim(0, 100)
    _style_axes(ax)
    fig.tight_layout()
    out = plots_dir / "phr_by_model_language.pdf"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.relative_to(ROOT)}")

    # --- Plot 2: PHR by strategy (aggregated) ---
    by_strat: Dict[str, Dict] = defaultdict(lambda: {"gen": 0, "valid": 0})
    for r in data:
        by_strat[r["strategy"]]["gen"] += r["n_generated"]
        by_strat[r["strategy"]]["valid"] += r["n_valid_with_stdlib"]

    strat_labels, strat_phrs = [], []
    for s in sorted(by_strat):
        acc = by_strat[s]
        phr = 100.0 * (acc["gen"] - acc["valid"]) / acc["gen"] if acc["gen"] else 0
        strat_labels.append(STRATEGY_DISPLAY.get(s, s))
        strat_phrs.append(phr)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = [LANG_COLORS[i % len(LANG_COLORS)] for i in range(len(strat_labels))]
    bar_hatches = [STRAT_HATCHES[i % len(STRAT_HATCHES)] for i in range(len(strat_labels))]
    bars = ax.barh(strat_labels, strat_phrs, color=bar_colors, edgecolor="black")
    for bar, hatch in zip(bars, bar_hatches):
        bar.set_hatch(hatch)
    ax.set_xlabel("PHR (%) with stdlib correction", fontsize=22)
    ax.set_title("PHR by Mitigation Strategy\n(all models, all languages)", fontsize=26, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.tick_params(axis="both", labelsize=18)
    for bar, val in zip(bars, strat_phrs):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=16)
    _style_axes(ax)
    fig.tight_layout()
    out = plots_dir / "phr_by_strategy.pdf"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.relative_to(ROOT)}")

    # --- Plot 3: stdlib delta by language ---
    by_lang_delta: Dict[str, List[float]] = defaultdict(list)
    for r in data:
        by_lang_delta[r["language"]].append(r["delta"])

    lang_labels = [l for l in LANGUAGES if by_lang_delta.get(l)]
    lang_means = [statistics.mean(by_lang_delta[l]) for l in lang_labels]
    lang_hatches = [LANG_HATCHES[i % len(LANG_HATCHES)] for i in range(len(lang_labels))]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(lang_labels, lang_means, color=LANG_COLORS[:len(lang_labels)], edgecolor="black")
    for bar, hatch in zip(bars, lang_hatches):
        bar.set_hatch(hatch)
    ax.set_ylabel("Mean Δ PHR (pp)", fontsize=22)
    ax.set_title("Stdlib Correction Impact by Language\n(positive = overestimation without correction)",
                 fontsize=24, fontweight="bold")
    ax.tick_params(axis="both", labelsize=18)
    ax.axhline(0, color="black", linewidth=0.8)
    _style_axes(ax)
    fig.tight_layout()
    out = plots_dir / "stdlib_delta.pdf"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.relative_to(ROOT)}")

    # --- Plot 4/5: valid vs hallucinated packages by model x language (task=packages, baseline) ---
    pkg_baseline = [r for r in data if r["task"] == "packages" and r["strategy"] == "baseline"]
    by_model_lang: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for r in pkg_baseline:
        by_model_lang[r["model"]][r["language"]] = r

    pkg_models = sorted(by_model_lang)
    pkg_langs = [l for l in LANGUAGES if any(l in by_model_lang[m] for m in pkg_models)]
    if pkg_models and pkg_langs:
        x = np.arange(len(pkg_models))
        width = 0.8 / max(len(pkg_langs), 1)

        for suffix, title_suffix, out_name in [
            ("with_stdlib", "(With Standard Packages)", "valid_vs_hallucinated_packages_by_language_and_model_all_packages.pdf"),
            ("without_stdlib", "", "valid_vs_hallucinated_packages_by_language_and_model.pdf"),
        ]:
            series = {}
            for lang in pkg_langs:
                valid_vals, hallu_vals = [], []
                for m in pkg_models:
                    r = by_model_lang[m].get(lang)
                    if not r:
                        valid_vals.append(0)
                        hallu_vals.append(0)
                        continue
                    if suffix == "with_stdlib":
                        valid = r["n_valid_with_stdlib"]
                    else:
                        valid = r["n_valid_without_stdlib"]
                    valid_vals.append(valid)
                    hallu_vals.append(r["n_generated"] - valid)
                series[lang] = (np.array(valid_vals), np.array(hallu_vals))

            fig, ax = plt.subplots(figsize=(18, 8))
            cats = _draw_grouped_stacked_bars(ax, pkg_langs, series, x, width, hatches=LANG_HATCHES)
            ax.set_xticks(x + width * (len(cats) - 1) / 2)
            ax.set_xticklabels([m.split("__")[0] for m in pkg_models], rotation=45, ha="right", fontsize=20)
            ax.legend(title="Language", fontsize=16, title_fontsize=18, ncol=2)
            title = "Valid vs Hallucinated Packages by Language and Model"
            if title_suffix:
                title += f"\n{title_suffix}"
            ax.set_title(title, fontsize=26, fontweight="bold")
            ax.set_xlabel("Model", fontsize=22)
            ax.set_ylabel("Package Count", fontsize=22)
            ax.tick_params(axis="y", labelsize=18)
            _style_axes(ax)
            fig.tight_layout()
            out = plots_dir / out_name
            fig.savefig(out, dpi=600, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out.relative_to(ROOT)}")

    # --- Plot 6: valid vs hallucinated packages (code task) by language x strategy ---
    code_rows = [r for r in data if r["task"] == "code"]
    by_lang_strat: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for r in code_rows:
        by_lang_strat[r["language"]][r["strategy"]] = r

    code_langs = [l for l in LANGUAGES if l in by_lang_strat]
    code_strats = [s for s in STRATEGY_DISPLAY if any(s in by_lang_strat[l] for l in code_langs)]
    if code_langs and code_strats:
        x = np.arange(len(code_langs))
        width = 0.8 / max(len(code_strats), 1)
        series = {}
        for s in code_strats:
            valid_vals, hallu_vals = [], []
            for lang in code_langs:
                r = by_lang_strat[lang].get(s)
                if not r:
                    valid_vals.append(0)
                    hallu_vals.append(0)
                    continue
                valid_vals.append(r["n_valid_with_stdlib"])
                hallu_vals.append(r["n_generated"] - r["n_valid_with_stdlib"])
            series[STRATEGY_DISPLAY[s]] = (np.array(valid_vals), np.array(hallu_vals))

        fig, ax = plt.subplots(figsize=(18, 10))
        cats = _draw_grouped_stacked_bars(ax, code_langs, series, x, width)
        ax.set_xticks(x + width * (len(cats) - 1) / 2)
        ax.set_xticklabels(code_langs, rotation=25, ha="right", fontsize=20)
        ax.legend(
            title="Strategy", fontsize=14, title_fontsize=16,
            loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3,
        )
        ax.set_title(
            "Valid vs Hallucinated Packages by Language and\nMitigation Strategy (Code Generation)",
            fontsize=26, fontweight="bold",
        )
        ax.set_ylabel("Package Count", fontsize=22)
        ax.tick_params(axis="y", labelsize=18)
        _style_axes(ax)
        fig.tight_layout()
        out = plots_dir / "valid_vs_hallucinated_packages_by_language_and_method_code.pdf"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.relative_to(ROOT)}")

    # --- Plot 7: valid vs hallucinated packages (code task) by model x strategy ---
    by_model_strat: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for r in code_rows:
        by_model_strat[r["model"]][r["strategy"]] = r

    code_models = sorted(by_model_strat)
    code_strats2 = [s for s in STRATEGY_DISPLAY if any(s in by_model_strat[m] for m in code_models)]
    if code_models and code_strats2:
        x = np.arange(len(code_models))
        width = 0.8 / max(len(code_strats2), 1)
        series = {}
        for s in code_strats2:
            valid_vals, hallu_vals = [], []
            for m in code_models:
                r = by_model_strat[m].get(s)
                if not r:
                    valid_vals.append(0)
                    hallu_vals.append(0)
                    continue
                valid_vals.append(r["n_valid_with_stdlib"])
                hallu_vals.append(r["n_generated"] - r["n_valid_with_stdlib"])
            series[STRATEGY_DISPLAY[s]] = (np.array(valid_vals), np.array(hallu_vals))

        fig, ax = plt.subplots(figsize=(18, 12))
        cats = _draw_grouped_stacked_bars(ax, code_models, series, x, width)
        ax.set_xticks(x + width * (len(cats) - 1) / 2)
        ax.set_xticklabels([m.split("__")[0] for m in code_models], rotation=45, ha="right", fontsize=20)
        ax.legend(
            title="Strategy", fontsize=14, title_fontsize=16,
            loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=3,
        )
        ax.set_title(
            "Valid vs Hallucinated Packages by Model and\nMitigation Strategy (Code Generation)",
            fontsize=26, fontweight="bold",
        )
        ax.set_ylabel("Package Count", fontsize=22)
        ax.tick_params(axis="y", labelsize=18)
        _style_axes(ax)
        fig.tight_layout()
        out = plots_dir / "valid_vs_hallucinated_packages_by_model_and_method_code.pdf"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.relative_to(ROOT)}")

    # --- Plot 8: syntax error rate & mean smell count by model x strategy ---
    quality_rows = [r for r in data if r["task"] == "code" and r["syntax_valid_rate"] is not None]
    by_model_strat_q: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for r in quality_rows:
        by_model_strat_q[r["model"]][r["strategy"]].append(r)

    quality_models = sorted(by_model_strat_q)
    quality_strats = [s for s in STRATEGY_DISPLAY if any(s in by_model_strat_q[m] for m in quality_models)]
    if quality_models and quality_strats:
        syntax_err: Dict[str, np.ndarray] = {}
        mean_smell: Dict[str, np.ndarray] = {}
        for s in quality_strats:
            err_vals, smell_vals = [], []
            for m in quality_models:
                rows = by_model_strat_q[m].get(s)
                if not rows:
                    err_vals.append(0.0)
                    smell_vals.append(0.0)
                    continue
                err_vals.append(100.0 * (1 - statistics.mean(r["syntax_valid_rate"] for r in rows)))
                smell_rows = [r["mean_smell_count"] for r in rows if r["mean_smell_count"] is not None]
                smell_vals.append(statistics.mean(smell_rows) if smell_rows else 0.0)
            syntax_err[s] = np.array(err_vals)
            mean_smell[s] = np.array(smell_vals)

        x = np.arange(len(quality_models))
        width = 0.8 / max(len(quality_strats), 1)

        for metric_name, values, ylabel, title, out_name in [
            ("syntax_error", syntax_err, "Syntax Valid Rate (%)",
             "Code Syntax Valid Rate by Model and\nMitigation Strategy",
             "syntax_error_by_model_and_strategy.pdf"),
            ("smell", mean_smell, "Mean Smell Count per Sample",
             "Code Smell Count by Model and\nMitigation Strategy",
             "code_smell_by_model_and_strategy.pdf"),
        ]:
            fig, ax = plt.subplots(figsize=(18, 8))
            for i, s in enumerate(quality_strats):
                hatch = STRAT_HATCHES[i % len(STRAT_HATCHES)]
                color = LANG_COLORS[i % len(LANG_COLORS)]
                ax.bar(x + i * width, values[s], width, label=STRATEGY_DISPLAY[s],
                       color=color, edgecolor="black", hatch=hatch)
            ax.set_xticks(x + width * (len(quality_strats) - 1) / 2)
            ax.set_xticklabels([m.split("__")[0] for m in quality_models], rotation=25, ha="right", fontsize=20)
            ax.legend(title="Strategy", fontsize=14, title_fontsize=16, ncol=2)
            ax.set_title(title, fontsize=26, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=22)
            ax.tick_params(axis="y", labelsize=18)
            _style_axes(ax)
            fig.tight_layout()
            out = plots_dir / out_name
            fig.savefig(out, dpi=600, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out.relative_to(ROOT)}")

    def _save_heatmap(
        cell: Dict[Tuple[str, str, str], float],
        hm_models: List[str],
        hm_strats: List[str],
        hm_langs: List[str],
        *,
        cmap: str,
        vmin: float,
        vmax: float,
        cbar_label: str,
        fmt_fn,
        white_threshold: float,
        title: str,
        out: Path,
    ) -> None:
        """Render and save one model×strategy×language heatmap."""
        strat_labels = [STRATEGY_DISPLAY.get(s, s) for s in hm_strats]
        ncols = 3
        nrows = math.ceil(len(hm_models) / ncols)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 5, nrows * 3.5 + 0.8),
            squeeze=False,
        )
        last_im = None
        for idx, model in enumerate(hm_models):
            ax = axes[idx // ncols][idx % ncols]
            matrix = np.full((len(hm_strats), len(hm_langs)), np.nan)
            for si, s in enumerate(hm_strats):
                for li, lang in enumerate(hm_langs):
                    v = cell.get((model, s, lang))
                    if v is not None:
                        matrix[si, li] = v
            last_im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                                vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(hm_langs)))
            ax.set_xticklabels(hm_langs, fontsize=14)
            ax.set_yticks(range(len(hm_strats)))
            ax.set_yticklabels(strat_labels, fontsize=14)
            ax.set_title(model.split("__")[0], fontsize=11, fontweight="bold")
            for si in range(len(hm_strats)):
                for li in range(len(hm_langs)):
                    v = matrix[si, li]
                    txt = fmt_fn(v) if not np.isnan(v) else "–"
                    fg = "white" if (not np.isnan(v) and v > white_threshold) else "black"
                    ax.text(li, si, txt, ha="center", va="center",
                            fontsize=8, color=fg, fontweight="bold")
        for idx in range(len(hm_models), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        fig.subplots_adjust(right=0.88, hspace=0.45, wspace=0.3)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label(cbar_label, fontsize=11)
        cbar.ax.tick_params(labelsize=9)
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.relative_to(ROOT)}")

    # --- Plot 9: syntax error rate heatmap — model × strategy × language ---
    quality_rows_lang = [r for r in data if r["task"] == "code"]
    if quality_rows_lang:
        hm_models = sorted({r["model"] for r in quality_rows_lang})

        # syntax error % — when syntax_valid_rate is None (all excluded) show 0
        syn_cell: Dict[Tuple[str, str, str], float] = {}
        for r in quality_rows_lang:
            key = (r["model"], r["strategy"], r["language"])
            rate = r.get("syntax_valid_rate")
            syn_cell[key] = 0.0 if rate is None else 100.0 * (1.0 - rate)

        hm_strats = [s for s in STRATEGY_DISPLAY if any(
            (m, s, lang) in syn_cell for m in hm_models for lang in LANGUAGES
        )]
        hm_langs = [lang for lang in LANGUAGES if any(
            (m, s, lang) in syn_cell for m in hm_models for s in hm_strats
        )]

        if hm_models and hm_strats and hm_langs:
            _save_heatmap(
                syn_cell, hm_models, hm_strats, hm_langs,
                cmap="RdYlGn_r", vmin=0, vmax=100,
                cbar_label="Syntax Error Rate (%)",
                fmt_fn=lambda v: f"{v:.0f}%",
                white_threshold=60,
                title="Syntax Error Rate (%) by Model, Strategy, and Language",
                out=plots_dir / "syntax_error_heatmap_by_model_strategy_language.pdf",
            )

        # --- Plot 10: mean smell count heatmap — model × strategy × language ---
        smell_cell: Dict[Tuple[str, str, str], float] = {}
        for r in quality_rows_lang:
            v = r.get("mean_smell_count")
            if v is not None:
                smell_cell[(r["model"], r["strategy"], r["language"])] = v

        sm_strats = [s for s in STRATEGY_DISPLAY if any(
            (m, s, lang) in smell_cell for m in hm_models for lang in LANGUAGES
        )]
        sm_langs = [lang for lang in LANGUAGES if any(
            (m, s, lang) in smell_cell for m in hm_models for s in sm_strats
        )]

        if hm_models and sm_strats and sm_langs:
            all_smell_vals = [v for v in smell_cell.values() if not np.isnan(v)]
            vmax_smell = max(all_smell_vals) if all_smell_vals else 1.0
            _save_heatmap(
                smell_cell, hm_models, sm_strats, sm_langs,
                cmap="RdYlGn_r", vmin=0, vmax=vmax_smell,
                cbar_label="Mean Smell Count per Sample",
                fmt_fn=lambda v: f"{v:.2f}",
                white_threshold=vmax_smell * 0.6,
                title="Mean Code Smell Count by Model, Strategy, and Language",
                out=plots_dir / "smell_heatmap_by_model_strategy_language.pdf",
            )



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze hallucination results and generate plots.")
    parser.add_argument(
        "--results-dir",
        default=str(ROOT / "output"),
        help="Directory of <strategy>/<task>/<model>/<Language>.json files enhanced by evaluate.py.",
    )
    parser.add_argument(
        "--plots-dir",
        default=str(ROOT / "results" / "plots"),
        help="Directory for plot PDFs.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument("--print-report", action="store_true", help="Print statistical report to stdout.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    n_backfilled = _backfill_syntax_errors(results_dir)
    if n_backfilled:
        print(f"Backfilled / corrected syntax data for {n_backfilled} sample(s) in {results_dir}.")

    data = load_result_dir(results_dir)
    print(f"Loaded {len(data)} rows from {args.results_dir}")

    if args.print_report:
        print_report(data)

    if not args.no_plots:
        print("\nGenerating plots ...")
        make_plots(data, Path(args.plots_dir))

    print("\nDone.")


if __name__ == "__main__":
    main()
