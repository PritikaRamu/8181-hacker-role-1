#!/usr/bin/env python3
"""Analyse variance across experimental runs in notebooks/tmp/run_X/.

Each file contains a mix of:
- list groups (each = 1 full run over ~498 seed packages) → variance source
- single-dict entries (individual samples, ignored here)

Generates two unified longtable files (one per task):
    results/tables/phr_variance_packages.tex
    results/tables/phr_variance_code.tex

Each file holds all strategies in a single table.  Only micro PHR and
macro PHR carry ±std; N_gen, N_hall, n_gen, n_hall show plain means.

Usage:
    python3 analyze_runs.py
    python3 analyze_runs.py --tasks packages
    python3 analyze_runs.py --strategies baseline actlcd
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "notebooks" / "tmp"
OUT_DIR  = ROOT / "results" / "tables"

STRATEGIES = [
    "baseline", "greedy_decoding", "dola",
    "contrastive_decoding", "nudging", "self_refine", "actlcd",
]
STRATEGY_LABELS = {
    "baseline":             "Vanilla",
    "greedy_decoding":      "Greedy",
    "dola":                 "DoLa",
    "contrastive_decoding": "CD",
    "nudging":              "Nudge",
    "self_refine":          "S-Ref",
    "actlcd":               "ActLCD",
}
# Order matching the stdlib-correction table in the paper
STDLIB_STRAT_ORDER = [
    "baseline", "greedy_decoding", "self_refine",
    "contrastive_decoding", "dola", "nudging", "actlcd",
]
MODELS = [
    "deepseek-1.3b", "deepseek-6.7b",
    "gemma-1b",      "gemma-4b",
    "llama-8b",      "mistral-7b",
    "qwen-1.5b",     "qwen-3b",
]
LANGUAGES = ["Python", "JavaScript", "Ruby", "Rust"]
TASKS     = ["packages", "code"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples(path: Path) -> list[dict]:
    """Return a flat list of sample dicts from a result file.

    Handles two formats:
    - New: flat list of dicts  [{"package_name": ..., "extracted_count": ...}, ...]
    - Old: nested list         [[{...}, ...], [{...}, ...], ...]  (one sub-list per batch)
    """
    data  = json.loads(path.read_text(encoding="utf-8"))
    inner = data.get("packages", data) if isinstance(data, dict) else data
    if not isinstance(inner, list) or not inner:
        return []
    first = inner[0]
    if isinstance(first, dict):          # new flat format
        return [e for e in inner if isinstance(e, dict)]
    if isinstance(first, list):          # old nested format — pool all batches
        return [e for batch in inner if isinstance(batch, list)
                  for e in batch if isinstance(e, dict)]
    return []


def run_stats(samples: list[dict]) -> dict | None:
    if not samples:
        return None
    n_ext  = sum(s.get("extracted_count",   0) for s in samples)
    n_hall = sum(s.get("hallucinated_count", 0) for s in samples)
    rates  = [s.get("hallucination_rate", 0.0) for s in samples]
    n_s    = len(samples)
    return {
        "N_gen":     n_ext,
        "N_hall":    n_hall,
        "micro_phr": (n_hall / n_ext * 100) if n_ext > 0 else 0.0,
        "n_gen":     n_ext / n_s,
        "n_hall":    n_hall / n_s,
        "macro_phr": float(np.mean(rates)) * 100,
    }


def collect_runs(run_dirs: list[Path], strategy: str, task: str,
                 model: str, lang: str) -> list[dict]:
    """One stats dict per run_X dir (entire file = one run)."""
    out = []
    for rd in run_dirs:
        f = rd / strategy / task / model / f"{lang}.json"
        if not f.exists():
            continue
        try:
            s = run_stats(load_samples(f))
            if s:
                out.append(s)
        except Exception as ex:
            print(f"  Warning {f}: {ex}")
    return out


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    a = np.array(values, dtype=float)
    return float(np.mean(a)), float(np.std(a, ddof=1) if len(a) > 1 else 0.0)


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def fmt_int(mean: float) -> str:
    """Plain integer mean (no variance)."""
    return "--" if math.isnan(mean) else f"{round(mean):,}"


def fmt_n(mean: float) -> str:
    """Plain 2-dp mean (no variance)."""
    return "--" if math.isnan(mean) else f"{mean:.2f}"


def fmt_phr(mean: float, std: float, bold: bool = False) -> str:
    """PHR with ±std subscript, optional bold."""
    if math.isnan(mean):
        return "--"
    inner = f"{mean:.1f}$_{{\\pm {std:.2f}}}$"
    return f"\\textbf{{{inner}}}" if bold else inner


# ---------------------------------------------------------------------------
# Model-block rows
# ---------------------------------------------------------------------------

def model_rows(model: str, strategy: str, task: str,
               run_dirs: list[Path]) -> list[str] | None:
    """Return LaTeX row strings for one model, or None if no data."""
    lang_stats: dict[str, list[dict]] = {
        l: collect_runs(run_dirs, strategy, task, model, l)
        for l in LANGUAGES
    }
    valid = [l for l in LANGUAGES if lang_stats[l]]
    if not valid:
        return None

    # Per-language stats
    lang_rec: dict[str, dict] = {}
    for lang in valid:
        rec: dict[str, tuple] = {}
        for key in ("N_gen", "N_hall", "micro_phr", "n_gen", "n_hall", "macro_phr"):
            rec[key] = mean_std([r[key] for r in lang_stats[lang]])
        lang_rec[lang] = rec

    # Bold = highest micro / macro PHR per model
    max_micro = max(lang_rec[l]["micro_phr"][0] for l in valid)
    max_macro = max(lang_rec[l]["macro_phr"][0] for l in valid)

    n_rows     = len(valid) + 1  # languages + Overall
    model_cell = f"\\multirow{{{n_rows}}}{{*}}{{\\texttt{{{model}}}}}"

    rows: list[str] = []
    for li, lang in enumerate(LANGUAGES):
        prefix = model_cell if li == 0 else ""
        if lang not in lang_rec:
            rows.append(f"      {prefix} & {lang} & -- & -- & -- & -- & -- & -- \\\\")
            continue
        rec = lang_rec[lang]
        b_micro = abs(rec["micro_phr"][0] - max_micro) < 1e-9
        b_macro = abs(rec["macro_phr"][0] - max_macro) < 1e-9
        is_last_lang = (lang == valid[-1])
        cmidrule = r" \cmidrule(lr){2-8}" if is_last_lang else ""
        rows.append(
            f"      {prefix} & {lang}"
            f" & {fmt_int(rec['N_gen'][0])}"
            f" & {fmt_int(rec['N_hall'][0])}"
            f" & {fmt_phr(*rec['micro_phr'], bold=b_micro)}"
            f" & {fmt_n(rec['n_gen'][0])}"
            f" & {fmt_n(rec['n_hall'][0])}"
            f" & {fmt_phr(*rec['macro_phr'], bold=b_macro)}"
            f" \\\\{cmidrule}"
        )

    # Overall — sum N across languages, mean PHR
    n_runs = max(len(lang_stats[l]) for l in valid)
    totals: list[dict] = []
    for ri in range(n_runs):
        row: dict = {"N_gen": 0.0, "N_hall": 0.0,
                     "mp": [], "mac": [], "ng": [], "nh": []}
        ok = True
        for l in valid:
            if ri >= len(lang_stats[l]):
                ok = False; break
            r = lang_stats[l][ri]
            row["N_gen"]  += r["N_gen"]
            row["N_hall"] += r["N_hall"]
            row["mp"].append(r["micro_phr"])
            row["mac"].append(r["macro_phr"])
            row["ng"].append(r["n_gen"])
            row["nh"].append(r["n_hall"])
        if ok:
            totals.append(row)

    if totals:
        ov_ng   = mean_std([float(np.mean(t["ng"]))  for t in totals])
        ov_nh   = mean_std([float(np.mean(t["nh"]))  for t in totals])
        ov_mi   = mean_std([float(np.mean(t["mp"]))  for t in totals])
        ov_ma   = mean_std([float(np.mean(t["mac"])) for t in totals])
        ov_Ng   = mean_std([t["N_gen"]  for t in totals])
        ov_Nh   = mean_std([t["N_hall"] for t in totals])
    else:
        ov_Ng = ov_Nh = ov_mi = ov_ma = ov_ng = ov_nh = (math.nan, math.nan)

    rows.append(
        f"      & \\textit{{Overall}}"
        f" & {fmt_int(ov_Ng[0])}"
        f" & {fmt_int(ov_Nh[0])}"
        f" & {fmt_phr(*ov_mi)}"
        f" & {fmt_n(ov_ng[0])}"
        f" & {fmt_n(ov_nh[0])}"
        f" & {fmt_phr(*ov_ma)}"
        r" \\"
    )
    return rows


# ---------------------------------------------------------------------------
# Unified longtable (one per task)
# ---------------------------------------------------------------------------

HEADER_COL = (
    r"    \textbf{Model} & \textbf{Lang.} & "
    r"\textbf{$N_{\text{gen}}$} & \textbf{$N_{\text{hall}}$} & "
    r"\textbf{micro PHR (\%)} & "
    r"\textbf{$\bar{n}_{\text{gen}}$} & \textbf{$\bar{n}_{\text{hall}}$} & "
    r"\textbf{macro PHR (\%)} \\"
)


def build_unified_table(task: str, strategies: list[str],
                        run_dirs: list[Path]) -> str:
    task_label = task.capitalize()
    lines: list[str] = []

    lines += [
        r"\begin{longtable}{@{}llrrp{0.72cm}rrp{0.72cm}@{}}",
        f"  \\caption{{PHR per model and language — {task_label} task. "
        r"micro PHR $= N_{\text{hall}} / N_{\text{gen}} \times 100$. "
        r"macro PHR $=$ mean per-sample PHR. "
        r"$\bar{n}$ = mean per prompt. "
        r"\textit{Overall} sums $N$ and averages PHR across languages. "
        r"Bold = highest PHR per model.}}"
        f"\\label{{tab:phr-variance-{task}}} \\\\",
        r"  \toprule",
        HEADER_COL,
        r"  \midrule",
        r"  \endfirsthead",
        r"  \multicolumn{8}{c}{\tablename\ \thetable\ -- continued} \\",
        r"  \toprule",
        HEADER_COL,
        r"  \midrule",
        r"  \endhead",
        r"  \midrule \multicolumn{8}{r}{Continued\ldots} \\",
        r"  \endfoot",
        r"  \bottomrule",
        r"  \endlastfoot",
    ]

    first_strat = True
    for strategy in strategies:
        strat_label = STRATEGY_LABELS.get(strategy, strategy)

        # collect model blocks for this strategy
        model_blocks: list[tuple[str, list[str]]] = []
        for model in MODELS:
            rows = model_rows(model, strategy, task, run_dirs)
            if rows is not None:
                model_blocks.append((model, rows))

        if not model_blocks:
            continue

        # Strategy header row
        if not first_strat:
            lines.append(r"  \midrule[1.2pt]")
        first_strat = False
        lines.append(
            f"  \\multicolumn{{8}}{{l}}"
            f"{{\\textbf{{\\small {strat_label}}}}} \\\\"
        )
        lines.append(r"  \midrule")

        for bi, (model, rows) in enumerate(model_blocks):
            if bi > 0:
                lines.append(r"    \midrule")
            lines.extend(rows)

    lines += [r"\end{longtable}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stdlib-correction table
# ---------------------------------------------------------------------------

def collect_stdlib_runs(run_dirs: list[Path], strategy: str, lang: str) -> list[dict]:
    """For each run_X dir, pool all models' samples for (strategy, packages, lang).

    Returns one dict per run with:
        N_gen, N_hall_w (with stdlib), N_hall_wo (without stdlib),
        micro_w, micro_wo, delta
    """
    out = []
    for rd in run_dirs:
        all_samples: list[dict] = []
        for model in MODELS:
            f = rd / strategy / "packages" / model / f"{lang}.json"
            if f.exists():
                try:
                    all_samples.extend(load_samples(f))
                except Exception as ex:
                    print(f"  Warning {f}: {ex}")
        if not all_samples:
            continue
        n_gen    = sum(s.get("extracted_count",   0) for s in all_samples)
        n_hall_w = sum(s.get("hallucinated_count", 0) for s in all_samples)
        n_std    = sum(s.get("stdlib_count",       0) for s in all_samples)
        n_hall_wo = n_hall_w + n_std
        if n_gen == 0:
            continue
        micro_w  = n_hall_w  / n_gen * 100
        micro_wo = n_hall_wo / n_gen * 100
        out.append({
            "N_gen":    n_gen,
            "micro_w":  micro_w,
            "micro_wo": micro_wo,
            "delta":    micro_wo - micro_w,
        })
    return out


def build_stdlib_table(strategies: list[str], run_dirs: list[Path]) -> str:
    """Build the micro-PHR stdlib-correction longtable (all languages)."""
    lines: list[str] = []
    lines += [
        r"\begin{longtable}{@{}llrrrr@{}}",
        r"  \caption{Micro PHR with and without standard library correction, "
        r"averaged over all models. "
        r"$\Delta>0$ means overestimation without the correction.}"
        r"\label{tab:stdlibs-delta-micro-variance} \\",
        r"  \toprule",
        r"  \textbf{Lang.} & \textbf{Strategy} & "
        r"\textbf{$N_{\text{gen}}$} & "
        r"\textbf{PHR$_{\text{w/ std}}$ (\%)} & "
        r"\textbf{PHR$_{\text{w/o std}}$ (\%)} & "
        r"\textbf{$\Delta$ (pp)} \\",
        r"  \midrule",
        r"  \endfirsthead",
        r"  \multicolumn{6}{c}{\tablename\ \thetable\ -- continued} \\",
        r"  \toprule",
        r"  \textbf{Lang.} & \textbf{Strategy} & "
        r"\textbf{$N_{\text{gen}}$} & "
        r"\textbf{PHR$_{\text{w/ std}}$ (\%)} & "
        r"\textbf{PHR$_{\text{w/o std}}$ (\%)} & "
        r"\textbf{$\Delta$ (pp)} \\",
        r"  \midrule",
        r"  \endhead",
        r"  \midrule \multicolumn{6}{r}{Continued\ldots} \\",
        r"  \endfoot",
        r"  \bottomrule",
        r"  \endlastfoot",
    ]

    for li, lang in enumerate(LANGUAGES):
        if li > 0:
            lines.append(r"  \midrule")

        strat_data: list[tuple[str, list[dict]]] = []
        for strat in strategies:
            runs = collect_stdlib_runs(run_dirs, strat, lang)
            if runs:
                strat_data.append((strat, runs))

        if not strat_data:
            continue

        n_rows = len(strat_data)
        # Bold = largest Δ per language
        max_delta = max(float(np.mean([r["delta"] for r in runs]))
                        for _, runs in strat_data)

        for si, (strat, runs) in enumerate(strat_data):
            lang_cell = (
                f"\\multirow{{{n_rows}}}{{*}}{{{lang}}}" if si == 0 else ""
            )
            label  = STRATEGY_LABELS.get(strat, strat)
            ng_m,  _     = mean_std([r["N_gen"]    for r in runs])
            w_m,   w_s   = mean_std([r["micro_w"]  for r in runs])
            wo_m,  wo_s  = mean_std([r["micro_wo"] for r in runs])
            d_m,   _     = mean_std([r["delta"]    for r in runs])

            bold = abs(d_m - max_delta) < 1e-9

            def phr_cell(m: float, s: float) -> str:
                if math.isnan(m): return "--"
                return f"{m:.1f}$_{{\\pm {s:.2f}}}$"

            delta_str = f"+{d_m:.1f}" if d_m >= 0 else f"{d_m:.1f}"
            delta_cell = f"\\textbf{{{delta_str}}}" if bold else delta_str

            lines.append(
                f"  {lang_cell} & {label}"
                f" & {fmt_int(ng_m)}"
                f" & {phr_cell(w_m, w_s)}"
                f" & {phr_cell(wo_m, wo_s)}"
                f" & {delta_cell} \\\\"
            )

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategies", nargs="+", default=STRATEGIES)
    p.add_argument("--tasks",      nargs="+", default=TASKS)
    p.add_argument("--runs-dir",   default=str(RUNS_DIR))
    p.add_argument("--output-dir", default=str(OUT_DIR))
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    runs_root = Path(args.runs_dir)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(
        d for d in runs_root.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    )
    if not run_dirs:
        print(f"No run_* directories found in {runs_root}")
        return
    print(f"Found {len(run_dirs)} run dirs: {[d.name for d in run_dirs]}")

    for task in args.tasks:
        print(f"  Building unified table for task: {task} ...")
        tex = build_unified_table(task, args.strategies, run_dirs)
        out = out_dir / f"phr_variance_{task}.tex"
        out.write_text(tex + "\n", encoding="utf-8")
        print(f"    → {out}")

    print("  Building stdlib-correction table ...")
    strat_order = [s for s in STDLIB_STRAT_ORDER if s in args.strategies]
    tex = build_stdlib_table(strat_order, run_dirs)
    out = out_dir / "phr_stdlib_correction_variance.tex"
    out.write_text(tex + "\n", encoding="utf-8")
    print(f"    → {out}")

    print("Done.")


if __name__ == "__main__":
    main()
