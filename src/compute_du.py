"""
Compute Dependency Utility (DU) scores for package-recommendation results.

Three subcommands:

  tfidf      Compute TF-IDF DU (P/R/F1) using the top-K similar_packages reference
             set and write scores back into each result JSON file.
             Per-entry: du_precision, du_recall, du
             In stats:  macro_du_precision, macro_du_recall, macro_du

  collective Compute collective-reference DU using packages validated across all
             (strategy × model) code runs.
             Per-entry: du_coll_precision, du_coll_recall, du_coll
             In stats:  macro_du_coll_precision, macro_du_coll_recall, macro_du_coll

  topk       Plot Strategy × Model heatmaps for DU at k=1,2,5,10 and with the
             full collective reference. Reads result files without writing back.

Usage:
    python compute_du.py tfidf
    python compute_du.py tfidf --top-k 5
    python compute_du.py collective
    python compute_du.py topk
    python compute_du.py topk --results-dir reformat_results --output-dir results/eval
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data" / "instruction"
RESULTS_DIR = ROOT / "reformat_results"

LANG_INSTRUCT = {
    "Python":     "packages_Python_instruct.json",
    "JavaScript": "packages_JavaScript_instruct.json",
    "Ruby":       "packages_Ruby_instruct.json",
    "Rust":       "packages_Rust_instruct.json",
}

STRATEGY_ORDER = [
    "baseline", "greedy_decoding", "dola",
    "contrastive_decoding", "nudging", "self_refine", "actlcd",
]
STRATEGY_LABELS = {
    "baseline":             "Vanilla",
    "greedy_decoding":      "Greedy",
    "dola":                 "DoLA",
    "contrastive_decoding": "CD",
    "nudging":              "Nudge",
    "self_refine":          "S-Ref",
    "actlcd":               "ALCD",
}
MODEL_ORDER = [
    "deepseek-1.3b", "deepseek-6.7b",
    "gemma-1b", "gemma-4b",
    "llama-8b", "mistral-7b",
    "qwen-1.5b", "qwen-3b",
]

TOP_K_LEVELS   = [1, 2, 5, 10]
COLLECTIVE_KEY = "collective"

sns.set_theme(style="whitegrid", font_scale=1.1)


# ── shared ───────────────────────────────────────────────────────────────────

def du_scores(G: set[str], R: set[str]) -> tuple[float, float, float]:
    """Return (DU_P, DU_R, DU_F1) for a single prompt."""
    if not G or not R:
        return 0.0, 0.0, 0.0
    U = G & R
    p = len(U) / len(G)
    r = len(U) / len(R)
    return p, r, (2 * p * r / (p + r) if p + r > 0 else 0.0)


def load_tfidf_index(lang: str, k: int) -> dict[str, set[str]]:
    """Return {seed_lower: R_i(k)} using the top-(k-1) TF-IDF similar_packages."""
    path = DATA_DIR / LANG_INSTRUCT[lang]
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("packages", data) if isinstance(data, dict) else data
    index: dict[str, set[str]] = {}
    for p in items:
        seed = p.get("name", "").lower()
        similar = [s["name"].lower() for s in (p.get("similar_packages") or [])]
        index[seed] = {seed} | set(similar[: k - 1])
    return index


def load_collective_index(lang: str) -> dict[str, set[str]]:
    """Return {seed_lower: R_i} using all collective_packages entries."""
    path = DATA_DIR / LANG_INSTRUCT[lang]
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("packages", data) if isinstance(data, dict) else data
    index: dict[str, set[str]] = {}
    for p in items:
        seed = p.get("name", "").lower()
        collective = {e["name"].lower() for e in (p.get("collective_packages") or [])}
        index[seed] = {seed} | collective
    return index


def _result_files(results_dir: Path) -> list[tuple[str, str, str, Path]]:
    """Return [(strategy, model, lang, path)] for all packages/ result files."""
    out = []
    for strat_dir in sorted(results_dir.iterdir()):
        if not strat_dir.is_dir():
            continue
        pkg_dir = strat_dir / "packages"
        if not pkg_dir.is_dir():
            continue
        for model_dir in sorted(pkg_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for path in sorted(model_dir.glob("*.json")):
                lang = path.stem
                if lang in LANG_INSTRUCT:
                    out.append((strat_dir.name, model_dir.name, lang, path))
    return out


# ── tfidf subcommand ─────────────────────────────────────────────────────────

def _write_tfidf(path: Path, ref_index: dict[str, set[str]]) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    p_list, r_list, f_list = [], [], []
    missing = 0
    for entry in data.get("packages", []):
        seed = (entry.get("package_name") or entry.get("name", "")).lower()
        G = {pkg.lower() for pkg in (entry.get("extracted_packages") or [])}
        R = ref_index.get(seed)
        if R is None:
            missing += 1
            entry["du_precision"] = entry["du_recall"] = entry["du"] = None
            continue
        dp, dr, du = du_scores(G, R)
        entry["du_precision"] = round(dp, 4)
        entry["du_recall"]    = round(dr, 4)
        entry["du"]           = round(du, 4)
        p_list.append(dp); r_list.append(dr); f_list.append(du)
    n = len(f_list)
    stats = data.setdefault("stats", {})
    if n:
        stats["macro_du_precision"] = round(sum(p_list) / n, 4)
        stats["macro_du_recall"]    = round(sum(r_list) / n, 4)
        stats["macro_du"]           = round(sum(f_list) / n, 4)
    else:
        stats["macro_du_precision"] = stats["macro_du_recall"] = stats["macro_du"] = None
    if missing:
        stats["du_missing_ref"] = missing
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run_tfidf(results_dir: Path, top_k: int) -> None:
    print(f"Loading TF-IDF reference indices (k={top_k}) ...")
    ref_indices = {lang: load_tfidf_index(lang, top_k) for lang in LANG_INSTRUCT}
    files = sorted(results_dir.rglob("packages/**/*.json"))
    print(f"Found {len(files)} result files.\n")
    for path in tqdm(files, desc="files", unit=" file"):
        lang = path.stem
        if lang not in ref_indices:
            tqdm.write(f"  Skipping {path} — unknown language '{lang}'")
            continue
        _write_tfidf(path, ref_indices[lang])
    print("\nDone.")


# ── collective subcommand ─────────────────────────────────────────────────────

def _write_collective(path: Path, ref_index: dict[str, set[str]]) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    p_list, r_list, f_list = [], [], []
    missing = 0
    for entry in data.get("packages", []):
        seed = (entry.get("package_name") or entry.get("name", "")).lower()
        G = {pkg.lower() for pkg in (entry.get("extracted_packages") or [])}
        R = ref_index.get(seed)
        if R is None:
            missing += 1
            entry["du_coll_precision"] = entry["du_coll_recall"] = entry["du_coll"] = None
            continue
        dp, dr, du = du_scores(G, R)
        entry["du_coll_precision"] = round(dp, 4)
        entry["du_coll_recall"]    = round(dr, 4)
        entry["du_coll"]           = round(du, 4)
        p_list.append(dp); r_list.append(dr); f_list.append(du)
    n = len(f_list)
    stats = data.setdefault("stats", {})
    if n:
        stats["macro_du_coll_precision"] = round(sum(p_list) / n, 4)
        stats["macro_du_coll_recall"]    = round(sum(r_list) / n, 4)
        stats["macro_du_coll"]           = round(sum(f_list) / n, 4)
    else:
        stats["macro_du_coll_precision"] = stats["macro_du_coll_recall"] = stats["macro_du_coll"] = None
    if missing:
        stats["du_coll_missing_ref"] = missing
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run_collective(results_dir: Path) -> None:
    print("Loading collective reference indices ...")
    ref_indices = {lang: load_collective_index(lang) for lang in LANG_INSTRUCT}
    files = sorted(results_dir.rglob("packages/**/*.json"))
    print(f"Found {len(files)} result files.\n")
    for path in tqdm(files, desc="files", unit=" file"):
        lang = path.stem
        if lang not in ref_indices:
            tqdm.write(f"  Skipping {path} — unknown language '{lang}'")
            continue
        _write_collective(path, ref_indices[lang])
    print("\nDone.")


# ── topk subcommand ──────────────────────────────────────────────────────────

def _compute_file(path: Path, ref_index: dict[str, set[str]]) -> tuple[float, float, float] | None:
    """Return (macro_P, macro_R, macro_F1) for a result file, or None if no valid samples."""
    data = json.loads(path.read_text(encoding="utf-8"))
    p_list, r_list, f_list = [], [], []
    for entry in data.get("packages", []):
        seed = (entry.get("package_name") or entry.get("name", "")).lower()
        R = ref_index.get(seed)
        if R is None:
            continue
        G = {pkg.lower() for pkg in (entry.get("extracted_packages") or [])}
        p, r, f = du_scores(G, R)
        p_list.append(p); r_list.append(r); f_list.append(f)
    if not p_list:
        return None
    n = len(p_list)
    return sum(p_list) / n, sum(r_list) / n, sum(f_list) / n


def _aggregate_files(result_files, ref_indices):
    accum: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    for strategy, model, lang, path in result_files:
        scores = _compute_file(path, ref_indices[lang])
        if scores is not None:
            accum[(strategy, model)].append(scores)
    return {
        key: tuple(sum(v[i] for v in vals) / len(vals) for i in range(3))
        for key, vals in accum.items()
    }


def compute_topk_data(results_dir: Path):
    rf = _result_files(results_dir)
    print(f"Found {len(rf)} result files.")
    out = {}
    for k in TOP_K_LEVELS:
        print(f"  Computing TF-IDF top-{k} ...")
        out[k] = _aggregate_files(rf, {lang: load_tfidf_index(lang, k) for lang in LANG_INSTRUCT})
    print("  Computing collective ...")
    out[COLLECTIVE_KEY] = _aggregate_files(rf, {lang: load_collective_index(lang) for lang in LANG_INSTRUCT})
    return out


def _build_pivot(data, metric_idx: int) -> np.ndarray:
    mat = np.full((len(STRATEGY_ORDER), len(MODEL_ORDER)), np.nan)
    for si, strat in enumerate(STRATEGY_ORDER):
        for mi, model in enumerate(MODEL_ORDER):
            val = data.get((strat, model))
            if val is not None:
                mat[si, mi] = val[metric_idx] * 100
    return mat


def _plot_metric(all_data, metric_idx: int, metric_name: str, out_dir: Path) -> None:
    panels = TOP_K_LEVELS + [COLLECTIVE_KEY]
    fig, axes = plt.subplots(1, 5, figsize=(48, 9.7),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.22]})
    xlabels = [m.replace("-", "\n") for m in MODEL_ORDER]
    ylabels = [STRATEGY_LABELS.get(s, s) for s in STRATEGY_ORDER]

    all_vals = []
    for key in panels:
        mat = _build_pivot(all_data[key], metric_idx)
        all_vals.extend(mat[~np.isnan(mat)].tolist())
    vmin, vmax = min(all_vals), max(all_vals)

    for i, (ax, key) in enumerate(zip(axes, panels)):
        mat = _build_pivot(all_data[key], metric_idx)
        hm = sns.heatmap(
            mat, annot=True, fmt=".1f", annot_kws={"fontsize": 34},
            cmap="YlGnBu", vmin=vmin, vmax=vmax,
            linewidths=0.5, linecolor="white",
            xticklabels=xlabels,
            yticklabels=ylabels if i == 0 else [],
            ax=ax,
            cbar=(i == len(panels) - 1),
            cbar_kws={"label": f"{metric_name} (×100)"},
            mask=np.isnan(mat),
        )
        if key == COLLECTIVE_KEY:
            title = "With code generated packages"
        elif key == 1:
            title = "Top-1  (seed only)"
        else:
            title = f"Top-{key}  (seed + {key-1} similar)"
        ax.set_title(title, fontsize=33, fontweight="bold", pad=8)
        ax.set_xlabel("Model", fontsize=26)
        ax.set_ylabel("Strategy" if i == 0 else "", fontsize=28)
        ax.tick_params(axis="x", labelsize=28)
        ax.tick_params(axis="y", labelsize=28)
        plt.setp(ax.get_yticklabels(), rotation=0, va="center")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        if i == len(panels) - 1:
            cbar = hm.collections[0].colorbar
            cbar.ax.tick_params(labelsize=20)
            cbar.set_label(f"{metric_name} (×100)", fontsize=26)

    fig.tight_layout()
    stem = f"du_topk_{metric_name.lower()}"
    for ext in ("pdf", "png"):
        fpath = out_dir / f"{stem}.{ext}"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"  Saved {fpath}")
    plt.close(fig)


def run_topk(results_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_data = compute_topk_data(results_dir)
    for metric_idx, metric_name in [(0, "Precision"), (1, "Recall"), (2, "F1")]:
        print(f"Plotting {metric_name} ...")
        _plot_metric(all_data, metric_idx, metric_name, out_dir)
    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR),
                        help="Root of the result corpus (default: reformat_results/)")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_tfidf = sub.add_parser("tfidf", help="Write TF-IDF DU scores back into result files.")
    p_tfidf.add_argument("--top-k", type=int, default=10,
                         help="Reference set size: seed + top-(k-1) similar (default: 10)")

    sub.add_parser("collective",
                   help="Write collective-reference DU scores back into result files.")

    p_topk = sub.add_parser("topk", help="Plot DU heatmaps at k=1,2,5,10 + collective.")
    p_topk.add_argument("--output-dir", default=str(ROOT / "results" / "eval"))

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if args.mode == "tfidf":
        run_tfidf(results_dir, args.top_k)
    elif args.mode == "collective":
        run_collective(results_dir)
    elif args.mode == "topk":
        run_topk(results_dir, Path(args.output_dir))


if __name__ == "__main__":
    main()
