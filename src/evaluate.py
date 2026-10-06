"""
Evaluation: tables and figures for PHR and DU metrics.

Outputs (under results/):
  Tables (LaTeX):
    table_by_strategy.tex
    table_by_model.tex
    table_by_language.tex
    table_strategy_model.tex

  Figures (PDF + PNG):
    fig_strategy_bars.pdf      — grouped bars: PHR / DU_coll / DU_tfidf by strategy
    fig_model_bars.pdf         — same, by model
    fig_language_bars.pdf      — same, by language
    fig_phr_vs_du.pdf          — scatter: PHR vs DU_coll per (strategy × model × language)
    fig_heatmap_strategy_model.pdf — heatmap of macro_du_coll over strategy × model
    fig_tfidf_vs_coll.pdf      — scatter: TF-IDF DU vs collective DU

Usage:
  python evaluate.py
  python evaluate.py --output-dir results/eval
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE = sns.color_palette("Set2")
FIGSIZE_BAR  = (10, 5)
FIGSIZE_HEAT = (9, 5)
FIGSIZE_SCAT = (7, 5)
DPI = 150

RESULTS_DIR = Path(__file__).parent / "results"

STRATEGY_ORDER = [
    "baseline", "greedy_decoding", "dola",
    "contrastive_decoding", "nudging", "self_refine",
]
STRATEGY_LABELS = {
    "baseline":             "Baseline",
    "greedy_decoding":      "Greedy",
    "dola":                 "DoLA",
    "contrastive_decoding": "Contrast.",
    "nudging":              "Nudging",
    "self_refine":          "Self-refine",
}
MODEL_ORDER = [
    "deepseek-1.3b", "deepseek-6.7b",
    "gemma-1b", "gemma-4b",
    "qwen-1.5b", "qwen-3b", "llama-8b", "mistral-7b"
]
LANG_ORDER = ["Python", "JavaScript", "Ruby", "Rust"]


# ── helpers ──────────────────────────────────────────────────────────────────

def save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"  Saved {stem}.pdf / .png")


def grouped_bar(
    df: pd.DataFrame,
    group_col: str,
    metrics: list[tuple[str, str, float]],   # [(col, label, scale), ...]
    order: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    out_dir: Path,
    stem: str,
    xticklabels: dict[str, str] | None = None,
) -> None:
    df = df.set_index(group_col).reindex(order).reset_index()
    x = np.arange(len(order))
    n = len(metrics)
    width = 0.75 / n

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    for i, (col, label, scale) in enumerate(metrics):
        vals = df[col].fillna(0).values * scale
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width, label=label,
               color=PALETTE[i], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    tick_labels = [(xticklabels or {}).get(v, v) for v in order]
    ax.set_xticklabels(tick_labels, rotation=20, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    fig.tight_layout()
    save(fig, out_dir, stem)


# ── tables ───────────────────────────────────────────────────────────────────

def fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.4f}"

def fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.2f}"


def latex_table(
    df: pd.DataFrame,
    index_col: str,
    order: list[str],
    label_map: dict[str, str] | None,
    caption: str,
    label: str,
) -> str:
    df = df.set_index(index_col).reindex(order)

    cols = [
        ("phr_without_stdlib",      r"PHR$\uparrow$",        fmt_pct),
        ("macro_du",                r"$\mathsf{DU}$",        fmt),
        ("macro_du_precision",      r"$\mathsf{DU_P}$",      fmt),
        ("macro_du_recall",         r"$\mathsf{DU_R}$",      fmt),
        ("macro_du_coll",           r"$\mathsf{DU^c}$",      fmt),
        ("macro_du_coll_precision", r"$\mathsf{DU^c_P}$",    fmt),
        ("macro_du_coll_recall",    r"$\mathsf{DU^c_R}$",    fmt),
    ]

    header_cols = " & ".join(h for _, h, _ in cols)
    n_cols = 1 + len(cols)
    col_spec = "l" + "r" * len(cols)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf" & {header_cols} \\",
        r"\midrule",
    ]

    best = {}
    for col, _, fn in cols:
        col_vals = [df.loc[k, col] for k in order if k in df.index]
        col_vals = [v for v in col_vals if v is not None and not np.isnan(v)]
        if col_vals:
            # PHR: lower is better; DU: higher is better
            best[col] = min(col_vals) if "phr" in col else max(col_vals)

    for k in order:
        if k not in df.index:
            continue
        row = df.loc[k]
        name = (label_map or {}).get(k, k)
        cells = []
        for col, _, fn in cols:
            val = row.get(col)
            fval = fn(val)
            if val is not None and not np.isnan(float(val if val is not None else float('nan'))) \
               and col in best and abs(val - best[col]) < 1e-9:
                fval = r"\textbf{" + fval + "}"
            cells.append(fval)
        lines.append(f"  {name} & " + " & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── figures ──────────────────────────────────────────────────────────────────

def fig_heatmap(df_detail: pd.DataFrame, metric: str, title: str,
                out_dir: Path, stem: str) -> None:
    pivot = (
        df_detail.groupby(["strategy", "model"])[metric]
        .mean()
        .unstack("model")
        .reindex(index=STRATEGY_ORDER, columns=MODEL_ORDER)
        * 100
    )
    fig, ax = plt.subplots(figsize=FIGSIZE_HEAT)
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="YlGnBu",
        linewidths=0.5, linecolor="white",
        xticklabels=[m.replace("-", "\n") for m in MODEL_ORDER],
        yticklabels=[STRATEGY_LABELS.get(s, s) for s in STRATEGY_ORDER],
        ax=ax,
        cbar_kws={"label": "Score (×100)"},
    )
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Strategy")
    fig.tight_layout()
    save(fig, out_dir, stem)


def fig_scatter_phr_du(df_detail: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_SCAT)
    strategies = df_detail["strategy"].unique()
    palette = dict(zip(STRATEGY_ORDER, PALETTE))
    for strat in STRATEGY_ORDER:
        sub = df_detail[df_detail["strategy"] == strat]
        ax.scatter(
            sub["phr_without_stdlib"],
            sub["macro_du_coll"] * 100,
            label=STRATEGY_LABELS.get(strat, strat),
            color=palette.get(strat, "gray"),
            alpha=0.75, s=60, edgecolors="white", linewidths=0.5,
        )
    ax.set_xlabel("PHR (without stdlib, %)")
    ax.set_ylabel(r"Macro-$\mathsf{DU^c}$ (×100)")
    ax.set_title("Safety–Utility Trade-off\n(each point = strategy × model × language)")
    ax.legend(title="Strategy", bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.9)
    fig.tight_layout()
    save(fig, out_dir, "fig_phr_vs_du")


def fig_scatter_tfidf_vs_coll(df_detail: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_SCAT)
    langs = df_detail["language"].unique()
    palette = dict(zip(LANG_ORDER, PALETTE))
    for lang in LANG_ORDER:
        sub = df_detail[df_detail["language"] == lang]
        ax.scatter(
            sub["macro_du"] * 100,
            sub["macro_du_coll"] * 100,
            label=lang,
            color=palette.get(lang, "gray"),
            alpha=0.75, s=60, edgecolors="white", linewidths=0.5,
        )
    lo = min(df_detail["macro_du"].min(), df_detail["macro_du_coll"].min()) * 100
    hi = max(df_detail["macro_du"].max(), df_detail["macro_du_coll"].max()) * 100
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5, label="y = x")
    ax.set_xlabel(r"$\mathsf{DU}$ TF-IDF (×100)")
    ax.set_ylabel(r"$\mathsf{DU^c}$ Collective (×100)")
    ax.set_title("TF-IDF vs Collective Reference")
    ax.legend(title="Language", framealpha=0.9)
    fig.tight_layout()
    save(fig, out_dir, "fig_tfidf_vs_coll")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "eval"))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df        = pd.read_csv(RESULTS_DIR / "du_scores.csv")
    by_strat  = pd.read_csv(RESULTS_DIR / "du_by_strategy.csv")
    by_model  = pd.read_csv(RESULTS_DIR / "du_by_model.csv")
    by_lang   = pd.read_csv(RESULTS_DIR / "du_by_language.csv")
    by_sm     = pd.read_csv(RESULTS_DIR / "du_by_strategy_model.csv")

    print("Generating tables ...\n")

    # LaTeX tables
    tables = [
        (by_strat,  "strategy", STRATEGY_ORDER, STRATEGY_LABELS,
         r"Results by mitigation strategy (macro-averaged over models and languages). "
         r"DU = TF-IDF reference; $\mathsf{DU^c}$ = collective reference. Bold = best.",
         "tab:by_strategy", "table_by_strategy"),
        (by_model, "model", MODEL_ORDER, None,
         r"Results by model (macro-averaged over strategies and languages). Bold = best.",
         "tab:by_model", "table_by_model"),
        (by_lang, "language", LANG_ORDER, None,
         r"Results by language (macro-averaged over strategies and models). Bold = best.",
         "tab:by_language", "table_by_language"),
    ]

    for df_t, idx_col, order, lmap, caption, label, stem in tables:
        tex = latex_table(df_t, idx_col, order, lmap, caption, label)
        path = out_dir / f"{stem}.tex"
        path.write_text(tex, encoding="utf-8")
        print(f"  Saved {stem}.tex")

    print("\nGenerating figures ...\n")

    # Bar: by strategy
    grouped_bar(
        by_strat, "strategy",
        [
            ("phr_without_stdlib",   "PHR (%)",                  1.0),
            ("macro_du_coll",        r"$\mathsf{DU^c}$ (×100)", 100.0),
            ("macro_du",             r"$\mathsf{DU}$ TF-IDF (×100)", 100.0),
        ],
        STRATEGY_ORDER,
        xlabel="Strategy", ylabel="Score",
        title="PHR and DU by Mitigation Strategy",
        out_dir=out_dir, stem="fig_strategy_bars",
        xticklabels=STRATEGY_LABELS,
    )

    # Bar: by model
    grouped_bar(
        by_model, "model",
        [
            ("phr_without_stdlib",   "PHR (%)",                  1.0),
            ("macro_du_coll",        r"$\mathsf{DU^c}$ (×100)", 100.0),
            ("macro_du",             r"$\mathsf{DU}$ TF-IDF (×100)", 100.0),
        ],
        MODEL_ORDER,
        xlabel="Model", ylabel="Score",
        title="PHR and DU by Model",
        out_dir=out_dir, stem="fig_model_bars",
    )

    # Bar: by language
    grouped_bar(
        by_lang, "language",
        [
            ("phr_without_stdlib",   "PHR (%)",                  1.0),
            ("macro_du_coll",        r"$\mathsf{DU^c}$ (×100)", 100.0),
            ("macro_du",             r"$\mathsf{DU}$ TF-IDF (×100)", 100.0),
        ],
        LANG_ORDER,
        xlabel="Language", ylabel="Score",
        title="PHR and DU by Language",
        out_dir=out_dir, stem="fig_language_bars",
    )

    # Heatmaps
    fig_heatmap(df, "macro_du_coll",
                r"Macro-$\mathsf{DU^c}$ (×100): Strategy × Model",
                out_dir, "fig_heatmap_du_coll")
    fig_heatmap(df, "phr_without_stdlib",
                "PHR (%) without stdlib: Strategy × Model",
                out_dir, "fig_heatmap_phr")

    # Scatters
    fig_scatter_phr_du(df, out_dir)
    fig_scatter_tfidf_vs_coll(df, out_dir)

    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
