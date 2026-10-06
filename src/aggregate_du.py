"""
Aggregate DU scores from all result files and group by strategy, model, language.

Outputs:
  - results/du_scores.csv        : one row per (strategy, model, language)
  - results/du_by_strategy.csv   : averaged across models and languages
  - results/du_by_model.csv      : averaged across strategies and languages
  - results/du_by_language.csv   : averaged across strategies and models

Usage:
  python aggregate_du.py
  python aggregate_du.py --output results/du_scores.csv   # override output dir
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "reformat_results"
OUTPUT_DIR  = Path(__file__).parent / "results"

METRICS = [
    "macro_du", "macro_du_precision", "macro_du_recall",
    "macro_du_coll", "macro_du_coll_precision", "macro_du_coll_recall",
    "phr_with_stdlib", "phr_without_stdlib",
]


# ── helpers ─────────────────────────────────────────────────────────────────

def collect_rows() -> list[dict]:
    rows = []
    for path in sorted(RESULTS_DIR.rglob("packages/**/*.json")):
        parts = path.parts
        strategy = parts[-4]
        model    = parts[-2]
        language = path.stem

        with open(path, encoding="utf-8") as f:
            stats = json.load(f).get("stats", {})

        row = {"strategy": strategy, "model": model, "language": language}
        for m in METRICS:
            row[m] = stats.get(m)
        rows.append(row)
    return rows


def avg(values: list):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def group_by(rows: list[dict], keys: list[str]) -> list[dict]:
    """Average all metrics grouped by the given key columns."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        bucket_key = tuple(row[k] for k in keys)
        buckets[bucket_key].append(row)

    result = []
    for key_vals, group in sorted(buckets.items()):
        out = dict(zip(keys, key_vals))
        out["n_files"] = len(group)
        for m in METRICS:
            out[m] = avg([r[m] for r in group])
        result.append(out)
    return result


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows):>3} rows → {path}")


def print_table(rows: list[dict], title: str) -> None:
    if not rows:
        return
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    keys = list(rows[0].keys())
    col_w = {k: max(len(k), max((len(str(r[k])) for r in rows), default=0)) for k in keys}
    header = "  " + "  ".join(k.ljust(col_w[k]) for k in keys)
    print(header)
    print("  " + "  ".join("-" * col_w[k] for k in keys))
    for row in rows:
        print("  " + "  ".join(str(row[k]).ljust(col_w[k]) for k in keys))


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="Directory to write CSV files (default: results/)")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    print("Collecting result files ...")
    rows = collect_rows()
    print(f"  {len(rows)} files found.\n")

    # Full detail
    write_csv(rows, out_dir / "du_scores.csv")

    # Grouped views
    by_strategy = group_by(rows, ["strategy"])
    by_model     = group_by(rows, ["model"])
    by_language  = group_by(rows, ["language"])
    by_strat_model = group_by(rows, ["strategy", "model"])
    by_strat_lang  = group_by(rows, ["strategy", "language"])

    write_csv(by_strategy,    out_dir / "du_by_strategy.csv")
    write_csv(by_model,       out_dir / "du_by_model.csv")
    write_csv(by_language,    out_dir / "du_by_language.csv")
    write_csv(by_strat_model, out_dir / "du_by_strategy_model.csv")
    write_csv(by_strat_lang,  out_dir / "du_by_strategy_language.csv")

    # Print summaries to terminal
    print_table(by_strategy, "DU by strategy  (avg across models & languages)")
    print_table(by_language, "DU by language  (avg across strategies & models)")
    print_table(by_model,    "DU by model     (avg across strategies & languages)")


if __name__ == "__main__":
    main()
