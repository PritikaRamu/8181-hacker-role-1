"""
Enrich instruction JSON files with package-recommendation reference data.

Two subcommands — run in order before compute_du.py:

  similar     Add a `similar_packages` field to each prompt: top-10 packages
              ranked by TF-IDF cosine similarity against the full language
              registry (resource NDJSON files). Parallelised across languages
              and similarity batches.

  collective  Add a `collective_packages` field: packages that valid models
              actually generated in (strategy × model) code experiments, with
              co-occurrence counts.  Run after `similar`.

  all         Run `similar` then `collective`.

Usage:
    python enrich_packages.py similar
    python enrich_packages.py similar --lang Python --workers 1
    python enrich_packages.py collective
    python enrich_packages.py all
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

DATA_DIR     = Path(__file__).parent / "data"
INSTRUCT_DIR = DATA_DIR / "instruction"
RESOURCE_DIR = DATA_DIR / "resources"
RESULTS_DIR  = Path(__file__).parent / "reformat_results"

LANGUAGES = [
    {
        "lang":          "Python",
        "instruct":      "packages_Python_instruct.json",
        "resource":      "bq-results-python-20260704-133336-1783172071559.json",
        "max_resources": 300_000,
    },
    {
        "lang":          "JavaScript",
        "instruct":      "packages_JavaScript_instruct.json",
        "resource":      "bq-results-npm-20260704-133930-1783172606939.json",
        "max_resources": 300_000,
    },
    {
        "lang":          "Ruby",
        "instruct":      "packages_Ruby_instruct.json",
        "resource":      "bq-results-ruby-20260704-121810-1783167518838.json",
        "max_resources": None,
    },
    {
        "lang":          "Rust",
        "instruct":      "packages_Rust_instruct.json",
        "resource":      "bq-results-rust.json",
        "max_resources": None,
    },
]

# ── similar: TF-IDF constants ─────────────────────────────────────────────────

BATCH_SIZE   = 256
MAX_FEATURES = 100_000
MIN_DESC_LEN = 5
TOP_K        = 10


# ── similar helpers ───────────────────────────────────────────────────────────

def _load_resources(path: Path, max_resources: int | None = None) -> tuple[list[str], list[str]]:
    names, descs = [], []
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc=f"  loading {path.name}", unit=" lines", leave=False):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            desc = (entry.get("description") or "").strip()
            if len(desc) >= MIN_DESC_LEN:
                names.append(entry["name"])
                descs.append(desc)
            if max_resources and len(names) >= max_resources:
                break
    return names, descs


def _similarity_batch(
    batch_descs: list[str],
    batch_names: list[str],
    resource_names: list[str],
    resource_tfidf,
    vectorizer: TfidfVectorizer,
    top_k: int,
) -> list[list[dict]]:
    query_vecs = vectorizer.transform(batch_descs)
    sims = cosine_similarity(query_vecs, resource_tfidf)
    results = []
    for i, qname in enumerate(batch_names):
        row = sims[i].copy()
        qname_lower = qname.lower()
        for j, rname in enumerate(resource_names):
            if rname.lower() == qname_lower:
                row[j] = -1.0
        top_idx = np.argpartition(row, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
        results.append([
            {"name": resource_names[k], "score": round(float(row[k]), 4)}
            for k in top_idx if row[k] >= 0.0
        ])
    return results


def process_language_similar(config: dict, inner_jobs: int = 4) -> str:
    """Process one language for the 'similar' subcommand. Runs inside a subprocess."""
    lang = config["lang"]
    instruct_path = INSTRUCT_DIR / config["instruct"]
    resource_path = RESOURCE_DIR / config["resource"]
    t_total = time.time()

    max_resources = config.get("max_resources")
    limit_note = f" (top {max_resources:,})" if max_resources else ""
    print(f"[{lang}] Loading {resource_path.name}{limit_note} ...", flush=True)
    res_names, res_descs = _load_resources(resource_path, max_resources=max_resources)
    print(f"[{lang}] {len(res_names):,} resources loaded", flush=True)

    print(f"[{lang}] Fitting TF-IDF ...", flush=True)
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES, sublinear_tf=True, strip_accents="unicode",
        analyzer="word", ngram_range=(1, 2), min_df=2,
    )
    res_tfidf = vectorizer.fit_transform(res_descs)
    print(f"[{lang}] TF-IDF shape {res_tfidf.shape}", flush=True)

    with open(instruct_path, encoding="utf-8") as f:
        data = json.load(f)
    packages = data["packages"]

    batches = [
        (
            [(p.get("description") or "").strip() for p in packages[s : s + BATCH_SIZE]],
            [p["name"] for p in packages[s : s + BATCH_SIZE]],
        )
        for s in range(0, len(packages), BATCH_SIZE)
    ]

    print(f"[{lang}] Computing similarities: {len(packages)} packages "
          f"in {len(batches)} batches (inner_jobs={inner_jobs}) ...", flush=True)

    batch_results = Parallel(n_jobs=inner_jobs, prefer="threads")(
        delayed(_similarity_batch)(descs, names, res_names, res_tfidf, vectorizer, TOP_K)
        for descs, names in tqdm(batches, desc=f"  [{lang}] batches", unit=" batch")
    )

    all_similar = [entry for batch in batch_results for entry in batch]
    for pkg, sim in zip(packages, all_similar):
        pkg["similar_packages"] = sim

    with open(instruct_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_total
    msg = f"[{lang}] Done — {len(packages)} packages enriched in {elapsed:.1f}s"
    print(msg, flush=True)
    return msg


def run_similar(lang_filter: str | None, workers: int | None, inner_jobs: int) -> None:
    configs = LANGUAGES
    if lang_filter:
        configs = [c for c in LANGUAGES if c["lang"].lower() == lang_filter.lower()]
        if not configs:
            raise ValueError(f"Unknown language '{lang_filter}'. Choose from: {[c['lang'] for c in LANGUAGES]}")

    workers = workers or len(configs)
    if len(configs) == 1:
        process_language_similar(configs[0], inner_jobs=inner_jobs)
    else:
        print(f"Running {len(configs)} languages with {workers} parallel worker(s).\n")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_language_similar, cfg, inner_jobs): cfg["lang"]
                       for cfg in configs}
            with tqdm(as_completed(futures), total=len(futures), desc="languages") as pbar:
                for fut in pbar:
                    lang = futures[fut]
                    try:
                        pbar.set_postfix_str(f"finished: {lang}")
                        print(fut.result())
                    except Exception as exc:
                        print(f"[{lang}] FAILED: {exc}")
    print("\nAll done.")


# ── collective helpers ────────────────────────────────────────────────────────

def _load_valid_names(resource_path: Path) -> set[str]:
    names: set[str] = set()
    with open(resource_path, encoding="utf-8") as f:
        for line in tqdm(f, desc=f"  loading {resource_path.name}", unit=" lines", leave=False):
            line = line.strip()
            if not line:
                continue
            try:
                names.add(json.loads(line)["name"].lower())
            except (json.JSONDecodeError, KeyError):
                continue
    return names


def _collect_for_language(lang: str, valid_names: set[str]) -> dict[str, dict[str, int]]:
    """Walk code/ result files and count valid co-occurring packages per seed."""
    code_files = sorted(RESULTS_DIR.rglob(f"*/code/*/{lang}.json"))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for path in tqdm(code_files, desc=f"  [{lang}] scanning code files", unit=" file"):
        try:
            packages = json.loads(path.read_text(encoding="utf-8")).get("packages", [])
        except json.JSONDecodeError:
            continue
        for entry in packages:
            seed = entry.get("package_name", "").lower()
            if not seed:
                continue
            for pkg in entry.get("extracted_packages") or []:
                pkg_lower = pkg.lower()
                if pkg_lower != seed and pkg_lower in valid_names:
                    counts[seed][pkg_lower] += 1
    return counts


def run_collective(lang_filter: str | None) -> None:
    lang_map = {c["lang"]: c for c in LANGUAGES}
    if lang_filter and lang_filter not in lang_map:
        raise ValueError(f"Unknown language '{lang_filter}'. Choose from: {list(lang_map)}")
    targets = {lang_filter: lang_map[lang_filter]} if lang_filter else lang_map

    for lang, cfg in targets.items():
        print(f"\n{'='*60}\n[{lang}]")
        valid_names = _load_valid_names(RESOURCE_DIR / cfg["resource"])
        print(f"[{lang}] {len(valid_names):,} valid names loaded.")
        counts = _collect_for_language(lang, valid_names)
        print(f"[{lang}] {len(counts)} prompts with at least one collective package.")

        instruct_path = INSTRUCT_DIR / cfg["instruct"]
        data = json.loads(instruct_path.read_text(encoding="utf-8"))
        for entry in data["packages"]:
            seed_lower = entry["name"].lower()
            pkg_counts = counts.get(seed_lower, {})
            entry["collective_packages"] = [
                {"name": pkg, "count": cnt}
                for pkg, cnt in sorted(pkg_counts.items(), key=lambda x: -x[1])
            ]
        instruct_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{lang}] Done.")

    print("\nAll done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    def _add_lang(p):
        p.add_argument("--lang", default=None, help="Process only one language (e.g. Python).")

    def _add_parallel(p):
        p.add_argument("--workers", type=int, default=None,
                       help="Language-level parallel workers (default: all 4).")
        p.add_argument("--inner-jobs", type=int, default=4,
                       help="Threads per language for batch similarity (default: 4).")

    p_sim = sub.add_parser("similar", help="Add similar_packages via TF-IDF.")
    _add_lang(p_sim)
    _add_parallel(p_sim)

    p_coll = sub.add_parser("collective", help="Add collective_packages from code experiments.")
    _add_lang(p_coll)

    p_all = sub.add_parser("all", help="Run similar then collective.")
    _add_lang(p_all)
    _add_parallel(p_all)

    args = parser.parse_args()

    if args.mode == "similar":
        run_similar(args.lang, args.workers, args.inner_jobs)
    elif args.mode == "collective":
        run_collective(args.lang)
    elif args.mode == "all":
        print("=== Step 1: similar packages ===")
        run_similar(args.lang, args.workers, args.inner_jobs)
        print("\n=== Step 2: collective packages ===")
        run_collective(args.lang)


if __name__ == "__main__":
    main()
