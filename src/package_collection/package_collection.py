"""
collect_registry_packages.py
─────────────────────────────────────────────────────────────────────────────
Fetch ALL packages (or top-N if --top-n is given) from four registries:

    PyPI        → pypi_packages.json
    npm         → npm_packages.json
    RubyGems    → rubygems_packages.json
    Crates.io   → cratesio_packages.json

Each output file has the structure:
    {
      "registry":      "<name>",
      "snapshot_date": "YYYY-MM-DD",
      "count":         <int>,
      "packages": [
        {
          "name":        "<str>",
          "description": "<str>",
          "version":     "<str>",
          "homepage":    "<str>",
          "downloads":   <int|float|null>
        },
        ...
      ]
    }

Usage
─────
    # all packages per registry → ./registry_snapshots/
    python collect_registry_packages.py

    # customise
    python collect_registry_packages.py --top-n 500 --output-dir ./data --delay 0.3

    # single registry
    python collect_registry_packages.py --registries pypi cratesio

Dependencies
────────────
    pip install requests tqdm

Rate-limit notes
────────────────
    PyPI      : individual JSON calls are parallelised (bounded pool).
    npm       : keyword sweep parallelised across all terms; deduplicates by score.
    RubyGems  : pages fetched in parallel batches; stops when a page returns empty.
    Crates.io : 1 req/s hard limit enforced by the API; sequential; delay ≥ 1 s.
                API caps at 10,000 results total (page 100 × per_page 100).
"""

import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import requests
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# Crates.io policy: https://crates.io/policies
# Must include a descriptive User-Agent with contact info.
CRATES_USER_AGENT = (
    "package-hallucination-research/1.0 "
    "(academic; contact: your-email@example.com)"
)

# Standard browser-like User-Agent for registries that require one.
DEFAULT_UA = "package-hallucination-research/1.0 (academic)"

TODAY = date.today().isoformat()

# npm keyword sweep: broad terms that together cover the most-downloaded packages.
# Each term fetches up to 250 results; duplicates are removed and the union
# is re-ranked by the registry's own popularity score.
NPM_SWEEP_TERMS = [
    "utils", "types", "core", "http", "async", "stream",
    "test", "cli", "react", "node", "webpack", "babel",
    "eslint", "typescript", "express", "parser", "format",
    "logger", "db", "crypto", "path", "fs", "buffer", "query",
    "server", "client", "api", "middleware", "router", "validate",
    "config", "build", "deploy", "plugin", "loader", "compiler",
    "schema", "model", "auth", "event", "cache", "queue",
]


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP helper
# ═══════════════════════════════════════════════════════════════════════════════

def _get(url: str,
         params: dict = None,
         headers: dict = None,
         retries: int = 3,
         backoff: float = 2.0) -> requests.Response:
    """GET with simple exponential-backoff retry."""
    _headers = {"User-Agent": DEFAULT_UA}
    if headers:
        _headers.update(headers)
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=_headers, timeout=20)
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            wait = backoff ** attempt
            log.warning("  attempt %d/%d failed (%s) — retrying in %.1fs",
                        attempt, retries, exc, wait)
            time.sleep(wait)


def _save(packages: list, registry: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "registry":      registry,
        "snapshot_date": TODAY,
        "count":         len(packages),
        "packages":      packages,
    }
    slug = registry.lower().replace(".", "_").replace(" ", "_")
    path = output_dir / f"{slug}_packages.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("  ✓  saved %d packages → %s", len(packages), path)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# PyPI
# ═══════════════════════════════════════════════════════════════════════════════

def _pypi_detail(name: str) -> dict:
    """Fetch metadata for a single PyPI package via the JSON API."""
    try:
        r = _get(f"https://pypi.org/pypi/{name}/json")
        info = r.json()["info"]
        return {
            "name":        info.get("name", name),
            "description": (info.get("summary") or "").strip(),
            "version":     info.get("version", ""),
            "homepage":    info.get("home_page") or info.get("project_url") or "",
            "downloads":   None,  # filled from the top-packages ranking below
        }
    except Exception as exc:
        log.debug("  PyPI detail failed for %s: %s", name, exc)
        return {
            "name": name, "description": "",
            "version": "", "homepage": "", "downloads": None,
        }


def fetch_pypi(n: int = None, delay: float = 0.1, workers: int = 8) -> list:
    """
    All PyPI packages by 30-day downloads (or top-N if n is set).

    Ranking source : Hugo van Kemenade's pre-computed dataset (raw GitHub).
    Detail source  : pypi.org/pypi/<name>/json  (parallelised fetch).
    """
    log.info("[PyPI] fetching %s packages …", f"top-{n}" if n else "all")

    # Step 1: popularity ranking — fetch the full list, slice only if n given
    ranking_url = (
        "https://raw.githubusercontent.com/hugovk/"
        "top-pypi-packages/main/top-pypi-packages-30-days.min.json"
    )
    r = _get(ranking_url)
    rows = r.json()["rows"]
    if n:
        rows = rows[:n]
    name_to_dl = {row["project"]: row["download_count"] for row in rows}
    names = list(name_to_dl.keys())
    log.info("[PyPI] ranking fetched — enriching %d entries via PyPI JSON API …",
             len(names))

    # Step 2: parallel detail fetches
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_pypi_detail, name): name for name in names}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="PyPI", unit="pkg"):
            pkg = fut.result()
            pkg["downloads"] = name_to_dl.get(pkg["name"], 0)
            results[pkg["name"]] = pkg
            time.sleep(delay / workers)

    # Restore original rank order
    ordered = [results.get(name, {"name": name, "description": "",
                                   "version": "", "homepage": "",
                                   "downloads": name_to_dl[name]})
               for name in names]
    return ordered


# ═══════════════════════════════════════════════════════════════════════════════
# npm
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_npm_term(term: str, seen: dict, lock: threading.Lock,
                    delay: float = 0.0) -> int:
    """Fetch one sweep term and merge new packages into `seen`. Returns new count."""
    try:
        r = _get(
            "https://registry.npmjs.org/-/v1/search",
            params={"text": term, "size": 250, "from": 0},
        )
        added = 0
        for obj in r.json().get("objects", []):
            pkg  = obj.get("package", {})
            name = pkg.get("name", "")
            if not name:
                continue
            with lock:
                if name in seen:
                    continue
                seen[name] = {
                    "name":        name,
                    "description": (pkg.get("description") or "").strip(),
                    "version":     pkg.get("version", ""),
                    "homepage":    pkg.get("links", {}).get("homepage", ""),
                    "downloads":   obj.get("score", {}).get("final", 0),
                }
                added += 1
        time.sleep(delay)
        return added
    except Exception as exc:
        log.warning("  npm sweep term %r failed: %s", term, exc)
        return 0


def fetch_npm(n: int = None, delay: float = 0.4, workers: int = 8) -> list:
    """
    All npm packages reachable via keyword sweep (or top-N if n is set).

    All NPM_SWEEP_TERMS are fetched in parallel (max `workers` at a time).
    Results are deduplicated by name and re-ranked by registry score.
    """
    log.info("[npm] fetching %s packages via parallel keyword sweep …",
             f"top-{n}" if n else "all")
    seen: dict[str, dict] = {}
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_npm_term, term, seen, lock, delay): term
                   for term in NPM_SWEEP_TERMS}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="npm sweep", unit="term"):
            fut.result()  # surface exceptions via tqdm

    ranked = sorted(seen.values(), key=lambda x: -x["downloads"])
    log.info("[npm] collected %d unique packages across all sweep terms", len(ranked))
    return ranked[:n] if n else ranked


# ═══════════════════════════════════════════════════════════════════════════════
# RubyGems
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_rubygems_page(page: int) -> list:
    """Fetch one page of the RubyGems search endpoint; returns parsed records."""
    try:
        r = _get(
            "https://rubygems.org/api/v1/search.json",
            params={"query": "", "page": page},
        )
        return [
            {
                "name":        gem.get("name", ""),
                "description": (gem.get("info") or "").strip(),
                "version":     gem.get("version", ""),
                "homepage":    gem.get("homepage_uri", ""),
                "downloads":   gem.get("downloads", 0),
            }
            for gem in r.json()
        ]
    except Exception as exc:
        log.warning("  RubyGems page %d failed: %s", page, exc)
        return []


def fetch_rubygems(n: int = None, delay: float = 0.5, workers: int = 8) -> list:
    """
    All RubyGems packages (or top-N if n is set).

    Pages are fetched in parallel batches of `workers` size. The loop
    stops when any page in a batch returns an empty response, ensuring
    we never miss results from pages that precede an empty one.
    """
    log.info("[RubyGems] fetching %s packages …", f"top-{n}" if n else "all")
    packages = []
    page = 1

    with tqdm(desc="RubyGems", unit="pkg") as bar:
        while True:
            batch_pages = list(range(page, page + workers))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_fetch_rubygems_page, p): p
                           for p in batch_pages}
                batch: dict[int, list] = {}
                for fut in as_completed(futures):
                    batch[futures[fut]] = fut.result()

            exhausted = False
            for p in batch_pages:
                gems = batch[p]
                if not gems:
                    exhausted = True
                    break
                packages.extend(gems)
                bar.update(len(gems))

            if exhausted:
                log.info("[RubyGems] exhausted at page %d", page + batch_pages.index(
                    next(p for p in batch_pages if not batch[p])))
                break
            if n and len(packages) >= n:
                break

            page += workers
            time.sleep(delay)

    return packages[:n] if n else packages


# ═══════════════════════════════════════════════════════════════════════════════
# Crates.io
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_cratesio(n: int = None, delay: float = 1.0, workers: int = 8) -> list:
    """
    All Crates.io packages sorted by all-time download count (or top-N if n is set).

    Crates.io enforces a strict 1 req/s rate limit on /api/v1/crates;
    delay is clamped to ≥ 1.0 s and fetching is sequential.
    The API caps results at 10,000 total (page 100 × per_page 100).
    """
    _ = workers  # sequential due to Crates.io 1 req/s rate limit
    log.info("[Crates.io] fetching %s packages …", f"top-{n}" if n else "all")
    packages = []
    page = 1
    per_page = 100  # API maximum

    with tqdm(desc="Crates.io", unit="pkg") as bar:
        while True:
            r = _get(
                "https://crates.io/api/v1/crates",
                params={"page": page, "per_page": per_page, "sort": "downloads"},
                headers={"User-Agent": CRATES_USER_AGENT},
            )
            crates = r.json().get("crates", [])
            if not crates:
                log.info("[Crates.io] no more results at page %d", page)
                break

            for crate in crates:
                packages.append({
                    "name":        crate.get("name", ""),
                    "description": (crate.get("description") or "").strip(),
                    "version":     crate.get("newest_version", ""),
                    "homepage":    crate.get("homepage") or "",
                    "downloads":   crate.get("downloads", 0),
                })

            bar.update(len(crates))
            page += 1
            time.sleep(max(delay, 1.0))

            if n and len(packages) >= n:
                break

    return packages[:n] if n else packages


# ═══════════════════════════════════════════════════════════════════════════════
# Registry dispatch table
# ═══════════════════════════════════════════════════════════════════════════════

REGISTRY_FETCHERS = {
    "pypi":     ("PyPI",      fetch_pypi),
    "npm":      ("npm",       fetch_npm),
    "rubygems": ("RubyGems",  fetch_rubygems),
    "cratesio": ("Crates.io", fetch_cratesio),
}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Collect ALL packages (or top-N via --top-n) from PyPI, npm, "
            "RubyGems, and Crates.io and store them as JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--top-n", type=int, default=None, metavar="N",
        help="Limit packages per registry (omit to fetch all available).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("registry_snapshots"),
        metavar="DIR",
        help="Directory where JSON snapshots are written.",
    )
    p.add_argument(
        "--delay", type=float, default=0.5, metavar="SEC",
        help=(
            "Base delay between HTTP requests (seconds). "
            "Crates.io uses max(delay, 1.0) to respect its 1 req/s limit."
        ),
    )
    p.add_argument(
        "--registries", nargs="+",
        choices=list(REGISTRY_FETCHERS.keys()) + ["all"],
        default=["all"],
        metavar="REG",
        help="Which registries to collect. Choices: pypi  npm  rubygems  cratesio  all.",
    )
    p.add_argument(
        "--workers", type=int, default=8, metavar="W",
        help="Parallel workers for PyPI detail calls, npm sweep, and RubyGems pages.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    targets = (
        list(REGISTRY_FETCHERS.keys())
        if "all" in args.registries
        else args.registries
    )

    log.info("═" * 60)
    log.info("Registry snapshot  |  top-N=%s  |  date=%s",
             args.top_n if args.top_n else "all", TODAY)
    log.info("Registries : %s", ", ".join(targets))
    log.info("Output dir : %s", args.output_dir.resolve())
    log.info("Workers    : %d", args.workers)
    log.info("═" * 60)

    results = {}
    for key in targets:
        display_name, fetcher = REGISTRY_FETCHERS[key]
        log.info("")
        try:
            packages = fetcher(n=args.top_n, delay=args.delay, workers=args.workers)
            path = _save(packages, display_name, args.output_dir)
            results[display_name] = {
                "status": "ok", "count": len(packages), "path": str(path),
            }
        except Exception as exc:
            log.error("[%s] collection failed: %s", display_name, exc,
                      exc_info=True)
            results[display_name] = {"status": "error", "error": str(exc)}

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("═" * 60)
    log.info("Summary")
    log.info("═" * 60)
    for reg, info in results.items():
        if info["status"] == "ok":
            log.info("  %-12s  ✓  %d packages  →  %s",
                     reg, info["count"], info["path"])
        else:
            log.info("  %-12s  ✗  %s", reg, info["error"])


if __name__ == "__main__":
    main()
