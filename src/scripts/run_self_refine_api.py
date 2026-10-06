#!/usr/bin/env python3
"""
Run the self_refine mitigation strategy via any OpenAI-compatible API
(vLLM, Ollama, LM Studio, …) without loading a model in-process.

Results are written to:
    reformat_results/self_refine/<task>/<alias>/<Language>.json

Then enhance_reformat_results.py is called automatically to compute PHR stats.

Usage examples
--------------
# vLLM (llama)
python scripts/run_self_refine_api.py \\
    --base-url http://localhost:8000 \\
    --inference-model meta-llama/Llama-3.1-8B-Instruct \\
    --alias llama-8b

# Ollama (mistral)
python scripts/run_self_refine_api.py \\
    --base-url http://localhost:11434 \\
    --inference-model mistral \\
    --alias mistral-7b

# Restrict to specific languages / tasks / sample count
python scripts/run_self_refine_api.py \\
    --base-url http://localhost:8000 \\
    --inference-model meta-llama/Llama-3.1-8B-Instruct \\
    --alias llama-8b \\
    --languages Python JavaScript \\
    --tasks packages \\
    --max-samples 500
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mitigation.api_generator import APIGenerator
from mitigation.self_refine import SelfRefineGenerator

LANGUAGES  = ["Python", "JavaScript", "Rust", "Ruby"]
TASKS      = ["packages", "code"]
STRATEGY   = "self_refine"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_instructions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "packages" in data:
        return data["packages"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected instruction format in {path}")


def instruction_text(row: dict) -> str:
    return row.get("instruction") or row.get("description") or ""


def write_output(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"packages": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Per-file runner
# ---------------------------------------------------------------------------

def run_one(
    instruct_path: Path,
    out_path: Path,
    gen: SelfRefineGenerator,
    alias: str,
    task: str,
    max_samples: int | None,
    dry_run: bool,
) -> int:
    instructions = load_instructions(instruct_path)
    if max_samples is not None:
        instructions = instructions[:max_samples]

    if dry_run:
        print(f"    [dry-run] would process {len(instructions)} samples → {out_path}")
        return len(instructions)

    # Skip if already fully done
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        done = len(existing.get("packages", []))
        if done >= len(instructions):
            print(f"    [skip] already complete ({done} samples)")
            return 0

    records: list[dict] = []
    for row in tqdm(instructions, desc=out_path.name, leave=False):
        text = instruction_text(row)
        if not text:
            continue
        t0 = time.perf_counter()
        answer, elapsed, rounds = gen._self_refine(text)
        record = {
            "name":        row.get("name", ""),
            "description": row.get("description", ""),
            "instruction": text,
            "answer":      answer,
            "time_sec":    elapsed,
            "rounds_used": rounds,
            "model":       alias,
            "strategy":    STRATEGY,
            "task":        task,
        }
        records.append(record)

    write_output(out_path, records)
    return len(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base-url", required=True,
                   help="OpenAI-compatible API root (e.g. http://localhost:8000).")
    p.add_argument("--inference-model", required=True,
                   help="Model id to send in the API payload.")
    p.add_argument("--alias", required=True,
                   help="Compare_v3 model slug used for the output directory (e.g. llama-8b).")
    p.add_argument("--api-key", default="",
                   help="Bearer token. Leave empty for local servers.")
    p.add_argument("--languages", nargs="+", default=LANGUAGES,
                   help="Languages to process (default: all four).")
    p.add_argument("--tasks", nargs="+", default=["packages"], choices=TASKS,
                   help="Tasks to run (default: packages).")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap samples per language (default: all).")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-rounds", type=int, default=3,
                   help="Maximum self-refinement rounds per sample.")
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--max-workers", type=int, default=4,
                   help="Parallel API requests (one per refinement step).")
    p.add_argument("--instruct-dir", default=str(ROOT / "data" / "instruction"),
                   help="Directory containing instruction JSON files.")
    p.add_argument("--output-dir", default=str(ROOT / "reformat_results"),
                   help="Root of the reformat_results tree.")
    p.add_argument("--skip-enhance", action="store_true",
                   help="Skip running enhance_reformat_results.py after generation.")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be written without calling the API.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    instruct_dir = Path(args.instruct_dir)
    output_dir   = Path(args.output_dir)

    inner = APIGenerator(
        base_url        = args.base_url,
        inference_model = args.inference_model,
        api_key         = args.api_key,
        max_new_tokens  = args.max_new_tokens,
        temperature     = args.temperature,
        max_workers     = args.max_workers,
    )

    written: list[Path] = []

    for task in args.tasks:
        for language in args.languages:
            instruct_path = instruct_dir / f"packages_{language}_instruct.json"
            if not instruct_path.exists():
                print(f"  [skip] instruction file not found: {instruct_path}")
                continue

            out_path = output_dir / STRATEGY / task / args.alias / f"{language}.json"

            gen = SelfRefineGenerator(
                inner    = inner,
                language = language,
                task     = task,
                config   = {"max_rounds": args.max_rounds,
                            "max_new_tokens": args.max_new_tokens},
            )

            print(f"\n[{task}] {language} → {out_path.relative_to(ROOT)}")
            n = run_one(instruct_path, out_path, gen, args.alias, task,
                        args.max_samples, args.dry_run)
            if n and not args.dry_run:
                written.append(out_path)
            print(f"    {n} samples written.")

    print(f"\nDone. {len(written)} file(s) written.")

    if written and not args.skip_enhance and not args.dry_run:
        print("\nRunning enhance_reformat_results.py ...")
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "enhance_reformat_results.py"),
                "--strategies", STRATEGY,
                "--tasks", *args.tasks,
            ],
            cwd=str(ROOT),
        )
        if result.returncode != 0:
            print("[warn] enhance_reformat_results.py exited with a non-zero status.")
            return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
