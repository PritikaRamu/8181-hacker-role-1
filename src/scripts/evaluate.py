#!/usr/bin/env python3
"""Enhance `output/` JSON files in place with per-sample and per-file
evaluation metrics: extracted packages, Package Hallucination Rate (PHR,
with/without stdlib correction), and - for the `code` task - syntax
validity and semgrep code-smell findings.

Expected layout (produced by `scripts/generate.py`, unchanged by this script):

    output/<strategy>/<task>/<model>/<Language>.json
    {"packages": [{"name": ..., "instruction": ..., "answer": ..., ...}, ...]}

No separate CSV is produced: results live on the same JSON files that
`scripts/generate.py` wrote, as a top-level "stats" dict plus per-sample
fields. `scripts/analyze.py` reads directly from `output/` to build tables
and plots. See `evaluation/enhance.py` for the shared implementation (also
used by `scripts/enhance_reformat_results.py` against `reformat_results/`).

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py --tasks packages
    python scripts/evaluate.py --strategies baseline greedy --models deepseek_1.3b
    python scripts/evaluate.py --tasks code --skip-smell   # syntax only, no semgrep
    python scripts/evaluate.py --dry-run                  # compute, don't write
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.enhance import build_arg_parser, run  # noqa: E402


def main() -> int:
    parser = build_arg_parser(default_root=ROOT / "output", description=__doc__)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
