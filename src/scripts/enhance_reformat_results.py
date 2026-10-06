#!/usr/bin/env python3
"""Enhance `reformat_results/` JSON files in place with per-sample and
per-file evaluation metrics: extracted packages, Package Hallucination Rate
(PHR, with/without stdlib correction), and - for the `code` task - syntax
validity and semgrep code-smell findings.

Expected layout (unchanged by this script):

    reformat_results/<strategy>/<task>/<model>/<Language>.json
    {"packages": [{"package_name": ..., "instruction": ..., "answer": ..., ...}, ...]}

See `evaluation/enhance.py` for the shared implementation (also used by
`scripts/evaluate.py` against `output/`).

Usage
-----
    python scripts/enhance_reformat_results.py
    python scripts/enhance_reformat_results.py --tasks packages
    python scripts/enhance_reformat_results.py --strategies baseline greedy_decoding --models deepseek-1.3b
    python scripts/enhance_reformat_results.py --tasks code --skip-smell   # syntax only, no semgrep
    python scripts/enhance_reformat_results.py --dry-run                  # compute, don't write
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.enhance import build_arg_parser, run  # noqa: E402


def main() -> int:
    parser = build_arg_parser(default_root=ROOT / "reformat_results", description=__doc__)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
