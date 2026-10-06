#!/usr/bin/env python3
"""
Two tests of whether hallucinated-name exposure is actually exploitable.

H5 (held-out / pre-registration). An attacker must choose target names BEFORE
    seeing a developer's prompts. Selecting the top-k from the same data that
    scores them is an oracle. Here targets are chosen from run_1 and exposure is
    measured on run_2 and run_3 -- independent generations. If held-out exposure
    stays close to oracle exposure, pre-registration is viable.

H4 (cross-model transfer). Targets derived from one model, scored against the
    others. High transfer means one registration reaches many model populations.
"""
import argparse, json, os, sys
from collections import Counter

STRATEGIES = ["baseline", "greedy_decoding", "dola", "nudging", "rag", "self_refine"]
MODELS = ["deepseek-1.3b", "deepseek-6.7b", "gemma-1b", "gemma-4b",
          "llama-8b", "mistral-7b", "qwen-1.5b", "qwen-3b"]


def prompt_sets(reg, out_root, run, strategy, model, lang):
    p = os.path.join(out_root, run, strategy, "packages", model, f"{lang}.json")
    if not os.path.exists(p):
        return None
    try:
        recs = json.load(open(p)).get("packages", [])
    except (json.JSONDecodeError, OSError):
        return None
    return [{n.strip().lower() for n in (r.get("extracted_packages") or [])
             if isinstance(n, str) and reg.classify(n, lang) == "hallucinated"}
            for r in recs]


def exposure(sets_, targets):
    if not sets_:
        return float("nan")
    return 100 * sum(1 for s in sets_ if s & targets) / len(sets_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="src")
    ap.add_argument("--languages", nargs="+", default=["Python", "Rust"])
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()

    sys.path.insert(0, args.src)
    from evaluation.registries import PackageRegistry
    reg = PackageRegistry(os.path.join(args.src, "data", "package_list"))
    out_root = os.path.join(args.src, "output")
    K = args.k

    for lang in args.languages:
        print(f"\n{'='*94}\nH5  PRE-REGISTRATION — targets chosen on run_1, scored on runs 2 & 3   [{lang}, k={K}]\n{'='*94}")
        print(f"{'strategy':20} {'oracle E@k (run_1)':>19} {'held-out run_2':>15} {'held-out run_3':>15} {'retained':>9}")
        for s in STRATEGIES:
            tr, te2, te3 = [], [], []
            for m in MODELS:
                a = prompt_sets(reg, out_root, "run_1", s, m, lang)
                b = prompt_sets(reg, out_root, "run_2", s, m, lang)
                c = prompt_sets(reg, out_root, "run_3", s, m, lang)
                if a: tr.extend(a)
                if b: te2.extend(b)
                if c: te3.extend(c)
            if not tr or not te2:
                continue
            freq = Counter()
            for st in tr: freq.update(st)
            targets = {n for n, _ in freq.most_common(K)}
            o, h2 = exposure(tr, targets), exposure(te2, targets)
            h3 = exposure(te3, targets) if te3 else float("nan")
            held = [x for x in (h2, h3) if x == x]
            ret = 100 * (sum(held)/len(held)) / o if o else float("nan")
            print(f"{s:20} {o:18.2f}% {h2:14.2f}% {h3:14.2f}% {ret:8.0f}%")

        print(f"\n{'='*94}\nH4  CROSS-MODEL TRANSFER — targets from one model, scored on the other seven   [{lang}, k={K}]\n{'='*94}")
        print(f"{'source model':16} {'E@k on itself':>14} {'E@k on others':>14} {'transfer':>9}")
        for src in MODELS:
            own = prompt_sets(reg, out_root, "run_1", "baseline", src, lang)
            if not own:
                continue
            freq = Counter()
            for st in own: freq.update(st)
            targets = {n for n, _ in freq.most_common(K)}
            others = []
            for m in MODELS:
                if m == src: continue
                o = prompt_sets(reg, out_root, "run_1", "baseline", m, lang)
                if o: others.extend(o)
            e_self, e_other = exposure(own, targets), exposure(others, targets)
            print(f"{src:16} {e_self:13.2f}% {e_other:13.2f}% {100*e_other/e_self if e_self else 0:8.0f}%")


if __name__ == "__main__":
    main()
