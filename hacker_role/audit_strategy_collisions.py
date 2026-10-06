#!/usr/bin/env python3
"""
Audit the released artifact of "Evaluating Inference-Time Defenses against
Package Hallucination in LLM-Generated Code" (Djire et al., ASE '26) for
degenerate strategy comparisons.

A "collision" is a (model, language) cell in which two nominally distinct
decoding strategies produced byte-identical generations for every prompt.
Such a cell cannot support a comparison between those two strategies.

Usage:  python3 audit_strategy_collisions.py [--output-root PATH] [--run run_1]
"""
import argparse, itertools, json, os
from collections import defaultdict

STRATEGIES = ["baseline", "greedy_decoding", "contrastive_decoding", "dola",
              "actlcd", "nudging", "rag", "self_refine"]
MODELS = ["deepseek-1.3b", "deepseek-6.7b", "gemma-1b", "gemma-4b",
          "llama-8b", "mistral-7b", "qwen-1.5b", "qwen-3b"]
LANGS = ["Python", "JavaScript", "Ruby", "Rust"]


def load_answers(root, run, strategy, model, lang, task="packages"):
    path = os.path.join(root, run, strategy, task, model, f"{lang}.json")
    if not os.path.exists(path):
        return None
    try:
        payload = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return None
    records = payload.get("packages")
    if not isinstance(records, list):
        return None
    return [r.get("answer") for r in records]


def phr(root, run, strategy, model, lang, task="packages"):
    path = os.path.join(root, run, strategy, task, model, f"{lang}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path)).get("stats", {}).get("phr_with_stdlib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="src/output")
    ap.add_argument("--run", default="run_1")
    ap.add_argument("--task", default="packages")
    args = ap.parse_args()

    collisions = defaultdict(set)
    for model, lang in itertools.product(MODELS, LANGS):
        got = {s: load_answers(args.output_root, args.run, s, model, lang, args.task)
               for s in STRATEGIES}
        got = {s: v for s, v in got.items() if v}
        for s1, s2 in itertools.combinations(sorted(got), 2):
            v1, v2 = got[s1], got[s2]
            n = min(len(v1), len(v2))
            if n and all(v1[i] == v2[i] for i in range(n)):
                collisions[(model, lang)].add((s1, s2))

    n_pairs = sum(len(v) for v in collisions.values())
    distinct_pairs = {p for v in collisions.values() for p in v}
    print(f"run={args.run} task={args.task}")
    print(f"  configuration-level collisions : {n_pairs}")
    print(f"  distinct strategy pairs        : {len(distinct_pairs)}")
    print(f"  affected (model,language) cells: {len(collisions)} / {len(MODELS)*len(LANGS)}")
    for (model, lang), pairs in sorted(collisions.items()):
        merged = sorted({s for pair in pairs for s in pair})
        print(f"    {model:14} {lang:11} {len(pairs):2} pairs among {{{', '.join(merged)}}}")

    # Impact of the degenerate cells on the headline Vanilla -> Greedy effect.
    rows = []
    for model, lang in itertools.product(MODELS, LANGS):
        v = phr(args.output_root, args.run, "baseline", model, lang, args.task)
        g = phr(args.output_root, args.run, "greedy_decoding", model, lang, args.task)
        if v is None or g is None:
            continue
        degenerate = ("baseline", "greedy_decoding") in collisions[(model, lang)]
        rows.append((v, g, degenerate))

    if rows:
        mean = lambda xs, i: sum(x[i] for x in xs) / len(xs)
        valid = [r for r in rows if not r[2]]
        deg = [r for r in rows if r[2]]
        print(f"\n  Vanilla -> Greedy mean PHR effect")
        print(f"    all {len(rows):2} cells : {mean(rows,0):5.2f} -> {mean(rows,1):5.2f}  "
              f"({mean(rows,0)-mean(rows,1):+.2f} pp)   [as published]")
        print(f"    {len(deg):2} degenerate: {mean(deg,0):5.2f} -> {mean(deg,1):5.2f}  "
              f"({mean(deg,0)-mean(deg,1):+.2f} pp)   [identical by construction]")
        print(f"    {len(valid):2} valid cells: {mean(valid,0):5.2f} -> {mean(valid,1):5.2f}  "
              f"({mean(valid,0)-mean(valid,1):+.2f} pp)   [corrected estimate]")


if __name__ == "__main__":
    main()
