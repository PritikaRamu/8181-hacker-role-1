#!/usr/bin/env python3
"""
Squat Exposure analysis for "Evaluating Inference-Time Defenses against Package
Hallucination in LLM-Generated Code" (Djire et al., ASE '26).

The paper evaluates defenses with Package Hallucination Rate (PHR), a frequency
metric. Its stated threat model is slopsquatting, whose economics depend not on
how OFTEN a model hallucinates but on whether it hallucinates the SAME name
repeatedly -- a name an attacker can pre-register.

This script re-scores the released generations with attacker-facing metrics:

  Exposure@k (E@k)  fraction of prompts that recommend at least one of the k
                    most frequently hallucinated names, i.e. the probability a
                    developer receives an attacker-controlled package if the
                    attacker registers the top k names.
  concentration     hallucination events per distinct name, and Gini.

All strategies are compared PAIRED on the identical set of (model, prompt)
pairs, because the released runs use unequal prompt counts across strategies.

No GPU required; reads only `src/output/`.
"""
import argparse, json, os, sys
from collections import Counter

STRATEGIES = ["baseline", "greedy_decoding", "contrastive_decoding", "dola",
              "actlcd", "nudging", "rag", "self_refine"]
MODELS = ["deepseek-1.3b", "deepseek-6.7b", "gemma-1b", "gemma-4b",
          "llama-8b", "mistral-7b", "qwen-1.5b", "qwen-3b"]


def gini(counts):
    xs = sorted(counts)
    n = len(xs)
    if n == 0 or sum(xs) == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * sum(xs)) - (n + 1) / n


def load_prompt_sets(registry, out_root, run, strategy, model, lang, task):
    """-> {instruction: set(hallucinated names)}"""
    path = os.path.join(out_root, run, strategy, task, model, f"{lang}.json")
    if not os.path.exists(path):
        return None
    try:
        records = json.load(open(path)).get("packages", [])
    except (json.JSONDecodeError, OSError):
        return None
    table = {}
    for rec in records:
        key = rec.get("instruction")
        if key is None:
            continue
        table[key] = {
            n.strip().lower()
            for n in (rec.get("extracted_packages") or [])
            if isinstance(n, str) and registry.classify(n, lang) == "hallucinated"
        }
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="src", help="path to the artifact's src/ directory")
    ap.add_argument("--run", default="run_1")
    ap.add_argument("--task", default="packages")
    ap.add_argument("--languages", nargs="+", default=["Python", "JavaScript", "Ruby", "Rust"])
    ap.add_argument("--ks", nargs="+", type=int, default=[10, 100, 1000])
    args = ap.parse_args()

    sys.path.insert(0, args.src)
    from evaluation.registries import PackageRegistry
    registry = PackageRegistry(os.path.join(args.src, "data", "package_list"))
    out_root = os.path.join(args.src, "output")

    for lang in args.languages:
        pooled = {s: [] for s in STRATEGIES}
        for model in MODELS:
            tables = {s: load_prompt_sets(registry, out_root, args.run, s, model, lang, args.task)
                      for s in STRATEGIES}
            tables = {s: t for s, t in tables.items() if t}
            if len(tables) < len(STRATEGIES):
                continue                      # model lacks a strategy -> skip, keeps pairing exact
            common = set.intersection(*(set(t) for t in tables.values()))
            for s, t in tables.items():
                pooled[s].extend(t[k] for k in sorted(common))

        n = len(pooled["baseline"])
        if n == 0:
            print(f"\n{lang}: no paired data\n")
            continue

        print(f"\n{'=' * 100}")
        print(f"{lang} — {n} paired (model, prompt) pairs, run={args.run}, task={args.task}")
        print(f"{'=' * 100}")
        hdr = f"{'strategy':22} {'halluc%':>8} {'events':>7} {'distinct':>9} {'ev/name':>8} {'gini':>6}"
        hdr += "".join(f" {'E@'+str(k):>8}" for k in args.ks)
        print(hdr)

        rows = []
        for s in STRATEGIES:
            sets_ = pooled[s]
            if not sets_:
                continue
            freq = Counter()
            for st in sets_:
                freq.update(st)
            events = sum(freq.values())
            distinct = len(freq)
            halluc = 100 * sum(1 for st in sets_ if st) / len(sets_)
            exposures = []
            for k in args.ks:
                target = {nm for nm, _ in freq.most_common(k)}
                exposures.append(100 * sum(1 for st in sets_ if st & target) / len(sets_))
            rows.append((s, halluc, exposures))
            line = (f"{s:22} {halluc:7.1f}% {events:7} {distinct:9} "
                    f"{events/max(distinct,1):8.2f} {gini(list(freq.values())):6.3f}")
            line += "".join(f" {e:7.2f}%" for e in exposures)
            print(line)

        # Does ranking by PHR agree with ranking by exposure?
        base = dict((s, (h, e)) for s, h, e in rows)
        for i, k in enumerate(args.ks):
            by_phr = [s for s, _, _ in sorted(rows, key=lambda r: r[1])]
            by_exp = [s for s, _, _ in sorted(rows, key=lambda r: r[2][i])]
            print(f"\n  ranking by PHR  : {' < '.join(by_phr)}")
            print(f"  ranking by E@{k:<4}: {' < '.join(by_exp)}")
            reversals = [s for s in base
                         if (base[s][0] < base['baseline'][0]) != (base[s][1][i] < base['baseline'][1][i])]
            if reversals:
                print(f"  !! sign reversal vs baseline at k={k}: {', '.join(sorted(reversals))}")


if __name__ == "__main__":
    main()
