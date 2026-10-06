# Hacker Role Report

**Paper.** Djire, Olatunji, Tessa, Barr, Klein, Bissyandé. *Evaluating Inference-Time Defenses against Package Hallucination in LLM-Generated Code.* ASE '26. arXiv:2608.22652v1.

Repo: https://github.com/PritikaRamu/8181-hacker-role-1.git

Name: Pritika Ramu

---

## 1. The research question

> **Does reducing the package hallucination rate actually reduce supply-chain attack exposure?**

The paper ranks seven defenses by how much they lower the Package Hallucination Rate (PHR). In this experiment, I check whether that ranking still holds when defenses are scored by the thing the paper says it cares about: how exposed a developer is to an attacker who has registered hallucinated package names.

---

## 2. The assumption being challenged

The paper's argument has three steps.

1. LLMs recommend packages that do not exist.
2. An attacker can register such a name and ship malware — *slopsquatting*.
3. **Therefore, lowering the rate of hallucination lowers the risk.**

Step 3 is NOT an assumption stated, but the entire evaluation rests on it: PHR, micro and macro, is the dependent variable in Tables 2–5, and defenses are recommended on that basis.

**Why it may not hold.** Slopsquatting is not a rate-based attack. An attacker cannot register "32% of names." They must register **specific strings, in advance**, and wait for a developer to install one. That makes the attack's economics depend on *repetition*:

- A model producing 10,000 hallucinations, every one a unique one-off string, is nearly worthless to an attacker — there is nothing to pre-register.
- A model producing 1,000 hallucinations concentrated on 50 recurring names is a rich target, despite a PHR ten times lower.

So a defense can cut PHR substantially and leave developers no safer — or worse off — if it funnels the surviving hallucinations onto a smaller, more repeatable set of names. PHR cannot distinguish these cases, because it counts events and discards identity.

The paper cites Spracklen et al. finding hallucinated names "persistent across repeated generations," and calls them "predictable targets for attackers." It then never measures persistence or concentration, and never checks whether its defenses change them. The information was in hand — the released records retain every package name in `extracted_packages` — and the pipeline reduced each record to a count.

---

## 3. Why it matters

If the assumption fails, the paper's practical advice is unsafe in a specific way: a practitioner reads Table 4, picks the lowest-PHR defense, and ships something that increases the chance their developers install an attacker's package. Because the defenses are *inference-time* and cheap to adopt, that advice travels.

It also matters for the field: PHR is the standard metric in this literature (Spracklen et al., Krishna et al., Haque et al.). If it is misaligned with the threat model, published defense rankings may need re-examination.

---

## 4. How the experiment was designed

### 4.1 Approach

A **secondary analysis** of the paper's own released generations was run. Hallucination labelling reuses the paper's `evaluation/registries.py`.

| | |
|---|---|
| Data | `src/output/run_{1,2,3}/<strategy>/packages/<model>/<lang>.json` |
| Grid | 8 models × 4 languages × 8 strategies × 3 independent runs |
| Ground truth | `src/data/package_list/*_with_standard.txt` (the paper's stdlib-corrected snapshot, 2026-03-04) |
| Paired sample | 3,167–3,969 (model, prompt) pairs per language |

### 4.2 Metric

> **Exposure@k (E@k)** — the fraction of prompts that recommend at least one of the *k* most frequently hallucinated names.

Interpretation: *the probability a developer receives an attacker-controlled package, if the attacker registers the top k names.* It combines rate and concentration into the quantity the threat model actually implies. Reported alongside Gini and events-per-name, which isolate concentration on its own.

### 4.3 Design decisions

**Pairing.** The released runs are unbalanced — baseline ran 1,000 prompts for four models while every other strategy ran 500. Every comparison is therefore restricted to the intersection of instructions present across all strategies for that model.

**Excluding degenerate cells.** A prior audit (§7) found that in 12 of 32 cells of `run_1`, nominally distinct strategies produced byte-identical outputs. Those cells cannot support a strategy comparison and are flagged.

**A held-out attacker.** Choosing the top *k* names from the same data that scores them is an oracle — a real attacker must commit first. So targets are selected from `run_1` and exposure is measured on `run_2` and `run_3`, which are independent generations. This is the primary specification; the oracle number is reported only as an upper bound.

### 4.4 What each hypothesis predicts

| | Hypothesis | Test |
|---|---|---|
| H1 | Hallucinated names are concentrated enough to target | E@100 ≥ 10% of prompts |
| H3 | PHR ranking and exposure ranking disagree | Kendall τ between the two rankings of 8 strategies; sign reversals vs. baseline |
| H4 | Names transfer across models — one registration, many victims | Targets from model A, exposure on the other 7 |
| H5 | Names persist across runs — pre-registration is viable | Held-out exposure retained vs. oracle |

---

## 5. Findings

### F1 — Hallucinated names are concentrated enough to be worth registering (H1: supported)

100 names reach **7–19% of all prompts** under baseline decoding, and up to 32% under one defense. Concentration is modest but real: the distribution is not a flat sea of one-off strings.

### F2 — PHR is a good proxy in two ecosystems and fails badly in a third (H3: partially supported)

| language | τ(PHR, E@10) | τ(PHR, E@100) | sign reversals vs. baseline |
|---|---|---|---|
| Ruby | 0.86 | **1.00** | none |
| JavaScript | 0.79 | 0.93 | none |
| Python | 0.93 | 0.64 | contrastive, rag |
| **Rust** | **0.21** | **0.21** | contrastive, dola |

This is the central result, and it is narrower than I expected. In Ruby and JavaScript the two metrics agree almost perfectly — **PHR is a sound proxy there, a conclusion the paper never argued for but is entitled to**. In Rust the rank correlation collapses to 0.21, effectively no relationship.

### F3 — In Rust, the two metrics invert outright

```
by PHR : rag < nudging < self_refine < actlcd < greedy < baseline < contrastive < dola
by E@10: rag < dola < nudging < contrastive < actlcd < self_refine < greedy < baseline
```

| strategy | PHR | E@10 | verdict |
|---|---|---|---|
| baseline | 47.9% | 11.38% | — |
| **dola** | **49.1%** (worst) | **4.74%** (best but one) | PHR says discard; exposure says deploy |
| contrastive | 48.7% (worse than baseline) | 6.01% | same reversal |
| self_refine | 44.3% (better) | 9.73% | better PHR, barely-moved exposure |
| rag | 28.7% | 3.13% | wins on both |

DoLa is the worst strategy in Rust on the paper's metric, yet it cuts concentrated attack exposure by **58%**. On the paper's evidence you would discard it; on the paper's threat model you would deploy it.

### F4 — The most dangerous case is one the paper reports but underestimates

RAG on JavaScript: the paper notes it degrades PHR (by 12.7 pp on average). The exposure cost is far worse.

| | baseline | RAG |
|---|---|---|
| halluc. prompts | 31.6% | **67.0%** |
| distinct names | 4,232 | 2,812 |
| events per name | 1.12 | **2.43** |
| Gini | 0.105 | **0.537** |
| **E@100** | 7.18% | **32.33%** |

RAG on JavaScript does not merely hallucinate more — it hallucinates *the same names repeatedly*, the worst possible security profile. Exposure rises **4.5×**.

### F5 — Pre-registration works (H5: supported — the key security finding)

Targets chosen on `run_1`, scored on independent runs:

| | oracle E@100 (run_1) | held-out run_2 | held-out run_3 | retained |
|---|---|---|---|---|
| Python baseline | 10.53% | 8.09% | 7.89% | **76%** |
| Python greedy | 9.95% | 8.69% | 8.69% | **87%** |
| Rust baseline | 16.59% | 13.33% | 14.23% | **83%** |
| Rust dola | 12.82% | 13.78% | 13.35% | **106%** |
| Rust self_refine | 18.91% | 14.58% | 14.53% | 77% |

An attacker who commits to 100 names based on observed generations retains **76–106%** of their exposure against completely fresh generations. The threat is not an artifact of hindsight — these names are stable. Rust/DoLa exceeds 100%, meaning its held-out exposure is *higher* than in the data used to pick the targets.

(RAG retains exactly 100% in both languages because its outputs are byte-identical across all three runs — deterministic retrieval, and a separate oddity worth flagging to the authors.)

### F6 — Cross-model targeting is weak in Python, substantial in Rust (H4: partially supported)

Transfer = exposure achieved against *other* models, as a fraction of exposure against the source model.

| source model | Python transfer | Rust transfer |
|---|---|---|
| deepseek-1.3b | 8% | 18% |
| gemma-4b | 33% | **60%** |
| qwen-3b | 28% | **58%** |
| qwen-1.5b | 28% | **51%** |

In Python, hallucinations are largely model-specific — an attacker must target one model. In Rust, up to 60% of exposure transfers, so one registration set reaches multiple model populations. This is consistent with F2: crates.io is the smallest registry studied, giving models the fewest real names to anchor on, so they converge on the same fabrications.

### Summary

| Claim | Verdict |
|---|---|
| Hallucinated names are concentrated enough to target | **Supported** |
| Lowering PHR lowers attack exposure | **Holds in Ruby and JavaScript; fails in Rust; partially fails in Python** |
| Defense rankings are stable across the two metrics | **Rejected for Rust** (DoLa, CD invert) |
| Hallucinated names are stable enough to pre-register | **Supported** (76–106% retention) |
| One registration reaches many models | **Rust yes (≤60%), Python no (≤33%)** |

**The assumption in §2 is not universally false — it is ecosystem-dependent.** The honest statement is: PHR is an adequate security proxy in large, mature registries, and an unreliable one in small registries where models converge on the same fabricated names. Since the paper's own headline is that Rust and Ruby are the hardest settings, that is precisely where its metric is least trustworthy — and it recommends defenses without checking.

---

## 8. Reproducing

```bash
python3 hacker_role/analyze_exposure.py --languages Python JavaScript Ruby Rust --ks 10 100
python3 hacker_role/analyze_transfer.py --languages Python Rust --k 100
python3 hacker_role/audit_strategy_collisions.py
```
