# Package Hallucination Evaluation

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate outputs for all enabled strategies and models
python scripts/generate.py

# 2. Evaluate — enriches output/*.json in place with PHR + code-quality stats
python scripts/evaluate.py

# 3. Analyze — LaTeX tables + plots + statistical report
python scripts/analyze.py --print-report
```

Run a quick subset during development:

```bash
# Single task, two strategies, one model, one language
python scripts/generate.py --tasks packages --strategies baseline greedy --models deepseek_1.3b --languages Python

# Both tasks, all models
python scripts/generate.py --tasks packages code
```

---

## Directory layout

```
configs/
  data.yml                    ← languages, tasks, data paths, max_samples, output dir
  <model_alias>/
    model.yml                 ← id, alias, torch_dtype, device_map, quantization
    baseline.yml              ← strategy params for this model (enabled: true/false)
    greedy.yml
    self_refine.yml
    dola.yml                  ← also holds mature_layer and early_exit_layers
    rag.yml
    nudging.yml               ← only on large models; names guide_model
    contrastive_decoding.yml  ← only on large models; names expert/amateur
    actlcd.yml                ← DoLa + BCQ gating policy; policy_path: null → standard DoLa
data/
  instruction/                ← per-language instruction datasets (JSON, enriched by enrich_*.py)
  package_list/               ← canonical registry lists for PHR validation
  resources/                  ← full NDJSON registry snapshots (used by enrich_similar_packages.py)
mitigation/                   ← strategy implementations
evaluation/                   ← PHR, pass@k, and code-quality metric helpers
  phr.py, pass_k.py
  registries.py               ← canonical/stdlib package name sets (with/without-stdlib pairs)
  extraction.py               ← package + import extraction from raw model responses
  code_quality.py             ← syntax validity (ast/tree-sitter) + semgrep code-smell checks
  enhance.py                  ← shared in-place enrichment pipeline (used by evaluate.py
                                   and enhance_reformat_results.py)
package_collection/           ← tools for building the registry snapshots
  package_collection.py       ← fetch and cache registry package lists
  cargo_collection.py         ← Rust/Cargo-specific collector
  rubygems_packages.txt       ← static RubyGems name list
prompts/                      ← Jinja2 prompt templates
  system_prompt_package_generation.jinja
  system_prompt_code_generation.jinja
  system_prompt_package_validation.jinja
  user_prompt_package_validation.jinja
scripts/
  generate.py                 ← run strategies → output/
  evaluate.py                 ← enrich output/*.json in place with PHR + code-quality metrics
  analyze.py                  ← tables + plots → results/tables/ and results/plots/ (reads output/)
  import_rag_results.py       ← import external RAG results into the reformat_results/ layout
  run_self_refine_api.py      ← self-refine via OpenAI-compatible API (no local GPU needed)
  train_actlcd_policy.py      ← collect DoLa features and train the BCQ gating policy

# Root-level analysis scripts (operate on reformat_results/ by default)
enrich_packages.py            ← enrich instruction files: similar subcommand (TF-IDF) and
                                 collective subcommand (co-occurring packages from code runs)
compute_du.py                 ← tfidf / collective / topk subcommands: compute DU (P/R/F1)
                                 and write back into result files, or plot heatmaps
aggregate_du.py               ← read DU stats from result files → results/du_scores.csv and CSVs
evaluate.py                   ← LaTeX tables + figures from the aggregated CSVs (results/eval/)
analyze_runs.py               ← variance analysis across run_X dirs → results/tables/phr_variance*.tex

output/                       ← generated at runtime
```

---

## Mitigation strategies

Eight strategies are implemented under `mitigation/`. Each is enabled independently in the config.

### Baseline (`mitigation/baseline.py`)
Standard sampling generation with no mitigation applied. Uses temperature, top-p, and top-k as configured. Serves as the reference point against which all other strategies are compared.

### Greedy (`mitigation/greedy.py`)
Deterministic decoding — always picks the highest-probability token (`do_sample=False`). Eliminates randomness entirely, trading diversity for reproducibility.

### Self-Refine (`mitigation/self_refine.py`)
Iterative self-correction loop. After an initial generation the model is asked to validate each suggested package (Yes/No). Packages flagged as invalid trigger a second generation pass that incorporates the feedback. Repeats up to `max_rounds` times.

### DoLa — Decoding by Contrasting Layers (`mitigation/dola.py`)
At each decoding step the model contrasts the logits of a mature (late) transformer layer against early-exit layers. Tokens whose probability increases monotonically through the layers are preferred, which tends to surface more factual outputs. `mature_layer` and `early_exit_layers` are **model-specific** and must be set in each model's `dola` block in the config (they depend on the model's total layer count).

### RAG — Retrieval-Augmented Generation (`mitigation/rag.py`)
Before generation, a retriever (sentence-transformers) fetches the `k` most relevant entries from a corpus and prepends them to the prompt. Grounds the model's output in verified package documentation.

### Nudging (`mitigation/nudging.py`)
A **paired** strategy: a larger *generator* model produces tokens normally, but whenever its confidence falls below `top_prob_thres`, a smaller *guide* model provides a corrective signal that steers the generator away from uncertain (potentially hallucinated) outputs. The generator and guide must be from the same model family. The larger model carries the `nudging` block in the config and names the guide via `guide_model`.

### Contrastive Decoding (`mitigation/contrastive_decoding.py`)
A **paired** strategy: at each step the final logit distribution is computed as `log p_expert − α · log p_amateur`. The expert is the larger model, the amateur is the smaller one from the same family. Tokens that the expert favors disproportionately over the amateur are up-weighted, suppressing superficial or hallucinated patterns that both models share. The larger model carries the `contrastive_decoding` block and must set `expert_model` to its own alias (validated at startup).

### ActLCD — Active Layer-Contrastive Decoding (`mitigation/actlcd.py`)
A **learned** extension of DoLa. Instead of applying the layer-contrastive adjustment at every token, a lightweight offline-RL policy (BCQ — Batch Constrained Q-learning) decides at each step whether contrasting is beneficial. When the policy is absent (`policy_path: null`) the strategy falls back to standard DoLa, making it a strict superset.

**State** fed to the policy: the top-5 token IDs and probabilities from each candidate premature layer plus the mature layer, concatenated into a flat vector.  
**Actions**: `0` = skip contrast (decode from mature layer directly), `1` = apply DoLa contrast.  
**Reward** (end of sequence): +1 for correct decisions, −4 for missing a needed correction (false negatives penalised heavily to preserve recall).

The policy must be trained before use with `scripts/train_actlcd_policy.py` (see below). Each model + language combination needs its own policy checkpoint. Set `policy_path` in `actlcd.yml` to the `.pth` file once trained. Because ActLCD processes one sequence at a time the config sets `batch_size: 1`.

Paper: Zhang et al., *Active Layer-Contrastive Decoding Reduces Hallucination in Large Language Model Generation*, EMNLP 2025.

---

## Configuration

Each model has its own directory under `configs/`. Each strategy it supports has its own `.yml` file inside that directory. **To edit an experiment, open the one file that corresponds to the model and strategy you care about.**

### `configs/data.yml` — shared data settings

```yaml
instruction_dir: data/instruction
package_list_dir: data/package_list
languages: [Python, JavaScript, Rust, Ruby]
max_samples:          # null = all samples; positive integer to cap per language
  Python: null
  JavaScript: null
  Rust: null
  Ruby: null
output:
  root_dir: output
```

### `configs/<model>/model.yml` — model identity

```yaml
# configs/deepseek_6.7b/model.yml
id: deepseek-ai/deepseek-coder-6.7b-instruct
alias: deepseek_6.7b
torch_dtype: bfloat16
device_map: auto
quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: nf4
```

### `configs/<model>/baseline.yml` — single-model strategy example

```yaml
enabled: true
max_new_tokens: 128
temperature: 0.7
batch_size: 8
generation_kwargs:
  do_sample: false
  top_p: 0.9
  top_k: 0
  num_beams: 1
```

### `configs/<model>/dola.yml` — model-specific layer params live here

```yaml
# configs/deepseek_6.7b/dola.yml
enabled: false
max_new_tokens: 128
relative_top: 0.1
repetition_penalty: 1.2
batch_size: 4
mature_layer: 32                                       # architecture-specific
early_exit_layers: [4, 8, 12, 16]
```

### `configs/<model>/actlcd.yml` — ActLCD (DoLa + BCQ gating policy)

```yaml
# configs/deepseek_6.7b/actlcd.yml
enabled: false
max_new_tokens: 128
relative_top: 0.1
repetition_penalty: 1.2
batch_size: 1              # one sequence at a time (BCQ state is per-token, per-sequence)
mature_layer: 32           # same as dola.yml
early_exit_layers: [4, 8, 12, 16]
# Path to a trained BCQAgent .pth checkpoint.
# null → standard DoLa behaviour (contrast applied at every token).
policy_path: null
bc_threshold: 0.3
```

**Training the BCQ policy** (required to unlock the selective gating):

```bash
# Step 1 — Collect per-token features from a DoLa run on labelled data
python scripts/train_actlcd_policy.py \
    --phase collect \
    --model-alias deepseek_1.3b \
    --language Python \
    --task packages \
    --max-samples 200 \
    --features-out data/actlcd/deepseek_1.3b_Python_packages.csv

# Step 2 — Train BCQ
python scripts/train_actlcd_policy.py \
    --phase train \
    --features-in data/actlcd/deepseek_1.3b_Python_packages.csv \
    --policy-out model/actlcd/deepseek_1.3b_Python_packages.pth

# Step 3 — Set policy_path in configs/deepseek_1.3b/actlcd.yml and enable: true
```

State dimension per model (derived from `len(early_exit_layers) + 1) × 5 × 2`):

| Model | Premature layers | state\_dim |
|---|---|---|
| deepseek\_1.3b | 8 | 90 |
| deepseek\_6.7b | 11 | 120 |
| qwen\_1.5b | 10 | 110 |
| qwen\_3b | 12 | 130 |

### `configs/<model>/nudging.yml` — paired strategy (large model only)

```yaml
# configs/deepseek_6.7b/nudging.yml
enabled: false
max_new_tokens: 128
batch_size: 4
guide_model: deepseek_1.3b   # alias of the smaller partner — validated at startup
top_prob_thres: 0.9
```

### `configs/<model>/contrastive_decoding.yml` — paired strategy (large model only)

```yaml
# configs/deepseek_6.7b/contrastive_decoding.yml
enabled: false
max_new_tokens: 128
batch_size: 4
expert_model: deepseek_6.7b  # must equal this model's alias — validated at startup
amateur_model: deepseek_1.3b # alias of the smaller partner — validated at startup
alpha: 0.1
repetition_penalty: 1.0
```

### Key config conventions

| Convention | Rule |
|---|---|
| File present + `enabled: true` | Strategy runs for this model |
| File absent | Strategy skipped for this model (small models have no `nudging`/`contrastive_decoding`) |
| `dola.yml` | Must exist for every model when dola is enabled anywhere |
| `contrastive_decoding.expert_model` | Must equal the enclosing model's alias — validated at startup |
| `nudging.guide_model` / `contrastive_decoding.amateur_model` | Must be a known alias — validated at startup |

---

## CLI reference

### `generate.py`

```bash
python scripts/generate.py [--config-dir DIR] [--tasks ...] [--strategies ...] [--models ...] [--languages ...]
```

| Flag | Default | Description |
|---|---|---|
| `--config-dir` | `configs/` | Root config directory |
| `--tasks` | from config | Run only these tasks (`packages`, `code`) |
| `--strategies` | all enabled | Run only these strategies |
| `--models` | all | Run only these model aliases |
| `--languages` | from config | Process only these languages |

### `evaluate.py` / `enhance_reformat_results.py`

Both enrich JSON result files **in place** (no CSV) and take the same flags —
they're thin wrappers around `evaluation/enhance.py`, differing only in
their default `--root` (`output/` vs `reformat_results/`).

```bash
python scripts/evaluate.py [--root DIR] [--package-list-dir DIR] [--strategies ...] [--tasks ...] [--models ...] [--languages ...] [--workers N] [--semgrep-timeout SECONDS] [--skip-syntax] [--skip-smell] [--dry-run]
```

| Flag | Default | Description |
|---|---|---|
| `--root` | `output/` (`reformat_results/` for the other script) | Root of the corpus to enhance |
| `--package-list-dir` | `data/package_list/` | Canonical registry lists |
| `--strategies` | all | Only process these strategy directories |
| `--tasks` | all | Only process these tasks (`packages`, `code`) |
| `--models` | all | Only process these model directories |
| `--languages` | all | Only process these languages |
| `--workers` | `4` | Files processed in parallel |
| `--semgrep-timeout` | `300` | Per-file semgrep timeout (seconds) |
| `--skip-syntax` | off | Skip syntax validity checks (`code` task) |
| `--skip-smell` | off | Skip semgrep smell detection (`code` task) |
| `--dry-run` | off | Compute and print stats without writing files |

See "Evaluation & code-quality metrics" below for what this adds to each file.

### `analyze.py`

```bash
python scripts/analyze.py [--results-dir DIR] [--plots-dir DIR] [--no-plots] [--print-report]
```

| Flag | Default | Description |
|---|---|---|
| `--results-dir` | `output/` | Directory of enhanced `<strategy>/<task>/<model>/<Language>.json` files |
| `--plots-dir` | `results/plots/` | Output directory for PDF plots |
| `--no-plots` | off | Skip plot generation (report only) |
| `--print-report` | off | Print statistical analysis to stdout |

---

## Variance analysis (`analyze_runs.py`)

When experiments are repeated across multiple run directories (`notebooks/tmp/run_1/`, `run_2/`, …), `analyze_runs.py` aggregates results and reports mean ± std across runs.

```bash
python analyze_runs.py
python analyze_runs.py --runs-dir notebooks/tmp --output-dir results/tables
```

| Flag | Default | Description |
|---|---|---|
| `--runs-dir` | `notebooks/tmp` | Parent directory containing `run_X/` subdirs |
| `--output-dir` | `results/tables` | Output directory for `.tex` files |

Outputs:

| File | Content |
|---|---|
| `phr_variance_packages.tex` | micro/macro PHR ± std for all 7 strategies, packages task |
| `phr_variance_code.tex` | same for the code-generation task |
| `phr_stdlib_correction_variance.tex` | PHR with vs without stdlib correction, grouped by language |

---

## Output layout

```
output/                               ← live pipeline output (scripts/generate.py → scripts/evaluate.py)
  <strategy>/
    <task>/
      <model>/
        Python.json
        JavaScript.json
        Ruby.json
        Rust.json
  nudging/
    packages/
      deepseek_6.7b__deepseek_1.3b/  ← generator__guide naming for paired strategies
        Python.json  ...
  contrastive_decoding/
    packages/
      deepseek_6.7b__deepseek_1.3b/  ← expert__amateur naming
        Python.json  ...
```

`scripts/evaluate.py` writes no separate results file — it adds
`extracted_packages`, PHR, and code-quality fields directly onto each
`output/<strategy>/<task>/<model>/<Language>.json`, including a top-level
`"stats"` block per file. `scripts/analyze.py` reads those `"stats"` blocks
straight out of `output/` to build the tables and plots above. See
"Evaluation & code-quality metrics" below.

---

## Evaluation & code-quality metrics

Both `output/` (the live `scripts/generate.py` pipeline) and
`reformat_results/` (see below) share the same
`<strategy>/<task>/<model>/<Language>.json` layout and
`{"packages": [{"answer": ..., ...}, ...]}` schema. `scripts/evaluate.py` and
`scripts/enhance_reformat_results.py` are thin wrappers around the same
`evaluation/enhance.py`, which evaluates a corpus **in place** — it adds
fields to the existing JSON files rather than producing a separate
CSV/results directory, so each file stays self-contained.

The raw `answer` text can be messy: some rows echo the full rendered chat
prompt (`system: ...user: ...assistant: <reply>user: <hallucinated follow-up>...`)
because small models keep inventing extra turns once they run out of real
content; others store only the cleaned final reply. Byte-level BPE artifacts
(`Ġ`/`Ċ`) leak into some decoded strings, and some models answer in prose
with a code block instead of the requested bracketed package list.
`evaluation/extraction.py` isolates the model's first real reply and handles
all of this before extracting package names (or import/require statements,
for the `code` task).

Each **sample** gains:

| Field | Meaning |
|---|---|
| `extracted_packages` | Package/import names parsed out of the response |
| `extracted_count` | `len(extracted_packages)` |
| `registry_valid_count` | Extracted names found in the registry (PyPI/npm/crates.io/RubyGems) |
| `stdlib_count` | Extracted names that are standard-library modules, not registry packages |
| `hallucinated_count` | Extracted names found in neither — true hallucinations |
| `hallucination_rate` | `hallucinated_count / extracted_count` |
| `syntax_valid` *(code task)* | Whether the generated snippet parses (`ast` for Python, tree-sitter for JS/Rust/Ruby) |
| `smell_issues` *(code task)* | Semgrep findings (rule, severity, message, line) |
| `smell_count` *(code task)* | `len(smell_issues)` |

Each **file** gains a top-level `"stats"` block aggregating the above:
`n_samples`, `n_extracted_packages`, `n_registry_valid`, `n_stdlib`,
`n_hallucinated`, `phr_with_stdlib`, `phr_without_stdlib`, `delta`
(the last three following the with/without-stdlib convention described in
[Configuration](#configuration) and mirrored in `stdlib_correction.tex`),
plus `syntax_valid_rate`, `mean_smell_count`, `total_smell_issues` for
`code`-task files. `scripts/analyze.py` reads these `"stats"` blocks
directly (see the `analyze.py` entry in the CLI reference above).

Semgrep is invoked **once per file** (not once per sample) by scanning a
temp directory of all of that file's code snippets at once — its ~2-3s
fixed startup cost dominates, so batching keeps a full-corpus run to
single-digit minutes instead of hours.

Additional dependencies beyond `requirements.txt`:

```bash
pip install tree-sitter tree-sitter-language-pack semgrep
```

Re-running either script is idempotent — it recomputes and overwrites the
`extracted_packages`/`stats` fields each time rather than appending to them.


---

## Adding a new model family

1. Add a **small** model entry with `id`, `alias`, and a `dola` block.
2. Add a **large** model entry with `id`, `alias`, `dola` block, plus `nudging` and `contrastive_decoding` blocks referencing the small model's alias.
3. Run the pipeline — no code changes needed.

```yaml
# Example: adding Gemma 1B + 4B
- id: google/gemma-3-1b-it
  alias: gemma_1b
  dola:
    mature_layer: 18
    early_exit_layers: [4, 8, 12, 16]

- id: google/gemma-3-4b-it
  alias: gemma_4b
  dola:
    mature_layer: 26
    early_exit_layers: [4, 8, 12, 16]
  nudging:
    guide_model: gemma_1b
    top_prob_thres: 0.9
  contrastive_decoding:
    expert_model: gemma_4b
    amateur_model: gemma_1b
    alpha: 0.1
    repetition_penalty: 1.0
```
