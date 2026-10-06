#!/usr/bin/env python3
"""
Generate outputs for all enabled (model, strategy, task) combinations defined under configs/.

Config layout
-------------
configs/
  data.yml                      ← languages, tasks, paths, max_samples, output dir
  <model_alias>/
    model.yml                   ← id, alias, torch_dtype, device_map, quantization
    baseline.yml                ← strategy params (enabled: true/false, ...)
    greedy.yml
    self_refine.yml
    dola.yml                    ← also holds mature_layer, early_exit_layers
    rag.yml
    nudging.yml                 ← only on larger models; names guide_model
    contrastive_decoding.yml    ← only on larger models; names expert/amateur

Two experiment tasks are supported (data.yml → tasks):
  packages  Generate a list of packages needed to solve the problem.
  code      Generate full code that solves the problem.

The presence of a strategy file means the strategy is applicable to that model.
The `enabled` field inside the file controls whether it actually runs.
Paired strategies (nudging, contrastive_decoding) are detected by the presence
of guide_model / expert_model keys respectively.

Usage examples
--------------
    python scripts/generate.py
    python scripts/generate.py --tasks packages
    python scripts/generate.py --tasks packages code --strategies baseline greedy --models deepseek_1.3b --languages Python
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jinja2
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mitigation.baseline import BaselineGenerator
from mitigation.greedy import GreedyGenerator
from mitigation.self_refine import SelfRefineGenerator
from mitigation.dola import DoLaGenerator
from mitigation.actlcd import ActLCDGenerator
from mitigation.rag import RagGenerator
from mitigation.nudging import NudgingGenerator
from mitigation.contrastive_decoding import ContrastiveDecodingGenerator

PAIRED_STRATEGIES = {"nudging", "contrastive_decoding"}

TASK_SYSTEM_PROMPTS = {
    "packages": "system_prompt_package_generation.jinja",
    "code": "system_prompt_code_generation.jinja",
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_config_dir(config_dir: Path) -> Dict[str, Any]:
    """
    Load all configs from the directory structure and return a merged dict:
    {
      "data": {...},
      "models": [model_cfg, ...],
      "strategies": {
          alias: {strategy_name: strategy_cfg, ...},
          ...
      }
    }
    """
    data_cfg = load_yaml(config_dir / "data.yml")

    models: List[Dict] = []
    strategies: Dict[str, Dict[str, Dict]] = {}

    for model_dir in sorted(config_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_yml = model_dir / "model.yml"
        if not model_yml.is_file():
            continue

        model_cfg = load_yaml(model_yml)
        alias = model_cfg["alias"]
        models.append(model_cfg)
        strategies[alias] = {}

        for strat_file in sorted(model_dir.glob("*.yml")):
            if strat_file.name == "model.yml":
                continue
            strategy_name = strat_file.stem
            strategies[alias][strategy_name] = load_yaml(strat_file)

    return {"data": data_cfg, "models": models, "strategies": strategies}


def resolve_alias(alias: str, models: List[Dict]) -> Dict:
    for m in models:
        if m["alias"] == alias:
            return m
    raise ValueError(f"Model alias '{alias}' not found in loaded configs.")


def validate_config(cfg: Dict) -> None:
    """Fail fast on obvious misconfigurations before loading any model."""
    models = cfg["models"]
    all_aliases = {m["alias"] for m in models}
    strategies = cfg["strategies"]

    for model_cfg in models:
        alias = model_cfg["alias"]
        model_strats = strategies.get(alias, {})

        cd_cfg = model_strats.get("contrastive_decoding")
        if cd_cfg and cd_cfg.get("enabled", False):
            if cd_cfg.get("expert_model") != alias:
                raise SystemExit(
                    f"[config] {alias}/contrastive_decoding.yml: "
                    f"expert_model must equal '{alias}' "
                    f"(got '{cd_cfg.get('expert_model')}')."
                )
            amateur = cd_cfg.get("amateur_model")
            if amateur and amateur not in all_aliases:
                raise SystemExit(
                    f"[config] {alias}/contrastive_decoding.yml: "
                    f"amateur_model '{amateur}' is not a known alias."
                )

        nudge_cfg = model_strats.get("nudging")
        if nudge_cfg and nudge_cfg.get("enabled", False):
            guide = nudge_cfg.get("guide_model")
            if guide and guide not in all_aliases:
                raise SystemExit(
                    f"[config] {alias}/nudging.yml: "
                    f"guide_model '{guide}' is not a known alias."
                )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_instructions(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("packages", [])


def save_results(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"packages": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def instruction_text(pkg: Dict) -> str:
    return pkg.get("instruction", "") or pkg.get("description", "")


def render_system_prompt(task: str, language: str) -> str:
    template_name = TASK_SYSTEM_PROMPTS[task]
    prompts_dir = ROOT / "prompts"
    text = (prompts_dir / template_name).read_text(encoding="utf-8")
    env = jinja2.Environment(autoescape=False)
    return env.from_string(text).render(language=language)


def build_chat_prompt(system_prompt: str, instruction: str) -> str:
    """Format a system + user turn the way the existing generators expect it."""
    return f"system: {system_prompt}\nuser: {instruction}\nassistant: "


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def batch_run(gen, packages: List[Dict], label: str, batch_size: int,
              system_prompt: str = "",
              extra_meta: Optional[Dict] = None) -> None:
    cur_bs = max(1, batch_size)
    idx = 0
    extra_meta = extra_meta or {}

    pbar = tqdm(total=len(packages), desc=label, leave=False)
    while idx < len(packages):
        chunk = packages[idx: idx + cur_bs]
        if system_prompt:
            prompts = [build_chat_prompt(system_prompt, instruction_text(p)) for p in chunk]
        else:
            prompts = [instruction_text(p) for p in chunk]
        try:
            t0 = time.perf_counter()
            answers = gen.batch_generate(prompts)
            elapsed = time.perf_counter() - t0
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and torch.cuda.is_available() and cur_bs > 1:
                torch.cuda.empty_cache()
                cur_bs = max(1, cur_bs // 2)
                continue
            raise
        per = elapsed / len(chunk) if chunk else 0.0
        for pkg, ans in zip(chunk, answers):
            pkg["answer"] = ans
            pkg["time_sec"] = per
            pkg.update(extra_meta)
        idx += cur_bs
        pbar.update(len(chunk))
    pbar.close()


def hf_kwargs(model_cfg: Dict) -> Dict:
    kw: Dict[str, Any] = {}
    if model_cfg.get("torch_dtype") in ("float16", "bfloat16"):
        kw["torch_dtype"] = model_cfg["torch_dtype"]
    if model_cfg.get("device_map") is not None:
        kw["device_map"] = model_cfg["device_map"]
    if model_cfg.get("quantization"):
        kw["quantization"] = model_cfg["quantization"]
    return kw


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

def run_baseline(packages, model_cfg, strat_cfg, system_prompt: str = ""):
    gen = BaselineGenerator(model_name=model_cfg["id"],
                            config={**strat_cfg, **hf_kwargs(model_cfg)})
    batch_run(gen, packages, "baseline", strat_cfg.get("batch_size", 1),
              system_prompt=system_prompt,
              extra_meta={"model": model_cfg["alias"],
                          "temperature": strat_cfg.get("temperature", 0.0),
                          "max_new_tokens": strat_cfg.get("max_new_tokens", 128)})
    return gen


def run_greedy(packages, model_cfg, strat_cfg, system_prompt: str = ""):
    gen = GreedyGenerator(model_name=model_cfg["id"],
                          config={**strat_cfg, **hf_kwargs(model_cfg)})
    batch_run(gen, packages, "greedy", strat_cfg.get("batch_size", 1),
              system_prompt=system_prompt,
              extra_meta={"model": model_cfg["alias"], "temperature": 0.0,
                          "max_new_tokens": strat_cfg.get("max_new_tokens", 128)})


def run_self_refine(packages, model_cfg, strat_cfg, task: str = "packages",
                    language: str = "", inner_gen=None):
    if inner_gen is None:
        inner_gen = BaselineGenerator(model_name=model_cfg["id"],
                                      config={**strat_cfg, **hf_kwargs(model_cfg)})
    gen = SelfRefineGenerator(inner=inner_gen, language=language,
                              task=task, config=strat_cfg)
    for pkg in tqdm(packages, desc="self_refine", leave=False):
        answer, elapsed, rounds = gen._self_refine(instruction_text(pkg))
        pkg.update({"answer": answer, "time_sec": elapsed, "rounds_used": rounds,
                    "model": model_cfg["alias"],
                    "max_rounds": strat_cfg.get("max_rounds", 3),
                    "max_new_tokens": strat_cfg.get("max_new_tokens", 128)})


def run_dola(packages, model_cfg, strat_cfg, system_prompt: str = ""):
    gen = DoLaGenerator(model_name=model_cfg["id"],
                        config={**strat_cfg, **hf_kwargs(model_cfg)})
    batch_run(gen, packages, "dola", strat_cfg.get("batch_size", 1),
              system_prompt=system_prompt,
              extra_meta={"model": model_cfg["alias"],
                          "max_new_tokens": strat_cfg.get("max_new_tokens", 128),
                          "mature_layer": strat_cfg.get("mature_layer")})


def run_actlcd(packages, model_cfg, strat_cfg, system_prompt: str = ""):
    gen = ActLCDGenerator(model_name=model_cfg["id"],
                          config={**strat_cfg, **hf_kwargs(model_cfg)})
    batch_run(gen, packages, "actlcd", strat_cfg.get("batch_size", 1),
              system_prompt=system_prompt,
              extra_meta={"model": model_cfg["alias"],
                          "max_new_tokens": strat_cfg.get("max_new_tokens", 128),
                          "mature_layer": strat_cfg.get("mature_layer"),
                          "policy_path": strat_cfg.get("policy_path")})


def run_rag(packages, model_cfg, strat_cfg, language, system_prompt: str = "",
            inner_gen=None):
    if inner_gen is None:
        inner_gen = BaselineGenerator(model_name=model_cfg["id"],
                                      config={**strat_cfg, **hf_kwargs(model_cfg)})
    gen = RagGenerator(inner=inner_gen, language=language, config=strat_cfg)
    batch_run(gen, packages, "rag", strat_cfg.get("batch_size", 1),
              system_prompt=system_prompt,
              extra_meta={"model": model_cfg["alias"],
                          "max_new_tokens": strat_cfg.get("max_new_tokens", 128)})


def run_nudging(packages, primary_cfg, partner_cfg, strat_cfg, system_prompt: str = ""):
    gen = NudgingGenerator(base_model_name=primary_cfg["id"],
                           nudging_model_name=partner_cfg["id"],
                           config={**strat_cfg, **hf_kwargs(primary_cfg)})
    pair_label = f"{primary_cfg['alias']}__{partner_cfg['alias']}"
    batch_run(gen, packages, "nudging", strat_cfg.get("batch_size", 1),
              system_prompt=system_prompt,
              extra_meta={"model": pair_label,
                          "generator_model": primary_cfg["alias"],
                          "guide_model": partner_cfg["alias"],
                          "top_prob_thres": strat_cfg.get("top_prob_thres"),
                          "max_new_tokens": strat_cfg.get("max_new_tokens", 128)})
    return pair_label


def run_contrastive_decoding(packages, primary_cfg, partner_cfg, strat_cfg,
                             system_prompt: str = ""):
    gen = ContrastiveDecodingGenerator(expert_model_name=primary_cfg["id"],
                                       amateur_model_name=partner_cfg["id"],
                                       config={**strat_cfg, **hf_kwargs(primary_cfg)})
    pair_label = f"{primary_cfg['alias']}__{partner_cfg['alias']}"
    batch_run(gen, packages, "contrastive_decoding", strat_cfg.get("batch_size", 1),
              system_prompt=system_prompt,
              extra_meta={"model": pair_label,
                          "expert_model": primary_cfg["alias"],
                          "amateur_model": partner_cfg["alias"],
                          "alpha": strat_cfg.get("alpha"),
                          "max_new_tokens": strat_cfg.get("max_new_tokens", 128)})
    return pair_label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run generation strategies as defined under configs/<model>/<strategy>.yml."
    )
    parser.add_argument(
        "--config-dir",
        default=str(ROOT / "configs"),
        help="Root config directory (default: configs/).",
    )
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Run only these strategies.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Run only these model aliases.")
    parser.add_argument("--languages", nargs="+", default=None,
                        help="Process only these languages.")
    parser.add_argument("--tasks", nargs="+", default=None,
                        choices=list(TASK_SYSTEM_PROMPTS),
                        help="Run only these tasks (default: all tasks listed in data.yml).")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    cfg = load_config_dir(config_dir)
    validate_config(cfg)

    data_cfg = cfg["data"]
    output_root = ROOT / data_cfg.get("output", {}).get("root_dir", "output")
    instruction_dir = ROOT / data_cfg.get("instruction_dir", "data/instruction")
    max_samples = data_cfg.get("max_samples", {}) or {}
    languages: List[str] = args.languages or data_cfg.get("languages", [])
    tasks: List[str] = args.tasks or data_cfg.get("tasks", list(TASK_SYSTEM_PROMPTS))

    all_models = cfg["models"]
    all_strategies = cfg["strategies"]  # {alias: {strategy_name: cfg}}

    # Apply CLI filters
    if args.models:
        all_models = [m for m in all_models if m["alias"] in args.models]
    if not all_models:
        raise SystemExit("No models matched.")

    # Collect the union of strategy names across all (filtered) models
    all_strategy_names = sorted({
        s for m in all_models
        for s in all_strategies.get(m["alias"], {})
        if args.strategies is None or s in args.strategies
    })
    if not all_strategy_names:
        raise SystemExit("No strategies found (check --strategies filter or config files).")

    print(f"Tasks      : {tasks}")
    print(f"Strategies : {all_strategy_names}")
    print(f"Models     : {[m['alias'] for m in all_models]}")
    print(f"Languages  : {languages}")

    for task in tasks:
        print(f"\n{'#'*60}\nTask: {task}\n{'#'*60}")

        for strategy_name in all_strategy_names:
            print(f"\n{'='*60}\nStrategy: {strategy_name}\n{'='*60}")

            for language in languages:
                in_path = instruction_dir / f"packages_{language}_instruct.json"
                if not in_path.is_file():
                    print(f"  [{language}] Instruction file not found, skipping.")
                    continue

                base_packages = load_instructions(in_path)
                n = max_samples.get(language)
                if isinstance(n, int) and n > 0:
                    base_packages = base_packages[:n]

                system_prompt = render_system_prompt(task, language)

                for model_cfg in all_models:
                    alias = model_cfg["alias"]
                    strat_cfg = all_strategies.get(alias, {}).get(strategy_name)

                    if strat_cfg is None:
                        continue  # this model has no config file for this strategy
                    force = args.strategies is not None and strategy_name in args.strategies
                    if not force and not strat_cfg.get("enabled", False):
                        continue  # file exists but disabled

                    packages = copy.deepcopy(base_packages)
                    is_paired = strategy_name in PAIRED_STRATEGIES

                    # Paired strategies
                    if is_paired:
                        if strategy_name == "nudging":
                            partner_alias = strat_cfg.get("guide_model")
                        else:  # contrastive_decoding
                            partner_alias = strat_cfg.get("amateur_model")

                        if not partner_alias:
                            print(f"  [{language}] {alias}/{strategy_name}: missing partner alias, skipping.")
                            continue

                        partner_cfg = resolve_alias(partner_alias, cfg["models"])
                        print(f"  [{language}] {alias}__{partner_alias} ...", end=" ", flush=True)

                        if strategy_name == "nudging":
                            pair_label = run_nudging(packages, model_cfg, partner_cfg,
                                                     strat_cfg, system_prompt)
                        else:
                            pair_label = run_contrastive_decoding(packages, model_cfg,
                                                                   partner_cfg, strat_cfg,
                                                                   system_prompt)

                        out = output_root / strategy_name / task / pair_label / f"{language}.json"

                    # Single-model strategies
                    else:
                        print(f"  [{language}] {alias} ...", end=" ", flush=True)

                        if strategy_name == "baseline":
                            run_baseline(packages, model_cfg, strat_cfg, system_prompt)
                        elif strategy_name == "greedy":
                            run_greedy(packages, model_cfg, strat_cfg, system_prompt)
                        elif strategy_name == "self_refine":
                            run_self_refine(packages, model_cfg, strat_cfg,
                                            task=task, language=language)
                        elif strategy_name == "dola":
                            run_dola(packages, model_cfg, strat_cfg, system_prompt)
                        elif strategy_name == "actlcd":
                            run_actlcd(packages, model_cfg, strat_cfg, system_prompt)
                        elif strategy_name == "rag":
                            run_rag(packages, model_cfg, strat_cfg, language, system_prompt)
                        else:
                            print(f"unknown strategy '{strategy_name}', skipping.")
                            continue

                        out = output_root / strategy_name / task / alias / f"{language}.json"

                    save_results(out, packages)
                    print(f"→ {out.relative_to(ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
