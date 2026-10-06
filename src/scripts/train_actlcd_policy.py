#!/usr/bin/env python3
"""
Train the BCQ policy for ActLCD.

Two-phase pipeline
------------------
Phase 1 — Feature collection  (--phase collect)
    Runs the full DoLa generation loop on a labelled dataset while recording,
    at each token step, the top-K logit features from the premature and mature
    layers together with a binary label indicating whether DoLa's contrastive
    adjustment produced a token that is part of a known-valid package name.

    Output: a CSV file with one row per generated token.

Phase 2 — BCQ training  (--phase train)
    Reads the CSV produced in Phase 1, constructs offline RL transitions, and
    trains the BCQAgent (behaviour-cloning pre-training followed by Q-network
    fine-tuning).  Saves the policy to ``--policy-out``.

Usage
-----
    # Collect features (requires a GPU with the target model)
    python scripts/train_actlcd_policy.py \\
        --phase collect \\
        --model-name deepseek-ai/deepseek-coder-1.3b-instruct \\
        --model-alias deepseek_1.3b \\
        --language Python \\
        --task packages \\
        --features-out data/actlcd/deepseek_1.3b_Python_packages.csv

    # Train BCQ
    python scripts/train_actlcd_policy.py \\
        --phase train \\
        --features-in data/actlcd/deepseek_1.3b_Python_packages.csv \\
        --policy-out model/actlcd/deepseek_1.3b_Python_packages.pth \\
        --hidden 1024 512 256 \\
        --bc-epochs 35 \\
        --q-epochs 130

    # Then point configs/deepseek_1.3b/actlcd.yml → policy_path to the .pth file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Phase 1 — Feature collection
# ---------------------------------------------------------------------------

def collect_features(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from mitigation.actlcd import BCQAgent
    from mitigation.decoding.actlcd import TOP_K_STATE, relative_top_filter

    cfg_path = ROOT / "configs" / args.model_alias / "actlcd.yml"
    if not cfg_path.is_file():
        raise SystemExit(f"actlcd.yml not found for model '{args.model_alias}': {cfg_path}")

    import yaml
    strat_cfg = yaml.safe_load(cfg_path.read_text()) or {}
    model_cfg = yaml.safe_load(
        (ROOT / "configs" / args.model_alias / "model.yml").read_text()
    ) or {}

    mature_layer     = int(strat_cfg["mature_layer"])
    premature_layers = [int(x) for x in strat_cfg["early_exit_layers"]]
    all_layers       = premature_layers + [mature_layer]
    state_dim        = BCQAgent.state_dim_for(len(premature_layers))

    # Load model
    model_name   = args.model_name or model_cfg["id"]
    tokenizer    = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    quant_cfg = model_cfg.get("quantization") or {}
    if quant_cfg.get("load_in_4bit") or quant_cfg.get("load_in_8bit"):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(**quant_cfg)
    model_kwargs["device_map"] = model_cfg.get("device_map", "auto")
    if model_cfg.get("torch_dtype") == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    # Load package registry for labelling
    pkg_list_dir = ROOT / "data" / "package_list"
    registry_files = {
        "Python":     "pypi_package_names.txt",
        "JavaScript": "npm_package_names.txt",
        "Rust":       "cargo_package_names.txt",
        "Ruby":       "rubygems_packages_names.txt",
    }
    pkg_file = pkg_list_dir / registry_files.get(args.language, "")
    valid_packages = set()
    if pkg_file.is_file():
        valid_packages = {ln.strip().lower() for ln in pkg_file.read_text().splitlines() if ln.strip()}

    # Load instruction data
    instr_path = ROOT / "data" / "instruction" / f"packages_{args.language}_instruct.json"
    instructions = json.loads(instr_path.read_text())["packages"]
    if args.max_samples:
        instructions = instructions[: args.max_samples]

    # Load system prompt
    import jinja2
    task = args.task
    tmpl_name = "system_prompt_package_generation.jinja" if task == "packages" else "system_prompt_code_generation.jinja"
    tmpl_text = (ROOT / "prompts" / tmpl_name).read_text()
    system_prompt = jinja2.Environment(autoescape=False).from_string(tmpl_text).render(language=args.language)

    out_path = Path(args.features_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Column names
    pre_cols = []
    for li, l in enumerate(premature_layers):
        for ki in range(TOP_K_STATE):
            pre_cols += [f"pre{li}_id{ki}", f"pre{li}_prob{ki}"]
    mat_cols = [f"mat_id{ki}" for ki in range(TOP_K_STATE)] + [f"mat_prob{ki}" for ki in range(TOP_K_STATE)]
    # interleaved like _build_state: id0,prob0,id1,prob1,...
    mat_cols_interleaved = []
    for ki in range(TOP_K_STATE):
        mat_cols_interleaved += [f"mat_id{ki}", f"mat_prob{ki}"]
    all_feature_cols = pre_cols + mat_cols_interleaved

    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["mature_top1_token"] + all_feature_cols + ["label"])

        for item in instructions:
            instruction = item.get("instruction", "") or item.get("description", "")
            prompt = f"system: {system_prompt}\nuser: {instruction}\nassistant: "

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            input_ids     = inputs["input_ids"].to(next(model.parameters()).device)
            attention_mask = inputs["attention_mask"].to(input_ids.device)
            max_new = int(strat_cfg.get("max_new_tokens", 128))

            for _ in range(max_new):
                pos_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
                pos_ids = pos_ids.masked_fill(attention_mask == 0, 1)

                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=pos_ids,
                        return_dict=True,
                        output_hidden_states=True,
                    )

                layer_logits = {
                    l: model.lm_head(outputs.hidden_states[l][:, -1:, :])[:, 0, :]
                    for l in all_layers
                }

                mature_logits = layer_logits[mature_layer]
                pre_logits    = {l: layer_logits[l] for l in premature_layers}

                # Build state row
                row_features: List[float] = []
                for l in premature_layers:
                    probs = torch.softmax(pre_logits[l][0].float(), dim=-1)
                    topk  = torch.topk(probs, k=TOP_K_STATE)
                    for tid, tp in zip(topk.indices.tolist(), topk.values.tolist()):
                        row_features += [tid, tp]

                mat_probs = torch.softmax(mature_logits[0].float(), dim=-1)
                mat_topk  = torch.topk(mat_probs, k=TOP_K_STATE)
                mat_top_token_id = mat_topk.indices[0].item()
                for tid, tp in zip(mat_topk.indices.tolist(), mat_topk.values.tolist()):
                    row_features += [tid, tp]

                # DoLa contrast to get the adjusted next token
                stacked_pre = torch.stack([pre_logits[l] for l in premature_layers], dim=0)
                sm_mature   = F.softmax(mature_logits, dim=-1)
                sm_pre      = F.softmax(stacked_pre, dim=-1)
                M           = 0.5 * (sm_mature[None] + sm_pre)
                kl1         = F.kl_div(F.log_softmax(mature_logits, dim=-1)[None], M, reduction="none").mean(-1)
                kl2         = F.kl_div(F.log_softmax(stacked_pre, dim=-1), M, reduction="none").mean(-1)
                js          = (0.5 * (kl1 + kl2)).mean(-1)
                best_pre    = premature_layers[int(js.argmax().cpu().item())]

                rel_top = float(strat_cfg.get("relative_top", 0.1))
                if rel_top > 0.0:
                    final_f = relative_top_filter(mature_logits.clone(), rel_top)
                    base_f  = pre_logits[best_pre].log_softmax(dim=-1)
                    base_f[final_f < -1e3] = -1e3
                    dola_logits = final_f - base_f
                else:
                    dola_logits = mature_logits.log_softmax(dim=-1) - pre_logits[best_pre].log_softmax(dim=-1)

                dola_token_id  = int(torch.argmax(dola_logits, dim=-1)[0].item())
                plain_token_id = mat_top_token_id

                # Label: 1 if DoLa chose a valid package token and plain did not,
                #        0 if plain was already valid or DoLa made it worse.
                dola_word  = tokenizer.decode([dola_token_id]).strip().lower()
                plain_word = tokenizer.decode([plain_token_id]).strip().lower()
                dola_valid  = dola_word in valid_packages
                plain_valid = plain_word in valid_packages

                if dola_valid and not plain_valid:
                    label = 1   # DoLa correction was beneficial
                elif not dola_valid and plain_valid:
                    label = 0   # DoLa made things worse
                else:
                    label = 1   # neutral: apply contrast (conservative default)

                mat_top1_text = tokenizer.decode([mat_top_token_id])
                writer.writerow([mat_top1_text] + row_features + [label])

                # Advance with DoLa token
                next_tok   = torch.tensor([[dola_token_id]], device=input_ids.device)
                input_ids  = torch.cat([input_ids, next_tok], dim=-1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=-1)

                if tokenizer.eos_token_id is not None and dola_token_id == tokenizer.eos_token_id:
                    break

    print(f"Features written to {out_path}  (state_dim={state_dim})")


# ---------------------------------------------------------------------------
# Phase 2 — BCQ training
# ---------------------------------------------------------------------------

def train_policy(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from mitigation.actlcd import BCQAgent, _MLP

    features_path = Path(args.features_in)
    if not features_path.is_file():
        raise SystemExit(f"Features CSV not found: {features_path}")

    df = pd.read_csv(features_path)
    feature_cols = [c for c in df.columns if c not in ("mature_top1_token", "label")]
    state_dim    = len(feature_cols)
    end_tokens   = {"<|end_of_text|>", "<eos>", "</s>", "<|endoftext|>"}

    # Group into per-sequence episodes
    sequences:    List[np.ndarray] = []
    token_labels: List[np.ndarray] = []
    cur_feats:    List[np.ndarray] = []
    cur_labels:   List[int]        = []

    for _, row in df.iterrows():
        cur_feats.append(row[feature_cols].values.astype(np.float32))
        cur_labels.append(int(row["label"]))
        if str(row["mature_top1_token"]) in end_tokens or row.name == len(df) - 1:
            if cur_feats:
                sequences.append(np.array(cur_feats))
                token_labels.append(np.array(cur_labels))
            cur_feats, cur_labels = [], []

    print(f"  {len(sequences)} sequences, state_dim={state_dim}")

    # Offline transition dataset
    class OfflineDataset(Dataset):
        def __init__(self, seqs, labels):
            self.transitions = []
            reward_tp, reward_fn, reward_fp, reward_tn = 1.0, -4.0, -1.0, 1.0
            for seq, lbls in zip(seqs, labels):
                T = len(seq)
                for t in range(T):
                    action = int(lbls[t])
                    next_s = seq[t + 1] if t < T - 1 else np.zeros(state_dim, np.float32)
                    done   = float(t == T - 1)
                    # Intermediate reward = 0; terminal = cumulative
                    if t < T - 1:
                        reward = 0.0
                    else:
                        reward = sum(
                            reward_tp if (p == 1 and l == 1) else
                            reward_fn if (p == 0 and l == 1) else
                            reward_fp if (p == 1 and l == 0) else
                            reward_tn
                            for p, l in zip(lbls, lbls)
                        )
                    self.transitions.append((seq[t], action, reward, next_s, done))

        def __len__(self): return len(self.transitions)

        def __getitem__(self, i):
            s, a, r, ns, d = self.transitions[i]
            return (torch.tensor(s, dtype=torch.float32),
                    torch.tensor(a, dtype=torch.long),
                    torch.tensor(r, dtype=torch.float32),
                    torch.tensor(ns, dtype=torch.float32),
                    torch.tensor(d, dtype=torch.float32))

    device  = torch.device(args.device if torch.cuda.is_available() else "cpu")
    agent   = BCQAgent(state_dim, hidden=args.hidden, bc_threshold=args.bc_threshold, device=str(device))
    dataset = OfflineDataset(sequences, token_labels)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Phase 2a — BC pre-training
    print("Pre-training behaviour-cloning network …")
    ce_loss = nn.CrossEntropyLoss()
    bc_opt  = torch.optim.AdamW(agent.bc_net.parameters(), lr=args.lr)
    agent.bc_net.train()
    for epoch in range(args.bc_epochs):
        total = 0.0
        for s, a, *_ in loader:
            s, a = s.to(device), a.to(device)
            loss = ce_loss(agent.bc_net(s), a)
            bc_opt.zero_grad(); loss.backward(); bc_opt.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  BC epoch {epoch+1}/{args.bc_epochs}  loss={total/len(loader):.4f}")
    agent.bc_net.eval()

    # Phase 2b — Q-network training
    print("Training Q-network …")
    q_opt   = torch.optim.AdamW(agent.q_net.parameters(), lr=args.lr)
    mse     = nn.MSELoss()
    steps   = 0
    agent.q_net.train()
    for epoch in range(args.q_epochs):
        total = 0.0
        for s, a, r, ns, d in loader:
            s, a, r, ns, d = s.to(device), a.to(device), r.to(device), ns.to(device), d.to(device)
            q_val = agent.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                bc_probs    = torch.softmax(agent.bc_net(ns), dim=1)
                mask        = (bc_probs > args.bc_threshold).float()
                no_allowed  = (mask.sum(dim=1) == 0).unsqueeze(1)
                mask        = mask + no_allowed
                q_next      = agent.q_target(ns) - (1 - mask) * 1e8
                target      = r + args.gamma * q_next.max(dim=1).values * (1 - d)
            loss = mse(q_val, target)
            q_opt.zero_grad(); loss.backward(); q_opt.step()
            total += loss.item(); steps += 1
            if steps % args.target_update == 0:
                agent.q_target.load_state_dict(agent.q_net.state_dict())
        if (epoch + 1) % 10 == 0:
            print(f"  Q epoch {epoch+1}/{args.q_epochs}  loss={total/len(loader):.4f}")
    agent.q_net.eval()

    out_path = Path(args.policy_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(out_path)
    print(f"BCQ policy saved to {out_path}  (state_dim={state_dim})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the BCQ policy for ActLCD (Phase 1: collect, Phase 2: train)."
    )
    parser.add_argument("--phase", required=True, choices=["collect", "train"])

    # Shared
    parser.add_argument("--model-alias", default=None, help="e.g. deepseek_1.3b")

    # Phase 1
    g1 = parser.add_argument_group("collect")
    g1.add_argument("--model-name",   default=None, help="HuggingFace model ID (overrides model.yml).")
    g1.add_argument("--language",     default="Python", choices=["Python", "JavaScript", "Rust", "Ruby"])
    g1.add_argument("--task",         default="packages", choices=["packages", "code"])
    g1.add_argument("--max-samples",  type=int, default=None)
    g1.add_argument("--features-out", default="data/actlcd/features.csv")

    # Phase 2
    g2 = parser.add_argument_group("train")
    g2.add_argument("--features-in",  default="data/actlcd/features.csv")
    g2.add_argument("--policy-out",   default="model/actlcd/policy.pth")
    g2.add_argument("--hidden",       type=int, nargs="+", default=[1024, 512, 256])
    g2.add_argument("--bc-epochs",    type=int,   default=35)
    g2.add_argument("--q-epochs",     type=int,   default=130)
    g2.add_argument("--batch-size",   type=int,   default=1024)
    g2.add_argument("--lr",           type=float, default=3e-4)
    g2.add_argument("--gamma",        type=float, default=1.0)
    g2.add_argument("--bc-threshold", type=float, default=0.3)
    g2.add_argument("--target-update",type=int,   default=200)
    g2.add_argument("--device",       default="cuda")

    args = parser.parse_args()

    if args.phase == "collect":
        if not args.model_alias:
            raise SystemExit("--model-alias is required for phase=collect.")
        collect_features(args)
    else:
        train_policy(args)


if __name__ == "__main__":
    main()
