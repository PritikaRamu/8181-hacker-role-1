"""
Active Layer-Contrastive Decoding (ActLCD) — Zhang et al., EMNLP 2025.
https://arxiv.org/abs/2505.23657

This module implements the token-by-token generation loop that sits on top of
the DoLa layer-contrast mechanism. At each decoding step a small BCQ policy
decides whether to apply the contrastive adjustment or fall back to plain
greedy decoding from the mature layer.

When no policy is supplied the behaviour is identical to DoLa.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
)
from transformers.generation.stopping_criteria import StoppingCriteriaList
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..actlcd import BCQAgent


# ---------------------------------------------------------------------------
# Shared helpers (duplicated from dola.py to keep modules independent)
# ---------------------------------------------------------------------------

def relative_top_filter(
    scores: torch.FloatTensor,
    relative_top: float = 0.1,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    scores_normalized = scores.log_softmax(dim=-1)
    sorted_logits, _ = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh).unsqueeze(-1)
    scores_normalized[scores_normalized < probs_thresh] = filter_value
    return scores_normalized


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

TOP_K_STATE = 5  # number of top tokens used to build the BCQ state vector


def _build_state(
    premature_logits: Dict[int, torch.Tensor],
    mature_logits: torch.Tensor,
    candidate_premature_layers: List[int],
) -> np.ndarray:
    """
    Build the flat feature vector fed to the BCQ policy at each token step.

    For every premature layer and for the mature layer the top-K token IDs
    and their softmax probabilities are concatenated in order:

        [premature_layer_0_id_0, premature_layer_0_prob_0, ...,
         premature_layer_0_id_K, premature_layer_0_prob_K,
         ...
         mature_id_0, mature_prob_0, ..., mature_id_K, mature_prob_K]

    State dimension = (num_premature_layers + 1) × TOP_K_STATE × 2.
    """
    parts: List[float] = []

    # Premature layers
    for layer in candidate_premature_layers:
        logits = premature_logits[layer][0]          # (vocab,)
        probs  = torch.softmax(logits.float(), dim=-1)
        topk   = torch.topk(probs, k=TOP_K_STATE)
        for tid, tp in zip(topk.indices.tolist(), topk.values.tolist()):
            parts.append(float(tid))
            parts.append(float(tp))

    # Mature layer
    probs  = torch.softmax(mature_logits[0].float(), dim=-1)
    topk   = torch.topk(probs, k=TOP_K_STATE)
    for tid, tp in zip(topk.indices.tolist(), topk.values.tolist()):
        parts.append(float(tid))
        parts.append(float(tp))

    return np.array(parts, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main ActLCD decoding function
# ---------------------------------------------------------------------------

def actlcd(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    max_new_tokens: int = 128,
    mature_layer: int = None,
    candidate_premature_layers: List[int] = None,
    relative_top: float = 0.1,
    repetition_penalty: float = 1.2,
    eos_token_id: int = None,
    early_stop: bool = True,
    stopping_criteria: StoppingCriteriaList = None,
    policy: Optional["BCQAgent"] = None,
) -> torch.LongTensor:
    """
    ActLCD decoding loop.

    Identical to the DoLa loop except that at each step a BCQ policy
    (``policy``) decides whether to apply the layer-contrastive adjustment:

      action = 1  →  apply DoLa contrast   (same as standard DoLa)
      action = 0  →  use mature-layer logits directly (no contrast)

    If ``policy`` is None every step applies contrast, recovering standard DoLa.

    Args:
        model:                      HuggingFace CausalLM (must support output_hidden_states).
        tokenizer:                  Corresponding tokenizer.
        input_ids:                  Prompt token ids, shape (1, seq_len).
                                    NOTE: ActLCD processes one sequence at a time
                                    because the BCQ policy operates on a single
                                    state vector per step.
        attention_mask:             Attention mask, shape (1, seq_len).
        max_new_tokens:             Maximum tokens to generate.
        mature_layer:               Index of the mature (final signal) layer.
        candidate_premature_layers: Candidate early layers for dynamic contrast.
        relative_top:               Relative-top filter threshold (0 = disabled).
        repetition_penalty:         Repetition penalty applied to logits.
        eos_token_id:               EOS token id override.
        early_stop:                 Stop generation at EOS when True.
        stopping_criteria:          Optional custom StoppingCriteriaList.
        policy:                     Optional pre-trained BCQAgent.  When None
                                    the method behaves identically to DoLa.

    Returns:
        Full token sequence including the original prompt, shape (1, seq_len + new_tokens).
    """
    if mature_layer is None:
        raise ValueError("mature_layer must be specified.")
    if input_ids.shape[0] != 1:
        raise ValueError(
            "ActLCD processes one sequence at a time (batch_size must be 1). "
            "Call generate() in a loop for multiple prompts."
        )

    candidate_premature_layers = candidate_premature_layers or []
    all_layers = candidate_premature_layers + [mature_layer]

    device = input_ids.device
    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    eos_tensor = (
        torch.tensor([eos_token_id], device=device) if eos_token_id is not None else None
    )
    unfinished = input_ids.new_ones(1)

    processors = LogitsProcessorList()
    if repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    stopping_criteria = stopping_criteria or StoppingCriteriaList()

    for _ in range(max_new_tokens):
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        position_ids = position_ids.masked_fill(attention_mask == 0, 1)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                output_hidden_states=True,
            )

        # Logits at the last generated position for every layer of interest
        layer_logits: Dict[int, torch.Tensor] = {
            layer: model.lm_head(outputs.hidden_states[layer][:, -1:, :])
            for layer in all_layers
        }

        mature_logits = layer_logits[mature_layer][:, 0, :]       # (1, V)
        premature_logits = {
            l: layer_logits[l][:, 0, :]                           # (1, V)
            for l in candidate_premature_layers
        }

        # ── Policy: decide whether to apply contrast ───────────────────────
        apply_contrast = True
        if policy is not None and candidate_premature_layers:
            state  = _build_state(premature_logits, mature_logits, candidate_premature_layers)
            action = policy.select_action(state, mature_logits=mature_logits)
            apply_contrast = bool(action == 1)

        # ── Logit computation ──────────────────────────────────────────────
        if apply_contrast and candidate_premature_layers:
            # Dynamic JS-divergence layer selection (same as DoLa)
            stacked_pre = torch.stack(
                [premature_logits[l] for l in candidate_premature_layers], dim=0
            )                                                       # (C, 1, V)

            sm_mature = F.softmax(mature_logits, dim=-1)            # (1, V)
            sm_pre    = F.softmax(stacked_pre,   dim=-1)            # (C, 1, V)

            M   = 0.5 * (sm_mature[None] + sm_pre)
            kl1 = F.kl_div(F.log_softmax(mature_logits, dim=-1)[None], M, reduction="none").mean(-1)
            kl2 = F.kl_div(F.log_softmax(stacked_pre,  dim=-1),       M, reduction="none").mean(-1)
            js  = (0.5 * (kl1 + kl2)).mean(-1)                     # (C,)

            best_pre_layer = candidate_premature_layers[int(js.argmax().cpu().item())]
            base_logits    = premature_logits[best_pre_layer]

            if relative_top > 0.0:
                final_filtered = relative_top_filter(mature_logits.clone(), relative_top)
                base_filtered  = base_logits.log_softmax(dim=-1)
                base_filtered[final_filtered < -1e3] = -1e3
                next_token_logits = final_filtered - base_filtered
            else:
                next_token_logits = mature_logits.log_softmax(dim=-1) - base_logits.log_softmax(dim=-1)
        else:
            # No contrast: decode directly from mature layer
            next_token_logits = mature_logits

        # ── Post-processing and token selection ────────────────────────────
        next_token_logits = next_token_logits.to(device)
        scores = processors(input_ids, next_token_logits)

        if not early_stop and eos_tensor is not None:
            scores[:, eos_token_id] = -float("inf")

        next_tokens = torch.argmax(scores, dim=-1)
        next_tokens = next_tokens * unfinished + tokenizer.pad_token_id * (1 - unfinished)

        input_ids      = torch.cat([input_ids,      next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=-1)

        if eos_tensor is not None:
            unfinished = unfinished.mul(
                next_tokens.tile(eos_tensor.shape[0], 1)
                .ne(eos_tensor.unsqueeze(1))
                .prod(dim=0)
            )
        unfinished = unfinished & ~stopping_criteria(input_ids, None)
        if unfinished.max() == 0:
            break

    return input_ids
