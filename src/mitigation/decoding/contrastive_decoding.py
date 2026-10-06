"""
Contrastive Decoding (CD) — Li et al. 2022
https://arxiv.org/abs/2210.15097

Core idea:
    At each step, pick the token that maximises:

        CD_score(x) = log p_EXP(x | context) - log p_AMA(x | context)

    subject to the adaptive plausibility constraint V_head:

        V_head = { x : p_EXP(x | context) >= alpha * max_w p_EXP(w | context) }

    Tokens outside V_head are masked to -inf before taking the argmax.

Intuition:
    Failure modes (repetition, incoherence, topic drift) are *more* common in
    smaller/weaker models. Subtracting the amateur log-probs from the expert's
    de-emphasises those failure modes while amplifying knowledge that only the
    expert has learnt.
"""

from typing import Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
)
from transformers.generation.stopping_criteria import StoppingCriteriaList


# ── Helpers ───────────────────────────────────────────────────────────────────

def _position_ids_from_mask(attention_mask: torch.LongTensor) -> torch.LongTensor:
    """
    Compute position_ids from attention_mask.
    Handles left-padded batches correctly and avoids RoPE size-mismatch errors
    that arise from using prepare_inputs_for_generation in a manual loop.
    """
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    position_ids = position_ids.masked_fill(attention_mask == 0, 1)
    return position_ids


def _forward_logprobs(
    model: AutoModelForCausalLM,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
) -> torch.FloatTensor:
    """
    Run a single forward pass and return log-softmax over the *last* token position.
    Shape: (batch_size, vocab_size)
    """
    position_ids = _position_ids_from_mask(attention_mask)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
    # log-softmax for numerical stability
    return outputs.logits[:, -1, :].log_softmax(dim=-1)


# ── Main function ─────────────────────────────────────────────────────────────

def contrastive_decoding(
    expert_model: AutoModelForCausalLM,
    amateur_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    max_new_tokens: int = 128,
    alpha: float = 0.1,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    early_stop: bool = True,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
) -> torch.LongTensor:
    """
    Contrastive Decoding (Li et al., 2022).

    Args:
        expert_model:      Large "strong" model (e.g. OPT-13B, Llama-7B).
        amateur_model:     Small "weak" model from the *same family*
                           (e.g. OPT-125M, Llama-1B).
                           Must share the same tokenizer / vocabulary.
        tokenizer:         Shared tokenizer for both models.
        input_ids:         Prompt token ids,   shape (batch, seq_len).
        attention_mask:    Attention mask,      shape (batch, seq_len).
        max_new_tokens:    Maximum tokens to generate.
        alpha:             Plausibility threshold in [0, 1].
                           Keeps only tokens x where:
                               p_EXP(x) >= alpha * max_w p_EXP(w)
                           alpha=0.1 is the default from the paper.
                           Lower → less filtering (more diversity).
                           Higher → more aggressive filtering (more conservative).
        temperature:       Softens/sharpens expert and amateur distributions
                           *before* computing the CD score. Not in original paper
                           but useful in practice. Set to 1.0 to match the paper.
        repetition_penalty: Multiplicative penalty on previously seen tokens
                           applied to the final CD scores (>1 discourages repeats).
        eos_token_id:      Override the tokenizer EOS id.
        early_stop:        If True, stop at EOS. If False, generate exactly
                           max_new_tokens regardless of EOS.
        stopping_criteria: Optional custom StoppingCriteriaList.

    Returns:
        Full token sequence (prompt + generated), shape (batch, seq_len + new).
    """
    if expert_model.device != amateur_model.device:
        raise ValueError(
            f"expert_model is on {expert_model.device} but "
            f"amateur_model is on {amateur_model.device}. "
            "Move both to the same device before calling contrastive_decoding."
        )

    batch_size = input_ids.shape[0]
    device = input_ids.device

    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    eos_tensor = (
        torch.tensor([eos_token_id], device=device) if eos_token_id is not None else None
    )
    unfinished = input_ids.new_ones(batch_size)  # 1 = still generating

    processors = LogitsProcessorList()
    if repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    stopping_criteria = stopping_criteria or StoppingCriteriaList()

    for step in range(max_new_tokens):
        # ── Step 1: Expert log-probs ──────────────────────────────────────────
        # Shape: (batch, vocab)
        expert_logprobs = _forward_logprobs(expert_model, input_ids, attention_mask)

        # ── Step 2: Adaptive plausibility constraint V_head ───────────────────
        # Keep only tokens x where p_EXP(x) >= alpha * max_w p_EXP(w)
        # Equivalently in log-space: log p_EXP(x) >= log(alpha) + max log p_EXP(w)
        #
        # Tokens outside V_head are set to -inf so they can never be selected.
        expert_max_logprob = expert_logprobs.max(dim=-1, keepdim=True).values
        plausibility_threshold = expert_max_logprob + torch.log(
            torch.tensor(alpha, device=device)
        )
        plausibility_mask = expert_logprobs < plausibility_threshold  # True → blocked

        # ── Step 3: Amateur log-probs ─────────────────────────────────────────
        amateur_logprobs = _forward_logprobs(amateur_model, input_ids, attention_mask)

        # ── Step 4: CD score ──────────────────────────────────────────────────
        # Apply temperature to raw log-probs before differencing.
        # (At temperature=1.0 this is a no-op and matches the paper exactly.)
        cd_scores = (expert_logprobs / temperature) - (amateur_logprobs / temperature)

        # Mask out implausible tokens
        cd_scores = cd_scores.masked_fill(plausibility_mask, -float("inf"))

        # ── Step 5: Optional repetition penalty on CD scores ─────────────────
        cd_scores = processors(input_ids, cd_scores)

        # ── Step 6: Suppress EOS if early_stop is off ────────────────────────
        if not early_stop and eos_tensor is not None:
            cd_scores[:, eos_token_id] = -float("inf")

        # ── Step 7: Greedy selection ──────────────────────────────────────────
        next_tokens = torch.argmax(cd_scores, dim=-1)  # (batch,)

        # Replace finished-sequence tokens with pad
        next_tokens = (
            next_tokens * unfinished
            + tokenizer.pad_token_id * (1 - unfinished)
        )

        # ── Step 8: Append token + extend mask (keep them in sync) ───────────
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1
        )

        # ── Step 9: Stopping conditions ───────────────────────────────────────
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


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Original paper uses OPT-13B (expert) vs OPT-125M (amateur).
    # Here we use small publicly available models for a quick smoke test.
    # For best results use models from the same family with a large size gap.
    EXPERT_ID  = "EleutherAI/pythia-410m-deduped"
    AMATEUR_ID = "EleutherAI/pythia-70m-deduped"

    REVISION = "step3000"
    DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(EXPERT_ID, revision=REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading expert model...")
    expert_model = AutoModelForCausalLM.from_pretrained(
        EXPERT_ID, revision=REVISION, device_map=DEVICE
    ).eval()

    print("Loading amateur model...")
    amateur_model = AutoModelForCausalLM.from_pretrained(
        AMATEUR_ID, revision=REVISION, device_map=DEVICE
    ).eval()

    prompt = "Barack Obama was born in"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    print(f"\nPrompt: {prompt!r}")
    print("-" * 60)

    # ── Contrastive Decoding ──────────────────────────────────────────────────
    cd_tokens = contrastive_decoding(
        expert_model=expert_model,
        amateur_model=amateur_model,
        tokenizer=tokenizer,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=64,
        alpha=0.1,           # plausibility threshold (paper default)
        repetition_penalty=1.2,
        early_stop=True,
    )
    cd_text = tokenizer.batch_decode(
        cd_tokens[:, inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    print(f"[CD]     {cd_text[0]}")

    # ── Greedy baseline (expert only) ─────────────────────────────────────────
    greedy_tokens = expert_model.generate(
        **inputs, do_sample=False, max_new_tokens=64
    )
    greedy_text = tokenizer.batch_decode(
        greedy_tokens[:, inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    print(f"[Greedy] {greedy_text[0]}")