"""
Nudging: Inference-time guided decoding for package hallucination mitigation.

Adapted from Package_guided_generation/nudging_utils.py. Uses two HuggingFace models:
- Base model: main generator
- Nudging model: guides when base model's top-1 token probability < threshold

At each generation step, if the base model is uncertain (low confidence), the nudging
model's suggestion is used instead. This reduces hallucinations by deferring to a
model trained to avoid inventing package names.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .interface import Generator, ChatMessage, ChatRole


def _messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Convert chat messages to a plain-text dialogue prompt (assistant continuation)."""
    if not messages:
        return ""
    lines = [f"{m.role.value}: {m.content}" for m in messages if m.content]
    return "\n".join(lines) + "\nassistant: "


@dataclass
class NudgingGenerator(Generator):
    """
    HuggingFace-based nudging generator. Uses base model + nudging model with
    token-level guided decoding: when base's top-1 prob < threshold, use nudging token.
    """

    base_model: AutoModelForCausalLM
    nudging_model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    config: Dict[str, Any]

    def __init__(
        self,
        base_model_name: str,
        nudging_model_name: str,
        tokenizer_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            base_model_name: HuggingFace model path for the base (main) generator.
            nudging_model_name: HuggingFace model path for the nudging (expert) model.
            tokenizer_name: HuggingFace tokenizer path. Defaults to base_model_name.
            config: Optional dict with keys:
                - top_prob_thres: float (default 0.4) – when base prob < this, use nudging.
                - max_new_tokens: int (default 256)
                - temperature: float (default 0.0) for base
                - nudging_temperature: float (default 0.0) for nudging
                - torch_dtype: "float16" | "bfloat16"
                - device_map: str (e.g. "auto")
        """
        self.config = dict(config or {})
        tokenizer_id = tokenizer_name or base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        elif self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
        if self.tokenizer.eos_token_id is None and self.tokenizer.pad_token_id is not None:
            self.tokenizer.eos_token_id = self.tokenizer.pad_token_id

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        torch_dtype = self.config.get("torch_dtype")
        if torch_dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16
        elif torch_dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16

        # Optional bitsandbytes quantization
        quant_cfg = self.config.get("quantization") or {}
        use_quant = bool(quant_cfg.get("load_in_4bit") or quant_cfg.get("load_in_8bit"))
        if use_quant:
            bnb_config = BitsAndBytesConfig(**quant_cfg)
            model_kwargs["quantization_config"] = bnb_config

        device_map = self.config.get("device_map")
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
        self.nudging_model = AutoModelForCausalLM.from_pretrained(nudging_model_name, **model_kwargs)

        if device_map is None:
            self.base_model.to(self.device)
            self.nudging_model.to(self.device)

        self.base_model.eval()
        self.nudging_model.eval()

        # Compatibility with Generator interface
        self.model = self.base_model

    def _get_next_token_logits(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute logits for the next token (last position)."""
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        return outputs.logits[:, -1, :]  # (batch, vocab)

    def _generate_with_nudging(self, prompt: str) -> str:
        """
        Token-by-token generation with nudging. When base's top-1 prob < threshold,
        use nudging model's top token.
        """
        top_prob_thres = float(self.config.get("top_prob_thres", 0.4))
        max_new_tokens = int(self.config.get("max_new_tokens", 256))
        temperature = float(self.config.get("temperature", 0.0))
        nudging_temperature = float(self.config.get("nudging_temperature", 0.0))

        # Some tokenizers expose an extremely large model_max_length, which can
        # overflow internal truncation logic. Cap to a reasonable window.
        max_len_raw = getattr(self.tokenizer, "model_max_length", 4096) or 4096
        max_len = min(int(max_len_raw), 4096)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        input_ids = inputs["input_ids"].to(self.base_model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.base_model.device)

        generated_ids: List[int] = []
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = self.tokenizer.pad_token_id

        for _ in range(max_new_tokens):
            # Base model forward
            logits_base = self._get_next_token_logits(
                self.base_model, input_ids, attention_mask
            )

            probs_base = F.softmax(logits_base, dim=-1)
            top_prob = probs_base.max().item()
            top_token_base = logits_base.argmax(dim=-1).item()

            if top_prob >= top_prob_thres:
                # Base is confident – use base token
                next_token = top_token_base
            else:
                # Base uncertain – use nudging model's suggestion
                logits_nudging = self._get_next_token_logits(
                    self.nudging_model, input_ids, attention_mask
                )
                if nudging_temperature > 0:
                    logits_nudging = logits_nudging / nudging_temperature
                    probs_nudging = F.softmax(logits_nudging, dim=-1)
                    next_token = torch.multinomial(probs_nudging, num_samples=1).item()
                else:
                    next_token = logits_nudging.argmax(dim=-1).item()

            if next_token == eos_id:
                break

            generated_ids.append(next_token)
            next_tensor = torch.tensor(
                [[next_token]],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, next_tensor], dim=1)
            if attention_mask is not None:
                attn_ones = torch.ones(
                    (1, 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, attn_ones], dim=1)

        if not generated_ids:
            return ""

        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

    def generate(self, prompt: str) -> str:
        return self._generate_with_nudging(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """Nudging is inherently sequential per sample; no parallel batching."""
        return [self.generate(p) for p in prompts]

    def chat_generation(self, messages: List[ChatMessage]) -> str:
        if not messages:
            return ""
        prompt = _messages_to_prompt(messages)
        return self.generate(prompt)

    def batch_chat_generation(self, messages: List[List[ChatMessage]]) -> List[str]:
        return [self.chat_generation(conv) for conv in messages]
