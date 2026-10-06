"""
Contrastive Decoding (CD) generator wrapper.

This module exposes a `ContrastiveDecodingGenerator` that loads an expert and
an amateur HuggingFace causal LM and runs the contrastive decoding loop from
`Compare.mitigation.decoding.contrastive_decoding`.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .interface import ChatMessage, ChatRole, Generator
from .decoding.contrastive_decoding import contrastive_decoding


def _messages_to_prompt(messages: List[ChatMessage]) -> str:
    if not messages:
        return ""
    lines = [f"{m.role.value}: {m.content}" for m in messages if m.content]
    return "\n".join(lines) + "\nassistant: "


@dataclass
class ContrastiveDecodingGenerator(Generator):
    """
    Generator that uses Contrastive Decoding (expert vs amateur model).
    """

    expert_model: AutoModelForCausalLM
    amateur_model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    config: Dict[str, Any]

    def __init__(
        self,
        expert_model_name: str,
        amateur_model_name: str,
        tokenizer_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            expert_model_name: HuggingFace id/path for the large "expert" model.
            amateur_model_name: HuggingFace id/path for the small "amateur" model.
            tokenizer_name: Optional separate tokenizer id; defaults to expert id.
            config: Optional dict with keys:
                - max_new_tokens: int (default 128)
                - alpha: float (default 0.1)
                - temperature: float (default 1.0)
                - repetition_penalty: float (default 1.0)
                - early_stop: bool (default True)
                - torch_dtype: "float16" | "bfloat16"
                - device_map: str or dict
        """
        self.config = dict(config or {})
        tok_id = tokenizer_name or expert_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)
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

        self.expert_model = AutoModelForCausalLM.from_pretrained(
            expert_model_name,
            **model_kwargs,
        )
        self.amateur_model = AutoModelForCausalLM.from_pretrained(
            amateur_model_name,
            **model_kwargs,
        )
        if device_map is None:
            self.expert_model.to(self.device)
            self.amateur_model.to(self.device)
        self.expert_model.eval()
        self.amateur_model.eval()

        # For base `Generator` compatibility
        self.model = self.expert_model

    def _postprocess_text(self, text: str) -> str:
        """
        Normalise common byte-level BPE markers (Ġ = space, Ċ/ģ = newline)
        that may appear in decoded outputs for some tokenisers/models.
        """
        return (
            text.replace("Ċ", "\n")
            .replace("ģ", "\n")
            .replace("Ġ", " ")
            .strip()
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _run_cd(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        # Some tokenizers expose an extremely large model_max_length which can
        # overflow internal truncation logic. Cap to a reasonable window.
        max_len_raw = getattr(self.tokenizer, "model_max_length", 4096) or 4096
        max_len = min(int(max_len_raw), 4096)
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        input_ids = inputs["input_ids"].to(self.expert_model.device)
        attention_mask = inputs["attention_mask"].to(self.expert_model.device)

        max_new_tokens = int(self.config.get("max_new_tokens", 128))
        alpha = float(self.config.get("alpha", 0.1))
        temperature = float(self.config.get("temperature", 1.0))
        repetition_penalty = float(self.config.get("repetition_penalty", 1.0))
        early_stop = bool(self.config.get("early_stop", True))

        cd_tokens = contrastive_decoding(
            expert_model=self.expert_model,
            amateur_model=self.amateur_model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            alpha=alpha,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            early_stop=early_stop,
        )

        prompt_lens = attention_mask.sum(dim=1).tolist()
        decoded: List[str] = []
        for i in range(cd_tokens.size(0)):
            start = prompt_lens[i]
            seq = cd_tokens[i, start:]
            text = self.tokenizer.decode(
                seq,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            decoded.append(self._postprocess_text(text))
        return decoded

    # ------------------------------------------------------------------ #
    # Generator interface
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str) -> str:
        res = self._run_cd([prompt])
        return res[0] if res else ""

    def batch_generate(self, prompts: List[str]) -> List[str]:
        return self._run_cd(prompts)

    def chat_generation(self, messages: List[ChatMessage]) -> str:
        prompt = _messages_to_prompt(messages)
        return self.generate(prompt)

    def batch_chat_generation(self, messages: List[List[ChatMessage]]) -> List[str]:
        return [self.chat_generation(conv) for conv in messages]

