"""
DoLa mitigation strategy — Decoding by Contrasting Layers.
Chuang et al., ICLR 2024.

Uses the HuggingFace community integration so generation is fully batched
through the native model.generate() pipeline rather than a custom Python loop.
https://huggingface.co/transformers-community/dola
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .interface import Generator, ChatMessage


@dataclass
class DoLaGenerator(Generator):
    """
    DoLa decoding via the HuggingFace community custom_generate integration.

    Config keys:
        early_exit_layers  (list | "low" | "high")  Passed as dola_layers.
                           List of ints → exact premature layers to contrast.
                           "high" → upper half of layers (best for short-answer tasks).
                           "low"  → lower half (better for long-form reasoning).
                           Defaults to "high".
        repetition_penalty (float)  Default 1.2.
        max_new_tokens     (int)    Default 128.
    """

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    config: Dict[str, Any]

    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = dict(config or {})

        tokenizer_id = tokenizer_name or model_name
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

        quant_cfg = self.config.get("quantization") or {}
        if quant_cfg.get("load_in_4bit") or quant_cfg.get("load_in_8bit"):
            model_kwargs["quantization_config"] = BitsAndBytesConfig(**quant_cfg)

        device_map = self.config.get("device_map")
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if device_map is None:
            self.model.to(self.device)
        self.model.eval()

    def _dola_layers(self) -> Any:
        layers = self.config.get("early_exit_layers", "high")
        # YAML gives a list of ints; "low"/"high" are string shortcuts accepted as-is.
        return layers

    def _generate(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []

        max_len_raw = getattr(self.tokenizer, "model_max_length", 4096) or 4096
        max_len = min(int(max_len_raw), 4096)
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(self.model.device)

        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

        output_ids = self.model.generate(
            **inputs,
            custom_generate="transformers-community/dola",
            trust_remote_code=True,
            dola_layers=self._dola_layers(),
            repetition_penalty=float(self.config.get("repetition_penalty", 1.2)),
            max_new_tokens=int(self.config.get("max_new_tokens", 128)),
            do_sample=False,
            output_hidden_states=True,
        )

        results: List[str] = []
        for i, seq in enumerate(output_ids):
            gen_ids = seq[int(prompt_lens[i]):]
            text = self.tokenizer.decode(
                gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            results.append(text.strip())
        return results

    def generate(self, prompt: str) -> str:
        return self._generate([prompt])[0]

    def batch_generate(self, prompts: List[str]) -> List[str]:
        return self._generate(prompts)

    def chat_generation(self, messages: List[ChatMessage]) -> str:
        if not messages:
            return ""
        lines = [f"{m.role.value}: {m.content}" for m in messages if m.content]
        return self.generate("\n".join(lines) + "\nassistant: ")

    def batch_chat_generation(self, messages: List[List[ChatMessage]]) -> List[str]:
        prompts = []
        for conv in messages:
            if not conv:
                prompts.append("")
                continue
            lines = [f"{m.role.value}: {m.content}" for m in conv if m.content]
            prompts.append("\n".join(lines) + "\nassistant: ")
        return self.batch_generate(prompts)
