from dataclasses import dataclass
from typing import Any, Dict, List
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .interface import Generator, ChatMessage


@dataclass
class GreedyGenerator(Generator):
    """
    Generator that uses greedy decoding to generate completions. Greedy decoding is a decoding strategy that selects the token with the highest probability at each step.
    It is mostly get at the top-1 token at each step without any sampling and without any temperature.
    It is also known as nucleus sampling with a probability of 1.
    It is a simple and fast decoding strategy that is often used as a baseline for other decoding strategies.
    Args:
        model: The model to use for the generator.
        tokenizer: The tokenizer to use for the generator.
        config: The configuration to use for the generator.
    Returns:
        GreedyGenerator: The greedy generator instance.
    """
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    config: Dict[str, Any]
    _is_deepseek: bool

    def __init__(self, model_name: str, tokenizer_name: str | None = None, config: Dict[str, Any] | None = None):
        """
        Greedy HuggingFace generator (argmax decoding, no sampling).
        """
        self.config = dict(config or {})
        self._is_deepseek = "deepseek" in model_name.lower()
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

        # Optional bitsandbytes quantization
        quant_cfg = self.config.get("quantization") or {}
        use_quant = bool(quant_cfg.get("load_in_4bit") or quant_cfg.get("load_in_8bit"))
        if use_quant:
            bnb_config = BitsAndBytesConfig(**quant_cfg)
            model_kwargs["quantization_config"] = bnb_config

        device_map = self.config.get("device_map")
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if device_map is None:
            self.model.to(self.device)
        self.model.eval()

    def _postprocess_text(self, text: str) -> str:
        """
        Normalise common byte-level BPE markers that some tokenisers leave in
        the decoded text (e.g. Ġ for space, Ċ for newline).
        """
        cleaned = (
            text.replace("Ċ", "\n")
            .replace("ģ", "\n")
            .replace("Ġ", " ")
        )
        if self._is_deepseek:
            # Some DeepSeek generations contain formatting/control artifacts.
            cleaned = cleaned.replace("```block```", "```")
            cleaned = re.sub(r"(?m)^##\s*\[\d{2}:\d{2}:\d{2}\]\s*$", "", cleaned)
            cleaned = re.sub(r"```([A-Za-z0-9_+-]+)\s*```", r"```\1\n", cleaned)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _build_generation_kwargs(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": int(self.config.get("max_new_tokens", 128)),
            "temperature": 0.0,
            "top_k": 50,
            "top_p": 0.95,
            "num_return_sequences": 1,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
            **self.config.get("generation_kwargs", {}),
        }

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)
        gen_kwargs = self._build_generation_kwargs()
        outputs = self.model.generate(**inputs, **gen_kwargs)
        raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._postprocess_text(raw)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        gen_kwargs = self._build_generation_kwargs()
        outputs = self.model.generate(**inputs, **gen_kwargs)
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [self._postprocess_text(t) for t in decoded]

    def chat_generation(self, messages: List[ChatMessage]) -> str:
        if not messages:
            return ""
        lines = [f"{m.role.value}: {m.content}" for m in messages if m.content]
        prompt = "\n".join(lines) + "\nassistant: "
        return self.generate(prompt)

    def batch_chat_generation(self, messages: List[List[ChatMessage]]) -> List[str]:
        if not messages:
            return []
        prompts: List[str] = []
        for conv in messages:
            if not conv:
                prompts.append("")
                continue
            lines = [f"{m.role.value}: {m.content}" for m in conv if m.content]
            prompts.append("\n".join(lines) + "\nassistant: ")
        return self.batch_generate(prompts)