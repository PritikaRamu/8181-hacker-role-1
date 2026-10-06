"""
OpenAI-compatible API generator.

Implements the Generator interface by calling any OpenAI-compatible REST
endpoint (vLLM, Ollama, LM Studio, OpenAI, …).  Use as the ``inner``
generator for SelfRefineGenerator, RagGenerator, or any other wrapper that
needs chat_generation() without loading a model in-process.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests

from .interface import ChatMessage, ChatRole, Generator


class APIGenerator(Generator):
    """
    Generator backed by an OpenAI-compatible /v1/chat/completions endpoint.

    Args:
        base_url:        API root, e.g. ``http://localhost:8000``.
        inference_model: Model id sent in the payload.
        api_key:         Bearer token (empty string for local servers).
        max_new_tokens:  Default ``max_tokens`` for each request.
        temperature:     Sampling temperature (0 = greedy).
        top_p:           Nucleus sampling threshold.
        max_workers:     Parallel requests for batch_generate / batch_chat_generation.
        timeout:         Per-request timeout in seconds.
        retry:           Number of retries on transient errors.
    """

    model = None       # no in-process model
    tokenizer = None

    def __init__(
        self,
        base_url: str,
        inference_model: str,
        api_key: str = "",
        max_new_tokens: int = 128,
        temperature: float = 0.01,
        top_p: float = 0.9,
        max_workers: int = 4,
        timeout: int = 120,
        retry: int = 2,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.base_url        = base_url.rstrip("/")
        self.inference_model = inference_model
        self.max_new_tokens  = max_new_tokens
        self.temperature     = temperature
        self.top_p           = top_p
        self.max_workers     = max_workers
        self.timeout         = timeout
        self.retry           = retry
        self.config          = dict(config or {})

        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        self._url = f"{self.base_url}/v1/chat/completions"

    # ------------------------------------------------------------------ #
    # Internal request helper
    # ------------------------------------------------------------------ #

    def _call(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model":       self.inference_model,
            "messages":    messages,
            "max_tokens":  self.max_new_tokens,
            "temperature": self.temperature,
            "top_p":       self.top_p,
        }
        last_exc: Exception | None = None
        for attempt in range(self.retry + 1):
            try:
                r = requests.post(
                    self._url,
                    headers=self._headers,
                    json=payload,
                    timeout=self.timeout,
                    verify=False,
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                last_exc = exc
                if attempt < self.retry:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"API call failed after {self.retry + 1} attempts: {last_exc}")

    # ------------------------------------------------------------------ #
    # Generator interface
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str) -> str:
        return self._call([{"role": "user", "content": prompt}])

    def chat_generation(self, messages: List[ChatMessage]) -> str:
        return self._call([{"role": m.role.value, "content": m.content} for m in messages])

    def batch_generate(self, prompts: List[str]) -> List[str]:
        results = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(prompts))) as ex:
            futures = {ex.submit(self.generate, p): i for i, p in enumerate(prompts)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        return results

    def batch_chat_generation(self, messages: List[List[ChatMessage]]) -> List[str]:
        results = [""] * len(messages)
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(messages))) as ex:
            futures = {ex.submit(self.chat_generation, msgs): i for i, msgs in enumerate(messages)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        return results
