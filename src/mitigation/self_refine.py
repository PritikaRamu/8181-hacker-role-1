from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import re
import time
import jinja2

from .interface import Generator, ChatMessage, ChatRole


def _extract_bracket_list(text: str) -> List[str]:
    """
    Extract a comma-separated list from the first [...] block in `text`.
    Falls back to splitting the whole string if no brackets are found.
    """
    match = re.search(r"\[([^\[\]]+)\]", text)
    raw = match.group(1) if match else text
    return [p.strip() for p in raw.split(",") if p.strip()]


def _extract_validities(text: str) -> List[str]:
    """
    Extract Yes/No labels from a validation response.
    Normalises to literal 'Yes' or 'No'.
    """
    items = _extract_bracket_list(text)
    return ["Yes" if "yes" in v.lower() else "No" for v in items]


def _extract_code_block(text: str, language: str) -> str:
    """Extract the content of the first fenced code block for the given language."""
    marker = f"```{language.lower()}"
    if marker in text:
        try:
            return text.split(marker, 1)[1].split("```", 1)[0]
        except Exception:
            pass
    return text


def _extract_packages_from_code(code_text: str, language: str) -> List[str]:
    """
    Extract imported package names from generated code using language-specific
    import patterns. Returns a deduplicated list in order of appearance.
    """
    code = _extract_code_block(code_text, language)
    if not code.strip():
        return []

    if language == "Python":
        imports = re.findall(r"^\s*import\s+([A-Za-z0-9_\.]+)", code, flags=re.MULTILINE)
        froms = re.findall(r"^\s*from\s+([A-Za-z0-9_\.]+)\s+import", code, flags=re.MULTILINE)
        pkgs = [x.split(".")[0] for x in imports + froms]
    elif language == "JavaScript":
        es_imports = re.findall(r"from\s+['\"]([^'\"]+)['\"]", code)
        req_imports = re.findall(r"require\(\s*['\"]([^'\"]+)['\"]\s*\)", code)
        pkgs = [x.split("/")[0] for x in es_imports + req_imports]
    elif language == "Rust":
        externs = re.findall(r"extern\s+crate\s+([A-Za-z0-9_]+)", code)
        uses = re.findall(r"^\s*use\s+([A-Za-z0-9_]+)", code, flags=re.MULTILINE)
        pkgs = externs + uses
    elif language == "Ruby":
        reqs = re.findall(r"require\s+['\"]([^'\"]+)['\"]", code)
        pkgs = [x.split("/")[0] for x in reqs]
    else:
        pkgs = []

    seen: set = set()
    uniq: List[str] = []
    for p in pkgs:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


@dataclass
class SelfRefineGenerator(Generator):
    """
    Wrapper that runs an underlying chat-capable `Generator` with an
    iterative self-refinement loop.

    Two tasks are supported (controlled by the `task` parameter):
      "packages"  Validate a package-list response and remove hallucinated entries.
      "code"      Validate packages imported in generated code and ask for a rewrite
                  that avoids any hallucinated imports.

    The wrapped generator is expected to implement `chat_generation`.
    This class only orchestrates prompting and refinement; it does not
    perform decoding itself.
    """

    inner: Generator
    language: str
    task: str
    config: Dict[str, Any]

    def __init__(self, inner: Generator, language: str,
                 task: str = "packages", config: Dict[str, Any] | None = None):
        self.inner = inner
        self.language = language
        self.task = task
        self.config = dict(config or {})
        self.model = getattr(inner, "model", None)
        self.tokenizer = getattr(inner, "tokenizer", None)

    # ------------------------------------------------------------------ #
    # Template rendering helpers
    # ------------------------------------------------------------------ #

    def _render_template(self, template_name: str, **kwargs: Any) -> str:
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        text = (prompts_dir / template_name).read_text(encoding="utf-8")
        try:
            env = jinja2.Environment(autoescape=False)
            return env.from_string(text).render(**kwargs)
        except Exception:
            return text.format(**kwargs)

    # ------------------------------------------------------------------ #
    # Prompt builders
    # ------------------------------------------------------------------ #

    def _system_prompt_generation(self) -> str:
        if self.task == "code":
            return self._render_template(
                "system_prompt_code_generation.jinja",
                language=self.language,
            )
        return self._render_template(
            "system_prompt_package_generation.jinja",
            language=self.language,
        )

    def _build_generation_messages(self, instruction: str) -> List[ChatMessage]:
        return [
            ChatMessage(role=ChatRole.SYSTEM, content=self._system_prompt_generation()),
            ChatMessage(role=ChatRole.USER, content=instruction),
        ]

    def _build_validation_messages(self, package_text: str) -> List[ChatMessage]:
        system = self._render_template("system_prompt_package_validation.jinja")
        user = self._render_template(
            "user_prompt_package_validation.jinja",
            language=self.language,
            package_text=package_text,
        )
        return [
            ChatMessage(role=ChatRole.SYSTEM, content=system),
            ChatMessage(role=ChatRole.USER, content=user),
        ]

    # ------------------------------------------------------------------ #
    # Core self-refinement loop — packages task
    # ------------------------------------------------------------------ #

    def _self_refine_packages(self, instruction: str) -> Tuple[str, float, int]:
        max_rounds: int = int(self.config.get("max_rounds", 3))
        t0 = time.perf_counter()

        messages = self._build_generation_messages(instruction)
        package_text = self.inner.chat_generation(messages)

        rounds_used = 0
        for _ in range(max_rounds):
            packages_list = _extract_bracket_list(package_text)
            if not packages_list:
                break

            val_messages = self._build_validation_messages(package_text)
            validation_text = self.inner.chat_generation(val_messages)
            validities = _extract_validities(validation_text)

            invalid = [pkg for pkg, v in zip(packages_list, validities) if v == "No"]
            if not invalid:
                break

            rounds_used += 1
            messages.append(ChatMessage(role=ChatRole.ASSISTANT, content=package_text))
            feedback = (
                f"The following packages do not exist or are not valid {self.language} packages "
                f"and must not appear in your answer: {invalid}. "
                f"Please provide a corrected list."
            )
            messages.append(ChatMessage(role=ChatRole.USER, content=feedback))
            package_text = self.inner.chat_generation(messages)

        return package_text, time.perf_counter() - t0, rounds_used

    # ------------------------------------------------------------------ #
    # Core self-refinement loop — code task
    # ------------------------------------------------------------------ #

    def _self_refine_code(self, instruction: str) -> Tuple[str, float, int]:
        max_rounds: int = int(self.config.get("max_rounds", 3))
        t0 = time.perf_counter()

        messages = self._build_generation_messages(instruction)
        code_text = self.inner.chat_generation(messages)

        rounds_used = 0
        for _ in range(max_rounds):
            imported_pkgs = _extract_packages_from_code(code_text, self.language)
            if not imported_pkgs:
                break

            pkg_list_str = "[" + ", ".join(imported_pkgs) + "]"
            val_messages = self._build_validation_messages(pkg_list_str)
            validation_text = self.inner.chat_generation(val_messages)
            validities = _extract_validities(validation_text)

            invalid = [pkg for pkg, v in zip(imported_pkgs, validities) if v == "No"]
            if not invalid:
                break

            rounds_used += 1
            messages.append(ChatMessage(role=ChatRole.ASSISTANT, content=code_text))
            feedback = (
                f"The following {self.language} packages imported in your code do not exist "
                f"and must not be used: {invalid}. "
                f"Please rewrite the code without importing those packages."
            )
            messages.append(ChatMessage(role=ChatRole.USER, content=feedback))
            code_text = self.inner.chat_generation(messages)

        return code_text, time.perf_counter() - t0, rounds_used

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def _self_refine(self, instruction: str) -> Tuple[str, float, int]:
        if self.task == "code":
            return self._self_refine_code(instruction)
        return self._self_refine_packages(instruction)

    # ------------------------------------------------------------------ #
    # Generator interface
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str) -> str:
        text, _, _ = self._self_refine(prompt)
        return text

    def batch_generate(self, prompts: List[str]) -> List[str]:
        return [self.generate(p) for p in prompts]

    def chat_generation(self, messages: List[ChatMessage]) -> str:
        if not messages:
            return ""
        instruction = ""
        for msg in reversed(messages):
            if msg.role == ChatRole.USER and msg.content:
                instruction = msg.content
                break
        if not instruction:
            return ""
        text, _, _ = self._self_refine(instruction)
        return text

    def batch_chat_generation(self, messages: List[List[ChatMessage]]) -> List[str]:
        return [self.chat_generation(conv) for conv in messages]
