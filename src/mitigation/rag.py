"""
RAG-based mitigation for package hallucination.

This module adapts the RAG setup from
`ref_works/package-hallucination-main/Mitigation/RAG_setup.py` and
`webui_api_package_query_RAG_Vector.py` to the `Compare` project.

It provides:
- `RagRetriever`: builds/loads a Chroma vector store over `Compare/data/rag/RAG_data.jsonl`
- `RagGenerator`: wraps an existing `Generator` and augments user queries with
  retrieved context before calling `inner.chat_generation`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from langchain_chroma import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

from .interface import ChatMessage, ChatRole, Generator


class RagRetriever:
    """
    Lightweight wrapper around a Chroma vector store built from RAG_data.jsonl.
    """

    def __init__(
        self,
        data_path: Optional[Path] = None,
        persist_dir: Optional[Path] = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        if data_path is None:
            data_path = base_dir / "data" / "rag" / "RAG_data.jsonl"
        if persist_dir is None:
            persist_dir = base_dir / "data" / "rag" / "chroma_db"

        self.data_path = Path(data_path)
        self.persist_dir = Path(persist_dir)
        self.embedding_function = SentenceTransformerEmbeddings(model_name=model_name)

        self._db = self._load_or_build_db()

    def _load_or_build_db(self) -> Chroma:
        """
        Load an existing Chroma DB if present; otherwise build it from JSONL.
        """
        if self.persist_dir.exists() and any(self.persist_dir.iterdir()):
            return Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embedding_function,
            )

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_json(self.data_path, lines=True)
        # Mirror original: ensure a 'descriptions' column
        if 0 in df.columns:
            df = df.rename(columns={0: "descriptions"})
        if "descriptions" not in df.columns:
            raise ValueError(
                f"'descriptions' column not found in {self.data_path}. "
                "RAG_data.jsonl must have descriptions in column 0 or named 'descriptions'."
            )
        df = df.drop_duplicates()
        loader = DataFrameLoader(df, page_content_column="descriptions")
        docs = loader.load()
        return Chroma.from_documents(
            docs,
            self.embedding_function,
            persist_directory=str(self.persist_dir),
        )

    def retrieve(self, query: str, k: int = 5) -> str:
        """
        Return a single concatenated string of the top-k retrieved descriptions.
        """
        if not query.strip():
            return ""
        results = self._db.similarity_search_with_score(query, k)
        return " ".join(doc.page_content for doc, _ in results)


def _last_user_content(messages: List[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == ChatRole.USER and msg.content:
            return msg.content
    return ""


@dataclass
class RagGenerator(Generator):
    """
    RAG-enhanced generator that wraps an underlying `Generator`.

    For a given query, it:
      1. Retrieves relevant statements from the RAG DB.
      2. Prepends a system message and a user prompt that includes both
         the original question and the retrieved statements.
      3. Delegates to `inner.chat_generation`.
    """

    inner: Generator
    language: str
    retriever: RagRetriever
    config: Dict[str, Any]

    def __init__(
        self,
        inner: Generator,
        language: str = "Python",
        retriever: Optional[RagRetriever] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.inner = inner
        self.language = language
        self.retriever = retriever or RagRetriever()
        self.config = dict(config or {})

        # For compatibility with the abstract `Generator` attributes
        self.model = getattr(inner, "model", None)
        self.tokenizer = getattr(inner, "tokenizer", None)

    # ------------------------------------------------------------------ #
    # Prompt builders
    # ------------------------------------------------------------------ #

    def _system_prompt(self) -> str:
        """
        System prompt, closely aligned with the original RAG webui script.
        """
        return (
            "You are a coding assistant that recommends {language} packages to help "
            "answer questions. Use the provided statements to help form your response, "
            "but do not limit your response to those statements. Respond with only a "
            "list of {language} packages, separated by commas and no additional text "
            "or formatting. Your response must begin with the name of a {language} "
            "package."
        ).format(language=self.language)

    def _build_messages(self, question: str) -> List[ChatMessage]:
        k = int(self.config.get("k", 5))
        retrieved = self.retriever.retrieve(question, k=k)
        if retrieved:
            user_content = (
                f"What {self.language} packages would be useful in solving the "
                f"following coding question: {question.strip()}\n\n"
                f"Here are some statements that may help answer the question:\n{retrieved}"
            )
        else:
            user_content = (
                f"What {self.language} packages would be useful in solving the "
                f"following coding question: {question.strip()}"
            )

        return [
            ChatMessage(role=ChatRole.SYSTEM, content=self._system_prompt()),
            ChatMessage(role=ChatRole.USER, content=user_content),
        ]

    # ------------------------------------------------------------------ #
    # Generator interface
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str) -> str:
        messages = self._build_messages(prompt)
        return self.inner.chat_generation(messages)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        return [self.generate(p) for p in prompts]

    def chat_generation(self, messages: List[ChatMessage]) -> str:
        question = _last_user_content(messages)
        if not question:
            return ""
        rag_messages = self._build_messages(question)
        return self.inner.chat_generation(rag_messages)

    def batch_chat_generation(self, messages: List[List[ChatMessage]]) -> List[str]:
        return [self.chat_generation(conv) for conv in messages]
