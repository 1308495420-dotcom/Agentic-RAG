"""Agent Toolkit — expose HybridSearch + AnswerGenerator + Verifier as tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from src.core.types import RetrievalResult


@dataclass
class SearchResult:
    """The result of one agent tool call."""

    query: str
    chunks: List[RetrievalResult] = field(default_factory=list)
    answer: str = ""
    verification_rate: float = 1.0

    @property
    def top_score(self) -> float:
        if self.chunks:
            return getattr(self.chunks[0], "score", 0.0)
        return 0.0


class AgentToolkit:
    """Wraps existing RAG components as Agent-callable tools.

    Tools: search, generate, verify — all stateless, called by RagAgent.
    """

    def __init__(
        self,
        hybrid_search: Any,
        answer_generator: Any,
        citation_verifier: Any = None,
    ) -> None:
        self._search = hybrid_search
        self._gen = answer_generator
        self._verify = citation_verifier

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Tool: search the knowledge base."""
        results = self._search.search(query=query, top_k=top_k)
        return results if isinstance(results, list) else results.results

    def generate(self, query: str, chunks: List[RetrievalResult]) -> str:
        """Tool: generate an answer from chunks."""
        ans = self._gen.generate(query=query, chunks=chunks)
        return ans.answer

    def verify(self, answer: str, chunks: List[RetrievalResult]) -> float:
        """Tool: verify claims in the answer against source chunks."""
        if self._verify is None:
            return 1.0
        report = self._verify.verify(answer=answer, chunks=chunks)
        return report.verification_rate

    def run_retrieval_pass(
        self,
        query: str,
        top_k: int = 5,
        confidence_threshold: float = 0.015,
    ) -> SearchResult:
        """Execute one full retrieval pass: search → generate → verify.

        Returns a SearchResult the Agent can inspect to decide whether
        to run additional passes.
        """
        chunks = self.search(query, top_k=top_k)
        answer = self.generate(query, chunks) if chunks else ""
        vr = self.verify(answer, chunks) if answer else 0.0

        return SearchResult(query=query, chunks=chunks, answer=answer, verification_rate=vr)
