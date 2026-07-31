"""Answer Generator — assemble context and generate cited answers via LLM.

Takes retrieved chunks + user query → formats a context-aware prompt →
calls the configured LLM → returns a structured answer with inline
source citations like [1], [2].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    """A generated answer with inline citations and source mapping.

    Attributes:
        answer: The full answer text with [N] citation markers.
        sources: List of source dicts keyed by citation number.
    """

    answer: str
    sources: List[Dict[str, str]] = field(default_factory=list)

    def to_text(self) -> str:
        """Return answer with per-source attribution appended."""
        if not self.sources:
            return self.answer
        lines = [self.answer, "", "---", "**参考来源:**"]
        for s in self.sources:
            num = s.get("num", "?")
            title = s.get("title", "未知")
            chunk_id = s.get("chunk_id", "")
            lines.append(f"  [{num}] {title}  (`{chunk_id[:24]}...`)")
        return "\n".join(lines)


class AnswerGenerator:
    """Generate cited answers from retrieved chunks using LLM.

    Example::

        gen = AnswerGenerator(settings)
        answer = gen.generate(
            query="什么是FinFET",
            chunks=[chunk1, chunk2, ...],
        )
        print(answer.to_text())
    """

    _SYSTEM_PROMPT = (
        "你是一个精确的知识助手。请仅根据下面提供的上下文资料回答用户的问题。\n"
        "\n"
        "规则：\n"
        "1. 每条关键信息后面标注来源编号，如：[1]、[2]\n"
        "2. 如果上下文资料不足以回答，请明确说\"根据现有资料无法确定\"\n"
        "3. 不要编造上下文资料中没有的信息\n"
        "4. 回答简洁直接，不添加无关背景"
    )

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._model = settings.llm.model
        self._client: Any = None

    def generate(
        self,
        query: str,
        chunks: List[Any],
        max_sources: int = 5,
        max_tokens: int = 1024,
    ) -> GeneratedAnswer:
        """Generate answer from retrieved chunks.

        Args:
            query: User question.
            chunks: Retrieved RetrievalResult objects (at least 1).
            max_sources: Max number of chunks to include as context.
            max_tokens: Max tokens for generated answer.

        Returns:
            GeneratedAnswer with citations.
        """
        if not chunks:
            return GeneratedAnswer(answer="未检索到相关内容，无法回答。")

        # Build context with numbered sources
        sources: List[Dict[str, str]] = []
        context_parts: List[str] = []
        for i, chunk in enumerate(chunks[:max_sources], 1):
            text = getattr(chunk, "text", "") if hasattr(chunk, "text") else str(chunk)
            meta = getattr(chunk, "metadata", {}) if hasattr(chunk, "metadata") else {}
            title = meta.get("title", "") if isinstance(meta, dict) else ""
            chunk_id = getattr(chunk, "chunk_id", "") if hasattr(chunk, "chunk_id") else ""

            context_parts.append(f"[{i}] {title}\n{text[:600]}")
            sources.append({"num": str(i), "title": title or "无标题", "chunk_id": chunk_id or str(i)})

        context = "\n\n".join(context_parts)
        user_prompt = f"## 上下文资料\n\n{context}\n\n## 用户问题\n\n{query}\n\n请根据上下文资料回答（附来源编号）："

        # Call LLM
        try:
            answer_text = self._call_llm(user_prompt, max_tokens)
        except Exception as exc:
            logger.warning("Answer generation failed: %s", exc)
            return GeneratedAnswer(
                answer=f"答案生成失败: {exc}",
                sources=sources,
            )

        return GeneratedAnswer(answer=answer_text, sources=sources)

    def _call_llm(self, prompt: str, max_tokens: int) -> str:
        """Call the configured LLM and return the response text."""
        from openai import OpenAI
        import os

        llm_cfg = self._settings.llm
        provider = llm_cfg.provider.lower()

        if provider == "deepseek":
            client = OpenAI(
                api_key=llm_cfg.api_key or os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
            )
        elif provider == "openai":
            client = OpenAI(api_key=llm_cfg.api_key)
        else:
            raise ValueError(f"AnswerGenerator unsupported provider: {provider}")

        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
