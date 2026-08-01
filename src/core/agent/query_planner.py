"""Query Planner — decompose complex queries into sub-questions via LLM."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, List

logger = logging.getLogger(__name__)


_SEPARATORS = ("；", ";", "\\n", "\n")


class QueryPlanner:
    """Decompose a complex query into multiple simple sub-queries.

    Uses the project LLM to split a multi-faceted question (e.g. comparisons,
    multi-hop) into independent retrieval-friendly sub-questions.

    Example::

        planner = QueryPlanner(settings)
        subs = planner.plan("FinFET 和 GAAFET 在 3nm 下谁更好")
        # ["FinFET 3nm 工艺优势", "GAAFET 3nm 工艺优势"]
    """

    _SYSTEM = (
        "你是查询拆分专家。用户会问一个复杂问题，请把它拆成 2-4 个独立的简单子问题，"
        "每个子问题都可以独立检索和回答。\n"
        "要求：\n"
        "1. 只返回子问题列表，每行一个，不要编号不要多余文字\n"
        "2. 子问题必须简洁、可直接用于检索\n"
        "3. 如果原问题已经很简单，就返回原问题本身（不拆分）\n"
        "4. 每行用中文分号或换行分隔"
    )

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._model = settings.llm.model

    def plan(self, query: str) -> List[str]:
        """Split a complex query into sub-questions.

        Args:
            query: The original user question.

        Returns:
            List of sub-queries (length ≥ 1).  Simple queries are returned
            as-is in a single-element list.
        """
        llm_cfg = self._settings.llm
        provider = llm_cfg.provider.lower()

        from openai import OpenAI

        if provider == "deepseek":
            client = OpenAI(
                api_key=llm_cfg.api_key or os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
            )
        elif provider == "openai":
            client = OpenAI(api_key=llm_cfg.api_key)
        else:
            # Fallback: do not split
            return [query]

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._SYSTEM},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Query planner failed: %s, returning original query", exc)
            return [query]

        # Parse the response: split on separators
        subs: List[str] = []
        for sep in _SEPARATORS:
            if sep in raw:
                subs = [s.strip() for s in raw.replace("\\n", "\n").split(sep) if s.strip()]
                break
        if not subs:
            subs = [raw]

        # Filter out numbering prefixes, empty strings
        cleaned: List[str] = []
        for s in subs:
            s = s.lstrip("0123456789. -)、").strip()
            if s and s not in cleaned:
                cleaned.append(s)

        if not cleaned:
            return [query]

        logger.debug("QueryPlanner: '%s' -> %d sub-queries", query[:40], len(cleaned))
        return cleaned
