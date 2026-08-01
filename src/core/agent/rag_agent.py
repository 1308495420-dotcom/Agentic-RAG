"""Agentic RAG — Think-Act-Observe loop with LLM-driven multi-pass retrieval.

The Agent uses LLM to judge whether retrieval results are sufficient.
If not, the LLM suggests better search terms for the next round.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from src.core.agent.query_planner import QueryPlanner
from src.core.agent.toolkit import AgentToolkit, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Final response from the RAG Agent."""

    answer: str
    sub_results: List[SearchResult] = field(default_factory=list)
    total_passes: int = 0
    sub_queries: List[str] = field(default_factory=list)


class RagAgent:
    """Agentic RAG: decompose → retrieve → LLM-judge → (re-retrieve) → answer.

    Example::

        agent = RagAgent(toolkit, settings)
        response = agent.run("FinFET 和 GAAFET 在 3nm 下谁更好？")
    """

    _JUDGE_PROMPT = (
        "你是检索质量判断专家。根据以下信息判断当前检索结果是否足以回答用户问题。\n"
        "\n"
        "用户原始问题：\n{original_query}\n"
        "\n"
        "当前搜索的子问题：\n{sub_query}\n"
        "\n"
        "检索到的 Top-3 文档：\n{chunks}\n"
        "\n"
        "基于这些文档生成的初步答案：\n{answer}\n"
        "\n"
        "引用核查率（答案中能在原文找到依据的比例）：{verify_rate:.0%}\n"
        "\n"
        "请判断：\n"
        "1. 检索结果是否足以回答用户问题？（YES / NO）\n"
        "2. 如果 NO，建议一个更好的搜索关键词（10 字以内）\n"
        "\n"
        "严格按以下格式回复，不要多余文字：\n"
        "DECISION: YES|NO\n"
        "SUGGESTION: （仅在 NO 时填写）"
    )

    def __init__(
        self,
        toolkit: AgentToolkit,
        settings: Any,
        max_rounds: int = 3,
    ) -> None:
        self._tk = toolkit
        self._planner = QueryPlanner(settings)
        self._gen = toolkit._gen
        self._settings = settings
        self._max_rounds = max_rounds

    def run(self, query: str, top_k: int = 5) -> AgentResponse:
        """Run the Agentic RAG pipeline.

        Flow: plan → retrieve → LLM judge → (re-retrieve with suggestion) → answer.
        """
        sub_queries = self._planner.plan(query)
        logger.info("Agent: '%s' -> %d sub-queries", query[:50], len(sub_queries))

        all_results: List[SearchResult] = []
        total_passes = 0

        for sq in sub_queries:
            result = self._tk.run_retrieval_pass(sq, top_k=top_k)
            total_passes += 1
            round_count = 1

            while round_count < self._max_rounds:
                should_continue, new_query = self._judge(
                    original_query=query,
                    sub_query=sq,
                    result=result,
                )
                if not should_continue:
                    break

                logger.info(
                    "Agent: re-retrieving '%s' → '%s' (round %d)",
                    sq[:40], new_query, round_count + 1,
                )
                result = self._tk.run_retrieval_pass(new_query, top_k=top_k + 5)
                total_passes += 1
                round_count += 1

            all_results.append(result)

        # Synthesize all results
        all_chunks: list = []
        for sr in all_results:
            for c in sr.chunks:
                if c.chunk_id not in {getattr(ec, "chunk_id", "") for ec in all_chunks}:
                    all_chunks.append(c)

        final_answer = self._gen.generate(query=query, chunks=all_chunks[:10])

        return AgentResponse(
            answer=final_answer.answer,
            sub_results=all_results,
            total_passes=total_passes,
            sub_queries=sub_queries,
        )

    # ── LLM confidence judge ────────────────────────────────────────

    def _judge(
        self,
        original_query: str,
        sub_query: str,
        result: SearchResult,
    ) -> Tuple[bool, str]:
        """Ask LLM: are the results sufficient? If not, suggest better keywords.

        Returns:
            (should_continue, new_search_query_or_empty_string)
        """
        # Format top-3 chunks for LLM
        chunk_lines = []
        for i, c in enumerate(result.chunks[:3], 1):
            title = (getattr(c, "metadata", {}) or {}).get("title", "")
            text = getattr(c, "text", "")[:200].replace("\n", " ")
            chunk_lines.append(f"[{i}] {title}\n   {text}...")

        prompt = self._JUDGE_PROMPT.format(
            original_query=original_query,
            sub_query=sub_query,
            chunks="\n\n".join(chunk_lines) if chunk_lines else "无结果",
            answer=result.answer[:500],
            verify_rate=result.verification_rate,
        )

        try:
            llm_cfg = self._settings.llm
            from openai import OpenAI

            if llm_cfg.provider.lower() == "deepseek":
                client = OpenAI(
                    api_key=llm_cfg.api_key or os.environ.get("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com",
                )
            else:
                client = OpenAI(api_key=llm_cfg.api_key)

            response = client.chat.completions.create(
                model=llm_cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=128,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Agent: LLM judge failed (%s) — stopping", exc)
            return False, ""

        # Parse structured response
        decision = "NO"
        suggestion = ""

        m = re.search(r"DECISION:\s*(YES|NO)", raw, re.IGNORECASE)
        if m:
            decision = m.group(1).upper()

        m = re.search(r"SUGGESTION:\s*(.+)", raw)
        if m:
            suggestion = m.group(1).strip()

        if decision == "NO" and suggestion:
            logger.info("Agent: NO — LLM suggests '%s'", suggestion)
            return True, suggestion

        logger.info("Agent: YES — result sufficient")
        return False, ""
