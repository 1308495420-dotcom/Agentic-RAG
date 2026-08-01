"""Chat page — conversational RAG with streaming responses."""

from __future__ import annotations

import logging
import os
from typing import Any, List

import streamlit as st

logger = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────


def _init_rag():
    """Lazy-init RAG components. Cached in session state."""
    if "rag_components" in st.session_state:
        return st.session_state["rag_components"]

    from src.core.settings import load_settings, resolve_path
    from src.core.query_engine.hybrid_search import create_hybrid_search
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.dense_retriever import create_dense_retriever
    from src.core.query_engine.sparse_retriever import create_sparse_retriever
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.core.response.answer_generator import AnswerGenerator
    from src.core.response.citation_verifier import CitationVerifier

    settings = load_settings()
    embedding = EmbeddingFactory.create(settings)
    vs = VectorStoreFactory.create(settings, collection_name="default")
    dense = create_dense_retriever(settings, embedding, vs)
    bm25 = BM25Indexer(index_dir=str(resolve_path("data/db/bm25/default")))
    sparse = create_sparse_retriever(settings, bm25, vs)
    sparse.default_collection = "default"

    hybrid = create_hybrid_search(settings, QueryProcessor(), dense, sparse)
    gen = AnswerGenerator(settings)
    verifier = CitationVerifier()

    comps = {
        "settings": settings,
        "hybrid": hybrid,
        "gen": gen,
        "verifier": verifier,
    }
    st.session_state["rag_components"] = comps
    return comps


def _stream_answer(query: str, chunks: list) -> str:
    """Stream LLM answer token-by-token. Returns the full answer."""
    from openai import OpenAI
    import os

    settings = st.session_state["rag_components"]["settings"]
    llm_cfg = settings.llm

    # Build context
    context_parts = []
    for i, chunk in enumerate(chunks[:5], 1):
        text = getattr(chunk, "text", str(chunk))
        title = (getattr(chunk, "metadata", {}) or {}).get("title", "")
        context_parts.append(f"[{i}] {title}\n{text[:600]}")

    ctx = "\n\n".join(context_parts)
    system_prompt = (
        "你是精确的知识助手。仅根据以下上下文回答。"
        "每条关键信息后标注来源编号 [1][2]。"
        "无法确定就说无法确定。\n\n"
        f"## 上下文\n{ctx}"
    )

    if llm_cfg.provider.lower() == "deepseek":
        client = OpenAI(
            api_key=llm_cfg.api_key or os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
    else:
        client = OpenAI(api_key=llm_cfg.api_key)

    response = client.chat.completions.create(
        model=llm_cfg.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=1024,
        stream=True,
    )

    placeholder = st.empty()
    full = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            full += chunk.choices[0].delta.content
            placeholder.markdown(full + "▌")
    placeholder.markdown(full)
    return full


# ── page ─────────────────────────────────────────────────────────────


def render() -> None:
    """Render the Chat page."""
    st.header("💬 RAG Chat")

    comps = _init_rag()
    verifier = comps["verifier"]

    # ── sidebar: retrieval settings ──────────────────────────────────
    with st.sidebar:
        st.markdown("### 检索设置")
        top_k = st.slider("Top-K", 3, 20, 5)
        collection = st.selectbox("知识库", ["default"])
        if st.button("🗑 清空对话"):
            st.session_state["chat_messages"] = []
            st.rerun()

    # ── message history ──────────────────────────────────────────────
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 参考来源"):
                    for s in msg["sources"]:
                        st.caption(
                            f"`{s.get('chunk_id', '?')[:24]}...` "
                            f"— {s.get('title', '无标题')}"
                        )

    # ── input ────────────────────────────────────────────────────────
    if prompt := st.chat_input("输入问题..."):
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Step 1: Retrieve
            with st.status("检索中...", expanded=False) as status:
                hybrid = comps["hybrid"]
                results = hybrid.search(query=prompt, top_k=top_k)
                results = results if isinstance(results, list) else results.results
                status.update(label=f"检索到 {len(results)} 条结果", state="complete")

            # Step 2: Stream answer
            full_answer = _stream_answer(prompt, results)

            # Step 3: Verify
            report = verifier.verify(answer=full_answer, chunks=results)

            # Step 4: Show sources + verification
            col1, col2 = st.columns(2)
            with col1:
                st.caption(
                    f"检索: {len(results)} chunks | "
                    f"核查: {report.verified_count}/{report.total_claims} 条可验证"
                )
            with col2:
                if report.verification_rate < 0.3:
                    st.warning("⚠️ 引用核查率偏低")
                else:
                    st.success(f"✅ {report.verification_rate:.0%}")

            sources = []
            for i, r in enumerate(results[:top_k], 1):
                meta = getattr(r, "metadata", {}) or {}
                sources.append({
                    "num": i,
                    "chunk_id": getattr(r, "chunk_id", ""),
                    "title": meta.get("title", "无标题"),
                })

            st.session_state["chat_messages"].append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources,
            })
