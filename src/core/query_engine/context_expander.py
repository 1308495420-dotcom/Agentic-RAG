"""Context Expander — Small-to-Big: retrieve small chunks, return with neighbors.

When a chunk is matched, its adjacent chunks from the same source document
are added as context, preserving semantic continuity.

Three strategies:
- "neighbors": Add chunk_index ± window chunks from same source
- "parent_doc": Add all chunks from the same source document
- "none": No expansion (passthrough)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from src.core.types import RetrievalResult

logger = logging.getLogger(__name__)


class ContextExpander:
    """Expand retrieval results with neighboring chunks from same document.

    Uses an existing ChromaDB collection for metadata queries.

    Example::

        expander = ContextExpander()
        expanded = expander.expand(results, collection=chroma_collection)
    """

    def expand(
        self,
        results: List[RetrievalResult],
        strategy: str = "neighbors",
        window: int = 2,
        collection: Any = None,
    ) -> List[RetrievalResult]:
        """Expand results with context chunks.

        Args:
            results: Original ranked RetrievalResult list.
            strategy: "neighbors" | "parent_doc" | "none"
            window: For neighbors strategy, ±N chunks to include.
            collection: ChromaDB collection name.

        Returns:
            Expanded list (original chunks first, then neighbors).
        """
        if strategy == "none" or not results:
            return results

        if strategy == "neighbors":
            return self._expand_neighbors(results, window, collection)
        if strategy == "parent_doc":
            return self._expand_parent_doc(results, collection)
        return results

    # ── internal ──────────────────────────────────────────────────

    def _expand_neighbors(
        self,
        results: List[RetrievalResult],
        window: int,
        col: Any,
    ) -> List[RetrievalResult]:
        """Add neighboring chunks (±window) from same source."""
        seen: Set[str] = {r.chunk_id for r in results}
        expanded: List[RetrievalResult] = list(results)

        for r in results:
            source = (r.metadata or {}).get("source_path", "")
            chunk_idx = (r.metadata or {}).get("chunk_index", -1)
            if not source or chunk_idx < 0:
                continue

            try:
                data = col.get(
                    where={"source_path": source},
                    include=["documents", "metadatas"],
                )
            except Exception:
                continue

            for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
                if cid in seen:
                    continue
                neighbor_idx = (meta or {}).get("chunk_index", -1)
                if abs(neighbor_idx - chunk_idx) <= window and neighbor_idx != chunk_idx:
                    seen.add(cid)
                    expanded.append(RetrievalResult(
                        chunk_id=cid,
                        text=doc,
                        score=r.score * 0.85,
                        metadata=meta or {},
                    ))

        logger.debug("ContextExpand: %d → %d chunks", len(results), len(expanded))
        return expanded

    def _expand_parent_doc(
        self,
        results: List[RetrievalResult],
        col: Any,
    ) -> List[RetrievalResult]:
        """Add ALL chunks from the same source document."""
        seen: Set[str] = {r.chunk_id for r in results}
        expanded: List[RetrievalResult] = list(results)

        for r in results:
            source = (r.metadata or {}).get("source_path", "")
            if not source:
                continue
            try:
                data = col.get(
                    where={"source_path": source},
                    include=["documents", "metadatas"],
                )
            except Exception:
                continue

            for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
                if cid not in seen:
                    seen.add(cid)
                    expanded.append(RetrievalResult(
                        chunk_id=cid,
                        text=doc,
                        score=r.score * 0.8,
                        metadata=meta or {},
                    ))

        logger.debug("ContextExpand(parent): %d → %d chunks", len(results), len(expanded))
        return expanded
