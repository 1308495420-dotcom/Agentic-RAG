"""Query result cache with TTL and ingestion-triggered invalidation.

Simple in-memory LRU cache that stores (query, top_k, collection) → results.
Automatically invalidated when new documents are ingested.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple


class QueryCache:
    """TTL-based query result cache.

    Caches (query, top_k, collection) → retrieval results so repeated
    queries hit memory instead of re-running full hybrid search.

    Attributes:
        max_size: Maximum number of cached entries.
        ttl_seconds: Time-to-live for each cache entry.
    """

    def __init__(self, max_size: int = 256, ttl_seconds: int = 300) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._version = 0  # bumped on each invalidation

    # ── public API ────────────────────────────────────────────────

    def get(self, query: str, top_k: int, collection: str) -> Optional[List[Any]]:
        """Return cached results or None if miss/expired."""
        key = self._make_key(query, top_k, collection)
        entry = self._store.get(key)
        if entry is None:
            return None
        timestamp, results = entry
        if time.monotonic() - timestamp > self._ttl:
            del self._store[key]
            return None
        # Move to end (most recently used)
        self._store.move_to_end(key)
        return results

    def put(self, query: str, top_k: int, collection: str, results: List[Any]) -> None:
        """Store results in cache."""
        key = self._make_key(query, top_k, collection)
        self._store[key] = (time.monotonic(), results)
        self._store.move_to_end(key)
        # Evict oldest if over capacity
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def invalidate(self) -> None:
        """Clear all cached entries (called after new document ingestion)."""
        self._store.clear()
        self._version += 1

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def version(self) -> int:
        return self._version

    # ── internal ──────────────────────────────────────────────────

    def _make_key(self, query: str, top_k: int, collection: str) -> str:
        raw = f"{query}|{top_k}|{collection}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
