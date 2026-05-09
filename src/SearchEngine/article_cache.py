"""
LRU cache for raw extracted article chunks.

Two separate caches:
  - ArticleCache: recently fetched article chunks (text only, no vectors)
  - FaissLruLoader: keeps at most `max_loaded` ZIM FAISS indexes in RAM;
    evicts the least-recently-used when the limit is exceeded.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import faiss as _faiss_t


class ArticleCache:
    """
    Thread-safe LRU cache of extracted article chunks keyed by (zim_name, path).
    Each entry is a list of chunk dicts (text, section_title, chunk_index, title).
    """

    def __init__(self, max_articles: int = 30):
        self._max = max_articles
        self._cache: OrderedDict[tuple[str, str], list[dict]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, zim_name: str, path: str) -> list[dict] | None:
        key = (zim_name, path)
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, zim_name: str, path: str, chunks: list[dict]) -> None:
        key = (zim_name, path)
        with self._lock:
            self._cache[key] = chunks
            self._cache.move_to_end(key)
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)

    def invalidate(self, zim_name: str) -> int:
        """Remove all entries for a ZIM (e.g. after rescan). Returns count removed."""
        with self._lock:
            keys = [k for k in self._cache if k[0] == zim_name]
            for k in keys:
                del self._cache[k]
            return len(keys)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)


class FaissLruLoader:
    """
    Keeps at most `max_loaded` FAISS indexes in RAM.
    Evicts the least-recently-used index when the cap is exceeded.

    Usage:
        loader = FaissLruLoader(max_loaded=3)
        idx = loader.get(handle)   # loads if needed, promotes if already loaded
        loader.touch(name)         # mark as recently used after a search
    """

    def __init__(self, max_loaded: int = 0):
        # 0 = unlimited (load all, original behavior)
        self._max = max_loaded
        self._loaded: OrderedDict[str, object] = OrderedDict()  # name → faiss index
        self._lock = threading.Lock()

    def get(self, name: str, idx_path: Path):
        """Return the FAISS index for `name`, loading from disk if necessary."""
        with self._lock:
            if name in self._loaded:
                self._loaded.move_to_end(name)
                return self._loaded[name]

        # Load outside lock so we don't block other searches during IO
        from SearchEngine import index as faiss_index
        idx = faiss_index.load_or_create(idx_path)

        with self._lock:
            self._loaded[name] = idx
            self._loaded.move_to_end(name)
            if self._max > 0:
                while len(self._loaded) > self._max:
                    evicted_name, _ = self._loaded.popitem(last=False)
                    print(f"[faiss_lru] evicted '{evicted_name}' from RAM", flush=True)
        return idx

    def touch(self, name: str) -> None:
        with self._lock:
            if name in self._loaded:
                self._loaded.move_to_end(name)

    def evict(self, name: str) -> bool:
        with self._lock:
            if name in self._loaded:
                del self._loaded[name]
                return True
            return False

    @property
    def loaded_names(self) -> list[str]:
        with self._lock:
            return list(self._loaded.keys())
