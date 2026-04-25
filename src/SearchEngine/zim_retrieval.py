"""
ZIM retrieval bridge for SearchEngine.

Loaded once at import time — discovers which configured ZIMs have a built
FAISS index and opens them. Provides search() with three modes:

  fast     — FAISS + BM25, small candidate pool, lightweight reranker
  balanced — FAISS + BM25, medium pool, stronger fusion + reranker (default)
  complex  — FAISS + BM25, large pool, stronger fusion + reranker

Retrieval is now two-stage:
  1. Candidate fusion with RRF over dense, paragraph BM25, title BM25,
     section/header overlap, and simple structural priors.
  2. Optional reranking over the fused pool. Set retrieval.reranker to "rrf"
     for pure RRF order, or "heuristic" for the weighted second stage.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import queue
import threading
import yaml
import numpy as np
import re
from pathlib import Path

from SearchEngine.article_cache import ArticleCache, FaissLruLoader

try:
    from libzim.reader import Archive as _ZimArchive
    from libzim.search import Query as _ZimQuery, Searcher as _ZimSearcher
    from libzim.suggestion import SuggestionSearcher as _ZimSuggester
    _LIBZIM_OK = True
except Exception:
    _LIBZIM_OK = False

from indexer import index as faiss_index
from indexer.db import (
    open_db, init_fts,
    title_search_scored, chunk_text_search, get_chunks_by_ids,
)
from SearchEngine.config import CFG
from SearchEngine.embedding import _st_encode as _encode
from SearchEngine.keywords import extract_keywords

_RETRIEVAL_CFG = CFG.get("retrieval", {})
_WEIGHT_CFG = _RETRIEVAL_CFG.get("weights", {})
_RRF_WEIGHT_CFG = _RETRIEVAL_CFG.get("rrf_weights", {})

_MIN_FAISS_SCORE         = float(_RETRIEVAL_CFG.get("min_faiss_score", 0.45))
_SUGGESTION_BOOST        = float(_RETRIEVAL_CFG.get("suggestion_boost", 0.50))
_FUSION_MAX_PER_ARTICLE  = int(  _RETRIEVAL_CFG.get("fusion_max_per_article", 3))
_NAV_LEAD_BOOST          = float(_RETRIEVAL_CFG.get("nav_lead_boost", 1.15))
_RERANKER                = str(_RETRIEVAL_CFG.get("reranker", "heuristic")).strip().lower()
_INFOBOX_MIN_SCORE       = float(_RETRIEVAL_CFG.get("infobox_min_score", 0.80))
_INFOBOX_MAX_PER_ARTICLE = int(_RETRIEVAL_CFG.get("infobox_max_per_article", 3))
_MAX_PROSE_CHARS         = int(_RETRIEVAL_CFG.get("max_prose_chars", 1200))
_RRF_K                   = int(_RETRIEVAL_CFG.get("rrf_k", 60))
_TITLE_ARTICLE_CHUNKS    = int(_RETRIEVAL_CFG.get("title_article_chunks", 6))
_RERANK_POOL_MIN         = int(_RETRIEVAL_CFG.get("rerank_pool_min", 24))
_RERANK_POOL_MULTIPLIER  = int(_RETRIEVAL_CFG.get("rerank_pool_multiplier", 6))
_ANSWER_FILTER_COVERAGE  = float(_RETRIEVAL_CFG.get("answer_filter_query_coverage", 0.34))
_ARTICLE_SCORE_MARGIN    = float(_RETRIEVAL_CFG.get("article_score_margin", 0.20))
_CROSS_ZIM_STRONG_SCORE  = float(_RETRIEVAL_CFG.get("cross_zim_strong_score", 0.80))
_CROSS_ZIM_MARGIN        = float(_RETRIEVAL_CFG.get("cross_zim_margin", 0.05))
_TITLE_ONLY_MIN_TITLE_OVERLAP = float(_RETRIEVAL_CFG.get("title_only_min_title_overlap", 0.35))
_TITLE_ONLY_MIN_QUERY_COVERAGE = float(_RETRIEVAL_CFG.get("title_only_min_query_coverage", 0.20))
_CROSS_ZIM_DOMINANCE_RERANK = float(_RETRIEVAL_CFG.get("cross_zim_dominance_rerank", 0.52))
_CROSS_ZIM_TITLE_OVERLAP = float(_RETRIEVAL_CFG.get("cross_zim_title_overlap", 0.35))
_CROSS_ZIM_KEEP_MARGIN = float(_RETRIEVAL_CFG.get("cross_zim_keep_margin", 0.08))
_SEMANTIC_RERANK_TOP_N = int(_RETRIEVAL_CFG.get("semantic_rerank_top_n", 18))
_SEMANTIC_RERANK_WEIGHT = float(_RETRIEVAL_CFG.get("semantic_rerank_weight", 0.30))
_SEMANTIC_SUMMARY_CHARS = int(_RETRIEVAL_CFG.get("semantic_summary_chars", 220))
_ARTICLE_SEMANTIC_RERANK_TOP_N = int(_RETRIEVAL_CFG.get("article_semantic_rerank_top_n", 12))
_ARTICLE_SEMANTIC_RERANK_WEIGHT = float(_RETRIEVAL_CFG.get("article_semantic_rerank_weight", 0.45))
_ARTICLE_SEMANTIC_BASE_WEIGHT = float(_RETRIEVAL_CFG.get("article_semantic_base_weight", 0.35))

# RRF source weights. Set a source to 0.0 to remove it from RRF fusion.
_RRF_W_SEMANTIC  = float(_RRF_WEIGHT_CFG.get("semantic", 1.0))
_RRF_W_PARA_BM25 = float(_RRF_WEIGHT_CFG.get("paragraph_bm25", 1.0))
_RRF_W_TITLE     = float(_RRF_WEIGHT_CFG.get("title_bm25", 0.9))
_RRF_W_SECTION   = float(_RRF_WEIGHT_CFG.get("section_overlap", 0.6))
_RRF_W_ASPECT    = float(_RRF_WEIGHT_CFG.get("aspect_section", 0.9))
_RRF_W_TITLE_TOK = float(_RRF_WEIGHT_CFG.get("title_query_overlap", 0.5))
_RRF_W_STRUCTURE = float(_RRF_WEIGHT_CFG.get("structure_prior", 0.3))

# Stage-2 heuristic reranker weights
_W_FUSION    = float(_WEIGHT_CFG.get("fusion", 0.34))
_W_SEMANTIC  = float(_WEIGHT_CFG.get("semantic", 0.24))
_W_PARA_BM25 = float(_WEIGHT_CFG.get("paragraph_bm25", 0.16))
_W_TITLE     = float(_WEIGHT_CFG.get("title_bm25", 0.08))
_W_SECTION   = float(_WEIGHT_CFG.get("section_overlap", 0.08))
_W_ASPECT    = float(_WEIGHT_CFG.get("aspect_section", 0.08))
_W_TITLE_TOK = float(_WEIGHT_CFG.get("title_query_overlap", 0.04))
_W_COVERAGE  = float(_WEIGHT_CFG.get("query_coverage", 0.03))
_W_STRUCTURE = float(_WEIGHT_CFG.get("structure_prior", 0.03))

ZIM_INDEX_BASE = Path(CFG.get("zim_index_base", "/library/zims/content"))
_MODES_DIR     = Path(__file__).parent / "modes"

# ── On-demand retrieval config ────────────────────────────────────────────────
_OD_CFG                  = CFG.get("zim_on_demand", {})
_OD_MAX_LOADED_ZIMS      = int(_OD_CFG.get("max_loaded_zims", 0))         # 0 = unlimited
_OD_RAM_CACHE_ARTICLES   = int(_OD_CFG.get("ram_cache_articles", 30))
_OD_TITLE_FETCH_COUNT    = int(_OD_CFG.get("title_fetch_articles", 3))    # articles to fetch per query
_OD_TITLE_SEARCH_TOP     = int(_OD_CFG.get("title_search_candidates", 20))
_OD_AUTO_EMBED           = bool(_OD_CFG.get("auto_embed", False))
_OD_AUTO_EMBED_THRESHOLD = float(_OD_CFG.get("auto_embed_score_threshold", 0.72))
_TOK_RE        = re.compile(r"[a-z0-9]+")
_ASPECT_SYNONYMS = {
    "development": {
        "development", "history", "economy", "economic", "growth", "urban",
        "urbanisation", "urbanization", "industry", "industrial", "trade",
        "port", "infrastructure", "modern", "modernisation", "modernization",
        "finance", "financial", "expansion",
    },
    "history": {
        "history", "historical", "background", "origins", "origin", "timeline",
        "colonial", "development",
    },
    "economy": {
        "economy", "economic", "finance", "financial", "trade", "industry",
        "industrial", "market", "business", "development",
    },
    "politics": {
        "politics", "political", "government", "administration", "law",
        "legal", "policy", "governance",
    },
    "demographics": {
        "demographics", "population", "people", "ethnic", "ethnicity",
        "migration",
    },
    "culture": {
        "culture", "cultural", "society", "language", "languages", "religion",
        "arts", "music", "media",
    },
    "transport": {
        "transport", "transportation", "rail", "road", "airport", "harbour",
        "harbor", "port", "infrastructure",
    },
}


# ── Mode configs ──────────────────────────────────────────────────────────────

def _load_mode(name: str) -> dict:
    path = _MODES_DIR / f"{name}.yaml"
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}

_MODE_CFGS = {
    "fast":     _load_mode("fast"),
    "balanced": _load_mode("balanced"),
    "complex":  _load_mode("complex"),
}

# ── Shared caches (module-level singletons) ───────────────────────────────────
_faiss_loader  = FaissLruLoader(max_loaded=_OD_MAX_LOADED_ZIMS)
_article_cache = ArticleCache(max_articles=_OD_RAM_CACHE_ARTICLES)

# Background embed queue: items are (zim_name, article_path)
_embed_queue: queue.Queue = queue.Queue()


def _embed_worker():
    while True:
        try:
            zim_name, path = _embed_queue.get(timeout=5)
            with _handles_lock:
                handle = next((h for h in _handles if h.name == zim_name), None)
            if handle:
                handle.embed_article(path)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[embed_worker] {e}", flush=True)


_embed_thread = threading.Thread(target=_embed_worker, daemon=True, name="embed-worker")
_embed_thread.start()


# ── ZIM handle ────────────────────────────────────────────────────────────────

class _ZimHandle:
    def __init__(self, name: str, zim_path: Path, enabled: bool = True, count: int = 5,
                 locked: bool = False, hidden: bool = False,
                 title_index: bool = False, max_articles: int = 2000, max_vectors: int = 500_000):
        self.name         = name
        self.zim_path     = zim_path
        self.enabled      = enabled
        self.count        = count
        self.locked       = locked
        self.hidden       = hidden
        self.title_index  = title_index   # whether title index is opt-in for this ZIM
        self.max_articles = max_articles  # on-demand embed limit
        self.max_vectors  = max_vectors   # FAISS vector cap

        out_dir        = ZIM_INDEX_BASE / name
        self._db_path   = out_dir / "data.db"
        self._idx_path  = out_dir / "faiss.index"

        self.is_indexed = (
            self._db_path.exists()
            and self._idx_path.exists()
            and self._idx_path.stat().st_size > 100
        )

        if self.is_indexed:
            try:
                self.con = open_db(self._db_path)
                init_fts(self.con)
                # FAISS is loaded lazily via the idx property / FaissLruLoader
            except Exception as e:
                print(f"[zim_retrieval] ⚠️  {name} index load failed: {e}", flush=True)
                self.con = None
                self.is_indexed = False
        else:
            self.con = None

        self.archive   = None
        self._suggester = None  # lazy SuggestionSearcher
        if _LIBZIM_OK:
            try:
                self.archive = _ZimArchive(str(zim_path))
            except Exception as e:
                print(f"[zim_retrieval] ⚠️  {name} archive open failed: {e}", flush=True)

        # Title index data — loaded lazily the first time title search is needed
        self._title_vecs:   np.ndarray | None = None
        self._title_paths:  list[str] | None  = None
        self._title_titles: list[str] | None  = None
        self._title_lock = threading.Lock()

    # ── Lazy FAISS via LRU loader ─────────────────────────────────────────────

    @property
    def idx(self):
        if not self.is_indexed:
            return None
        return _faiss_loader.get(self.name, self._idx_path)

    # ── Title index helpers ───────────────────────────────────────────────────

    @property
    def has_title_index(self) -> bool:
        from indexer.title_index import title_db_exists
        return title_db_exists(self.zim_path)

    def _load_title_data(self) -> bool:
        """Load title vecs into RAM if not already loaded. Returns True if available."""
        if self._title_vecs is not None:
            return True
        if not self.has_title_index:
            return False
        with self._title_lock:
            if self._title_vecs is not None:
                return True
            try:
                from indexer.title_index import open_title_db, load_all_vecs
                con = open_title_db(self.zim_path)
                vecs, paths, titles = load_all_vecs(con)
                con.close()
                if vecs.shape[0] == 0:
                    return False
                self._title_vecs   = vecs
                self._title_paths  = paths
                self._title_titles = titles
                print(f"[zim_retrieval] 🔍 {self.name}: {len(paths):,} title vectors loaded", flush=True)
                return True
            except Exception as e:
                print(f"[zim_retrieval] title index load failed ({self.name}): {e}", flush=True)
                return False

    def title_search(self, query_vec: np.ndarray, top_k: int = 20) -> list[dict]:
        """Semantic title search. Returns [{path, title, score}]."""
        if not self._load_title_data():
            return []
        from indexer.title_index import search_titles
        return search_titles(self._title_vecs, self._title_paths, self._title_titles,
                             query_vec, top_k=top_k)

    # ── Article fetch & on-demand embed ──────────────────────────────────────

    def fetch_chunks(self, path: str) -> list[dict]:
        """Fetch and extract chunks for an article path via libzim. Uses article cache."""
        cached = _article_cache.get(self.name, path)
        if cached is not None:
            return cached

        if not self.archive:
            return []
        try:
            from indexer.extract import extract as _extract
            entry  = self.archive.get_entry_by_path(path)
            item   = entry.get_item()
            html   = bytes(item.content).decode("utf-8", errors="replace")
            parsed = _extract(html)
            if not parsed:
                return []
            title  = parsed.get("title") or entry.title
            chunks: list[dict] = []
            lead_texts = parsed.get("lead_paragraphs") or []
            for i, text in enumerate(lead_texts):
                chunks.append({"section_title": "lead", "chunk_index": i,
                               "text": text, "title": title, "url": path})
            for sec in parsed.get("sections", []):
                for i, text in enumerate(sec.get("paragraphs", [])):
                    chunks.append({"section_title": sec.get("title", ""),
                                   "chunk_index": i, "text": text,
                                   "title": title, "url": path})
            _article_cache.put(self.name, path, chunks)
            return chunks
        except Exception as e:
            print(f"[zim_retrieval] fetch_chunks failed ({self.name}/{path}): {e}", flush=True)
            return []

    def embed_article(self, path: str) -> int:
        """
        Embed a specific article and add its chunks to FAISS + SQLite.
        Returns number of chunks newly embedded, or -1 on error.
        Respects max_articles and max_vectors limits.
        """
        if not self.archive or not self.is_indexed or self.con is None:
            return -1

        # Check limits
        try:
            art_count = self.con.execute(
                "SELECT COUNT(DISTINCT article_id) FROM chunks"
            ).fetchone()[0]
            if art_count >= self.max_articles:
                print(f"[zim_retrieval] {self.name}: max_articles ({self.max_articles}) reached", flush=True)
                return -1

            idx = self.idx
            if idx and idx.ntotal >= self.max_vectors:
                print(f"[zim_retrieval] {self.name}: max_vectors ({self.max_vectors}) reached", flush=True)
                return -1

            # Skip if already embedded
            existing = self.con.execute(
                "SELECT a.id FROM articles a JOIN chunks c ON c.article_id = a.id "
                "WHERE a.url = ? AND c.embedded = 1 LIMIT 1", (path,)
            ).fetchone()
            if existing:
                return 0
        except Exception:
            pass

        chunks = self.fetch_chunks(path)
        if not chunks:
            return -1

        from indexer.db import insert_article, insert_chunks, mark_embedded, article_exists
        from indexer import index as faiss_index
        from SearchEngine.embedding import embed_batch

        try:
            title = chunks[0]["title"] if chunks else path
            if not article_exists(self.con, title):
                art_id = insert_article(self.con, title, path, str(self.zim_path))
            else:
                art_id = self.con.execute(
                    "SELECT id FROM articles WHERE title = ?", (title,)
                ).fetchone()[0]

            db_chunks = [{"section_title": c["section_title"],
                          "chunk_index": c["chunk_index"], "text": c["text"]}
                         for c in chunks]
            insert_chunks(self.con, art_id, db_chunks)
            self.con.commit()

            # Fetch the newly inserted chunk ids
            pending = self.con.execute(
                "SELECT id, text FROM chunks WHERE article_id = ? AND embedded = 0",
                (art_id,)
            ).fetchall()
            if not pending:
                return 0

            ids   = [r[0] for r in pending]
            texts = [r[1] for r in pending]
            vecs  = embed_batch(texts)

            idx = self.idx
            idx.add(vecs)
            faiss_index.save(idx, self._idx_path)
            mark_embedded(self.con, ids)

            print(f"[zim_retrieval] ✅ embedded '{title}' ({len(ids)} chunks) in {self.name}", flush=True)
            return len(ids)
        except Exception as e:
            print(f"[zim_retrieval] embed_article failed ({self.name}/{path}): {e}", flush=True)
            return -1

    def embed_limits(self) -> dict:
        """Return current usage vs configured limits."""
        art_count = vec_count = 0
        if self.con:
            try:
                art_count = self.con.execute(
                    "SELECT COUNT(DISTINCT article_id) FROM chunks"
                ).fetchone()[0]
            except Exception:
                pass
        idx = self.idx
        if idx:
            vec_count = idx.ntotal
        return {
            "embedded_articles":  art_count,
            "max_articles":       self.max_articles,
            "embedded_vectors":   vec_count,
            "max_vectors":        self.max_vectors,
            "articles_pct":       round(art_count / max(self.max_articles, 1) * 100, 1),
            "vectors_pct":        round(vec_count  / max(self.max_vectors,  1) * 100, 1),
        }

    def status(self) -> dict:
        from indexer.title_index import title_db_count
        zim_size  = self.zim_path.stat().st_size if self.zim_path.exists() else 0
        idx_size  = self._idx_path.stat().st_size if self._idx_path.exists() else 0
        db_size   = self._db_path.stat().st_size  if self._db_path.exists()  else 0

        # Use SQLite for counts — avoids loading FAISS just for status
        vector_count  = 0
        chunk_count   = 0
        article_count = 0
        if self.con:
            try:
                chunk_count   = self.con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                vector_count  = self.con.execute("SELECT COUNT(*) FROM chunks WHERE embedded=1").fetchone()[0]
                article_count = self.con.execute("SELECT COUNT(DISTINCT article_id) FROM chunks").fetchone()[0]
            except Exception:
                pass

        archive_articles = 0
        archive_title    = self.name
        if self.archive:
            try:
                archive_articles = self.archive.article_count
            except Exception:
                pass
            try:
                archive_title = self.archive.metadata.get("Title", self.name)
            except Exception:
                pass

        return {
            "name":                  self.name,
            "title":                 archive_title,
            "enabled":               self.enabled,
            "count":                 self.count,
            "locked":                self.locked,
            "is_indexed":            self.is_indexed,
            "zim_size_mb":           round(zim_size  / 1024**2, 1),
            "db_size_mb":            round(db_size   / 1024**2, 1),
            "index_size_mb":         round(idx_size  / 1024**2, 1),
            "vector_count":          vector_count,
            "chunk_count":           chunk_count,
            "indexed_article_count": article_count,
            "archive_article_count": archive_articles,
            # On-demand / title index
            "title_index_enabled":   self.title_index,
            "title_index_ready":     self.has_title_index,
            "title_index_count":     title_db_count(self.zim_path),
            "max_articles":          self.max_articles,
            "max_vectors":           self.max_vectors,
        }

    def xapian_search(self, query_text: str, count: int = 5) -> list[dict]:
        if not self.archive or not _LIBZIM_OK:
            return []
        try:
            searcher = _ZimSearcher(self.archive)
            results  = searcher.search(_ZimQuery(query_text))
            paths    = list(results.getResults(0, count * 3))
        except Exception as e:
            print(f"[zim_retrieval] Xapian search failed ({self.name}): {e}", flush=True)
            return []

        from indexer.extract import extract
        from bs4 import BeautifulSoup

        chunks = []
        for path in paths:
            if len(chunks) >= count:
                break
            try:
                entry   = self.archive.get_entry_by_path(path)
                item    = entry.get_item()
                html    = bytes(item.content).decode("utf-8", errors="replace")
                parsed  = extract(html)
                if not parsed:
                    continue
                title = parsed.get("title") or entry.title
                for sec in parsed.get("sections", [])[:3]:
                    for text in sec.get("paragraphs", [])[:2]:
                        chunks.append({
                            "chunk_id":          abs(hash(path + text)) % (2**31),
                            "article_id":        abs(hash(path)) % (2**31),
                            "title":             title,
                            "section":           sec.get("title", ""),
                            "text":              text,
                            "chunk_index":       0,
                            "zim_name":          self.name,
                            "faiss_score":       0.0,
                            "para_bm25_score":   0.0,
                            "title_bm25_score":  0.0,
                            "rerank_score":      0.45,
                            "candidate_sources": ["xapian"],
                            "url":               path,
                            "type":              "prose",
                        })
                        if len(chunks) >= count:
                            break
                    if len(chunks) >= count:
                        break
            except Exception:
                continue
        return chunks

    def suggestion_article_ids(self, keyword: str, top_n: int = 3) -> set[int]:
        """Return DB article_ids whose titles match keyword via libzim suggestion search."""
        if not self.archive or not self.con or not _LIBZIM_OK:
            return set()
        try:
            if self._suggester is None:
                self._suggester = _ZimSuggester(self.archive)
            sugg = self._suggester.suggest(keyword)
            if sugg.getEstimatedMatches() == 0:
                return set()
            titles = []
            for path in list(sugg.getResults(0, top_n)):
                try:
                    titles.append(self.archive.get_entry_by_path(path).title)
                except Exception:
                    pass
            if not titles:
                return set()
            placeholders = ",".join("?" * len(titles))
            rows = self.con.execute(
                f"SELECT id FROM articles WHERE title IN ({placeholders})", titles
            ).fetchall()
            return {row[0] for row in rows}
        except Exception:
            return set()


_handles: list[_ZimHandle] = []
_handles_lock = threading.RLock()

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _zim_settings_from_config() -> dict[str, dict]:
    """Return per-ZIM settings from config.yaml zims list."""
    out: dict[str, dict] = {}
    for entry in CFG.get("zims", []):
        name = entry.get("name", "")
        if name:
            out[name] = {
                "enabled":      entry.get("enabled", True),
                "count":        int(entry.get("count", 5)),
                "locked":       bool(entry.get("locked", False)),
                "hidden":       bool(entry.get("hidden", False)),
                "title_index":  bool(entry.get("title_index", False)),
                "max_articles": int(entry.get("max_articles", _OD_CFG.get("max_articles_per_zim", 2000))),
                "max_vectors":  int(entry.get("max_vectors",  _OD_CFG.get("max_vectors_per_zim", 500_000))),
            }
    return out


def _load_all():
    settings = _zim_settings_from_config()
    for zim_path in sorted(ZIM_INDEX_BASE.glob("*.zim")):
        name = zim_path.stem
        cfg  = settings.get(name, {})
        try:
            h = _ZimHandle(
                name, zim_path,
                enabled=cfg.get("enabled", True),
                count=cfg.get("count", 5),
                locked=cfg.get("locked", False),
                hidden=cfg.get("hidden", False),
                title_index=cfg.get("title_index", False),
                max_articles=cfg.get("max_articles", 2000),
                max_vectors=cfg.get("max_vectors", 500_000),
            )
            _handles.append(h)
            ti = " + title-index" if h.has_title_index else ""
            if h.is_indexed:
                print(f"[zim_retrieval] ✅ {name}{ti}", flush=True)
            else:
                print(f"[zim_retrieval] 📄 {name}: Xapian-only{ti}", flush=True)
        except Exception as e:
            print(f"[zim_retrieval] ⚠️  {name}: {e}", flush=True)


_load_all()


# ── Public API for libraries management ───────────────────────────────────────

def get_all_handles() -> list[dict]:
    with _handles_lock:
        return [h.status() for h in _handles if not h.hidden]


def rescan() -> int:
    """Re-discover ZIM files on disk and reload handles. Returns new handle count."""
    global _handles
    settings = _zim_settings_from_config()
    new_handles: list[_ZimHandle] = []
    for zim_path in sorted(ZIM_INDEX_BASE.glob("*.zim")):
        name   = zim_path.stem
        cfg    = settings.get(name, {})
        locked = cfg.get("locked", False)
        with _handles_lock:
            existing = next((h for h in _handles if h.name == name), None)
        # Locked ZIMs: config is always authoritative. Unlocked: preserve in-memory state.
        enabled = cfg.get("enabled", True) if locked else (existing.enabled if existing else cfg.get("enabled", True))
        count   = existing.count if existing else cfg.get("count", 5)
        try:
            h = _ZimHandle(
                name, zim_path,
                enabled=enabled, count=count,
                locked=locked,
                hidden=cfg.get("hidden", False),
                title_index=cfg.get("title_index", False),
                max_articles=cfg.get("max_articles", 2000),
                max_vectors=cfg.get("max_vectors", 500_000),
            )
            new_handles.append(h)
        except Exception as e:
            print(f"[zim_retrieval] ⚠️  rescan {name}: {e}", flush=True)
    with _handles_lock:
        _handles = new_handles
    print(f"[zim_retrieval] rescan complete — {len(new_handles)} ZIM(s)", flush=True)
    return len(new_handles)


def update_handle(name: str, enabled: bool | None = None, count: int | None = None) -> str | bool:
    """Update a handle. Returns True on success, 'locked' if admin-locked, False if not found."""
    with _handles_lock:
        for h in _handles:
            if h.name == name:
                if h.locked:
                    return "locked"
                if enabled is not None:
                    h.enabled = enabled
                if count is not None:
                    h.count = count
                return True
    return False


def write_config_zims():
    """Persist current handle enabled/count state back to config.yaml."""
    with _handles_lock:
        snapshot = [(h.name, h.enabled, h.count, h.locked, h.hidden) for h in _handles]
    try:
        with open(_CONFIG_PATH) as f:
            raw = yaml.safe_load(f) or {}
        existing = {e["name"]: e for e in raw.get("zims", []) if "name" in e}
        raw["zims"] = [
            {**existing.get(name, {}), "name": name, "enabled": enabled, "count": count,
             "locked": locked, "hidden": hidden}
            for name, enabled, count, locked, hidden in snapshot
        ]
        with open(_CONFIG_PATH, "w") as f:
            yaml.dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except Exception as e:
        print(f"[zim_retrieval] config write failed: {e}", flush=True)


def is_available() -> bool:
    with _handles_lock:
        return any(h.enabled for h in _handles)


# ── Reranker ──────────────────────────────────────────────────────────────────

def _minmax(d: dict) -> dict:
    """Min-max normalise a {key: float} dict to [0, 1]."""
    if not d:
        return d
    lo, hi = min(d.values()), max(d.values())
    if hi == lo:
        return {k: 1.0 for k in d}
    span = hi - lo
    return {k: (v - lo) / span for k, v in d.items()}


def _norm_faiss(score: float) -> float:
    """
    Normalise a FAISS cosine score using a fixed meaningful range
    [_MIN_FAISS_SCORE, 1.0] → [0, 1].

    Using a fixed range rather than min-max within the candidate pool
    prevents the normalization from being distorted by:
      - BM25-only chunks (faiss=0.0) dragging the min to zero
      - Narrow score spreads amplifying tiny differences between candidates
    """
    if score <= 0:
        return 0.0
    return min(1.0, max(0.0, (score - _MIN_FAISS_SCORE) / (1.0 - _MIN_FAISS_SCORE)))


def _section_overlap(query: str, section_title: str) -> float:
    """Word-level overlap between query and section title, normalised to [0,1]."""
    q = set(_tokenize(query))
    s = set(_tokenize(section_title))
    if not s:
        return 0.0
    return len(q & s) / max(len(q), len(s))


def _tokenize(text: str) -> list[str]:
    return _TOK_RE.findall((text or "").lower())


def _query_aspects(query_text: str) -> set[str]:
    tokens = set(_tokenize(query_text))
    aspects: set[str] = set()
    for aspect, synonyms in _ASPECT_SYNONYMS.items():
        if tokens & synonyms:
            aspects.add(aspect)
    return aspects


def _aspect_section_score(query_text: str, section_title: str) -> float:
    aspects = _query_aspects(query_text)
    if not aspects:
        return 0.0

    section_tokens = set(_tokenize(section_title))
    if not section_tokens:
        return 0.0

    best = 0.0
    for aspect in aspects:
        synonyms = _ASPECT_SYNONYMS[aspect]
        overlap = len(section_tokens & synonyms)
        if overlap:
            best = max(best, overlap / max(1, min(len(section_tokens), 4)))
    return min(1.0, best)


def _rank_map(score_map: dict, reverse: bool = True) -> dict:
    ordered = sorted(score_map.items(), key=lambda kv: kv[1], reverse=reverse)
    return {key: rank for rank, (key, _) in enumerate(ordered, start=1)}


def _rrf_add(total: dict[int, float], rank_map: dict[int, int], weight: float = 1.0) -> None:
    for key, rank in rank_map.items():
        total[key] = total.get(key, 0.0) + (weight / (_RRF_K + rank))


def _title_token_overlap(query_text: str, title: str) -> float:
    q = set(_tokenize(query_text))
    t = set(_tokenize(title))
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)


def _query_coverage(query_text: str, chunk: dict) -> float:
    q = set(_tokenize(query_text))
    if not q:
        return 0.0
    hay = set(_tokenize(chunk["title"])) | set(_tokenize(chunk["section_title"])) | set(_tokenize(chunk["text"]))
    return len(q & hay) / len(q)


def _body_query_coverage(query_text: str, chunk: dict) -> float:
    q = set(_tokenize(query_text))
    if not q:
        return 0.0
    hay = set(_tokenize(chunk["section_title"])) | set(_tokenize(chunk["text"]))
    return len(q & hay) / len(q)


def _answer_signal(query_text: str, chunk: dict) -> float:
    if chunk.get("para_bm25_score", 0.0) > 0.0:
        return 1.0
    return max(
        _body_query_coverage(query_text, chunk),
        _section_overlap(query_text, chunk["section_title"]),
        _aspect_section_score(query_text, chunk["section_title"]),
    )


def _filter_weak_answer_chunks(chunks: list[dict], query_text: str) -> list[dict]:
    """
    Keep title-expansion recall from flooding the final context with article-relevant
    but question-irrelevant chunks when direct answer-bearing chunks exist.
    """
    if len(chunks) <= 1:
        return chunks

    signals = {c["chunk_id"]: _answer_signal(query_text, c) for c in chunks}
    has_direct = any(score >= _ANSWER_FILTER_COVERAGE for score in signals.values())
    if not has_direct:
        return chunks

    filtered = [
        c for c in chunks
        if signals.get(c["chunk_id"], 0.0) >= _ANSWER_FILTER_COVERAGE
        or (c.get("section_title") or "").lower() == "lead"
    ]
    return filtered or chunks


def _is_weak_title_only_chunk(query_text: str, chunk: dict) -> bool:
    """Reject title-only expansion hits unless they look topically anchored."""
    sources = set(chunk.get("candidate_sources", []))
    if sources != {"title_bm25"}:
        return False
    if _title_token_overlap(query_text, chunk.get("title", "")) >= _TITLE_ONLY_MIN_TITLE_OVERLAP:
        return False
    if _query_coverage(query_text, chunk) >= _TITLE_ONLY_MIN_QUERY_COVERAGE:
        return False
    return True


def _candidate_summary_text(chunk: dict) -> str:
    """Compact semantic summary for lightweight query-time reranking."""
    text = (chunk.get("raw_text") or chunk.get("text") or "").strip()
    summary = text[:_SEMANTIC_SUMMARY_CHARS].strip()
    if summary and len(text) > _SEMANTIC_SUMMARY_CHARS:
        summary = summary.rsplit(" ", 1)[0].strip() + " ..."
    parts = [
        f"Title: {chunk.get('title', '').strip()}",
        f"Section: {chunk.get('section_title', '').strip()}",
    ]
    if summary:
        parts.append(f"Summary: {summary}")
    return "\n".join(parts)


def _article_summary_text(chunks: list[dict]) -> str:
    """Build an article-level summary from the strongest available chunk."""
    if not chunks:
        return ""
    lead_chunk = next((c for c in chunks if (c.get("section_title") or "").lower() == "lead"), None)
    best_chunk = lead_chunk or max(chunks, key=lambda c: c.get("rerank_score", 0.0))
    return _candidate_summary_text(best_chunk)


def _semantic_rerank_articles(chunks: list[dict], query_vec: np.ndarray) -> list[dict]:
    """
    Query-time article rerank using article summaries.
    Embeddings are ephemeral and are not persisted to any index.
    """
    if len(chunks) <= 1 or _ARTICLE_SEMANTIC_RERANK_TOP_N <= 0 or _ARTICLE_SEMANTIC_RERANK_WEIGHT <= 0.0:
        return chunks

    article_map: dict[tuple[str, int], list[dict]] = {}
    for chunk in chunks:
        key = (chunk.get("zim_name", ""), int(chunk.get("article_id", 0)))
        article_map.setdefault(key, []).append(chunk)

    ranked_articles = sorted(
        article_map.items(),
        key=lambda item: max(c.get("rerank_score", 0.0) for c in item[1]),
        reverse=True,
    )
    head_articles = ranked_articles[:_ARTICLE_SEMANTIC_RERANK_TOP_N]
    article_texts = [_article_summary_text(article_chunks) for _, article_chunks in head_articles]

    try:
        article_vecs = _encode(article_texts)
    except Exception as e:
        print(f"[zim_retrieval] article semantic rerank skipped: {e}", flush=True)
        return chunks

    sims = np.dot(article_vecs, query_vec.astype(np.float32))
    sim_map: dict[tuple[str, int], float] = {
        key: float(score) for (key, _), score in zip(head_articles, sims, strict=False)
    }
    n_sim = _minmax(sim_map)
    base_map = {
        key: max(c.get("rerank_score", 0.0) for c in article_chunks)
        for key, article_chunks in head_articles
    }
    n_base = _minmax(base_map)

    article_score: dict[tuple[str, int], float] = {}
    for key, _article_chunks in head_articles:
        article_score[key] = (
            _ARTICLE_SEMANTIC_BASE_WEIGHT * n_base.get(key, 0.0)
            + (1.0 - _ARTICLE_SEMANTIC_BASE_WEIGHT) * n_sim.get(key, 0.0)
        )

    for chunk in chunks:
        key = (chunk.get("zim_name", ""), int(chunk.get("article_id", 0)))
        if key not in article_score:
            continue
        chunk["article_semantic_score"] = round(sim_map.get(key, 0.0), 5)
        chunk["rerank_score"] = round(
            (1.0 - _ARTICLE_SEMANTIC_RERANK_WEIGHT) * chunk.get("rerank_score", 0.0)
            + _ARTICLE_SEMANTIC_RERANK_WEIGHT * article_score[key],
            5,
        )

    chunks.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
    return chunks


def _semantic_rerank_candidates(chunks: list[dict], query_vec: np.ndarray) -> list[dict]:
    """
    Query-time semantic rerank over a small prose pool using title + section + snippet.
    Embeddings are ephemeral and are not persisted to any index.
    """
    if len(chunks) <= 1 or _SEMANTIC_RERANK_TOP_N <= 0 or _SEMANTIC_RERANK_WEIGHT <= 0.0:
        return chunks

    limit = min(len(chunks), _SEMANTIC_RERANK_TOP_N)
    head = list(chunks[:limit])
    tail = list(chunks[limit:])
    summaries = [_candidate_summary_text(c) for c in head]

    try:
        cand_vecs = _encode(summaries)
    except Exception as e:
        print(f"[zim_retrieval] semantic rerank skipped: {e}", flush=True)
        return chunks

    sims = np.dot(cand_vecs, query_vec.astype(np.float32))
    sim_map = {c["chunk_id"]: float(score) for c, score in zip(head, sims, strict=False)}
    n_sim = _minmax(sim_map)

    for c in head:
        cid = c["chunk_id"]
        c["semantic_summary_score"] = round(sim_map.get(cid, 0.0), 5)
        c["rerank_score"] = round(
            (1.0 - _SEMANTIC_RERANK_WEIGHT) * c.get("rerank_score", 0.0)
            + _SEMANTIC_RERANK_WEIGHT * n_sim.get(cid, 0.0),
            5,
        )

    head.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
    return head + tail


def _structure_prior(chunk: dict) -> float:
    section = (chunk.get("section_title") or "").lower()
    idx     = int(chunk.get("chunk_index", 0))
    score   = 0.20
    if section == "lead":
        score += 0.45
    elif section.startswith("infobox:"):
        score += 0.15
    elif any(k in section for k in ("overview", "summary", "background", "history")):
        score += 0.20
    elif any(k in section for k in ("references", "external links", "see also", "notes")):
        score -= 0.12
    score += max(0.0, 0.20 - min(idx, 8) * 0.025)
    return max(0.0, min(1.0, score))


def _mention_strength(query_text: str, chunk_text: str) -> float:
    """
    Multiplier [0.6, 1.0] penalising chunks where query terms appear only
    incidentally — single occurrence and/or only in the latter half of text.
    Chunks that contain the term early and repeatedly are not penalised.
    Chunks that don't contain the term at all (FAISS-only hits) are not penalised.
    """
    terms = set(_TOK_RE.findall(query_text.lower()))
    if not terms:
        return 1.0
    text_lower   = chunk_text.lower()
    early_cutoff = max(1, len(text_lower) // 2)
    multipliers  = []
    for term in terms:
        count = text_lower.count(term)
        if count == 0:
            continue  # not present — FAISS hit, no penalty
        m = 1.0
        if count == 1:
            m *= 0.82
        if term not in text_lower[:early_cutoff]:
            m *= 0.88
        multipliers.append(m)
    return sum(multipliers) / len(multipliers) if multipliers else 1.0


def _article_title_chunk_ids(con, article_scores: dict[int, float], per_article: int = _TITLE_ARTICLE_CHUNKS) -> list[int]:
    if not article_scores:
        return []
    article_ids = list(article_scores)
    ph = ",".join("?" * len(article_ids))
    rows = con.execute(
        f"SELECT id, article_id, chunk_index "
        f"FROM chunks "
        f"WHERE embedded = 1 AND article_id IN ({ph}) "
        f"ORDER BY article_id ASC, chunk_index ASC",
        article_ids,
    ).fetchall()
    out = []
    counts: dict[int, int] = {}
    for chunk_id, article_id, _chunk_index in rows:
        if counts.get(article_id, 0) >= per_article:
            continue
        counts[article_id] = counts.get(article_id, 0) + 1
        out.append(int(chunk_id))
    return out


def _fuse_candidates(chunks: list[dict],
                     query_text: str,
                     faiss_scores: dict[int, float],
                     para_bm25: dict[int, float],
                     title_bm25: dict[int, float]) -> dict[int, float]:
    fused: dict[int, float] = {}

    sem_rank = _rank_map({c["chunk_id"]: faiss_scores.get(c["chunk_id"], 0.0) for c in chunks if c["chunk_id"] in faiss_scores})
    par_rank = _rank_map({c["chunk_id"]: para_bm25.get(c["chunk_id"], 0.0) for c in chunks if c["chunk_id"] in para_bm25})
    title_rank = _rank_map({c["chunk_id"]: title_bm25.get(c["article_id"], 0.0) for c in chunks if c["article_id"] in title_bm25})
    sec_rank = _rank_map({c["chunk_id"]: _section_overlap(query_text, c["section_title"])
                          for c in chunks
                          if _section_overlap(query_text, c["section_title"]) > 0.0})
    aspect_rank = _rank_map({c["chunk_id"]: _aspect_section_score(query_text, c["section_title"])
                             for c in chunks
                             if _aspect_section_score(query_text, c["section_title"]) > 0.0})
    tit_rank = _rank_map({c["chunk_id"]: _title_token_overlap(query_text, c["title"])
                          for c in chunks
                          if _title_token_overlap(query_text, c["title"]) > 0.0})
    struct_rank = _rank_map({c["chunk_id"]: _structure_prior(c) for c in chunks})

    _rrf_add(fused, sem_rank, _RRF_W_SEMANTIC)
    _rrf_add(fused, par_rank, _RRF_W_PARA_BM25)
    _rrf_add(fused, title_rank, _RRF_W_TITLE)
    _rrf_add(fused, sec_rank, _RRF_W_SECTION)
    _rrf_add(fused, aspect_rank, _RRF_W_ASPECT)
    _rrf_add(fused, tit_rank, _RRF_W_TITLE_TOK)
    _rrf_add(fused, struct_rank, _RRF_W_STRUCTURE)
    return fused


def _rerank(chunks: list[dict],
            query_text: str,
            faiss_scores: dict[int, float],
            para_bm25:    dict[int, float],
            title_bm25:   dict[int, float],
            fused_scores: dict[int, float]) -> list[dict]:
    """
    Compute a second-stage heuristic rerank over the fused candidate pool.
    Mutates each chunk dict in-place to add 'fusion_score' and 'rerank_score'.
    """
    if not chunks:
        return []

    n_fused = _minmax({c["chunk_id"]: fused_scores.get(c["chunk_id"], 0.0) for c in chunks})
    n_sem = {c["chunk_id"]: _norm_faiss(faiss_scores.get(c["chunk_id"], 0.0))
             for c in chunks}
    para_raw    = {c["chunk_id"]: para_bm25.get(c["chunk_id"],    0.0) for c in chunks}
    title_raw   = {c["chunk_id"]: title_bm25.get(c["article_id"], 0.0) for c in chunks}
    section_raw = {c["chunk_id"]: _section_overlap(query_text, c["section_title"])
                   for c in chunks}
    aspect_raw = {c["chunk_id"]: _aspect_section_score(query_text, c["section_title"])
                  for c in chunks}
    title_tok_raw = {c["chunk_id"]: _title_token_overlap(query_text, c["title"])
                     for c in chunks}
    coverage_raw = {c["chunk_id"]: _query_coverage(query_text, c)
                    for c in chunks}
    struct_raw = {c["chunk_id"]: _structure_prior(c)
                  for c in chunks}

    n_par = _minmax(para_raw)
    n_tit = _minmax(title_raw)
    n_sec = _minmax(section_raw)
    n_aspect = _minmax(aspect_raw)
    n_title_tok = _minmax(title_tok_raw)
    n_cov = _minmax(coverage_raw)
    n_struct = _minmax(struct_raw)

    for c in chunks:
        cid = c["chunk_id"]
        c["fusion_score"] = round(fused_scores.get(cid, 0.0), 5)
        base = (
            _W_FUSION    * n_fused.get(cid, 0.0)
            + _W_SEMANTIC  * n_sem.get(cid, 0.0)
            + _W_PARA_BM25 * n_par.get(cid, 0.0)
            + _W_TITLE     * n_tit.get(cid, 0.0)
            + _W_SECTION   * n_sec.get(cid, 0.0)
            + _W_ASPECT    * n_aspect.get(cid, 0.0)
            + _W_TITLE_TOK * n_title_tok.get(cid, 0.0)
            + _W_COVERAGE  * n_cov.get(cid, 0.0)
            + _W_STRUCTURE * n_struct.get(cid, 0.0)
        )
        mention = _mention_strength(query_text, c.get("raw_text") or c.get("text", ""))
        c["rerank_score"] = round(base * mention, 5)

    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return chunks


def _rank_by_rrf(chunks: list[dict], fused_scores: dict[int, float]) -> list[dict]:
    """Use pure RRF fusion order as the final ranking."""
    for c in chunks:
        cid = c["chunk_id"]
        c["fusion_score"] = round(fused_scores.get(cid, 0.0), 5)
        c["rerank_score"] = c["fusion_score"]
    chunks.sort(key=lambda x: x["fusion_score"], reverse=True)
    return chunks


# ── Title-based on-demand retrieval ──────────────────────────────────────────

def _keyword_rank_chunks(chunks: list[dict], query_text: str) -> list[dict]:
    """Score fetched chunks by keyword overlap + structure prior, return sorted."""
    q = set(_TOK_RE.findall(query_text.lower()))
    if not q:
        return chunks
    for c in chunks:
        hay = set(_TOK_RE.findall((c.get("text", "") + " " + c.get("section_title", "")).lower()))
        kw_score  = len(q & hay) / len(q)
        structure = _structure_prior(c) if "chunk_index" in c else 0.3
        c["rerank_score"] = round(kw_score * 0.7 + structure * 0.3, 4)
    chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
    return chunks


def _title_fetch_chunks(handles: list, query_vec: np.ndarray, query_text: str,
                        already_fetched_paths: set[str]) -> list[dict]:
    """
    For each handle with a title index, find top article candidates,
    fetch their raw chunks from libzim, keyword-rank, and return chunk dicts.
    Skips articles already covered by FAISS search.
    """
    result: list[dict] = []

    for h in handles:
        if not h.enabled or not h.has_title_index:
            continue

        candidates = h.title_search(query_vec, top_k=_OD_TITLE_SEARCH_TOP)
        # Filter out articles already in FAISS results
        fresh = [c for c in candidates if c["path"] not in already_fetched_paths]
        if not fresh:
            continue

        fetched = 0
        for cand in fresh:
            if fetched >= _OD_TITLE_FETCH_COUNT:
                break
            path   = cand["path"]
            tscore = cand["score"]
            chunks = h.fetch_chunks(path)
            if not chunks:
                continue
            fetched += 1

            ranked = _keyword_rank_chunks(chunks, query_text)
            for c in ranked[:h.count]:
                result.append({
                    "chunk_id":          abs(hash(path + c["text"])) % (2**31),
                    "article_id":        abs(hash(path)) % (2**31),
                    "title":             c["title"],
                    "section_title":     c["section_title"],
                    "text":              c["text"],
                    "chunk_index":       c["chunk_index"],
                    "zim_name":          h.name,
                    "faiss_score":       0.0,
                    "para_bm25_score":   0.0,
                    "title_bm25_score":  tscore,
                    "rerank_score":      c.get("rerank_score", 0.4),
                    "fusion_score":      0.0,
                    "candidate_sources": ["title_search"],
                    "url":               path,
                    "type":              "prose",
                })

            # Optionally enqueue for background embedding
            if _OD_AUTO_EMBED and tscore >= _OD_AUTO_EMBED_THRESHOLD:
                try:
                    _embed_queue.put_nowait((h.name, path))
                except queue.Full:
                    pass

    return result


# ── Public search entry point ─────────────────────────────────────────────────

def search(query_text: str, top_k: int = 10,
           mode: str = "balanced",
           query_vec: np.ndarray | None = None,
           debug: bool = False,
           zim_name: str | None = None) -> list[dict] | tuple[list[dict], list[dict]]:
    """
    Search all loaded ZIM indexes.

    mode      — "fast" | "balanced" | "complex"
    query_vec — pre-computed (384,) L2-normalised embedding; pass to avoid
                re-embedding when the caller already has the vector.

    Returns list of chunk dicts sorted by rerank_score, up to top_k prose
    chunks + up to _INFOBOX_GLOBAL_MAX infobox chunks appended:
      text, title, url, section_title, chunk_index,
      rerank_score, faiss_score, zim_name
    """
    with _handles_lock:
        handles = list(_handles)

    active = [h for h in handles if h.enabled and (not zim_name or h.name == zim_name)]
    if not active:
        return ([], []) if debug else []

    cfg    = _MODE_CFGS.get(mode, _MODE_CFGS["balanced"])
    r      = cfg.get("retrieval", {})
    faiss_n = r.get("faiss_candidates", 40)
    bm25_n  = r.get("bm25_candidates",  30)
    top_k   = r.get("top_k", top_k)
    answer_filter = bool(r.get("answer_filter", mode == "balanced"))

    if query_vec is None:
        query_vec = _encode([query_text])[0]

    bm25_query     = extract_keywords(query_text)
    is_navigational = bm25_query != query_text.strip()

    all_prose:   list[dict] = []
    all_infobox: list[dict] = []
    debug_pools: list[dict] = []

    for h in active:
        if not h.is_indexed:
            xapian_chunks = h.xapian_search(query_text, count=h.count)
            all_prose.extend(xapian_chunks)
            continue

        # ── 1. Candidate gathering ────────────────────────────────────────────
        faiss_hits   = faiss_index.search(h.idx, query_vec, top_k=faiss_n)
        faiss_scores = {cid: score for cid, score in faiss_hits}

        para_hits = chunk_text_search(h.con, bm25_query, limit=bm25_n)
        para_bm25 = {cid: score for cid, score in para_hits}

        title_hits = title_search_scored(h.con, bm25_query, limit=20)
        title_bm25 = {art_id: score for art_id, score in title_hits}
        title_chunk_ids = _article_title_chunk_ids(h.con, title_bm25)

        candidate_ids = list(set(faiss_scores) | set(para_bm25) | set(title_chunk_ids))
        if not candidate_ids:
            continue

        # ── 2. Batch fetch metadata ────────────────────────────────────────────
        chunks = get_chunks_by_ids(h.con, candidate_ids)
        for c in chunks:
            c["zim_name"]    = h.name
            c["raw_text"]    = c.get("text", "")
            c["faiss_score"] = round(faiss_scores.get(c["chunk_id"], 0.0), 4)
            c["candidate_sources"] = [
                src for src, present in (
                    ("faiss", c["chunk_id"] in faiss_scores),
                    ("paragraph_bm25", c["chunk_id"] in para_bm25),
                    ("title_bm25", c["article_id"] in title_bm25),
                )
                if present
            ]
            c["para_bm25_score"] = round(para_bm25.get(c["chunk_id"], 0.0), 5)
            c["title_bm25_score"] = round(title_bm25.get(c["article_id"], 0.0), 5)

        # ── 3. Filter obvious non-matches ──────────────────────────────────────
        chunks = [
            c for c in chunks
            if faiss_scores.get(c["chunk_id"], 0.0) >= _MIN_FAISS_SCORE
            or c["chunk_id"] in para_bm25
            or c["article_id"] in title_bm25
        ]
        if not chunks:
            continue

        fused_scores = _fuse_candidates(chunks, bm25_query or query_text, faiss_scores, para_bm25, title_bm25)
        fused_ranked = sorted(chunks, key=lambda c: fused_scores.get(c["chunk_id"], 0.0), reverse=True)
        for rank, c in enumerate(fused_ranked, start=1):
            c["fusion_rank"] = rank
            c["fusion_score"] = round(fused_scores.get(c["chunk_id"], 0.0), 5)
        # Diversity cap: limit chunks per article so no single article
        # monopolises the rerank pool before scoring has a chance to run.
        _pool_target = max(top_k * _RERANK_POOL_MULTIPLIER, _RERANK_POOL_MIN)
        _seen_art: dict[int, int] = {}
        _diverse:  list[dict]    = []
        _overflow: list[dict]    = []
        for _c in fused_ranked:
            _aid = _c["article_id"]
            if _seen_art.get(_aid, 0) < _FUSION_MAX_PER_ARTICLE:
                _seen_art[_aid] = _seen_art.get(_aid, 0) + 1
                _diverse.append(_c)
            else:
                _overflow.append(_c)
        chunks = _diverse[:_pool_target]
        if len(chunks) < _pool_target:
            chunks.extend(_overflow[:_pool_target - len(chunks)])

        # ── 4. Split prose / infobox ───────────────────────────────────────────
        prose   = [c for c in chunks if not c["section_title"].startswith("Infobox:")]
        infobox = [c for c in chunks if     c["section_title"].startswith("Infobox:")]

        # ── 5. Rerank each bucket ──────────────────────────────────────────────
        rerank_query = bm25_query or query_text
        if _RERANKER == "rrf":
            prose   = _rank_by_rrf(prose, fused_scores)
            infobox = _rank_by_rrf(infobox, fused_scores)
        else:
            prose   = _rerank(prose,   rerank_query, faiss_scores, para_bm25, title_bm25, fused_scores)
            infobox = _rerank(infobox, rerank_query, faiss_scores, para_bm25, title_bm25, fused_scores)

        # ── 5b. Navigational adjustments — "tell me about X" queries ──────────
        # When extract_keywords stripped question framing, the user wants a
        # specific article. Two boosts apply:
        #   1. Suggestion boost — chunks from libzim title-matched articles rise
        #   2. Lead section boost — overview sections preferred for entity queries
        if is_navigational:
            nav_ids = h.suggestion_article_ids(bm25_query)
            needs_sort = False
            if nav_ids:
                for c in prose:
                    if c["article_id"] in nav_ids:
                        c["rerank_score"] = min(1.0, c["rerank_score"] + _SUGGESTION_BOOST)
                needs_sort = True
            for c in prose:
                if (c.get("section_title") or "").lower() == "lead":
                    c["rerank_score"] = min(1.0, c["rerank_score"] * _NAV_LEAD_BOOST)
                    needs_sort = True
            if needs_sort:
                prose.sort(key=lambda c: c["rerank_score"], reverse=True)

        if debug:
            ranked_debug = sorted(
                prose + infobox,
                key=lambda c: c.get("rerank_score", c.get("fusion_score", 0.0)),
                reverse=True,
            )
            debug_pools.append({
                "zim_name": h.name,
                "faiss_candidates": len(faiss_hits),
                "paragraph_bm25_candidates": len(para_hits),
                "title_bm25_articles": len(title_hits),
                "unique_candidate_chunks": len(candidate_ids),
                "post_filter_chunks": len(fused_ranked),
                "rerank_pool_chunks": len(ranked_debug),
                "chunks": [
                    {
                        "chunk_id": c.get("chunk_id"),
                        "article_id": c.get("article_id"),
                        "title": c.get("title", ""),
                        "section_title": c.get("section_title", ""),
                        "chunk_index": c.get("chunk_index"),
                        "candidate_sources": c.get("candidate_sources", []),
                        "faiss_score": c.get("faiss_score", 0.0),
                        "para_bm25_score": c.get("para_bm25_score", 0.0),
                        "title_bm25_score": c.get("title_bm25_score", 0.0),
                        "fusion_rank": c.get("fusion_rank"),
                        "fusion_score": c.get("fusion_score", 0.0),
                        "rerank_score": c.get("rerank_score", 0.0),
                        "text": c.get("raw_text") or c.get("text", ""),
                    }
                    for c in ranked_debug
                ],
            })

        # ── 6. Intra-article filter (prose only) ───────────────────────────────
        # Drop articles whose best rerank score is too far below the top.
        # Prevents semantically-adjacent articles (e.g. The Beatles for a MJ query)
        # from crowding out chunks from the dominant article.
        if prose:
            best_by_article: dict[int, float] = {}
            for c in prose:
                aid = c["article_id"]
                if c["rerank_score"] > best_by_article.get(aid, 0.0):
                    best_by_article[aid] = c["rerank_score"]
            top_rerank = max(best_by_article.values())
            cutoff     = top_rerank - _ARTICLE_SCORE_MARGIN
            prose = [c for c in prose
                     if best_by_article[c["article_id"]] >= cutoff]

        if answer_filter:
            prose = _filter_weak_answer_chunks(prose, bm25_query or query_text)
        prose = [c for c in prose if not _is_weak_title_only_chunk(query_text, c)]

        # Infobox: apply semantic/BM25 evidence + per-article cap. BM25 can be
        # the strongest signal for terse facts such as dates, names, and places.
        infobox = [c for c in infobox
                   if faiss_scores.get(c["chunk_id"], 0.0) >= _INFOBOX_MIN_SCORE
                   or c["chunk_id"] in para_bm25]
        seen_ib: dict[str, int] = {}
        capped_infobox = []
        for c in infobox:
            art = c["title"]
            if seen_ib.get(art, 0) < _INFOBOX_MAX_PER_ARTICLE:
                seen_ib[art] = seen_ib.get(art, 0) + 1
                capped_infobox.append(c)

        all_prose.extend(prose[:top_k])
        all_infobox.extend(capped_infobox)

    # ── Title-search path (for ZIMs with title index, covers unembedded articles) ──
    faiss_urls = {c.get("url", "") for c in all_prose}
    title_chunks = _title_fetch_chunks(active, query_vec, query_text, faiss_urls)
    all_prose.extend(title_chunks)

    if not all_prose and not all_infobox:
        return ([], debug_pools) if debug else []

    # ── 6. Sort across ZIMs ────────────────────────────────────────────────────
    all_prose.sort(  key=lambda x: x["rerank_score"], reverse=True)
    all_infobox.sort(key=lambda x: (x.get("rerank_score", 0.0), x["faiss_score"]), reverse=True)

    # ── 7. Cross-ZIM false-positive filter (prose only) ────────────────────────
    if all_prose:
        top_chunk = all_prose[0]
        prose_top = top_chunk["faiss_score"]
        top_rerank = top_chunk.get("rerank_score", 0.0)
        top_title_overlap = _title_token_overlap(query_text, top_chunk.get("title", ""))
        dominant_title_hit = (
            top_rerank >= _CROSS_ZIM_DOMINANCE_RERANK
            and top_title_overlap >= _CROSS_ZIM_TITLE_OVERLAP
        )
        if prose_top >= _CROSS_ZIM_STRONG_SCORE or dominant_title_hit:
            top_zim = top_chunk["zim_name"]
            faiss_threshold = prose_top - _CROSS_ZIM_MARGIN
            rerank_threshold = top_rerank - _CROSS_ZIM_KEEP_MARGIN

            def _keep_cross_zim_chunk(c: dict) -> bool:
                if c["zim_name"] == top_zim:
                    return True
                if c.get("faiss_score", 0.0) >= faiss_threshold:
                    return True
                if dominant_title_hit:
                    title_overlap = _title_token_overlap(query_text, c.get("title", ""))
                    coverage = _query_coverage(query_text, c)
                    if (
                        c.get("rerank_score", 0.0) >= rerank_threshold
                        and (
                            title_overlap >= _CROSS_ZIM_TITLE_OVERLAP
                            or coverage >= _TITLE_ONLY_MIN_QUERY_COVERAGE
                        )
                    ):
                        return True
                return False

            all_prose = [c for c in all_prose if _keep_cross_zim_chunk(c)]
            all_infobox = [c for c in all_infobox if _keep_cross_zim_chunk(c)]

    if all_prose:
        all_prose = _semantic_rerank_articles(all_prose, query_vec)
        all_prose = _semantic_rerank_candidates(all_prose, query_vec)

    # ── 8. Truncate long prose chunks ──────────────────────────────────────────
    for c in all_prose:
        if len(c["text"]) > _MAX_PROSE_CHARS:
            c["text"] = c["text"][:_MAX_PROSE_CHARS].rsplit(" ", 1)[0] + " …"

    # ── 9. Merge infoboxes into the first prose chunk for their article ────────
    # Infobox stands alone only if its article appears in prose — merge it in
    # so the LLM sees the structured facts alongside the article text, not as
    # a separate floating entry.
    prose_top_k = all_prose[:top_k]
    prose_article_ids = {c["article_id"] for c in prose_top_k}
    all_infobox = [c for c in all_infobox if c["article_id"] in prose_article_ids]

    # Best infobox facts per article (already sorted by rerank/semantic score)
    best_infobox: dict[int, list[str]] = {}
    for c in all_infobox:
        aid = c["article_id"]
        facts = best_infobox.setdefault(aid, [])
        if len(facts) < _INFOBOX_MAX_PER_ARTICLE:
            facts.append(c["text"])

    # Attach infobox text separately so the context formatter can preserve
    # chunk boundaries instead of blending structured facts into prose.
    seen_articles: set[int] = set()
    for c in prose_top_k:
        aid = c["article_id"]
        if aid not in seen_articles and aid in best_infobox:
            c["infobox_text"] = "\n".join(best_infobox[aid])
        seen_articles.add(aid)

    return (prose_top_k, debug_pools) if debug else prose_top_k


# ── Public on-demand API ──────────────────────────────────────────────────────

def embed_article(zim_name: str, path: str) -> dict:
    """
    Embed a specific article by ZIM name + article path.
    Returns {ok, chunks_added, error}.
    """
    with _handles_lock:
        handle = next((h for h in _handles if h.name == zim_name), None)
    if not handle:
        return {"ok": False, "error": f"ZIM '{zim_name}' not found"}
    if not handle.is_indexed:
        return {"ok": False, "error": f"ZIM '{zim_name}' has no chunk index (run full indexer first)"}
    n = handle.embed_article(path)
    if n == -1:
        return {"ok": False, "error": "Embedding failed or limit reached"}
    return {"ok": True, "chunks_added": n}


def build_title_index(zim_name: str, progress_cb=None) -> dict:
    """
    Build the title index for a ZIM. Blocking — call from a background thread.
    Returns {ok, indexed, error}.
    """
    with _handles_lock:
        handle = next((h for h in _handles if h.name == zim_name), None)
    if not handle:
        return {"ok": False, "error": f"ZIM '{zim_name}' not found"}
    try:
        from indexer.title_index import build
        n = build(handle.zim_path, progress_cb=progress_cb)
        # Invalidate cached title vecs so next search reloads from updated DB
        handle._title_vecs   = None
        handle._title_paths  = None
        handle._title_titles = None
        return {"ok": True, "indexed": n}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_embed_limits(zim_name: str) -> dict | None:
    """Return current on-demand embed usage for a ZIM, or None if not found."""
    with _handles_lock:
        handle = next((h for h in _handles if h.name == zim_name), None)
    return handle.embed_limits() if handle else None
