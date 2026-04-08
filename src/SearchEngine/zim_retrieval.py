"""
ZIM retrieval bridge for SearchEngine.

Loaded once at import time — discovers which configured ZIMs have a built
FAISS index and opens them.  Provides search() with three modes:

  fast     — FAISS + BM25, small candidate pool, lightweight reranker
  balanced — FAISS + BM25, medium pool, weighted reranker (default)
  complex  — FAISS + BM25, large pool, weighted reranker

All modes use the same reranker:
  final_score = 0.55 * semantic
              + 0.25 * paragraph_bm25
              + 0.10 * title_bm25
              + 0.10 * section_overlap
Each signal is min-max normalised within the candidate pool before weighting.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import numpy as np
from pathlib import Path

from zim_indexer import index as faiss_index
from zim_indexer.db import (
    open_db, init_fts,
    title_search_scored, chunk_text_search, get_chunks_by_ids,
)
from SearchEngine.config import CFG
from SearchEngine.embedding import _st_encode as _encode
from SearchEngine.keywords import extract_keywords

_MIN_FAISS_SCORE         = 0.45
_INFOBOX_MIN_SCORE       = 0.80
_INFOBOX_MAX_PER_ARTICLE = 3
_MAX_PROSE_CHARS         = 1200

# Reranker weights (signals are min-max normalised before combining)
_W_SEMANTIC  = 0.55
_W_PARA_BM25 = 0.25
_W_TITLE     = 0.10
_W_SECTION   = 0.10

ZIM_INDEX_BASE = Path(CFG.get("zim_index_base", "/library/zims/content"))
_MODES_DIR     = Path(__file__).parent / "modes"


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


# ── ZIM handle ────────────────────────────────────────────────────────────────

class _ZimHandle:
    __slots__ = ("name", "con", "idx")

    def __init__(self, name: str, db_path: Path, idx_path: Path):
        self.name = name
        self.con  = open_db(db_path)
        self.idx  = faiss_index.load_or_create(idx_path)
        init_fts(self.con)


_handles: list[_ZimHandle] = []


def _load_all():
    for zim in CFG.get("zims", []):
        name     = zim["name"]
        out_dir  = ZIM_INDEX_BASE / name
        db_path  = out_dir / "data.db"
        idx_path = out_dir / "faiss.index"
        if db_path.exists() and idx_path.exists() and idx_path.stat().st_size > 100:
            try:
                h = _ZimHandle(name, db_path, idx_path)
                _handles.append(h)
                print(f"[zim_retrieval] ✅ {name}: {h.idx.ntotal:,} vectors", flush=True)
            except Exception as e:
                print(f"[zim_retrieval] ⚠️  {name}: {e}", flush=True)
        else:
            print(f"[zim_retrieval] ⏭  {name}: no index yet — skipped", flush=True)


_load_all()


def is_available() -> bool:
    return bool(_handles)


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
    q = set(query.lower().split())
    s = set(section_title.lower().split())
    if not s:
        return 0.0
    return len(q & s) / max(len(q), len(s))


def _rerank(chunks: list[dict],
            query_text: str,
            faiss_scores: dict[int, float],
            para_bm25:    dict[int, float],
            title_bm25:   dict[int, float]) -> list[dict]:
    """
    Compute weighted rerank score for each chunk and sort descending.
    Mutates each chunk dict in-place to add 'rerank_score'.
    """
    if not chunks:
        return []

    # FAISS: fixed-range normalisation — stable regardless of candidate pool spread
    n_sem = {c["chunk_id"]: _norm_faiss(faiss_scores.get(c["chunk_id"], 0.0))
             for c in chunks}
    # BM25 / section: min-max within the pool (query-relative signals)
    para_raw    = {c["chunk_id"]: para_bm25.get(c["chunk_id"],    0.0) for c in chunks}
    title_raw   = {c["chunk_id"]: title_bm25.get(c["article_id"], 0.0) for c in chunks}
    section_raw = {c["chunk_id"]: _section_overlap(query_text, c["section_title"])
                   for c in chunks}
    n_par = _minmax(para_raw)
    n_tit = _minmax(title_raw)
    n_sec = _minmax(section_raw)

    for c in chunks:
        cid = c["chunk_id"]
        c["rerank_score"] = round(
            _W_SEMANTIC  * n_sem.get(cid, 0.0)
            + _W_PARA_BM25 * n_par.get(cid, 0.0)
            + _W_TITLE     * n_tit.get(cid, 0.0)
            + _W_SECTION   * n_sec.get(cid, 0.0),
            5,
        )

    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return chunks


# ── Public search entry point ─────────────────────────────────────────────────

def search(query_text: str, top_k: int = 10,
           mode: str = "balanced",
           query_vec: np.ndarray | None = None) -> list[dict]:
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
    if not _handles:
        return []

    cfg    = _MODE_CFGS.get(mode, _MODE_CFGS["balanced"])
    r      = cfg.get("retrieval", {})
    faiss_n = r.get("faiss_candidates", 40)
    bm25_n  = r.get("bm25_candidates",  30)
    top_k   = r.get("top_k", top_k)

    if query_vec is None:
        query_vec = _encode([query_text])[0]

    # FAISS gets the full query (model handles stop words via embeddings).
    # BM25 gets only content keywords — avoids stop words ("tell", "me", "about")
    # matching unrelated articles via FTS5.
    bm25_query = extract_keywords(query_text)

    all_prose:   list[dict] = []
    all_infobox: list[dict] = []

    for h in _handles:

        # ── 1. Candidate gathering ────────────────────────────────────────────
        faiss_hits   = faiss_index.search(h.idx, query_vec, top_k=faiss_n)
        faiss_scores = {cid: score for cid, score in faiss_hits}

        para_hits = chunk_text_search(h.con, bm25_query, limit=bm25_n)
        para_bm25 = {cid: score for cid, score in para_hits}

        title_hits = title_search_scored(h.con, bm25_query, limit=20)
        title_bm25 = {art_id: score for art_id, score in title_hits}

        candidate_ids = list(set(faiss_scores) | set(para_bm25))
        if not candidate_ids:
            continue

        # ── 2. Batch fetch metadata ────────────────────────────────────────────
        chunks = get_chunks_by_ids(h.con, candidate_ids)
        for c in chunks:
            c["zim_name"]    = h.name
            c["faiss_score"] = round(faiss_scores.get(c["chunk_id"], 0.0), 4)

        # ── 3. Filter obvious non-matches ──────────────────────────────────────
        chunks = [
            c for c in chunks
            if faiss_scores.get(c["chunk_id"], 0.0) >= _MIN_FAISS_SCORE
            or c["chunk_id"] in para_bm25
        ]
        if not chunks:
            continue

        # ── 4. Split prose / infobox ───────────────────────────────────────────
        prose   = [c for c in chunks if not c["section_title"].startswith("Infobox:")]
        infobox = [c for c in chunks if     c["section_title"].startswith("Infobox:")]

        # ── 5. Rerank each bucket ──────────────────────────────────────────────
        prose   = _rerank(prose,   query_text, faiss_scores, para_bm25, title_bm25)
        infobox = _rerank(infobox, query_text, faiss_scores, para_bm25, title_bm25)

        # ── 6. Intra-article filter (prose only) ───────────────────────────────
        # Drop articles whose best rerank score is more than 0.20 below the top.
        # Prevents semantically-adjacent articles (e.g. The Beatles for a MJ query)
        # from crowding out chunks from the dominant article.
        if prose:
            best_by_article: dict[int, float] = {}
            for c in prose:
                aid = c["article_id"]
                if c["rerank_score"] > best_by_article.get(aid, 0.0):
                    best_by_article[aid] = c["rerank_score"]
            top_rerank = max(best_by_article.values())
            cutoff     = top_rerank - 0.20
            prose = [c for c in prose
                     if best_by_article[c["article_id"]] >= cutoff]

        # Infobox: apply FAISS floor + per-article cap
        infobox = [c for c in infobox
                   if faiss_scores.get(c["chunk_id"], 0.0) >= _INFOBOX_MIN_SCORE]
        seen_ib: dict[str, int] = {}
        capped_infobox = []
        for c in infobox:
            art = c["title"]
            if seen_ib.get(art, 0) < _INFOBOX_MAX_PER_ARTICLE:
                seen_ib[art] = seen_ib.get(art, 0) + 1
                capped_infobox.append(c)

        all_prose.extend(prose[:top_k])
        all_infobox.extend(capped_infobox)

    if not all_prose and not all_infobox:
        return []

    # ── 6. Sort across ZIMs ────────────────────────────────────────────────────
    all_prose.sort(  key=lambda x: x["rerank_score"], reverse=True)
    all_infobox.sort(key=lambda x: x["faiss_score"],  reverse=True)

    # ── 7. Cross-ZIM false-positive filter (prose only) ────────────────────────
    if all_prose:
        prose_top = all_prose[0]["faiss_score"]
        if prose_top >= 0.80:
            top_zim   = all_prose[0]["zim_name"]
            threshold = prose_top - 0.05
            all_prose   = [c for c in all_prose
                           if c["zim_name"] == top_zim or c["faiss_score"] >= threshold]
            all_infobox = [c for c in all_infobox
                           if c["zim_name"] == top_zim or c["faiss_score"] >= threshold]

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

    # Best infobox per article (already sorted by faiss_score descending)
    best_infobox: dict[int, str] = {}
    for c in all_infobox:
        aid = c["article_id"]
        if aid not in best_infobox:
            best_infobox[aid] = c["text"]

    # Prepend infobox text to the first prose chunk of each article
    seen_articles: set[int] = set()
    for c in prose_top_k:
        aid = c["article_id"]
        if aid not in seen_articles and aid in best_infobox:
            c["text"] = best_infobox[aid] + "\n\n" + c["text"]
        seen_articles.add(aid)

    return prose_top_k
