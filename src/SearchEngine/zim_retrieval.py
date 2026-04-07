"""
ZIM retrieval bridge for SearchEngine.

Loaded once at import time — discovers which configured ZIMs have a built
FAISS index and opens them.  Provides search() with three modes:

  fast     — FAISS + BM25 + RRF, small candidate pool, no expansion
  balanced — FAISS + BM25 + RRF, medium pool, Phase 1 chunks only (default)
  complex  — FAISS + BM25 + RRF, large pool, then embed ALL sections of the
             top article on-the-fly and cosine-rank for best chunks

Uses SearchEngine's shared embed model (Snowflake/snowflake-arctic-embed-xs, 384-dim)
— same instance as the semantic cache, so only one ONNX model loads at startup.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import numpy as np
from pathlib import Path

from zim_indexer import index as faiss_index
from zim_indexer.db import (
    open_db, init_fts, title_search,
    get_chunks_for_article, get_chunk_by_id,
)
from SearchEngine.config import CFG
from SearchEngine.embedding import _st_encode as _encode

_RRF_K         = 60
_MIN_FAISS_SCORE = 0.55   # chunks below this cosine score are semantically irrelevant
ZIM_INDEX_BASE = Path(CFG.get("zim_index_base", "/library/zims/content"))
_MODES_DIR     = Path(__file__).parent / "modes"

# ── Load mode configs ─────────────────────────────────────────────────────────

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


# ── Core RRF search ───────────────────────────────────────────────────────────

def _rrf_search(h: _ZimHandle, query_vec: np.ndarray,
                faiss_n: int, bm25_n: int, query_text: str,
                top_k: int) -> list[dict]:
    """
    FAISS + BM25 + RRF for a single ZIM handle.
    Returns deduped list of chunk dicts sorted by rrf_score desc.
    """
    # FAISS semantic
    faiss_results   = faiss_index.search(h.idx, query_vec, top_k=faiss_n)
    faiss_rank_map  = {cid: rank for rank, (cid, _) in enumerate(faiss_results)}
    faiss_score_map = {cid: score for cid, score in faiss_results}

    # BM25 title → expand to chunk IDs
    bm25_articles  = title_search(h.con, query_text, limit=bm25_n)
    bm25_rank_map: dict[int, int] = {}
    for article_id, article_rank in bm25_articles:
        for chunk_id, _ in get_chunks_for_article(h.con, article_id):
            if chunk_id not in bm25_rank_map:
                bm25_rank_map[chunk_id] = article_rank

    # RRF fusion
    all_cids = set(faiss_rank_map) | set(bm25_rank_map)
    n        = len(all_cids)
    rrf = {
        cid: (1 / (_RRF_K + faiss_rank_map.get(cid, n)) +
              1 / (_RRF_K + bm25_rank_map.get(cid, n)))
        for cid in all_cids
    }
    top_cids = sorted(rrf, key=rrf.__getitem__, reverse=True)[:top_k * 2]

    hits  = []
    seen: set[tuple] = set()
    for cid in top_cids:
        if len(hits) >= top_k:
            break
        chunk = get_chunk_by_id(h.con, cid)
        if not chunk:
            continue
        faiss_score = faiss_score_map.get(cid, 0.0)
        bm25_rank   = bm25_rank_map.get(cid, n)
        # Keep if FAISS score is decent (semantic match),
        # OR BM25 ranked the article #1 (exact title match) with some FAISS signal.
        if faiss_score < _MIN_FAISS_SCORE and not (bm25_rank == 0 and faiss_score > 0.3):
            continue
        key = (chunk["title"], chunk["section_title"])
        if key in seen:
            continue
        seen.add(key)
        hits.append({
            **chunk,
            "rrf_score":   round(rrf[cid], 6),
            "faiss_score": round(faiss_score, 4),
            "zim_name":    h.name,
        })
    return hits


# ── Complex: on-the-fly full article expansion ────────────────────────────────

def _expand_article(h: _ZimHandle, article_id: int,
                    query_vec: np.ndarray, top_n: int) -> list[dict]:
    """
    Fetch ALL stored sections for article_id from SQLite,
    embed them in one batch, cosine-rank, return top_n best chunks.
    """
    rows = h.con.execute(
        "SELECT c.id, c.section_title, c.chunk_index, c.text, a.title, a.url "
        "FROM chunks c JOIN articles a ON c.article_id = a.id "
        "WHERE c.article_id = ? ORDER BY c.chunk_index",
        (article_id,),
    ).fetchall()

    if not rows:
        return []

    texts = [r[3] for r in rows]
    vecs  = _encode(texts)                       # (N, 384) L2-normalised
    sims  = (vecs @ query_vec).tolist()          # cosine scores

    ranked = sorted(zip(sims, rows), key=lambda x: x[0], reverse=True)

    seen: set[str] = set()
    hits = []
    for sim, row in ranked:
        if len(hits) >= top_n:
            break
        key = row[2]  # section_title
        if key in seen:
            continue
        seen.add(key)
        hits.append({
            "chunk_id":      row[0],
            "article_id":    article_id,
            "section_title": row[1],
            "chunk_index":   row[2],
            "text":          row[3],
            "title":         row[4],
            "url":           row[5],
            "rrf_score":     0.0,
            "faiss_score":   round(sim, 4),
            "zim_name":      h.name,
        })
    return hits


# ── Public search entry point ─────────────────────────────────────────────────

def search(query_text: str, top_k: int = 10,
           mode: str = "balanced") -> list[dict]:
    """
    Search all loaded ZIM indexes.

    mode: "fast" | "balanced" | "complex"

    Returns list of chunk dicts sorted by rrf_score (faiss_score for complex
    expansion chunks), up to top_k results:
      text, title, url, section_title, chunk_index,
      rrf_score, faiss_score, zim_name
    """
    if not _handles:
        return []

    cfg = _MODE_CFGS.get(mode, _MODE_CFGS["balanced"])
    r   = cfg.get("retrieval", {})

    faiss_n = r.get("faiss_candidates", 40)
    bm25_n  = r.get("bm25_candidates",  30)
    top_k   = r.get("top_k", top_k)

    query_vec = _encode([query_text])[0]   # (384,) L2-normalised

    all_hits: list[dict] = []

    for h in _handles:
        if mode == "complex":
            expand_n       = r.get("expand_top_n_articles", 1)
            expansion_top  = r.get("expansion_top_chunks",  5)

            # Stage 1: RRF to find the top article(s)
            rrf_hits = _rrf_search(h, query_vec, faiss_n, bm25_n,
                                   query_text, top_k)

            # Stage 2: expand top N articles with full on-the-fly embedding
            expanded_ids: list[int] = []
            expanded_hits: list[dict] = []
            for hit in rrf_hits:
                aid = hit["article_id"]
                if aid not in expanded_ids:
                    expanded_ids.append(aid)
                    expanded_hits.extend(
                        _expand_article(h, aid, query_vec, expansion_top)
                    )
                if len(expanded_ids) >= expand_n:
                    break

            # Fill remaining slots from RRF hits (other articles)
            seen_aids = set(expanded_ids)
            for hit in rrf_hits:
                if hit["article_id"] not in seen_aids:
                    expanded_hits.append(hit)

            # Sort by faiss_score (expansion chunks) then rrf_score
            expanded_hits.sort(
                key=lambda x: (x["faiss_score"], x["rrf_score"]), reverse=True
            )
            all_hits.extend(expanded_hits[:top_k])

        else:
            hits = _rrf_search(h, query_vec, faiss_n, bm25_n,
                               query_text, top_k)
            all_hits.extend(hits)

    # Global sort when multiple ZIMs contribute
    all_hits.sort(key=lambda x: (x["faiss_score"], x["rrf_score"]), reverse=True)
    return all_hits[:top_k]
