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

_RRF_K                  = 60
_MIN_FAISS_SCORE        = 0.45  # min cosine score for regular chunks
_INFOBOX_MIN_SCORE      = 0.80  # raised: short infobox text is easy false positive
_INFOBOX_MAX_PER_ARTICLE = 3    # hard cap per article
_INFOBOX_GLOBAL_MAX     = 5     # hard cap across all articles
_MAX_PROSE_CHARS        = 1200  # truncate long chunks (e.g. multi-paragraph leads)
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

    # Scan all RRF candidates — two separate buckets
    regular_hits: list[dict] = []   # lead + section paragraphs, up to top_k
    infobox_hits: list[dict] = []   # infobox rows passing threshold, additive
    seen_regular: set[tuple] = set()
    infobox_count: dict[str, int] = {}

    all_cids_sorted = sorted(rrf, key=rrf.__getitem__, reverse=True)

    for cid in all_cids_sorted:

        chunk = get_chunk_by_id(h.con, cid)
        if not chunk:
            continue
        faiss_score = faiss_score_map.get(cid, 0.0)
        bm25_rank   = bm25_rank_map.get(cid, n)

        if chunk["section_title"].startswith("Infobox:"):
            # Infobox: only include if cosine similarity is high enough to be relevant
            if faiss_score < _INFOBOX_MIN_SCORE:
                continue
            art = chunk["title"]
            if infobox_count.get(art, 0) >= _INFOBOX_MAX_PER_ARTICLE:
                continue
            infobox_count[art] = infobox_count.get(art, 0) + 1
            infobox_hits.append({
                **chunk,
                "rrf_score":   round(rrf[cid], 6),
                "faiss_score": round(faiss_score, 4),
                "zim_name":    h.name,
            })
        else:
            # Regular chunks: fill top_k slots
            if len(regular_hits) >= top_k:
                continue
            if faiss_score < _MIN_FAISS_SCORE and not (bm25_rank == 0 and faiss_score > 0.3):
                continue
            key = (chunk["title"], chunk["section_title"])
            if key in seen_regular:
                continue
            seen_regular.add(key)
            regular_hits.append({
                **chunk,
                "rrf_score":   round(rrf[cid], 6),
                "faiss_score": round(faiss_score, 4),
                "zim_name":    h.name,
            })

    # Infobox rows first (short facts), then regular prose chunks
    return infobox_hits + regular_hits


# ── Complex: on-the-fly full article expansion ────────────────────────────────

def _expand_article(h: _ZimHandle, article_id: int,
                    query_vec: np.ndarray, top_n: int,
                    prose_only: bool = False) -> list[dict]:
    """
    Fetch phase-1 embedded chunks for article_id, cosine-rank, return top_n.
    prose_only=True skips infobox rows (they're handled in a separate bucket).
    """
    rows = h.con.execute(
        "SELECT c.id, c.section_title, c.chunk_index, c.text, a.title, a.url "
        "FROM chunks c JOIN articles a ON c.article_id = a.id "
        "WHERE c.article_id = ? AND c.embedded = 1 ORDER BY c.chunk_index",
        (article_id,),
    ).fetchall()

    if not rows:
        return []

    article_title = rows[0][4]
    print(f"[expand] '{article_title}' — {len(rows)} embedded chunks"
          f"{' (prose only)' if prose_only else ''}", flush=True)

    texts = [r[3] for r in rows]
    vecs  = _encode(texts)                       # (N, 384) L2-normalised
    sims  = (vecs @ query_vec).tolist()          # cosine scores

    ranked = sorted(zip(sims, rows), key=lambda x: x[0], reverse=True)

    seen: set[str] = set()
    hits = []
    for sim, row in ranked:
        if len(hits) >= top_n:
            break
        section = row[1]
        if prose_only and section.startswith("Infobox:"):
            continue                              # infobox handled by RRF bucket
        key = section  # dedup so one section doesn't fill all slots
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

    # Keep prose (lead + section paragraphs) and infobox rows as separate buckets.
    # Prose fills top_k slots; infobox is additive on top — never crowds out prose.
    all_prose:   list[dict] = []
    all_infobox: list[dict] = []

    def _split(hits: list[dict]) -> tuple[list[dict], list[dict]]:
        prose   = [h for h in hits if not h["section_title"].startswith("Infobox:")]
        infobox = [h for h in hits if     h["section_title"].startswith("Infobox:")]
        return prose, infobox

    for h in _handles:
        if mode == "complex":
            expansion_top     = r.get("expansion_top_chunks",       5)
            high_conf_thresh  = r.get("high_conf_threshold",     0.75)
            cross_article_slots = r.get("cross_article_slots",      2)

            rrf_hits = _rrf_search(h, query_vec, faiss_n, bm25_n,
                                   query_text, top_k)
            if not rrf_hits:
                continue

            top_score = rrf_hits[0]["faiss_score"]
            expand_n  = 1 if top_score >= high_conf_thresh else 2
            print(f"[complex] top faiss={top_score:.3f} → expanding {expand_n} article(s)",
                  flush=True)

            expanded_ids: list[int] = []
            expanded_hits: list[dict] = []
            for hit in rrf_hits:
                aid = hit["article_id"]
                if aid not in expanded_ids:
                    expanded_ids.append(aid)
                    expanded_hits.extend(
                        _expand_article(h, aid, query_vec, expansion_top,
                                        prose_only=True)
                    )
                if len(expanded_ids) >= expand_n:
                    break

            expanded_hits.sort(key=lambda x: x["faiss_score"], reverse=True)
            expansion_slots = max(top_k - cross_article_slots, top_k // 2)
            final_prose: list[dict] = expanded_hits[:expansion_slots]

            # Fill cross-article slots from OTHER articles (prose only)
            seen_aids = set(expanded_ids)
            rrf_prose, rrf_infobox = _split(rrf_hits)
            for hit in rrf_prose:
                if len(final_prose) >= top_k:
                    break
                if hit["article_id"] not in seen_aids:
                    final_prose.append(hit)

            all_prose.extend(final_prose)
            all_infobox.extend(rrf_infobox)

        elif mode == "balanced":
            rrf_hits = _rrf_search(h, query_vec, faiss_n, bm25_n,
                                   query_text, top_k)
            if not rrf_hits:
                continue
            rrf_prose, rrf_infobox = _split(rrf_hits)
            expansion_threshold = r.get("expansion_threshold", 0.75)
            expansion_top       = r.get("expansion_top_chunks", 2)
            top_score = rrf_hits[0]["faiss_score"]
            if top_score >= expansion_threshold:
                print(f"[balanced] top faiss={top_score:.3f} ≥ {expansion_threshold} "
                      f"→ expanding top article (prose only)", flush=True)
                exp_prose = _expand_article(h, rrf_hits[0]["article_id"],
                                            query_vec, expansion_top,
                                            prose_only=True)
                # Best expanded prose first, then remaining RRF prose
                all_prose.extend(exp_prose + rrf_prose)
            else:
                all_prose.extend(rrf_prose)
            all_infobox.extend(rrf_infobox)

        else:  # fast
            hits = _rrf_search(h, query_vec, faiss_n, bm25_n,
                               query_text, top_k)
            prose, infobox = _split(hits)
            all_prose.extend(prose)
            all_infobox.extend(infobox)

    # Sort each bucket by score independently
    all_prose.sort(key=lambda x:   (x["faiss_score"], x["rrf_score"]), reverse=True)
    all_infobox.sort(key=lambda x: x["faiss_score"], reverse=True)

    # Cross-ZIM false positive filter: use the TRUE top score across both buckets
    # (expansion prose often scores lower than infobox, so we must check infobox too)
    prose_top   = all_prose[0]["faiss_score"]   if all_prose   else 0.0
    infobox_top = all_infobox[0]["faiss_score"] if all_infobox else 0.0
    global_top  = max(prose_top, infobox_top)

    if global_top >= 0.80:
        # Determine the dominant ZIM (whichever produced the top score)
        if infobox_top >= prose_top and all_infobox:
            top_zim = all_infobox[0]["zim_name"]
        else:
            top_zim = all_prose[0]["zim_name"]

        threshold = global_top - 0.05
        all_prose   = [h for h in all_prose
                       if h["zim_name"] == top_zim or h["faiss_score"] >= threshold]
        all_infobox = [h for h in all_infobox
                       if h["zim_name"] == top_zim or h["faiss_score"] >= threshold]

    # Truncate long prose chunks (e.g. multi-paragraph leads stored as one block)
    for h in all_prose:
        if len(h["text"]) > _MAX_PROSE_CHARS:
            h["text"] = h["text"][:_MAX_PROSE_CHARS].rsplit(" ", 1)[0] + " …"

    # Infobox only for articles that also have prose — never from unrelated articles
    prose_article_ids = {h["article_id"] for h in all_prose[:top_k]}
    all_infobox = [h for h in all_infobox if h["article_id"] in prose_article_ids]

    # Prose fills top_k slots; infobox appended as additive context (global cap)
    return all_prose[:top_k] + all_infobox[:_INFOBOX_GLOBAL_MAX]
