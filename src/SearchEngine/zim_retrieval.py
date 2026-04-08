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
  2. Heuristic reranking over the fused pool.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import numpy as np
import re
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
_RRF_K                   = 60
_TITLE_ARTICLE_CHUNKS    = 6

# Stage-2 heuristic reranker weights
_W_FUSION    = 0.34
_W_SEMANTIC  = 0.24
_W_PARA_BM25 = 0.16
_W_TITLE     = 0.08
_W_SECTION   = 0.08
_W_TITLE_TOK = 0.04
_W_COVERAGE  = 0.03
_W_STRUCTURE = 0.03

ZIM_INDEX_BASE = Path(CFG.get("zim_index_base", "/library/zims/content"))
_MODES_DIR     = Path(__file__).parent / "modes"
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

    _rrf_add(fused, sem_rank, 1.0)
    _rrf_add(fused, par_rank, 1.0)
    _rrf_add(fused, title_rank, 0.9)
    _rrf_add(fused, sec_rank, 0.6)
    _rrf_add(fused, aspect_rank, 0.9)
    _rrf_add(fused, tit_rank, 0.5)
    _rrf_add(fused, struct_rank, 0.3)
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
        c["rerank_score"] = round(
            _W_FUSION    * n_fused.get(cid, 0.0)
            + _W_SEMANTIC  * n_sem.get(cid, 0.0)
            + _W_PARA_BM25 * n_par.get(cid, 0.0)
            + _W_TITLE     * n_tit.get(cid, 0.0)
            + _W_SECTION   * n_sec.get(cid, 0.0),
            5,
        )
        c["rerank_score"] = round(
            c["rerank_score"]
            + _W_SECTION   * n_aspect.get(cid, 0.0)
            + _W_TITLE_TOK * n_title_tok.get(cid, 0.0)
            + _W_COVERAGE  * n_cov.get(cid, 0.0)
            + _W_STRUCTURE * n_struct.get(cid, 0.0),
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
        title_chunk_ids = _article_title_chunk_ids(h.con, title_bm25)

        candidate_ids = list(set(faiss_scores) | set(para_bm25) | set(title_chunk_ids))
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
            or c["article_id"] in title_bm25
        ]
        if not chunks:
            continue

        fused_scores = _fuse_candidates(chunks, bm25_query or query_text, faiss_scores, para_bm25, title_bm25)
        fused_ranked = sorted(chunks, key=lambda c: fused_scores.get(c["chunk_id"], 0.0), reverse=True)
        chunks = fused_ranked[:max(top_k * 6, 24)]

        # ── 4. Split prose / infobox ───────────────────────────────────────────
        prose   = [c for c in chunks if not c["section_title"].startswith("Infobox:")]
        infobox = [c for c in chunks if     c["section_title"].startswith("Infobox:")]

        # ── 5. Rerank each bucket ──────────────────────────────────────────────
        prose   = _rerank(prose,   query_text, faiss_scores, para_bm25, title_bm25, fused_scores)
        infobox = _rerank(infobox, query_text, faiss_scores, para_bm25, title_bm25, fused_scores)

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
