"""
RAG pipelines — chunk building, standard /ai pipeline, and advanced /ai-mode pipeline.

rag_pipeline        — standard: fetch top N articles → BM25+cosine chunk rank
rag_pipeline_advanced — progressive: title embed → semantic title rank →
                        first-para rerank → full chunk pipeline on winners only
"""
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from retrieval.kiwix_client import fetch_article_sections
from retrieval.rerank import rank_chunks
from ingest.article_cleaner import last_sentence, first_sentence

from SearchEngine.config import RAG, AI_MODE_CFG, NAV_RE
from SearchEngine.embedding import embed, embed_batch_ai_mode, embedder, ai_mode_embedder
from SearchEngine.keywords import extract_keywords

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def _split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_RE.split(text) if p.strip()]
    return parts or [text]


def _section_kind(section: str) -> str:
    sec = (section or "").strip().lower()
    if sec == "introduction":
        return "lead"
    if any(k in sec for k in ("history", "background", "overview", "summary")):
        return "overview"
    if any(k in sec for k in ("references", "external links", "see also", "notes")):
        return "support"
    return "body"


def _semantic_units(paragraphs: list[str], soft_chars: int, min_chars: int) -> list[str]:
    """
    Build chunkable units from paragraph text while preserving semantic boundaries:
    - never cross section boundaries here
    - split long paragraphs on sentence boundaries
    - merge very short adjacent units inside the same section
    """
    units: list[str] = []
    for para in paragraphs:
        para = (para or "").strip()
        if not para:
            continue
        if len(para) <= soft_chars:
            units.append(para)
            continue

        sentences = _split_sentences(para)
        if len(sentences) <= 1:
            units.append(para)
            continue

        buf: list[str] = []
        buf_len = 0
        for sent in sentences:
            sent_len = len(sent) + (1 if buf else 0)
            if buf and buf_len + sent_len > soft_chars:
                units.append(" ".join(buf).strip())
                buf = [sent]
                buf_len = len(sent)
            else:
                buf.append(sent)
                buf_len += sent_len
        if buf:
            units.append(" ".join(buf).strip())

    merged: list[str] = []
    i = 0
    while i < len(units):
        current = units[i]
        if len(current) < min_chars and i + 1 < len(units):
            current = (current + " " + units[i + 1]).strip()
            i += 1
        merged.append(current)
        i += 1
    return merged


def _format_chunk(title: str, section: str, kind: str, unit_idx: int, unit_total: int,
                  left: str, body: str, right: str) -> str:
    structure = f"Structure: kind={kind}; position={unit_idx + 1}/{unit_total}"
    text = f"{left}{body}{right}".strip()
    return f"[{title} | {section}]\n{structure}\n{text}"


def build_chunks(title: str, sections: list, chunk_cfg: dict | None = None) -> list:
    cfg         = chunk_cfg or RAG
    use_overlap = cfg.get("chunk_overlap", True)
    min_chars   = cfg.get("min_paragraph_chars", 50)
    soft_chars  = cfg.get("max_chunk_chars", 800)
    chunks = []
    for sec in sections:
        section = sec["section"]
        kind = _section_kind(section)
        units = _semantic_units(sec.get("paragraphs") or [], soft_chars, min_chars)
        for i, unit in enumerate(units):
            if len(unit) < min_chars:
                continue
            left  = (last_sentence(units[i - 1]) + " ") if i > 0 and use_overlap else ""
            right = (" " + first_sentence(units[i + 1])) if i < len(units) - 1 and use_overlap else ""
            chunks.append(_format_chunk(title, section, kind, i, len(units), left, unit, right))
    return chunks


def rag_pipeline(query: str, ranked_results: list, mark=None) -> tuple:
    """
    Fetch + chunk top N articles, rank chunks with BM25+cosine+section bias.
    Returns (top_chunks: list[str], first_para: str).

    Speed levers (config.yaml rag section):
      use_embeddings: false  → BM25-only chunk ranking, no embed calls
      max_chunks_per_article → caps chunk pool size before BM25 filter
      top_articles           → fewer articles = smaller pool
    """
    top_n           = RAG.get("top_articles", 3)
    top_k           = RAG.get("top_chunks", 8)
    max_chars       = RAG.get("max_chunk_chars", 800)
    max_per_article = RAG.get("max_chunks_per_article", 20)
    use_embeddings  = RAG.get("use_embeddings", True)

    content_results = [r for r in ranked_results if not NAV_RE.search(r["title"])]
    top_articles    = (content_results or ranked_results)[:top_n]

    def _fetch(r):
        return r["title"], fetch_article_sections(r["url"], max_chars)

    # Parallel: embed query at the same time as fetching articles
    sections_by_title = {}
    query_vec = None
    with ThreadPoolExecutor(max_workers=top_n + 1) as ex:
        query_future    = ex.submit(embed, query) if use_embeddings else None
        article_futures = {ex.submit(_fetch, r): r for r in top_articles}
        for fut in as_completed(article_futures):
            title, sections = fut.result()
            sections_by_title[title] = sections
        if query_future:
            query_vec = query_future.result()

    if mark:
        art_summary = " | ".join(
            f"{t!r} {sum(len(s['paragraphs']) for s in sections_by_title.get(t, []))}p"
            for t in [a["title"] for a in top_articles]
        )
        mark("3_fetch_articles", art_summary)

    first_para = ""
    top_secs   = sections_by_title.get(top_articles[0]["title"], [])
    if top_secs and top_secs[0]["paragraphs"]:
        first_para = top_secs[0]["paragraphs"][0]

    all_chunks = []
    for art in top_articles:
        sections = sections_by_title.get(art["title"], [])
        all_chunks.extend(build_chunks(art["title"], sections, RAG)[:max_per_article])

    if mark:
        mark("4_build_chunks", f"{len(all_chunks)} chunks")

    if not all_chunks:
        return [], first_para

    keywords     = extract_keywords(query)
    title_scores = {r["title"]: r.get("_title_score", 1.0) for r in ranked_results}
    if use_embeddings:
        top_chunks = rank_chunks(keywords, all_chunks, top_k, embedder, query_vec, title_scores=title_scores)
    else:
        top_chunks = rank_chunks(keywords, all_chunks, top_k, title_scores=title_scores)

    if mark:
        mark("5_rank_chunks", f"{len(top_chunks)} selected from {len(all_chunks)}")
        for i, c in enumerate(top_chunks):
            header  = c.split("\n")[0]
            preview = c.split("\n", 1)[1][:80].replace("\n", " ") if "\n" in c else c[:80]
            print(f"  [{i+1:02d}] {header}  {preview!r}", flush=True)

    return top_chunks, first_para


def rag_pipeline_advanced(query: str, query_vec: np.ndarray, ranked_results: list, mark=None) -> list:
    """
    Progressive AI Mode RAG pipeline:

    Stage A (parallel):
      - Embed all candidate title texts in one batch
      - Fetch full article sections for all candidates (parallel HTTP)

    Stage B: Semantic title ranking  →  reorder candidates by cosine(title, query)

    Stage C: First-paragraph rerank  →  embed first para of top N candidates,
             pick article_rerank_top winners by cosine(first_para, query)

    Stage D: Full chunk pipeline on winners only
             (BM25 → cosine → section-bias → dedup)

    All section fetches happen once in Stage A — no re-fetch needed in Stage D.
    BM25 title scores from rank_by_title are blended with semantic title cosines
    as title_scores for the chunk reranker.
    """
    n_candidates    = AI_MODE_CFG.get("title_rank_candidates", 8)
    n_para_rerank   = AI_MODE_CFG.get("para_rerank_candidates", 5)
    n_winners       = AI_MODE_CFG.get("article_rerank_top", 2)
    top_k           = AI_MODE_CFG.get("top_chunks", 8)
    max_chars       = AI_MODE_CFG.get("max_chunk_chars", 600)
    max_per_article = AI_MODE_CFG.get("max_chunks_per_article", 5)

    content_results = [r for r in ranked_results if not NAV_RE.search(r["title"])]
    candidates      = (content_results or ranked_results)[:n_candidates]

    # ── Stage A: parallel title-embed batch + section fetch ───────────────────
    def _fetch_sections(r):
        return r["title"], fetch_article_sections(r["url"], max_chars)

    sections_by_title = {}
    # Embed title + snippet so the model sees article content, not just the title string.
    title_texts = [r["title"] + ". " + r.get("snippet", "") for r in candidates]
    with ThreadPoolExecutor(max_workers=n_candidates + 1) as ex:
        title_embed_fut = ex.submit(embed_batch_ai_mode, title_texts)
        fetch_futs      = {ex.submit(_fetch_sections, r): r for r in candidates}
        title_vecs      = title_embed_fut.result()    # (N, dim)
        for fut in as_completed(fetch_futs):
            title, sections = fut.result()
            sections_by_title[title] = sections

    if mark:
        mark("3_title_embed_fetch",
             f"{len(candidates)} titles embedded | sections fetched in parallel")
        for r in candidates:
            secs = sections_by_title.get(r["title"], [])
            print(f"  → {r['title']!r:<45} {len(secs)} sec  "
                  f"{sum(len(s['paragraphs']) for s in secs)} para", flush=True)

    # ── Stage B: semantic title ranking ───────────────────────────────────────
    title_cosines = title_vecs @ query_vec    # (N,)
    title_order   = np.argsort(title_cosines)[::-1]
    sem_ranked    = [candidates[i] for i in title_order]

    if mark:
        mark("4_semantic_title_rank", f"best: {sem_ranked[0]['title']!r}")
        for i, idx in enumerate(title_order):
            print(f"  [{i+1}] cos={title_cosines[idx]:.4f}  {candidates[idx]['title']!r}", flush=True)

    # ── Stage C: first-paragraph rerank ───────────────────────────────────────
    para_pool   = sem_ranked[:n_para_rerank]
    first_paras = []
    for r in para_pool:
        secs = sections_by_title.get(r["title"], [])
        para = secs[0]["paragraphs"][0] if secs and secs[0]["paragraphs"] else r["title"]
        first_paras.append(para)

    para_vecs    = embed_batch_ai_mode(first_paras)
    para_cosines = para_vecs @ query_vec
    para_order   = np.argsort(para_cosines)[::-1]
    winners      = [para_pool[i] for i in para_order[:n_winners]]

    if mark:
        mark("5_first_para_rerank", f"winners → {[w['title'] for w in winners]}")
        for i, idx in enumerate(para_order[:n_winners]):
            print(f"  [{i+1}] cos={para_cosines[idx]:.4f}  {sem_ranked[idx]['title']!r}", flush=True)

    # ── Stage D: full chunk pipeline on winners ───────────────────────────────
    all_chunks = []
    for art in winners:
        sections = sections_by_title.get(art["title"], [])
        all_chunks.extend(build_chunks(art["title"], sections, AI_MODE_CFG)[:max_per_article])

    if mark:
        mark("6_build_chunks", f"{len(all_chunks)} chunks from {len(winners)} articles")

    if not all_chunks:
        return []

    # Blend BM25 title scores with semantic cosines for chunk reranker
    title_scores = {r["title"]: r.get("_title_score", 1.0) for r in ranked_results}
    for i, idx in enumerate(title_order):
        r    = candidates[idx]
        sem  = max(0.0, float(title_cosines[idx]))
        bm25 = r.get("_title_score", 1.0)
        title_scores[r["title"]] = 0.4 * bm25 + 0.6 * sem

    keywords   = extract_keywords(query)
    top_chunks = rank_chunks(keywords, all_chunks, top_k, ai_mode_embedder, query_vec,
                             title_scores=title_scores)

    if mark:
        mark("7_rank_chunks", f"{len(top_chunks)} selected from {len(all_chunks)}")
        for i, c in enumerate(top_chunks):
            header  = c.split("\n")[0]
            preview = c.split("\n", 1)[1][:80].replace("\n", " ") if "\n" in c else c[:80]
            print(f"  [{i+1:02d}] {header}  {preview!r}", flush=True)

    return top_chunks
