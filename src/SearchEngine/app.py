"""
IIAB Search Engine — Google-style search over Kiwix ZIMs with RAG AI summaries.

Pipeline:
  /search  → parallel scrape across all ZIMs → batch-embed title ranking → cache
  /ai      → cache check → fetch+chunk top N articles → BM25+cosine chunk rank → LLM 2-sentence summary → stream → cache
  /images  → scrape images from top ranked article pages
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import requests
from lxml import html as lxml_html
from urllib.parse import urljoin
import ollama
import numpy as np
import time
import sqlite3
import json
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from concurrent.futures import ThreadPoolExecutor, as_completed

import re
import spacy

from rank_bm25 import BM25Okapi
from retrieval.kiwix_client import fetch_article_sections
from retrieval.rerank import rank_chunks
from ingest.article_cleaner import last_sentence, first_sentence


# ─── Config ───────────────────────────────────────────────────────────────────

_CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def _load_cfg() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)


def _load_prompt(rel_path: str) -> str:
    """Load a system prompt from a .md file path relative to the config directory."""
    full_path = os.path.join(os.path.dirname(_CFG_PATH), rel_path)
    with open(full_path) as f:
        return f.read().strip()

CFG = _load_cfg()

_BASE_URL = CFG.get("base_url", "http://box")
_NAV_RE   = re.compile(r"^(List(s)? of|Index of|Outline of|Portal:|Category:)|\(disambiguation\)$", re.I)
_EMBED_MODEL = CFG.get("embed_model", "snowflake-arctic-embed:xs")
_LLM_MODEL   = CFG.get("llm_model", "qwen2.5:1.5b-instruct-q4_K_M")
_LLM_OPTIONS = CFG.get("llm_options", {})
_ZIMS        = CFG.get("zims", [])
_RAG         = CFG.get("rag", {})

_SEARCH_CFG          = CFG.get("search", {})
_SEARCH_MAX_RESULTS  = _SEARCH_CFG.get("max_results", 25)
_TITLE_WEIGHT_FLOOR  = _SEARCH_CFG.get("title_weight_floor", 0.2)

_AI_PROMPT      = _load_prompt(_RAG.get("system_prompt", "prompts/ai.md"))
_AI_MODE_CFG    = CFG.get("ai_mode", {})
_AI_MODE_PROMPT = _load_prompt(_AI_MODE_CFG.get("system_prompt", "prompts/ai_mode.md"))

_AI_MODE_EMBED_MODEL = _AI_MODE_CFG.get("embed_model", "snowflake-arctic-embed:xs")
_AI_MODE_LLM_MODEL   = _AI_MODE_CFG.get("llm_model", "smollm2:360m")
_AI_MODE_LLM_OPTIONS = _AI_MODE_CFG.get("llm_options", {})
_AI_MODE_SIM_THRESH  = _AI_MODE_CFG.get("semantic_cache", {}).get("similarity_threshold", 0.92)
_AI_MODE_KEEP_ALIVE  = _AI_MODE_CFG.get("llm_options", {}).get("keep_alive", "15m")

# Ollama client with timeout — prevents hanging threads if Ollama stalls (OOM, model load stuck, etc.)
_OLLAMA_TIMEOUT = CFG.get("ollama_timeout", 120)
_ollama = ollama.Client(timeout=_OLLAMA_TIMEOUT)


# ─── SQLite cache ─────────────────────────────────────────────────────────────

_DB_PATH   = CFG.get("db_path", "cache.db")
_CACHE_TTL = CFG.get("cache_ttl", 60 * 60 * 24 * 7)


def _init_db():
    con = sqlite3.connect(_DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS search_cache (
            query TEXT PRIMARY KEY, results TEXT NOT NULL, timestamp REAL NOT NULL
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_query ON search_cache(query)")
    con.execute("""
        CREATE TABLE IF NOT EXISTS ai_cache (
            query TEXT PRIMARY KEY, answer TEXT NOT NULL, timestamp REAL NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ai_mode_cache (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            query     TEXT    NOT NULL,
            query_vec BLOB    NOT NULL,
            answer    TEXT    NOT NULL,
            timestamp REAL    NOT NULL
        )
    """)
    con.commit()
    con.close()


def _db_get_results(q: str):
    con = sqlite3.connect(_DB_PATH)
    row = con.execute(
        "SELECT results, timestamp FROM search_cache WHERE query = ?", (q,)
    ).fetchone()
    con.close()
    if row and (time.time() - row[1]) < _CACHE_TTL:
        return json.loads(row[0])
    return None


def _db_set_results(q: str, ranked: list):
    con = sqlite3.connect(_DB_PATH)
    con.execute(
        "INSERT OR REPLACE INTO search_cache (query, results, timestamp) VALUES (?, ?, ?)",
        (q, json.dumps(ranked), time.time()),
    )
    con.commit()
    con.close()


def _db_get_ai(q: str):
    con = sqlite3.connect(_DB_PATH)
    row = con.execute(
        "SELECT answer, timestamp FROM ai_cache WHERE query = ?", (q,)
    ).fetchone()
    con.close()
    if row and (time.time() - row[1]) < _CACHE_TTL:
        return row[0]
    return None


def _db_set_ai(q: str, answer: str):
    con = sqlite3.connect(_DB_PATH)
    con.execute(
        "INSERT OR REPLACE INTO ai_cache (query, answer, timestamp) VALUES (?, ?, ?)",
        (q, answer, time.time()),
    )
    con.commit()
    con.close()


def _db_suggest(q: str, limit: int = 8) -> list:
    con = sqlite3.connect(_DB_PATH)
    rows = con.execute(
        "SELECT query FROM search_cache WHERE query LIKE ? ORDER BY timestamp DESC LIMIT ?",
        (q + "%", limit),
    ).fetchall()
    con.close()
    return [r[0] for r in rows]


def _db_ai_mode_lookup(query_vec: np.ndarray) -> str | None:
    """Semantic similarity cache lookup for AI Mode answers.
    Loads all stored query vectors, finds the closest by cosine similarity.
    Returns the cached answer if similarity >= _AI_MODE_SIM_THRESH, else None.
    """
    con = sqlite3.connect(_DB_PATH)
    rows = con.execute(
        "SELECT query_vec, answer, timestamp FROM ai_mode_cache ORDER BY timestamp DESC"
    ).fetchall()
    con.close()
    if not rows:
        return None
    best_sim    = -1.0
    best_answer = None
    for vec_bytes, answer, ts in rows:
        if (time.time() - ts) >= _CACHE_TTL:
            continue
        stored_vec = np.frombuffer(vec_bytes, dtype=np.float32)
        sim = float(np.dot(query_vec, stored_vec))
        if sim > best_sim:
            best_sim    = sim
            best_answer = answer
    if best_sim >= _AI_MODE_SIM_THRESH:
        print(f"✅ AI Mode semantic cache hit (sim={best_sim:.4f})", flush=True)
        return best_answer
    print(f"  AI Mode cache miss (best sim={best_sim:.4f})", flush=True)
    return None


def _db_set_ai_mode(query: str, query_vec: np.ndarray, answer: str):
    con = sqlite3.connect(_DB_PATH)
    con.execute(
        "INSERT INTO ai_mode_cache (query, query_vec, answer, timestamp) VALUES (?, ?, ?, ?)",
        (query, query_vec.tobytes(), answer, time.time()),
    )
    con.commit()
    con.close()


# ─── FastAPI app ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    print("✅ SQLite cache initialised", flush=True)
    # Pre-warm the AI Mode embed model so stage 1 of the first AI Mode request
    # doesn't pay the full cold-load penalty (~30s).
    def _warmup():
        try:
            _ollama.embeddings(model=_AI_MODE_EMBED_MODEL, prompt="warmup", keep_alive=_AI_MODE_KEEP_ALIVE)
            print("✅ AI Mode embed model warmed", flush=True)
        except Exception as e:
            print(f"⚠️  Embed warmup failed: {e}", flush=True)
    threading.Thread(target=_warmup, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)


# ─── Embedding ────────────────────────────────────────────────────────────────

def embed(text: str) -> np.ndarray:
    """Single embed via ollama.embeddings() — L2-normalised."""
    vec = np.array(
        _ollama.embeddings(model=_EMBED_MODEL, prompt=text)["embedding"],
        dtype=np.float32,
    )
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def embed_batch(texts: list) -> np.ndarray:
    """Batch embed in one Ollama call — L2-normalised, shape (n, dim)."""
    resp  = _ollama.embed(model=_EMBED_MODEL, input=texts)
    vecs  = np.array(resp["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


class _Embedder:
    """Minimal adapter so rank_chunks() can call embed_batch()."""
    def embed_batch(self, texts):
        return embed_batch(texts)

_embedder = _Embedder()


def _embed_ai_mode(text: str) -> np.ndarray:
    """Single embed using ollama.embed() — same endpoint as batch so keep_alive
    carries over between the query embed (stage 1) and chunk batch (stage 6).
    """
    resp = _ollama.embed(model=_AI_MODE_EMBED_MODEL, input=[text], keep_alive=_AI_MODE_KEEP_ALIVE)
    vec  = np.array(resp["embeddings"][0], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _embed_batch_ai_mode(texts: list) -> np.ndarray:
    """Batch embed in one ollama.embed() call — same endpoint as _embed_ai_mode
    so the model loaded at stage 1 is still in RAM here at stage 6.
    """
    resp  = _ollama.embed(model=_AI_MODE_EMBED_MODEL, input=texts, keep_alive=_AI_MODE_KEEP_ALIVE)
    vecs  = np.array(resp["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


class _AiModeEmbedder:
    """Adapter so rank_chunks() can call embed_batch() with the AI Mode model."""
    def embed_batch(self, texts):
        return _embed_batch_ai_mode(texts)

_ai_mode_embedder = _AiModeEmbedder()


# ─── Keyword extraction (spaCy) ───────────────────────────────────────────────

try:
    _nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy en_core_web_sm loaded", flush=True)
except OSError:
    _nlp = None
    print("⚠️  spaCy model not found — run: python -m spacy download en_core_web_sm", flush=True)

_keyword_cache: dict[str, str] = {}

def _extract_keywords(q: str) -> str:
    """
    Extract core search keywords using spaCy — zero LLM cost, runs in <5ms.

    Priority: named entities (GPE, ORG, PERSON, WORK_OF_ART…) → noun chunks → raw query.
    "tell me about Houston"      → "Houston"       (GPE entity)
    "what is the speed of light" → "speed of light" (noun chunk)
    "how does MIT work"          → "MIT"            (ORG entity)
    Falls back to raw query if spaCy not loaded.
    """
    if q in _keyword_cache:
        return _keyword_cache[q]
    keywords = q
    if _nlp is not None:
        doc = _nlp(q)
        ents = [e.text for e in doc.ents if e.label_ in {
            "GPE", "LOC", "ORG", "PERSON", "NORP", "FAC",
            "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
        }]
        if ents:
            keywords = " ".join(ents[:3])
        else:
            # Exclude pronoun-only chunks ("me", "you", "it", "we"…) — they are
            # never the search topic, just conversational filler.
            chunks = [
                c.text for c in doc.noun_chunks
                if len(c.text.split()) <= 4
                and not all(t.pos_ == "PRON" for t in c)
            ]
            if chunks:
                keywords = " ".join(chunks[:2])
    _keyword_cache[q] = keywords
    return keywords


# ─── Kiwix scraping ───────────────────────────────────────────────────────────

def _scrape_zim(q: str, zim_name: str, count: int, has_images: bool) -> list:
    url = f"{_BASE_URL}/kiwix/search?pattern={q}&books.name={zim_name}&start=0&pageLength={count}"
    try:
        response = requests.get(url, timeout=8)
        tree     = lxml_html.fromstring(response.content)
        results  = []
        for li in tree.xpath('//div[@class="results"]//li'):
            try:
                title     = li.xpath('.//a/text()')[0].strip()
                href      = li.xpath('.//a/@href')[0]
                snippet   = " ".join(li.xpath('.//cite//text()')).strip()
                wordcount = li.xpath('.//div[@class="informations"]/text()')[0].strip()
                results.append({
                    "title":      title,
                    "url":        _BASE_URL + href,
                    "snippet":    snippet,
                    "wordcount":  wordcount,
                    "source_zim": zim_name,
                    "has_images": has_images,
                })
            except Exception:
                continue
        return results
    except Exception:
        return []


def _scrape_all_zims(q: str) -> tuple[list, dict]:
    """Returns (all_results, {zim_name: hit_count})."""
    all_results = []
    zim_counts  = {}
    with ThreadPoolExecutor(max_workers=len(_ZIMS)) as ex:
        futures = {
            ex.submit(_scrape_zim, q, z["name"], z["count"], z["has_images"]): z
            for z in _ZIMS
        }
        for future in as_completed(futures):
            zim     = futures[future]
            results = future.result()
            zim_counts[zim["name"]] = len(results)
            all_results.extend(results)
    return all_results, zim_counts


# ─── Title ranking ────────────────────────────────────────────────────────────

def _rank_by_title(query: str, results: list, mark=None) -> list:
    """
    BM25 ranking over title + snippet text — zero embedding cost, runs in milliseconds.
    Including the snippet gives richer signal: "tell me about Houston" matches the
    Houston, Texas snippet content far more than a song title that contains 'tell me about'.
    Returns up to 25 results sorted by BM25 score descending.
    """
    if not results:
        return []
    tokenized_docs  = [(r["title"] + " " + r.get("snippet", "")).lower().split() for r in results]
    tokenized_query = query.lower().split()
    scores     = BM25Okapi(tokenized_docs).get_scores(tokenized_query)
    # Exact-title boost: if the article title equals the query (case-insensitive),
    # push it to the front. Prevents "Banana (disambiguation)" outranking "Banana".
    q_lower = query.lower().strip()
    for i, r in enumerate(results):
        if r["title"].lower().strip() == q_lower:
            scores[i] = float(max(scores)) * 1.5
    ranked_idx = np.argsort(scores)[::-1][:_SEARCH_MAX_RESULTS]
    ranked     = [results[i] for i in ranked_idx]
    # Normalise scores to [0.2, 1.0] — floor at 0.2 so a low-ranked article's
    # chunks still get considered (title_w=0 would zero out the final score entirely).
    max_score = float(max(scores)) if max(scores) > 0 else 1.0
    for i, r in enumerate(results):
        normalized = max(0.0, float(scores[i])) / max_score
        r["_title_score"] = _TITLE_WEIGHT_FLOOR + (1.0 - _TITLE_WEIGHT_FLOOR) * normalized
    if mark:
        mark("2_bm25_rank", f"{len(ranked)} results ranked")
    for rank, i in enumerate(ranked_idx):
        print(f"  [{rank+1:02d}] {scores[i]:.4f}  {results[i]['title']!r:<40}  ({results[i]['source_zim']})", flush=True)
    return ranked


def _full_search_pipeline(q: str) -> list:
    mark = _make_marker("search", q)
    keywords = _extract_keywords(q)        # spaCy: <5ms, no Ollama
    results, zim_counts = _scrape_all_zims(keywords)  # Kiwix gets clean keywords
    mark("1_scrape_zims", f"{len(results)} total | {q!r} → {keywords!r}")
    for zim, count in sorted(zim_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  → {zim:<45} {count} hits", flush=True)
    return _rank_by_title(keywords, results, mark=mark)


# ─── Pipeline logging ─────────────────────────────────────────────────────────

def _make_marker(prefix: str, q: str):
    """Returns a mark(stage, detail) closure that prints timing per stage."""
    print(f"\n[{prefix}] {'═'*6} q={q!r} {'═'*6}", flush=True)
    t0   = time.time()
    t_   = [t0]
    def mark(stage: str, detail: str = ""):
        now   = time.time()
        took  = now - t_[0]
        total = now - t0
        line  = f"[{prefix}] {stage:<22} | took={took:.3f}s | total={total:.3f}s"
        if detail:
            line += f" | {detail}"
        print(line, flush=True)
        t_[0] = now
    return mark


# ─── Pipeline deduplication ───────────────────────────────────────────────────

_inflight_lock = threading.Lock()
_inflight: dict[str, threading.Event] = {}

# Queries where /ai generation should abort — set by /ai-mode to free Ollama immediately.
_ai_cancelled: set = set()
_ai_cancelled_lock = threading.Lock()

_ai_mode_inflight_lock = threading.Lock()
_ai_mode_inflight: dict[str, threading.Event] = {}
_ai_mode_results:  dict[str, str]            = {}  # short-lived in-memory share between waiter and first caller

def _get_or_run_pipeline(q: str) -> list:
    """
    Guarantees the scrape+rank pipeline runs at most once per query at a time.

    /search and /ai fire simultaneously on every page load. Without this, both
    hit a cache miss together and scrape all ZIMs twice in parallel.

    First caller runs the pipeline and saves to SQLite.
    Any concurrent caller for the same query blocks on the event, then reads
    the result from SQLite once the first caller is done.
    """
    cached = _db_get_results(q)
    if cached:
        return cached

    with _inflight_lock:
        if q in _inflight:
            ev = _inflight[q]
            is_first = False
        else:
            ev = threading.Event()
            _inflight[q] = ev
            is_first = True

    if not is_first:
        ev.wait(timeout=120)             # block until first caller finishes (120s safety timeout)
        return _db_get_results(q) or [] # read what first caller saved

    ranked = []
    try:
        ranked = _full_search_pipeline(q)
        if ranked:
            _db_set_results(q, ranked)  # save synchronously so waiters can read
    finally:
        with _inflight_lock:
            _inflight.pop(q, None)
        ev.set()                        # wake all waiters

    return ranked


# ─── Chunking ─────────────────────────────────────────────────────────────────

def _build_chunks(title: str, sections: list) -> list:
    use_overlap = _RAG.get("chunk_overlap", True)
    min_chars   = _RAG.get("min_paragraph_chars", 50)
    all_paras   = [(sec["section"], p) for sec in sections for p in sec["paragraphs"]]
    chunks = []
    for i, (section, para) in enumerate(all_paras):
        left  = (last_sentence(all_paras[i - 1][1]) + " ") if i > 0 and use_overlap else ""
        right = (" " + first_sentence(all_paras[i + 1][1])) if i < len(all_paras) - 1 and use_overlap else ""
        chunk = f"[{title} | {section}]\n{left}{para}{right}"
        if len(para) >= min_chars:
            chunks.append(chunk)
    return chunks


# ─── RAG pipeline ─────────────────────────────────────────────────────────────

def _rag_pipeline(query: str, ranked_results: list, mark=None) -> tuple:
    """
    Fetch + chunk top N articles, rank chunks with BM25+cosine+section bias.
    Returns (top_chunks: list[str], first_para: str).
    first_para is the intro paragraph of the #1 article — used as fallback context.

    Speed levers (all in config.yaml rag section):
      use_embeddings: false  → BM25-only chunk ranking, no embed calls (~25s faster)
      max_chunks_per_article → caps chunk pool size before BM25 filter
      top_articles           → fewer articles = smaller pool
    """
    top_n            = _RAG.get("top_articles", 3)
    top_k            = _RAG.get("top_chunks", 8)
    max_chars        = _RAG.get("max_chunk_chars", 800)
    max_per_article  = _RAG.get("max_chunks_per_article", 20)
    use_embeddings   = _RAG.get("use_embeddings", True)

    content_results = [r for r in ranked_results if not _NAV_RE.search(r["title"])]
    top_articles = (content_results or ranked_results)[:top_n]

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

    # First paragraph of top article (summary fallback)
    first_para = ""
    top_secs   = sections_by_title.get(top_articles[0]["title"], [])
    if top_secs and top_secs[0]["paragraphs"]:
        first_para = top_secs[0]["paragraphs"][0]

    all_chunks = []
    for art in top_articles:
        sections = sections_by_title.get(art["title"], [])
        all_chunks.extend(_build_chunks(art["title"], sections)[:max_per_article])

    if mark:
        mark("4_build_chunks", f"{len(all_chunks)} chunks")

    if not all_chunks:
        return [], first_para

    keywords     = _extract_keywords(query)  # free — already cached from search stage
    title_scores = {r["title"]: r.get("_title_score", 1.0) for r in ranked_results}
    if use_embeddings:
        top_chunks = rank_chunks(keywords, all_chunks, top_k, _embedder, query_vec, title_scores=title_scores)
    else:
        top_chunks = rank_chunks(keywords, all_chunks, top_k, title_scores=title_scores)

    if mark:
        mark("5_rank_chunks", f"{len(top_chunks)} selected from {len(all_chunks)}")
        for i, c in enumerate(top_chunks):
            header  = c.split("\n")[0]
            preview = c.split("\n", 1)[1][:80].replace("\n", " ") if "\n" in c else c[:80]
            print(f"  [{i+1:02d}] {header}  {preview!r}", flush=True)

    return top_chunks, first_para


def _rag_pipeline_advanced(query: str, query_vec: np.ndarray, ranked_results: list, mark=None) -> list:
    """
    Progressive AI Mode RAG pipeline:

    Stage A (parallel):
      - Embed all candidate title texts in one batch
      - Fetch full article sections for all candidates (parallel HTTP)

    Stage B: Semantic title ranking  →  reorder candidates by cosine(title, query)

    Stage C: First-paragraph rerank  →  embed first para of each candidate,
             pick article_rerank_top winners by cosine(first_para, query)

    Stage D: Full chunk pipeline on winners only
             (BM25 → cosine → section-bias → dedup)

    All section fetches happen once in Stage A — no re-fetch needed in Stage D.
    BM25 title scores from _rank_by_title are blended with semantic title cosines
    as title_scores for the chunk reranker.
    """
    n_candidates    = _AI_MODE_CFG.get("title_rank_candidates", 8)
    n_para_rerank   = _AI_MODE_CFG.get("para_rerank_candidates", 5)  # top-N of title-ranked to embed for first-para rerank
    n_winners       = _AI_MODE_CFG.get("article_rerank_top", 2)
    top_k           = _AI_MODE_CFG.get("top_chunks", 8)
    max_chars       = _AI_MODE_CFG.get("max_chunk_chars", 600)
    max_per_article = _AI_MODE_CFG.get("max_chunks_per_article", 5)
    use_overlap     = _AI_MODE_CFG.get("chunk_overlap", True)
    min_chars       = _AI_MODE_CFG.get("min_paragraph_chars", 50)

    content_results = [r for r in ranked_results if not _NAV_RE.search(r["title"])]
    candidates      = (content_results or ranked_results)[:n_candidates]

    # ── Stage A: parallel title-embed batch + section fetch ───────────────────
    def _fetch_sections(r):
        return r["title"], fetch_article_sections(r["url"], max_chars)

    sections_by_title = {}
    # Embed title + snippet so the model sees article content, not just the title string.
    # "This Is My Truth Tell Me Yours" has "Tell Me" in the title but its snippet reveals
    # it's a rock album — the snippet prevents false matches on conversational query words.
    title_texts = [r["title"] + ". " + r.get("snippet", "") for r in candidates]
    with ThreadPoolExecutor(max_workers=n_candidates + 1) as ex:
        title_embed_fut = ex.submit(_embed_batch_ai_mode, title_texts)
        fetch_futs      = {ex.submit(_fetch_sections, r): r for r in candidates}
        title_vecs      = title_embed_fut.result()          # (N, dim)
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
    title_cosines = title_vecs @ query_vec          # (N,)
    title_order   = np.argsort(title_cosines)[::-1]
    sem_ranked    = [candidates[i] for i in title_order]

    if mark:
        mark("4_semantic_title_rank", f"best: {sem_ranked[0]['title']!r}")
        for i, idx in enumerate(title_order):
            print(f"  [{i+1}] cos={title_cosines[idx]:.4f}  {candidates[idx]['title']!r}", flush=True)

    # ── Stage C: first-paragraph rerank ───────────────────────────────────────
    # Only embed the top n_para_rerank articles from the title ranking — avoids
    # embedding all n_candidates first paragraphs when the pool is large.
    para_pool   = sem_ranked[:n_para_rerank]
    first_paras = []
    for r in para_pool:
        secs = sections_by_title.get(r["title"], [])
        para = secs[0]["paragraphs"][0] if secs and secs[0]["paragraphs"] else r["title"]
        first_paras.append(para)

    para_vecs    = _embed_batch_ai_mode(first_paras)
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
        sections  = sections_by_title.get(art["title"], [])
        all_paras = [(sec["section"], p) for sec in sections for p in sec["paragraphs"]]
        art_chunks = []
        for i, (section, para) in enumerate(all_paras):
            if len(art_chunks) >= max_per_article:
                break
            left  = (last_sentence(all_paras[i - 1][1]) + " ") if i > 0 and use_overlap else ""
            right = (" " + first_sentence(all_paras[i + 1][1])) if i < len(all_paras) - 1 and use_overlap else ""
            chunk = f"[{art['title']} | {section}]\n{left}{para}{right}"
            if len(para) >= min_chars:
                art_chunks.append(chunk)
        all_chunks.extend(art_chunks)

    if mark:
        mark("6_build_chunks", f"{len(all_chunks)} chunks from {len(winners)} articles")

    if not all_chunks:
        return []

    # Use semantic title cosines (blended with BM25) as title_scores for chunk reranker
    title_scores = {r["title"]: r.get("_title_score", 1.0) for r in ranked_results}
    for i, idx in enumerate(title_order):
        r    = candidates[idx]
        sem  = max(0.0, float(title_cosines[idx]))
        bm25 = r.get("_title_score", 1.0)
        title_scores[r["title"]] = 0.4 * bm25 + 0.6 * sem   # blend BM25 + semantic

    keywords   = _extract_keywords(query)
    top_chunks = rank_chunks(keywords, all_chunks, top_k, _ai_mode_embedder, query_vec,
                             title_scores=title_scores)

    if mark:
        mark("7_rank_chunks", f"{len(top_chunks)} selected from {len(all_chunks)}")
        for i, c in enumerate(top_chunks):
            header  = c.split("\n")[0]
            preview = c.split("\n", 1)[1][:80].replace("\n", " ") if "\n" in c else c[:80]
            print(f"  [{i+1:02d}] {header}  {preview!r}", flush=True)

    return top_chunks


# ─── Image scraping ───────────────────────────────────────────────────────────

def _scrape_images(rank: int, url: str, title: str) -> list:
    try:
        response = requests.get(url, timeout=8)
        tree     = lxml_html.fromstring(response.content)
        images   = []
        for img in tree.xpath("//img"):
            src    = img.get("src", "")
            alt    = img.get("alt", title)
            width  = img.get("width", "")
            height = img.get("height", "")
            try:
                if int(width) < 50:
                    continue
            except Exception:
                pass
            try:
                if int(height) < 50:
                    continue
            except Exception:
                pass
            if not src or src.endswith(".svg"):
                continue
            if "math" in src.lower() or "formula" in src.lower():
                continue
            if src.startswith("//"):
                src = "http:" + src
            elif not src.startswith("http"):
                src = urljoin(url, src)
            images.append({"src": src, "alt": alt or title, "title": title, "source": url, "rank": rank})
        return images
    except Exception:
        return []


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/suggest")
async def suggest(q: str):
    if not q or len(q) < 2:
        return {"suggestions": []}
    return {"suggestions": _db_suggest(q.lower())}


@app.get("/search")
async def search(q: str):
    was_cached = _db_get_results(q) is not None
    ranked = _get_or_run_pipeline(q)
    print(f"{'✅ Search cache hit' if was_cached else '🔍 Search cache miss'}: {q}", flush=True)
    return {"query": q, "results": ranked, "cached": was_cached}


@app.get("/ai")
async def ai_answer(q: str):
    # Stage 1: check AI answer cache — return immediately if hit
    cached_answer = _db_get_ai(q)
    if cached_answer:
        print(f"✅ AI cache hit: {q}", flush=True)
        def _stream_cached():
            yield cached_answer
        return StreamingResponse(_stream_cached(), media_type="text/plain")

    # Stage 2: no hit — do RAG and stream the generated answer
    def _generate():
        mark = _make_marker("ai", q)

        ranked = _get_or_run_pipeline(q)
        src = "cached" if _db_get_results(q) else "fresh"
        mark("1_search_results", f"{len(ranked)} results ({src})")
        if not ranked:
            yield "No relevant content found."
            return

        top_chunks, first_para = _rag_pipeline(q, ranked, mark=mark)
        context = "\n\n".join(top_chunks) if top_chunks else first_para
        if not context:
            yield "No relevant content found."
            return

        mark("6_llm_start", f"context={len(context)} chars | model={_LLM_MODEL}")
        full_answer = ""
        stream = _ollama.chat(
            model=_LLM_MODEL,
            options=_LLM_OPTIONS,
            messages=[
                {"role": "system", "content": _AI_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {q}",
                },
            ],
            stream=True,
        )
        for chunk in stream:
            with _ai_cancelled_lock:
                if q in _ai_cancelled:
                    print(f"[ai] cancelled by ai-mode: {q!r}", flush=True)
                    _ai_cancelled.discard(q)
                    return
            token = chunk["message"]["content"]
            full_answer += token
            yield token

        mark("7_done", f"{len(full_answer)} chars generated")
        if full_answer:
            _db_set_ai(q, full_answer)

    return StreamingResponse(_generate(), media_type="text/plain")


@app.get("/images")
async def images(q: str, background_tasks: BackgroundTasks):
    ranked = _db_get_results(q) or _full_search_pipeline(q)
    if ranked:
        background_tasks.add_task(_db_set_results, q, ranked)
    image_candidates = [r for r in ranked if r.get("has_images")]
    all_images = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {
            ex.submit(_scrape_images, rank, r["url"], r["title"]): rank
            for rank, r in enumerate(image_candidates)
        }
        for future in as_completed(futures):
            all_images.extend(future.result())
    all_images.sort(key=lambda x: x["rank"])
    return {"query": q, "images": all_images}


@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.get("/results")
async def results_page():
    return FileResponse("static/results.html")

@app.get("/images-page")
async def images_page():
    return FileResponse("static/images.html")


@app.get("/ai-mode")
async def ai_mode_answer(q: str):
    """
    Full semantic RAG endpoint for AI Mode tab.
    1. Embed query (AI Mode model)
    2. Semantic similarity cache lookup (cosine ≥ threshold → stream cached answer)
    3. Miss → fetch+chunk top N articles → BM25+cosine+section-bias rank → LLM stream
    4. Save (query_vec, answer) to ai_mode_cache
    """
    def _generate():
        # Initialize before try so finally can safely reference them even if
        # an exception fires before the lock block assigns them.
        is_first = False
        ev: threading.Event | None = None
        try:
            # Deduplicate: if another request is already running this query,
            # block until it finishes then serve the saved answer from cache.
            with _ai_mode_inflight_lock:
                if q in _ai_mode_inflight:
                    ev = _ai_mode_inflight[q]
                    is_first = False
                else:
                    ev = threading.Event()
                    _ai_mode_inflight[q] = ev
                    is_first = True

            if not is_first:
                print(f"[ai-mode] duplicate request — waiting for in-flight result: {q!r}", flush=True)
                ev.wait(timeout=120)     # 120s safety timeout — never block forever
                answer = _ai_mode_results.get(q, "")
                if answer:
                    yield answer
                return

            mark = _make_marker("ai-mode", q)

            # Cancel any in-flight /ai generation for this query so Ollama is free.
            with _ai_cancelled_lock:
                _ai_cancelled.add(q)

            # Stage 1: embed query + Kiwix search in parallel
            # They hit different services (Ollama vs Kiwix) so there's no contention.
            with ThreadPoolExecutor(max_workers=2) as ex:
                vec_fut    = ex.submit(_embed_ai_mode, q)
                search_fut = ex.submit(_get_or_run_pipeline, q)
                query_vec  = vec_fut.result()
                ranked     = search_fut.result()
            src = "cached" if _db_get_results(q) else "fresh"
            mark("1_embed_and_search", f"vec={query_vec.shape} | {len(ranked)} results ({src})")

            cached = _db_ai_mode_lookup(query_vec)
            mark("2_cache_lookup", "hit" if cached else f"miss (threshold={_AI_MODE_SIM_THRESH})")
            if cached:
                yield cached
                return
            if not ranked:
                yield "No relevant content found."
                return

            top_chunks = _rag_pipeline_advanced(q, query_vec, ranked, mark=mark)
            if not top_chunks:
                yield "No relevant content found."
                return

            context = "\n\n".join(top_chunks)
            mark("8_llm_start", f"context={len(context)} chars | model={_AI_MODE_LLM_MODEL}")
            print("─" * 60, flush=True)
            print(context, flush=True)
            print("─" * 60, flush=True)

            full_answer = ""
            stream = _ollama.chat(
                model=_AI_MODE_LLM_MODEL,
                options=_AI_MODE_LLM_OPTIONS,
                messages=[
                    {"role": "system", "content": _AI_MODE_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {q}",
                    },
                ],
                stream=True,
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                full_answer += token
                yield token

            if full_answer:
                mark("8_done", f"{len(full_answer)} chars generated")
                _ai_mode_results[q] = full_answer   # share with any waiting duplicate requests
                _db_set_ai_mode(q, query_vec, full_answer)

        except Exception as e:
            print(f"[ai-mode] ERROR: {e}", flush=True)
            yield f"\n[AI Mode error: {e}]"

        finally:
            if is_first and ev is not None:
                with _ai_mode_inflight_lock:
                    _ai_mode_inflight.pop(q, None)
                    _ai_mode_results.pop(q, None)
                ev.set()   # wake all waiters regardless of success or failure

    return StreamingResponse(_generate(), media_type="text/plain")

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
