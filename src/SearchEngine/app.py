"""
IIAB Search Engine — Google-style search over Kiwix ZIMs with RAG AI summaries.

Pipeline:
  /search   → parallel scrape across all ZIMs → BM25 title rank → cache
  /ai       → cache check → fetch+chunk top N articles → BM25+cosine chunk rank → LLM stream → cache
  /ai-mode  → embed query → semantic cache → FAISS+BM25 RRF (ZIM index) → LLM stream → cache
             (fallback: Kiwix scrape + progressive RAG when no FAISS index is ready)
  /images   → scrape images from top ranked article pages
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import json
import threading
import httpx
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse, JSONResponse
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import time

from SearchEngine.config import (
    AI_PROMPT, AI_MODE_PROMPT,
    LLM_MODEL, LLM_OPTIONS,
    AI_MODE_LLM_MODEL, AI_MODE_LLM_OPTIONS,
    AI_MODE_CFG,
    ollama_client,
    _load_prompt,
)
from SearchEngine.cache import (
    init_db,
    db_get_results, db_set_results,
    db_get_ai, db_set_ai,
    db_suggest,
    db_ai_mode_lookup, db_set_ai_mode,
    db_clear_ai,
)
from SearchEngine.embedding import embed_ai_mode
from SearchEngine.search import get_or_run_pipeline, full_search_pipeline
from SearchEngine.rag import rag_pipeline, rag_pipeline_advanced
from SearchEngine.images import scrape_images
from SearchEngine.utils import make_marker
import SearchEngine.zim_retrieval as zim_retrieval
from SearchEngine.metrics import llm_metrics
from SearchEngine.metrics.llm_client import _extract_timings as _llm_extract_timings


# ─── Cancellation set ─────────────────────────────────────────────────────────
# Queries where /ai generation should abort — set by /ai-mode to free Ollama immediately.
_ai_cancelled      : set             = set()
_ai_cancelled_lock : threading.Lock = threading.Lock()

# ─── AI Mode deduplication ────────────────────────────────────────────────────
_ai_mode_inflight_lock = threading.Lock()
_ai_mode_inflight: dict[str, threading.Event] = {}
_ai_mode_results:  dict[str, str]             = {}


# ─── FastAPI app ──────────────────────────────────────────────────────────────

_CFG_DIR = os.path.dirname(os.path.abspath(__file__))

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("✅ SQLite cache initialised", flush=True)
    llm_metrics.init(_CFG_DIR)
    yield

app = FastAPI(lifespan=lifespan)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/suggest")
async def suggest(q: str):
    if not q or len(q) < 2:
        return {"suggestions": []}
    return {"suggestions": db_suggest(q.lower())}


@app.get("/search")
async def search(q: str):
    was_cached = db_get_results(q) is not None
    ranked     = get_or_run_pipeline(q)
    print(f"{'✅ Search cache hit' if was_cached else '🔍 Search cache miss'}: {q}", flush=True)
    return {"query": q, "results": ranked, "cached": was_cached}


@app.get("/ai")
async def ai_answer(q: str):
    cached_answer = db_get_ai(q)
    if cached_answer:
        print(f"✅ AI cache hit: {q}", flush=True)
        def _stream_cached():
            yield cached_answer
        return StreamingResponse(_stream_cached(), media_type="text/plain")

    def _generate():
        mark   = make_marker("ai", q)
        ranked = get_or_run_pipeline(q)
        src    = "cached" if db_get_results(q) else "fresh"
        mark("1_search_results", f"{len(ranked)} results ({src})")
        if not ranked:
            yield "No relevant content found."
            return

        top_chunks, first_para = rag_pipeline(q, ranked, mark=mark)
        context = "\n\n".join(top_chunks) if top_chunks else first_para
        if not context:
            yield "No relevant content found."
            return

        mark("6_llm_start", f"context={len(context)} chars | model={LLM_MODEL}")
        full_answer = ""
        stream = ollama_client.chat(
            model=LLM_MODEL,
            options={k: v for k, v in LLM_OPTIONS.items() if k != "keep_alive"},
            keep_alive=LLM_OPTIONS.get("keep_alive", "15m"),
            messages=[
                {"role": "system", "content": AI_PROMPT},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {q}"},
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
            if getattr(chunk, "done", False):
                try: llm_metrics.record("answer", LLM_MODEL, _llm_extract_timings(chunk, LLM_MODEL))
                except Exception: pass
            yield token

        mark("7_done", f"{len(full_answer)} chars generated")
        if full_answer:
            db_set_ai(q, full_answer)

    return StreamingResponse(_generate(), media_type="text/plain",
                            headers={"X-Accel-Buffering": "no",
                                     "Cache-Control": "no-cache"})


@app.get("/images")
async def images(q: str, background_tasks: BackgroundTasks):
    ranked = db_get_results(q) or full_search_pipeline(q)
    if ranked:
        background_tasks.add_task(db_set_results, q, ranked)
    image_candidates = [r for r in ranked if r.get("has_images")]
    all_images = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {
            ex.submit(scrape_images, rank, r["url"], r["title"]): rank
            for rank, r in enumerate(image_candidates)
        }
        for future in as_completed(futures):
            all_images.extend(future.result())
    all_images.sort(key=lambda x: x["rank"])
    return {"query": q, "images": all_images}


@app.get("/")
async def index():
    return FileResponse(os.path.join(_CFG_DIR, "static", "index.html"))

@app.get("/results")
async def results_page():
    return FileResponse(os.path.join(_CFG_DIR, "static", "results.html"))

@app.get("/images-page")
async def images_page():
    return FileResponse(os.path.join(_CFG_DIR, "static", "images.html"))


@app.get("/cache/clear")
async def cache_clear():
    result = db_clear_ai()
    print(f"[cache] cleared — {result}", flush=True)
    return JSONResponse(result)


@app.get("/zim-sources")
async def zim_sources(q: str, mode: str = "balanced"):
    """
    Returns the top ZIM chunks for a query as source cards for the WebUI panel.
    Called by the WebUI server before streaming the answer.
    """
    if not zim_retrieval.is_available():
        return JSONResponse({"sources": []})
    # Use fast mode — sources only need article titles/URLs, not expansion chunks
    hits = zim_retrieval.search(q, mode="fast")
    # Deduplicate by article title — one source card per article
    seen: set[str] = set()
    sources = []
    for h in hits:
        if h["title"] in seen:
            continue
        seen.add(h["title"])
        slug = h["title"].replace(" ", "_")
        sources.append({
            "title":   h["title"],
            "url":     f"/kiwix/viewer#{h['zim_name']}/{slug}",
            "snippet": h["text"][:200],
        })
    return JSONResponse({"sources": sources})


async def _wake_ollama(model: str):
    """Fire-and-forget: tell Ollama to load the model now so it's warm by the time we need it."""
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            await c.post("http://localhost:11434/api/generate",
                         json={"model": model, "keep_alive": "25m", "prompt": ""})
    except Exception:
        pass


@app.get("/ai-mode")
async def ai_mode_answer(q: str, mode: str = "balanced"):
    """
    Full semantic RAG endpoint for AI Mode tab.
    1. Embed query + run search pipeline in parallel
    2. Semantic similarity cache lookup (cosine ≥ threshold → stream cached answer)
    3. Miss → progressive RAG (title embed → semantic rank → first-para rerank → chunks) → LLM stream
    4. Save (query_vec, answer) to ai_mode_cache
    """
    # Kick off Ollama model load immediately — overlaps with embed+search below
    _mode_cfg  = zim_retrieval._MODE_CFGS.get(mode, {})
    _wake_model = _mode_cfg.get("llm_model", AI_MODE_LLM_MODEL)
    asyncio.create_task(_wake_ollama(_wake_model))

    def _generate():
        is_first = False
        ev: threading.Event | None = None
        try:
            with _ai_mode_inflight_lock:
                if q in _ai_mode_inflight:
                    ev       = _ai_mode_inflight[q]
                    is_first = False
                else:
                    ev                  = threading.Event()
                    _ai_mode_inflight[q] = ev
                    is_first            = True

            if not is_first:
                print(f"[ai-mode] duplicate request — waiting for in-flight result: {q!r}", flush=True)
                ev.wait(timeout=120)
                answer = _ai_mode_results.get(q, "")
                if answer:
                    yield answer
                return

            mark = make_marker("ai-mode", q)

            # Cancel any in-flight /ai generation for this query so Ollama is free.
            with _ai_cancelled_lock:
                _ai_cancelled.add(q)

            # Load per-mode LLM config — falls back to global ai_mode config
            _mode_cfg     = zim_retrieval._MODE_CFGS.get(mode, {})
            _llm_model    = _mode_cfg.get("llm_model",    AI_MODE_LLM_MODEL)
            _llm_options  = _mode_cfg.get("llm_options",  AI_MODE_LLM_OPTIONS)
            _top_k        = _mode_cfg.get("retrieval", {}).get("top_k",
                            AI_MODE_CFG.get("top_chunks", 3))
            _prompt_path  = _mode_cfg.get("system_prompt", None)
            _system_prompt = (
                _load_prompt(_prompt_path) if _prompt_path else AI_MODE_PROMPT
            )

            if zim_retrieval.is_available():
                # ── FAISS path: embed once, reuse vector for cache + search ──
                # Expansion now uses FAISS reconstruct() — no re-embedding at query time.
                query_vec = embed_ai_mode(q)
                mark("1a_embed", f"vec={query_vec.shape}")
                zim_hits  = zim_retrieval.search(q, _top_k, mode, query_vec=query_vec)
                mark("1b_search", f"{len(zim_hits)} ZIM chunks ({mode})")

                cached = db_ai_mode_lookup(query_vec)
                mark("2_cache_lookup", "hit" if cached else "miss")
                if cached:
                    yield cached
                    return
                if not zim_hits:
                    yield "No relevant content found."
                    return

                mark("3_zim_chunks", f"{len(zim_hits)} chunks from ZIM index")
                for i, h in enumerate(zim_hits):
                    print(f"  [{i+1:02d}] rerank={h['rerank_score']:.5f} faiss={h['faiss_score']:.3f}"
                          f"  {h['title']!r} §{h['section_title']!r}", flush=True)

                top_chunks = [h["text"] for h in zim_hits]
            else:
                # ── Fallback: Kiwix scrape + progressive RAG ──────────────────
                with ThreadPoolExecutor(max_workers=2) as ex:
                    vec_fut    = ex.submit(embed_ai_mode, q)
                    search_fut = ex.submit(get_or_run_pipeline, q)
                    query_vec  = vec_fut.result()
                    ranked     = search_fut.result()
                src = "cached" if db_get_results(q) else "fresh"
                mark("1_embed_and_search",
                     f"vec={query_vec.shape} | {len(ranked)} results ({src}) [kiwix fallback]")

                cached = db_ai_mode_lookup(query_vec)
                mark("2_cache_lookup", "hit" if cached else "miss")
                if cached:
                    yield cached
                    return
                if not ranked:
                    yield "No relevant content found."
                    return

                top_chunks = rag_pipeline_advanced(q, query_vec, ranked, mark=mark)
                if not top_chunks:
                    yield "No relevant content found."
                    return

            context = "\n\n".join(top_chunks)
            mark("8_llm_start", f"context={len(context)} chars | model={_llm_model} | mode={mode}")
            print("─" * 60, flush=True)
            print(context, flush=True)
            print("─" * 60, flush=True)

            full_answer = ""
            stream = ollama_client.chat(
                model=_llm_model,
                options={k: v for k, v in _llm_options.items() if k != "keep_alive"},
                keep_alive=_llm_options.get("keep_alive", "15m"),
                messages=[
                    {"role": "system", "content": _system_prompt},
                    {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {q}"},
                ],
                stream=True,
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                full_answer += token
                if getattr(chunk, "done", False):
                    try: llm_metrics.record("answer", _llm_model, _llm_extract_timings(chunk, _llm_model))
                    except Exception: pass
                yield token

            if full_answer:
                mark("8_done", f"{len(full_answer)} chars generated")
                _ai_mode_results[q] = full_answer
                db_set_ai_mode(q, query_vec, full_answer)

        except Exception as e:
            print(f"[ai-mode] ERROR: {e}", flush=True)
            yield f"\n[AI Mode error: {e}]"

        finally:
            if is_first and ev is not None:
                with _ai_mode_inflight_lock:
                    _ai_mode_inflight.pop(q, None)
                    _ai_mode_results.pop(q, None)
                ev.set()

    return StreamingResponse(_generate(), media_type="text/plain",
                            headers={"X-Accel-Buffering": "no",
                                     "Cache-Control": "no-cache"})


# ─── Metrics ──────────────────────────────────────────────────────────────────

_DASHBOARD_HTML = os.path.join(_CFG_DIR, "metrics", "metrics_dashboard.html")
_BOOT_TIME      = psutil.boot_time()
_OWN_PROC       = psutil.Process(os.getpid())


@app.get("/metrics/dashboard", response_class=HTMLResponse)
async def metrics_dashboard():
    with open(_DASHBOARD_HTML, encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), headers={"Cache-Control": "no-store"})


@app.get("/api/metrics")
def get_metrics():
    def _safe(fn, fallback=None):
        try: return fn()
        except Exception: return fallback

    def _cpu():
        pcts = psutil.cpu_percent(percpu=True)
        freq = psutil.cpu_freq()
        return {
            "per_core_pct":   pcts,
            "avg_pct":        round(sum(pcts) / len(pcts), 1) if pcts else 0,
            "count_logical":  psutil.cpu_count(logical=True),
            "count_physical": psutil.cpu_count(logical=False),
            "freq_mhz":       round(freq.current) if freq else None,
            "freq_max_mhz":   round(freq.max)     if freq else None,
        }

    def _memory():
        ram  = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "ram_used_mb":   round(ram.used   / 1024**2),
            "ram_total_mb":  round(ram.total  / 1024**2),
            "ram_pct":       ram.percent,
            "ram_avail_mb":  round(ram.available / 1024**2),
            "swap_used_mb":  round(swap.used  / 1024**2),
            "swap_total_mb": round(swap.total / 1024**2),
            "swap_pct":      swap.percent,
        }

    def _disk():
        u  = psutil.disk_usage(os.getcwd())
        io = psutil.disk_io_counters()
        return {
            "used_gb":  round(u.used  / 1024**3, 2),
            "total_gb": round(u.total / 1024**3, 2),
            "free_gb":  round(u.free  / 1024**3, 2),
            "pct":      u.percent,
            "read_mb":  round(io.read_bytes  / 1024**2) if io else None,
            "write_mb": round(io.write_bytes / 1024**2) if io else None,
        }

    def _temperatures():
        try:
            out = []
            for chip, entries in (psutil.sensors_temperatures() or {}).items():
                for e in entries:
                    out.append({"chip": chip, "label": e.label or chip,
                                "current": e.current, "high": e.high, "critical": e.critical})
            return out
        except Exception:
            return []

    def _process():
        mem = _OWN_PROC.memory_info()
        return {
            "pid":     _OWN_PROC.pid,
            "rss_mb":  round(mem.rss / 1024**2, 1),
            "cpu_pct": _OWN_PROC.cpu_percent(),
            "threads": _OWN_PROC.num_threads(),
        }

    records = llm_metrics.get_last(100)
    return JSONResponse(content={
        "uptime_s":     round(time.time() - _BOOT_TIME),
        "cpu":          _safe(_cpu, {}),
        "memory":       _safe(_memory, {}),
        "disk":         _safe(_disk, {}),
        "temperatures": _safe(_temperatures, []),
        "process":      _safe(_process, {}),
        "llm":          {"count": len(records), "records": records},
    }, headers={"Cache-Control": "no-store"})


@app.get("/api/metrics/stream")
async def metrics_stream():
    queue: asyncio.Queue = asyncio.Queue()
    llm_metrics.subscribe(queue)

    async def _event_gen():
        try:
            while True:
                record = await queue.get()
                yield f"data: {json.dumps(record, default=str)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            llm_metrics.unsubscribe(queue)

    return StreamingResponse(
        _event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


_STATIC_DIR = os.path.join(_CFG_DIR, "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)
