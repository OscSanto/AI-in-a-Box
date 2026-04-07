"""
WebUI server — port 5050.

Serves the llama.cpp-style chat WebUI and proxies /v1/chat/completions
to the SearchEngine server (port 8000).

  mode "ai"    → SearchEngine /ai-mode  (FAISS+BM25 RRF → LLM)
  mode "kiwix" → SearchEngine /ai       (Kiwix scrape → LLM)

Plain-text streaming from SearchEngine is reformatted to OpenAI SSE so the
WebUI receives the format it expects.
"""
import asyncio
import json
import os
import time
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

# ── Config ────────────────────────────────────────────────────────────────────
WEBUI_DIR      = os.path.join(os.path.dirname(__file__), "..", "webui")
SE_BASE        = os.environ.get("SEARCHENGINE_URL", "http://localhost:5050")
ZIM_CONTENT_ID = os.environ.get("ZIM_CONTENT_ID",   "wikipedia_en_astronomy_maxi_2025-11")

app = FastAPI()

# ── Static assets ─────────────────────────────────────────────────────────────
app.mount("/_app", StaticFiles(directory=os.path.join(WEBUI_DIR, "_app")), name="app_static")

# ── Root — inject interceptor script into WebUI HTML ─────────────────────────
@app.get("/")
def index():
    html = open(os.path.join(WEBUI_DIR, "index.html")).read()
    html = html.replace(
        "</body>",
        '<script src="/webgpu-interceptor.js" defer></script></body>',
    )
    return Response(html, media_type="text/html")

# ── Stubs required by llama.cpp WebUI on startup ─────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/props")
def props():
    return {}

@app.get("/v1/models")
def list_models():
    return {}

# ── Config — read by webgpu-interceptor.js for citation links + mode ──────────
@app.get("/config")
def client_config():
    return {
        "inference_default_mode":     "server",
        "inference_allow_user_toggle": True,
        "zim_content_id":             ZIM_CONTENT_ID,
    }

# ── WebGPU interceptor script ─────────────────────────────────────────────────
@app.get("/webgpu-interceptor.js")
def webgpu_interceptor():
    return FileResponse(
        os.path.join(WEBUI_DIR, "webgpu-interceptor.js"),
        media_type="application/javascript",
    )

# ── /v1/chat/completions — proxy to SearchEngine, reformat to OpenAI SSE ──────
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body     = await request.json()
    messages = body.get("messages", [])
    mode     = body.get("mode", "kiwix")

    # Extract the latest user message
    query = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            query   = content if isinstance(content, str) else " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
            break

    if not query:
        return Response('data: [DONE]\n\n', media_type="text/event-stream")

    # Map WebUI mode buttons → SearchEngine endpoint + retrieval mode
    # "fast" | "balanced" | "complex" → /ai-mode?q=...&mode=...
    # anything else (legacy "kiwix") → /ai
    if mode in ("fast", "balanced", "complex"):
        se_path   = "/ai-mode"
        se_params = {"q": query, "mode": mode}
    else:
        se_path   = "/ai"
        se_params = {"q": query}

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def _sse(text: str, finish: bool = False) -> str:
        return "data: " + json.dumps({
            "id": chat_id, "object": "chat.completion.chunk",
            "created": created, "model": "searchengine",
            "choices": [{"index": 0,
                         "delta": {"content": text},
                         "finish_reason": "stop" if finish else None}],
        }) + "\n\n"

    def _sse_sources(sources: list) -> str:
        return "data: " + json.dumps({
            "id": chat_id, "object": "chat.completion.chunk",
            "created": created, "model": "searchengine",
            "sources": sources,
            "choices": [{"index": 0, "delta": {"content": ""}}],
        }) + "\n\n"

    async def _stream():
        # For ZIM modes, kick off sources + answer stream in parallel.
        # Sources are fetched concurrently so we don't pay an extra round-trip
        # before the first token.
        if mode in ("fast", "balanced", "complex"):
            sources_task = asyncio.create_task(
                httpx.AsyncClient(timeout=15).get(
                    f"{SE_BASE}/zim-sources", params={"q": query, "mode": mode}
                )
            )
        else:
            sources_task = None

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream("GET", f"{SE_BASE}{se_path}",
                                         params=se_params) as resp:
                    first_chunk = True
                    async for chunk in resp.aiter_bytes():
                        # Emit sources before the first token (but don't wait for them)
                        if first_chunk and sources_task is not None:
                            first_chunk = False
                            if sources_task.done():
                                try:
                                    sr = sources_task.result()
                                    sources = sr.json().get("sources", [])
                                    if sources:
                                        yield _sse_sources(sources)
                                except Exception:
                                    pass
                        text = chunk.decode("utf-8", errors="replace")
                        if text:
                            yield _sse(text)
        except Exception as e:
            yield _sse(f"[Error contacting SearchEngine: {e}]", finish=True)
            yield "data: [DONE]\n\n"
            return
        finally:
            # If sources arrived after streaming started, send them now
            if sources_task is not None and not sources_task.cancelled():
                try:
                    sr = await sources_task
                    sources = sr.json().get("sources", [])
                    if sources:
                        yield _sse_sources(sources)
                except Exception:
                    pass

        yield _sse("", finish=True)
        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream",
                            headers={"X-Accel-Buffering": "no",
                                     "Cache-Control": "no-cache"})


# ── Cache clear — proxy to SearchEngine ──────────────────────────────────────
@app.get("/cache/clear")
async def cache_clear():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{SE_BASE}/cache/clear")
        return Response(r.content, media_type="application/json")
    except Exception as e:
        return Response(f'{{"error":"{e}"}}', media_type="application/json")


# ── Metrics — proxy to SearchEngine ──────────────────────────────────────────
@app.get("/metrics/dashboard")
async def metrics_dashboard():
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{SE_BASE}/metrics/dashboard")
    return Response(r.content, media_type="text/html")

@app.get("/api/metrics")
async def api_metrics():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{SE_BASE}/api/metrics")
        return Response(r.content, media_type="application/json")
    except Exception:
        return Response('{"error":"SearchEngine unavailable"}', media_type="application/json")

@app.get("/api/metrics/stream")
async def metrics_stream():
    async def _proxy():
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("GET", f"{SE_BASE}/api/metrics/stream") as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
    return StreamingResponse(_proxy(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── /context + /remember — stubs (server-side inference only) ─────────────────
@app.post("/context")
async def context(_: Request):
    return {"cache_hit": False, "error": "WebGPU mode not enabled"}

@app.post("/remember")
async def remember(_: Request):
    return {"ok": False}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050, log_level="info")
