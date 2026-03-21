import json
import os
import sys
import time
import uuid
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Ensure src/ is on path
sys.path.insert(0, os.path.dirname(__file__))

from utils.config import load_config, config_from_dict
from models.embedder import Embedder
from models import llm_client as llm
import retrieval.kiwix_client as kiwixClient

from graphs.content_graph import ContentGraph
from graphs.query_memory import QueryMemory
from runner import Runner
from routers import metrics as metrics_router

app = FastAPI()
app.include_router(metrics_router.router)



# webui/models
WEBUI_DIR = os.path.join(os.path.dirname(__file__), "..", "webui")
app.mount("/_app",    StaticFiles(directory=os.path.join(WEBUI_DIR, "_app")),    name="static")
app.mount("/webllm",  StaticFiles(directory=os.path.join(WEBUI_DIR, "webllm")),  name="webllm")
app.mount("/models",  StaticFiles(directory=os.path.join(WEBUI_DIR, "models")),  name="models")
#app.mount("/metrics",  StaticFiles(directory=os.path.join(WEBUI_DIR, "metrics")),  name="metrics")


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "exp_01_mini_query_memory_graph.yaml")
configpath = load_config(os.path.abspath(CONFIG_PATH))
_config = config_from_dict(configpath)

OLLAMA_SERVER_URL = "http://localhost:11434"

# The webui and our applications PATHS used via FastAPI server
LOCAL_PATHS = {"/", "/health", "/props", "/admin", "/admin/reset-cache", "/v1/models", "/v1/chat/completions",
               "/config", "/context", "/remember", "/relations",
               "/webgpu-interceptor.js", "/api/metrics"}

# ----- Initialize global components -----
#config.kiwix_endpoint is probed and updated to correct URL. Verifies and configures kiwix_endpoint for the rest of the app lifecycle.
    #dependent on IIAB on android or not.
_config.kiwix_endpoint = kiwixClient.probe_kiwix_endpoint(_config.kiwix_endpoint, _config.zim_content_id)
_zim_meta = kiwixClient.get_zim_metadata(_config.kiwix_endpoint, _config.zim_content_id)

_embedder      = Embedder(_config)
_content_graph = ContentGraph(_config, _embedder)
_query_memory  = QueryMemory(_config, _embedder)

_runner = Runner(_config, _embedder, llm, _content_graph, _query_memory)

# ------ App start -----
os.makedirs(_config.data_dir, exist_ok=True)

@app.on_event("startup")
async def startup():
    print(f"[MAIN] Experiment: {_config.experiment_name}  data_dir: {_config.data_dir}")
    print(f"[MAIN] ZIM metadata: {_zim_meta}")
    print(f"[MAIN] Embedder keep_alive: {_config.embedding_keep_alive}")
    if _content_graph.is_built():
        print("[MAIN] Content graph found.")
    else:
        print("[MAIN] Content graph not found. Build with first query.")

# ----- Middleware & LOCAL PATHS-----

# ON EVERY REQUEST: MIDDLEWARE HANDLES IT FIRST!!!
# Local or ollama forwarding

# Is path on LOCAL_PATHS?
    #NO -> Ollama port
    #YES -> FastAPI endpoint
"""
A "middleware" is a function that works with every 
    request before it is processed by any specific path operation. 
    And also with every response before returning it.

https://fastapi.tiangolo.com/tutorial/middleware/

"""
# webLLM will require this
class RelayMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request:Request, call_next):
        # on request
        path = request.url.path

        # If path is ours. Pass to call_next handler.
        # This routes to the FASTAPI endpoint we designated
        if path in LOCAL_PATHS or path.startswith("/_app/"):
            return await call_next(request)
        
        # The Ollama server call.
        # Forward to ollama. 
        # gets Ollama response
        ollama_url = f"{OLLAMA_SERVER_URL}{request.url.path}"
        body = b""
        async for chunk in request.stream():
            body += chunk
        async with httpx.AsyncClient() as client:
            req_data = {
                "method": request.method,
                "url": ollama_url,
                "headers": request.headers.raw,
                "params": request.query_params,
                "content": body,
            }
            # Ollama response
            response = await client.request(**req_data)
            return Response(response.content, status_code=response.status_code, headers=dict(response.headers))

app.add_middleware(RelayMiddleware)

# ----- APP ENDPOINTS -----

@app.get("/")
async def index():
    html = open(os.path.join(WEBUI_DIR, "index.html")).read()
    html = html.replace("</body>", '<script src="/webgpu-interceptor.js" defer></script></body>')
    return Response(html, media_type="text/html")

@app.get("/admin")
async def admin():
    return {"status": "ok"}


@app.post("/admin/reset-cache")
async def reset_cache():
    """Delete and reinitialize query memory (FAISS + SQLite). Fixes desync corruption."""
    db_path    = _runner.query_memory.db_path
    faiss_path = _runner.query_memory.faiss_path
    removed = []
    for path in [db_path, faiss_path]:
        if os.path.exists(path):
            os.remove(path)
            removed.append(os.path.basename(path))
    _runner.reset_query_memory(QueryMemory(_config, _embedder))
    print(f"[ADMIN] Cache reset. Removed: {removed}", flush=True)
    return {"ok": True, "removed": removed}



# ----- OLLAMA ENDPOINTS -----
"""
    DO NOT REMOVE
    -llama.cpp webui calls this to display and set UI defaults.
    -Seems to be required. Unknown why but removal ->  404 error code
    -TODO: fix. use our models, _config, yaml 
    -Apparently, (/props & /v1/models) are standard endpoints which webui index.html calls on startup
    -TODO: Check llama.cpp API source code
"""
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/props")
async def props():
    return {}

@app.get("/v1/models")
async def list_models():
    return {}


def _openai_stream(query: str, mode: str = "kiwix"):
    global _last_sources
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}" # Random UUID
    created = int(time.time()) # Timestamp this response
    model = _config.llm_model

    #Helper function
    def _make_chunk(text, finish_reason=None):
        return {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish_reason}],
        }

    _pending_sources: list = []
    _sources_emitted = False

    def _on_sources(sources):
        _pending_sources.extend(sources)

    try:
        for text_chunk in _runner.run(query, mode=mode, zim_meta=_zim_meta, on_sources=_on_sources):
            # Emit sources chunk before the first text token so the UI can show
            # article links while generation is still in progress.
            if not _sources_emitted and _pending_sources:
                sources_chunk = {
                    "id": chat_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "sources": _pending_sources,
                    "choices": [{"index": 0, "delta": {"content": ""}}],
                }
                yield f"data: {json.dumps(sources_chunk)}\n\n"
                _sources_emitted = True
            yield f"data: {json.dumps(_make_chunk(text_chunk))}\n\n"

    except Exception as e:
        err_chunk = _make_chunk(f"[ERROR] {e}", finish_reason="stop")
        yield f"data: {json.dumps(err_chunk)}\n\n"

    yield f"data: {json.dumps(_make_chunk('', finish_reason='stop'))}\n\n"
    yield "data: [DONE]\n\n"

# ----- MAIN CHAT ENDPOINT -----
# User's chat POST
# All user content gets POST every query
# TODO: Handle images, pdfs, other media.
# TODO: Code is unreasonable unless paired with mutli context
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):

    body = await request.json()
    messages = body.get("messages", [])
    mode = body.get("mode", "kiwix")   # set by UI mode buttons; defaults to kiwix

    query = ""
    # Messages: OLD -> RECENT
    # On TURN, resend history
    for m in  reversed(messages):
        if m.get("role") ==  "user":
            content = m.get("content", "")

            query = content if isinstance(content, str) else " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
            break
    return StreamingResponse(_openai_stream(query, mode=mode), media_type="text/event-stream")


# ----- WebGPU client endpoints -----
"""
SCRIPT INJECTION PATH
Script injection. Serve .js file 
Client loads this script to intercept WebGPU calls (hook into browser's GPU API before WebLLM)
-Monkey-patches intercepts any call to /v1/chat/complection before it reaches server
-Happens only on '[x] Device GPU
-Inference would work on the UI thread

WebLLM
-Only runs model compiled in WebGPU format (MLC-compiled .wasm + weight shards)
https://webllm.mlc.ai/docs/user/basic_usage.html

https://github.com/mlc-ai/web-llm/tree/main/examples/simple-chat-upload

CACHE API BUCKETS 
Servers have built in storage called Cache API that acts like key-value store
-key is a URL and value is a file (response object client receives). 
-A "bucket' is a named container (similar to a folder)

WebLLM uses SPECIFIC BUCKET NAMES
-webllm/model, webllm/wasm, webllm/config
-On Load: buckets checked first before network

Worker.ts gets compiled into a web worker on seperate thread. 
-Use WebWorkerMLCEngine + worker.ts for threading leaves UI responsiveness safe.
-Upload model files to webui/models/
-Client fetches /models/ and cache them into same Cache API buckets
-Client next visit: WebLLM finds them cached. No redownload. Offline after first load



"""
@app.get("/webgpu-interceptor.js")
async def webgpu_interceptor():
    return FileResponse(os.path.join(WEBUI_DIR, "webgpu-interceptor.js"),
                        media_type="application/javascript")

#    """Serve the WebGPU client page."""
#@app.get("/{_config.experiment_alias}/webgpu")
#async def exp01_page():
#    return FileResponse(os.path.join(WEBUI_DIR, "exp01.html"))


@app.get("/config")
async def client_config():
    """
    Returns client config 
    """
    return {
        "inference_default_mode": _config.inference_default_mode,
        "inference_allow_user_toggle": _config.inference_allow_user_toggle,
    }


@app.post("/context")
async def client_context(request: Request):
    """
    Runs RAG stages 1–10, returns context + assembled prompt as JSON.
    Client uses this to run LLM generation locally via WebGPU (WebLLM).

    on cache_hit: returns the cached answer directly thus no client generation needed.
    query_vec is included so the client can POST it back to /remember.
    """
    body = await request.json()
    query = body.get("query", "").strip()
    mode  = body.get("mode", "kiwix")
    if not query:
        return {"error": "query is required"}

    result = _runner.build_context(query, zim_meta=_zim_meta, mode=mode)

    if result["cache_hit"]:
        return {"cache_hit": True, "answer": result["cached_answer"]}

    return {
        "cache_hit": False,
        "query": query,
        "context": result["context"],
        "prompt": f"Context:\n{result['context']}\n\nQuestion: {query}",
        "query_vec": result["query_vec"],
        "entities": result["entity_title_candidates"],  # returned so client stores for next turn's rewrite
        "webllm_model": "SmolLM2-360M-Instruct-q4f16_1-MLC",
        # tells client whether to extract relations and POST to /relations
        "extract_relations": _config.entity_llm_enabled and _config.entity_llm_mode == "client",
    }


@app.post("/relations")
async def store_relations(request: Request):
    """
    Client-mode relation extraction: WebLLM extracts entity→relation→entity triples
    from the chunks returned by /context and POSTs them here for graph storage.
    Body: {"triples": [{"from": "...", "relation": "...", "to": "..."}]}
    """
    body = await request.json()
    triples = body.get("triples", [])
    if not triples:
        return {"ok": True, "stored": 0}
    _content_graph.store_relations(triples, _zim_meta)
    return {"ok": True, "stored": len(triples)}


@app.post("/remember")
async def exp01_remember(request: Request):
    """
    Stores a client-generated (WebGPU) answer into query memory.
    Client POSTs {query, answer, query_vec} after local generation completes.
    This should occur twice if client uses WebLLM semantic understanding and again on generation
    This allows for client to do attempt cache-hit or future identical/similar queries to get a cache hit.
    """
    body = await request.json()
    query = body.get("query", "").strip()
    answer = body.get("answer", "").strip()
    query_vec = body.get("query_vec", [])
    if not query or not answer or not query_vec:
        return {"error": "query, answer, and query_vec are required"}

    llm.store_client_answer(query, answer, query_vec, _config, _runner.query_memory)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    host = _config.server_host
    port = _config.server_port
    uvicorn.run(app, host=host, port=port, log_level="info")
