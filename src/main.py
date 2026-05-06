"""
Single-process WebUI + RAG server.

Serves the built WebUI and implements the OpenAI-compatible
/v1/chat/completions endpoint directly. Retrieval/generation is imported as
Python engine code instead of proxied to a second SearchEngine HTTP server.

Concerns are split across api/ routers:
  api/backends.py — inference backend priority chain (Android mDNS + user remotes + Pi Local)
  api/models.py   — model discovery, runtime model switching, curated store, HF search
  api/chat.py     — /v1/chat/completions (RAG pipeline + direct chat)
  api/metrics.py  — LLM performance metrics
  api/library.py  — ZIM library management + on-demand article embedding + title index build
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from SearchEngine.cache import init_db
from SearchEngine.metrics import llm_metrics
from api.backends import router as backends_router
from api.backends import start as backends_start, stop as backends_stop
from api.chat import router as chat_router
from api.library import router as library_router
from api.metrics import router as metrics_router
from api.models import router as models_router

# ── Config ────────────────────────────────────────────────────────────────────

WEBUI_DIR  = os.path.join(os.path.dirname(__file__), "..", "webui")
_CFG_DIR   = os.path.dirname(os.path.abspath(__file__))
_RUNTIME_DIR   = Path(_CFG_DIR).parent / "runtime"
_BACKENDS_FILE = _RUNTIME_DIR / "backends.json"


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _RUNTIME_DIR.mkdir(exist_ok=True)
    init_db(str(_RUNTIME_DIR / "cache.db"))
    llm_metrics.init(str(_RUNTIME_DIR))
    await backends_start(_BACKENDS_FILE)
    yield
    await backends_stop()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan)

app.include_router(backends_router)
app.include_router(models_router)
app.include_router(chat_router)
app.include_router(metrics_router)
app.include_router(library_router)

app.mount("/_app", StaticFiles(directory=os.path.join(WEBUI_DIR, "_app")), name="app_static")


# ── Static + health routes ────────────────────────────────────────────────────

@app.get("/")
def index():
    html = open(os.path.join(WEBUI_DIR, "index.html"), encoding="utf-8").read()
    html = html.replace(
        "</body>",
        '<script src="/webgpu-interceptor.js" defer></script></body>',
    )
    return Response(html, media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok"}

# Note: the client needs to know the available ZIMs and some config options at runtime, so we expose a simple endpoint for that.
# 
@app.get("/config")
@app.get("/api/config")
def client_config():
    """Runtime config for the WebUI (mode, ZIM list)."""
    from api.chat import _available_zim_names
    from SearchEngine.config import CFG, _load_prompt
    from SearchEngine.zim_retrieval import _MODE_CFGS

    def _mode_entry(name: str) -> dict:
        mode_cfg = _MODE_CFGS.get(name, {})
        prompt_path = mode_cfg.get("system_prompt")
        return {
            "system_prompt_path": prompt_path,
            "system_prompt": _load_prompt(prompt_path) if prompt_path else "",
        }

    return {
        "inference_default_mode": "server",
        "inference_allow_user_toggle": True,
        "zims": _available_zim_names(),
        "flash_attention": bool(CFG.get("ollama_flash_attention", False)),
        "modes": {name: _mode_entry(name) for name in ("chat", "fast", "balanced", "complex")},
    }


@app.get("/webgpu-interceptor.js")
def webgpu_interceptor():
    return FileResponse(
        os.path.join(WEBUI_DIR, "webgpu-interceptor.js"),
        media_type="application/javascript",
    )


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5050"))
    uvicorn.run(app, host=host, port=port, log_level="info")
