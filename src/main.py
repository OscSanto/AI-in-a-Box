"""
WebUI + RAG server.
Concerns are split across api/ routers:
  api/models.py   — model discovery and runtime model switching
  api/chat.py     — /v1/chat/completions (RAG pipeline + direct chat)
  api/metrics.py  — system metrics
  api/library.py  — ZIM library management + on-demand article embedding
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
# from starlette.middleware.proxy_headers import ProxyHeadersMiddleware
# ProxyHeadersMiddleware: Cloudflare and nginx forward the real scheme/host via
# X-Forwarded-* headers. This patches request.url so url_for() produces correct
# absolute URLs (e.g. https://ai-iab.com/auth/callback instead of localhost:5050).
# trusted_hosts="*" is safe while Cloudflare/nginx is the only upstream — if this
# server is ever exposed directly to the internet, lock it to known proxy IPs.
# Uncomment when adding OAuth or other IIAB services that need absolute callback URLs:
# app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

from SearchEngine.cache import init_db
from api.chat import router as chat_router
from api.library import router as library_router
from api.metrics import router as metrics_router
from api.models import router as models_router

WEBUI_DIR    = os.path.join(os.path.dirname(__file__), "..", "webui")
# _RRUNTIME_DIR stores files (cache.db, llm_metrics, etc.) needed at runtime, 
# KEEP GENEREATED FILES OUT OF THE REPO. 
_RUNTIME_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent / "runtime"

@asynccontextmanager
async def lifespan(app: FastAPI):
    _RUNTIME_DIR.mkdir(exist_ok=True)
    init_db(str(_RUNTIME_DIR / "cache.db"))
    from SearchEngine.config import LLAMACPP_CFG
    model = LLAMACPP_CFG.get("model", "").strip()
    if model:
        import threading
        from api.llamacpp_process import start as _lc_start
        def _boot():
            try:
                _lc_start(model)
                print(f"[llamacpp] Server ready with {model}", flush=True)
            except Exception as e:
                print(f"[llamacpp] Failed to start llama-server: {e}", flush=True)
        threading.Thread(target=_boot, daemon=True).start()
    else:
        print("[llamacpp] No model set in config.yaml — set llamacpp.model to auto-start", flush=True)
    yield
    from api.llamacpp_process import stop as _lc_stop
    _lc_stop()

app = FastAPI(lifespan=lifespan)
app.include_router(models_router)
app.include_router(chat_router)
app.include_router(metrics_router)
app.include_router(library_router)

# Serve WebUI static files (JS/CSS) from /_app, and index.html from webui root.
app.mount("/_app", StaticFiles(directory=os.path.join(WEBUI_DIR, "_app")))


@app.get("/")
def index():
    return FileResponse(os.path.join(WEBUI_DIR, "index.html"))

# TODO: (optional) check if endpoint is REQUIRED by Ollama? 
# can be used to check server /health before routing traffic. 
# if ollama or nginx is not doing health check, can be removed.
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/config")
def client_config():
    """ Called on every request to /api/config — typically once when the frontend
    loads (chat.svelte.ts fetches it on mount) and again if the page is refreshed.
    Imports are deferred here to avoid circular imports between main, api.chat,
    and SearchEngine at module load time. """
    from api.chat import _available_zim_names

    return {
        "zims": _available_zim_names(),
    }

if __name__ == "__main__":
    import uvicorn
    from SearchEngine.config import CFG as _CFG
    host = os.environ.get("HOST") or _CFG.get("host", "0.0.0.0")
    port = int(os.environ.get("PORT") or _CFG.get("port", 5050))
    uvicorn.run(app, host=host, port=port, log_level="info")
