"""LLM model management — list models and get/set the active model."""
import threading

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from SearchEngine.config import CFG, LLAMACPP_CFG

router = APIRouter()


def get_active_model() -> str:
    from api.llamacpp_process import active_model
    return active_model()


def _is_embedding_model(model_id: str, families: list[str]) -> bool:
    low = model_id.lower()
    if any(h in low for h in ("embed", "minilm", "e5-", "bge-", "gte-", "stella", "nomic-embed")):
        return True
    return bool({"bert", "nomic-bert", "nomic-new-embed"} & {f.lower() for f in (families or [])})


@router.get("/v1/models")
def list_models():
    """List available GGUF models from model_dir."""
    from api.llamacpp_process import active_model, list_gguf_models
    current = active_model()
    files = list_gguf_models()
    models_data, models_detail = [], []
    for f in files:
        models_data.append({
            "id": f, "object": "model", "owned_by": "llamacpp",
            "in_cache": True, "path": f,
            "status": {"value": "loaded" if f == current else "idle"},
            "backend": "Pi Local (llama.cpp)",
        })
        models_detail.append({
            "id": f, "name": f, "model": f,
            "description": f,
            "capabilities": [], "backend": "Pi Local (llama.cpp)",
            "details": {},
        })
    return {"object": "list", "data": models_data, "models": models_detail, "backend": "Pi Local (llama.cpp)"}


@router.get("/api/model-store")
def model_store():
    """Return the curated list of recommended models for the Download panel."""
    return CFG.get("model_store", {"families": []})


@router.get("/api/hf/search")
async def hf_search(q: str = "", limit: int = 20):
    """Proxy Hugging Face model search, filtered to GGUF models."""
    params = {
        "filter": "gguf", "sort": "downloads", "direction": "-1",
        "limit": min(limit, 50), "full": "false",
    }
    if q.strip():
        params["search"] = q.strip()
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get("https://huggingface.co/api/models", params=params)
            r.raise_for_status()
        results = [
            {
                "id":           m.get("modelId") or m.get("id") or "",
                "ollama_tag":   f"hf.co/{m.get('modelId') or m.get('id') or ''}",
                "downloads":    m.get("downloads", 0),
                "likes":        m.get("likes", 0),
                "pipeline_tag": m.get("pipeline_tag", ""),
            }
            for m in r.json()
            if m.get("modelId") or m.get("id")
        ]
        return {"models": results}
    except Exception as e:
        return JSONResponse({"error": str(e), "models": []}, status_code=502)


@router.post("/api/set-model")
async def set_model_endpoint(request: Request):
    """Switch the active generation model at runtime."""
    body = await request.json()
    model = (body.get("model") or "").strip()
    if not model:
        return JSONResponse({"error": "model required"}, status_code=400)

    from api.llamacpp_process import start
    print(f"[model] Restarting llama-server with model → {model}", flush=True)
    def _restart():
        try:
            start(model)
            print(f"[model] llama-server ready with {model}", flush=True)
        except Exception as e:
            print(f"[model] llama-server failed to start: {e}", flush=True)
    threading.Thread(target=_restart, daemon=True).start()
    return {"model": model, "status": "starting"}


@router.get("/api/llamacpp/repo-files")
async def llamacpp_repo_files(repo: str = ""):
    """List GGUF files available in a HuggingFace repo."""
    if not repo.strip():
        return JSONResponse({"error": "repo required"}, status_code=400)
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"https://huggingface.co/api/models/{repo.strip()}")
            r.raise_for_status()
        data = r.json()
        files = [
            {
                "filename": s["rfilename"],
                "size":     s.get("size", 0),
                "url":      f"https://huggingface.co/{repo.strip()}/resolve/main/{s['rfilename']}",
            }
            for s in data.get("siblings", [])
            if s.get("rfilename", "").endswith(".gguf")
        ]
        files.sort(key=lambda f: f["filename"])
        return {"repo": repo.strip(), "files": files}
    except Exception as e:
        return JSONResponse({"error": str(e), "files": []}, status_code=502)


@router.post("/api/llamacpp/download")
async def llamacpp_download(request: Request):
    """Stream download of a GGUF file into model_dir. Yields SSE progress events."""
    import json as _json

    body = await request.json()
    url      = (body.get("url") or "").strip()
    filename = (body.get("filename") or url.split("/")[-1]).strip()
    if not url:
        return JSONResponse({"error": "url required"}, status_code=400)
    if not filename.endswith(".gguf"):
        return JSONResponse({"error": "filename must end in .gguf"}, status_code=400)

    from api.llamacpp_process import _model_dir
    dest = _model_dir() / filename

    async def _stream():
        try:
            async with httpx.AsyncClient(timeout=600) as c:
                async with c.stream("GET", url, follow_redirects=True) as r:
                    r.raise_for_status()
                    total = int(r.headers.get("content-length", 0))
                    received = 0
                    with open(dest, "wb") as f:
                        async for chunk in r.aiter_bytes(65536):
                            f.write(chunk)
                            received += len(chunk)
                            pct = int(received * 100 / total) if total else 0
                            yield f"data: {_json.dumps({'received': received, 'total': total, 'pct': pct})}\n\n"
            yield f"data: {_json.dumps({'done': True, 'filename': filename})}\n\n"
        except Exception as e:
            try:
                dest.unlink(missing_ok=True)
            except Exception:
                pass
            yield f"data: {_json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.get("/api/llamacpp/status")
def llamacpp_status():
    """Return whether llama-server is ready to accept requests."""
    from api.llamacpp_process import is_running, active_model
    url = LLAMACPP_CFG.get("url", "http://localhost:8080")
    ready = False
    if is_running():
        try:
            r = httpx.get(f"{url}/health", timeout=2)
            ready = r.status_code == 200
        except Exception:
            pass
    return {"ready": ready, "model": active_model()}
