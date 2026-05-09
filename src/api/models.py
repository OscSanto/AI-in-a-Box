"""LLM model management — list models and get/set the active model."""
import threading

import httpx
import ollama
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from SearchEngine.config import AI_MODE_LLM_MODEL, CFG

router = APIRouter()

_OLLAMA_SAMPLING_DEFAULTS: dict = {
    "temperature":        0.8,
    "top_k":              40,
    "top_p":              0.95,
    "min_p":              0.05,
    "repeat_last_n":      64,
    "repeat_penalty":     1.1,
    "presence_penalty":   0.0,
    "frequency_penalty":  0.0,
    "dry_multiplier":     0.0,
    "dry_base":           1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": -1,
    "num_predict":        -1,
}

OLLAMA_SUPPORTED_KEYS: frozenset = frozenset(_OLLAMA_SAMPLING_DEFAULTS) | {"num_predict"}

_active_model_lock = threading.Lock()
_active_model: str = AI_MODE_LLM_MODEL


def get_active_model() -> str:
    with _active_model_lock:
        return _active_model


def _is_embedding_model(model_id: str, families: list[str]) -> bool:
    low = model_id.lower()
    if any(h in low for h in ("embed", "minilm", "e5-", "bge-", "gte-", "stella", "nomic-embed")):
        return True
    return bool({"bert", "nomic-bert", "nomic-new-embed"} & {f.lower() for f in (families or [])})


@router.get("/v1/models")
def list_models():
    """List available LLMs from local Ollama (excludes embedding models)."""
    try:
        client = ollama.Client(host="http://localhost:11434", timeout=10)
        result = client.list()
        models_data, models_detail = [], []
        for m in result.models:
            mid = m.model or m.name or ""
            det = m.details
            families = list(det.families) if det and det.families else []
            if _is_embedding_model(mid, families):
                continue
            models_data.append({
                "id": mid, "object": "model", "owned_by": "ollama",
                "created": int(m.modified_at.timestamp()) if m.modified_at else 0,
                "in_cache": True, "path": mid,
                "status": {"value": "loaded"}, "backend": "Pi Local",
            })
            models_detail.append({
                "id": mid, "name": mid, "model": mid,
                "description": " · ".join(filter(None, [
                    det.family if det else None,
                    det.parameter_size if det else None,
                    det.quantization_level if det else None,
                ])) or mid,
                "capabilities": [], "backend": "Pi Local",
                "details": {
                    "parameter_size":     det.parameter_size if det else None,
                    "family":             det.family if det else None,
                    "families":           families,
                    "quantization_level": det.quantization_level if det else None,
                    "size":               m.size,
                },
            })
        return {"object": "list", "data": models_data, "models": models_detail, "backend": "Pi Local"}
    except Exception:
        return {"object": "list", "data": [], "models": [], "backend": "Pi Local"}


@router.get("/api/sampling-config")
def sampling_config():
    """Sampling parameter defaults and per-mode caps for the settings UI."""
    from SearchEngine.zim_retrieval import _MODE_CFGS
    modes = {}
    for name, cfg in _MODE_CFGS.items():
        modes[name] = {
            "llm_options": cfg.get("llm_options", {}),
            "caps": cfg.get("caps", {}),
        }
    return {"ollama_defaults": _OLLAMA_SAMPLING_DEFAULTS, "modes": modes}


@router.get("/api/model-store")
def model_store():
    """Return the curated list of recommended models for the Download panel."""
    return CFG.get("model_store", {"families": []})


@router.get("/api/hf/search")
async def hf_search(q: str = "", limit: int = 20):
    """Proxy Hugging Face model search, filtered to GGUF models pullable by Ollama."""
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


@router.post("/api/ollama/pull")
async def pull_model(request: Request):
    """Stream an Ollama model pull, forwarding progress events to the UI."""
    body = await request.json()
    model = (body.get("model") or "").strip()
    if not model:
        return JSONResponse({"error": "model required"}, status_code=400)

    import json as _json

    async def _stream_pull():
        try:
            async with httpx.AsyncClient(timeout=600) as c:
                async with c.stream("POST", "http://localhost:11434/api/pull",
                                    json={"name": model}) as r:
                    async for line in r.aiter_lines():
                        if line:
                            yield f"data: {line}\n\n"
        except Exception as e:
            yield f"data: {_json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream_pull(), media_type="text/event-stream")


@router.post("/api/set-model")
async def set_model_endpoint(request: Request):
    """Switch the active generation model at runtime (no restart needed)."""
    global _active_model
    body = await request.json()
    model = (body.get("model") or "").strip()
    if not model:
        return JSONResponse({"error": "model required"}, status_code=400)
    with _active_model_lock:
        _active_model = model
    print(f"[model] Switched active model → {model}", flush=True)
    return {"model": model}
