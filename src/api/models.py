"""LLM model management — list models and get/set the active model."""
import threading

import ollama
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from SearchEngine.config import AI_MODE_LLM_MODEL

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
