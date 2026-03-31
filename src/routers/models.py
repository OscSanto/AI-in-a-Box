"""
Model management endpoints — /api/models/*

GET    /api/models/installed     list models installed in local Ollama
GET    /api/models/registry      fetch model list from ollama.com (with disk cache)
GET    /api/models/active        current llm_model and embedding_model from config
POST   /api/models/pull          pull a model from Ollama (streams progress via SSE)
DELETE /api/models/{name}        delete a model from Ollama
POST   /api/models/set-active    update config.yaml → restart server
"""

import json
import os
import sys
import threading
import time

import httpx
import yaml
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter()

OLLAMA_BASE = "http://localhost:11434"
REGISTRY_URL = "https://ollama.com/api/search"

# Name fragments that identify embedding models
_EMBED_PATTERNS = ["embed", "bge", "e5-", "arctic-embed", "minilm", "rerank", "jina"]
_EMBED_FAMILIES = {"bert", "nomic-bert"}


def _is_embedding(name: str, family: str = "") -> bool:
    n = name.lower()
    return any(p in n for p in _EMBED_PATTERNS) or family.lower() in _EMBED_FAMILIES


def _size_label(size_bytes: int) -> str:
    gb = size_bytes / 1024 ** 3
    if gb >= 1:
        return f"{gb:.1f} GB"
    return f"{size_bytes / 1024 ** 2:.0f} MB"


# ── Installed models ──────────────────────────────────────────────────────────

@router.get("/api/models/installed")
async def get_installed():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_BASE}/api/tags")
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        return {"error": str(e), "models": []}

    out = []
    for m in data.get("models", []):
        details = m.get("details", {})
        name = m.get("name", "")
        out.append({
            "name":         name,
            "size_bytes":   m.get("size", 0),
            "size_label":   _size_label(m.get("size", 0)),
            "param_size":   details.get("parameter_size", ""),
            "quantization": details.get("quantization_level", ""),
            "family":       details.get("family", ""),
            "is_embedding": _is_embedding(name, details.get("family", "")),
            "modified_at":  m.get("modified_at", ""),
        })
    return {"models": out}


# ── Registry (ollama.com) ─────────────────────────────────────────────────────

@router.get("/api/models/registry")
async def get_registry(request: Request):
    data_dir = request.app.state.config.data_dir
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, "models_registry_cache.json")

    # Always try fresh fetch; fall back to cache on failure
    try:
        all_models: list = []
        async with httpx.AsyncClient(timeout=12) as client:
            for page in range(1, 4):      # fetch up to 3 pages (~300 models)
                r = await client.get(REGISTRY_URL, params={"q": "", "p": page, "per_page": 100})
                r.raise_for_status()
                page_data = r.json()
                if isinstance(page_data, list):
                    if not page_data:
                        break
                    all_models.extend(page_data)
                elif isinstance(page_data, dict):
                    batch = page_data.get("models", [])
                    if not batch:
                        break
                    all_models.extend(batch)
                else:
                    break

        result = {"models": all_models, "offline": False, "cached_at": time.time()}
        with open(cache_path, "w") as f:
            json.dump(result, f)
        return result

    except Exception:
        pass

    # Offline fallback
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            cached["offline"] = True
            return cached
        except Exception:
            pass

    return {"models": [], "offline": True}


# ── Active config ─────────────────────────────────────────────────────────────

@router.get("/api/models/active")
async def get_active(request: Request):
    cfg = request.app.state.config
    return {
        "llm_model":       cfg.llm_model,
        "embedding_model": cfg.embedding_model,
    }


# ── Pull a model (streams Ollama pull progress as SSE) ───────────────────────

@router.post("/api/models/pull")
async def pull_model(request: Request):
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        return {"error": "name is required"}

    async def _stream():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST", f"{OLLAMA_BASE}/api/pull",
                    json={"name": name, "stream": True},
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line:
                            yield f"data: {line}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ── Delete a model ────────────────────────────────────────────────────────────

@router.delete("/api/models/{name:path}")
async def delete_model(name: str):
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.request(
                "DELETE", f"{OLLAMA_BASE}/api/delete",
                json={"name": name},
            )
            r.raise_for_status()
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}


# ── Set active model (updates config.yaml → restarts server) ─────────────────

def _wipe_vector_db(data_dir: str) -> list[str]:
    """Delete all FAISS / SQLite files so they rebuild from scratch after restart."""
    removed = []
    for fname in [
        "content_graph.db", "content_graph.faiss",
        "query_memory.db",  "query_memory.faiss",
    ]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            os.remove(path)
            removed.append(fname)
    return removed


def _schedule_restart():
    def _do():
        time.sleep(0.8)
        os.execv(sys.executable, [sys.executable] + sys.argv)
    threading.Thread(target=_do, daemon=True).start()


@router.post("/api/models/set-active")
async def set_active_model(request: Request):
    """
    Body: { type: "llm" | "embedder", model: "<name:tag>", wipe_db?: bool }

    For embedder changes, caller should pass wipe_db=true so the vector stores
    (built with the old model's embedding space) are deleted before restart.
    """
    body       = await request.json()
    model_type = body.get("type", "")          # "llm" or "embedder"
    model_name = body.get("model", "").strip()
    wipe_db    = body.get("wipe_db", False)

    if not model_type or not model_name:
        return {"error": "type and model are required"}
    if model_type not in ("llm", "embedder"):
        return {"error": "type must be 'llm' or 'embedder'"}

    cfg         = request.app.state.config
    config_path = request.app.state.config_path

    # ── Update config.yaml ──────────────────────────────────────────────────
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if model_type == "llm":
        raw.setdefault("llm", {})["model"] = model_name
        cfg.llm_model = model_name
    else:
        raw.setdefault("query_embedding", {})["model"] = model_name
        cfg.embedding_model = model_name

    with open(config_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # ── Optionally wipe vector stores ───────────────────────────────────────
    wiped = []
    if wipe_db:
        wiped = _wipe_vector_db(cfg.data_dir)

    _schedule_restart()
    return {"ok": True, "restarting": True, "wiped": wiped}
