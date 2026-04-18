"""
Inference backend management.

Supports a priority-ordered chain of Ollama-compatible backends:
  1. Android phone running the MLC inference app (priority 1, auto-discovered via mDNS)
  2. User-configured remote machines (priority 2..N, persisted in backends.json)
  3. Pi Local Ollama (priority 999, always-present fallback)

At query time, _select_backend() health-checks backends in priority order and
returns the first healthy one. This means a phone on the same Wi-Fi automatically
offloads generation; when the phone leaves, the Pi falls back to local Ollama.
"""
import asyncio
import json
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

# ── Backend state ─────────────────────────────────────────────────────────────

_PI_LOCAL_BACKEND = {
    "id": "pi-local", "name": "Pi Local", "url": "http://localhost:11434",
    "priority": 999, "builtin": True, "enabled": True,
}

# Android backend is managed automatically by mDNS — never persisted.
_android_backend: dict = {
    "id": "android", "name": "Phone (MLC)", "url": "",
    "priority": 1, "builtin": True, "enabled": False,
}

_user_backends: list[dict] = []
_backends_lock = asyncio.Lock()

# Set by start() from main.py's runtime dir
_backends_file: Path | None = None

# Background task handles (managed by lifespan)
_mdns_manager: object = None
_android_health_task: asyncio.Task | None = None


# ── Lifespan helpers (called from main.py) ────────────────────────────────────

async def start(backends_file: Path) -> None:
    """Load persisted backends and start mDNS + Android health loop."""
    global _user_backends, _backends_file, _mdns_manager, _android_health_task
    _backends_file = backends_file
    _user_backends = _load_user_backends()

    try:
        from mdns_manager import MdnsManager
        _mdns_manager = MdnsManager(
            on_android_found=_on_android_found,
            on_android_lost=_on_android_lost,
        )
        await _mdns_manager.start()
    except Exception as e:
        print(f"[mdns] Failed to start (zeroconf installed?): {e}", flush=True)

    _android_health_task = asyncio.create_task(_android_health_loop())


async def stop() -> None:
    """Cancel health loop and stop mDNS on app shutdown."""
    if _android_health_task:
        _android_health_task.cancel()
    if _mdns_manager:
        await _mdns_manager.stop()


# ── Backend helpers ───────────────────────────────────────────────────────────

def _load_user_backends() -> list[dict]:
    if _backends_file and _backends_file.exists():
        try:
            return json.loads(_backends_file.read_text())
        except Exception:
            pass
    return []


def _save_user_backends(backends: list[dict]) -> None:
    if _backends_file:
        _backends_file.write_text(json.dumps(backends, indent=2))


def _all_backends_sorted() -> list[dict]:
    """All enabled backends with a URL, sorted by priority (lower = first)."""
    candidates = [_android_backend] + _user_backends + [_PI_LOCAL_BACKEND]
    return sorted(
        [b for b in candidates if b.get("enabled") and b.get("url")],
        key=lambda b: b["priority"],
    )


async def _backend_healthy(url: str, timeout: float = 2.5) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"{url}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


async def select_backend() -> tuple[str, str] | None:
    """Return (name, url) of the first healthy backend in priority order, or None."""
    for b in _all_backends_sorted():
        if await _backend_healthy(b["url"]):
            return b["name"], b["url"]
    return None


# ── Android / mDNS callbacks ──────────────────────────────────────────────────
# The Android MLC inference app broadcasts itself via mDNS (_ollama._tcp).
# When found it's promoted to priority-1 backend; when lost it's disabled.

async def _on_android_found(host: str, port: int) -> None:
    global _android_backend
    url = f"http://{host}:{port}"
    async with _backends_lock:
        _android_backend = {**_android_backend, "url": url, "enabled": True}
    print(f"[mdns] Android inference app found at {url}", flush=True)


async def _on_android_lost() -> None:
    global _android_backend
    async with _backends_lock:
        _android_backend = {**_android_backend, "url": "", "enabled": False}
    print("[mdns] Android inference app lost", flush=True)


async def _android_health_loop() -> None:
    """Poll Android backend every 15 s; mark unavailable on failure."""
    while True:
        await asyncio.sleep(15)
        async with _backends_lock:
            url = _android_backend.get("url", "")
        if not url:
            continue
        if not await _backend_healthy(url):
            await _on_android_lost()
            print("[mdns] Android health check failed — marked unavailable", flush=True)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/api/backends")
async def get_backends():
    """Return all configured backends with live health status."""
    async def _with_health(b: dict) -> dict:
        healthy = await _backend_healthy(b["url"]) if b.get("url") else False
        return {**b, "healthy": healthy}

    all_known = [_android_backend, *_user_backends, _PI_LOCAL_BACKEND]
    results = await asyncio.gather(*[_with_health(b) for b in all_known])
    return {"backends": list(results)}


@router.post("/api/backends")
async def save_backends(request: Request):
    """Persist user-configured (non-builtin) backends."""
    global _user_backends
    body = await request.json()
    incoming: list[dict] = body.get("backends", [])

    cleaned: list[dict] = []
    for i, b in enumerate(incoming):
        url = (b.get("url") or "").strip()
        name = (b.get("name") or "").strip() or f"Backend {i + 1}"
        if not url:
            continue
        cleaned.append({
            "id":       b.get("id") or f"user-{uuid.uuid4().hex[:8]}",
            "name":     name,
            "url":      url,
            "priority": int(b.get("priority", 10 + i)),
            "builtin":  False,
            "enabled":  bool(b.get("enabled", True)),
        })

    async with _backends_lock:
        _user_backends = cleaned
    _save_user_backends(cleaned)
    return {"ok": True, "count": len(cleaned)}


@router.get("/api/backends/health")
async def probe_backend_health(url: str = ""):
    """Test reachability of an arbitrary URL (used by the Test button in Settings)."""
    if not url:
        return JSONResponse({"error": "url required"}, status_code=400)
    healthy = await _backend_healthy(url.strip(), timeout=4.0)
    models: list[str] = []
    if healthy:
        try:
            async with httpx.AsyncClient(timeout=4.0) as c:
                r = await c.get(f"{url.strip()}/api/tags")
                models = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception:
            pass
    return {"healthy": healthy, "url": url.strip(), "models": models}
