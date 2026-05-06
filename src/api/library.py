"""
ZIM library management and on-demand article embedding.

GET  /api/libraries                              — list all ZIM handles
POST /api/libraries/rescan                       — rescan ZIM directory
PUT  /api/libraries/{name}                       — enable/disable or set count

POST /api/embed/{zim_name}                       — embed one article into FAISS
POST /api/libraries/{zim_name}/build-title-index — start title index build
GET  /api/libraries/{zim_name}/title-index-status
GET  /api/libraries/{zim_name}/embed-limits
"""
import threading
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from SearchEngine import zim_retrieval

router = APIRouter()

# Active title-index builds: {zim_name: {done, total, building, finished, error}}
_builds: dict[str, dict] = {}
_builds_lock = threading.Lock()


# ── Library management ────────────────────────────────────────────────────────

@router.get("/api/libraries")
def get_libraries():
    return JSONResponse({"libraries": zim_retrieval.get_all_handles()})


@router.post("/api/libraries/rescan")
def rescan_libraries():
    count = zim_retrieval.rescan()
    return JSONResponse({"ok": True, "count": count, "libraries": zim_retrieval.get_all_handles()})


@router.put("/api/libraries/{name}")
async def update_library(name: str, request: Request):
    body    = await request.json()
    enabled = body.get("enabled")
    count   = body.get("count")

    if count is not None:
        try:
            count = max(1, int(count))
        except (TypeError, ValueError):
            return JSONResponse({"ok": False, "error": "count must be an integer"}, status_code=400)

    result = zim_retrieval.update_handle(name, enabled=enabled, count=count)
    if result == "locked":
        return JSONResponse({"ok": False, "error": f"ZIM '{name}' is managed by admin"}, status_code=403)
    if not result:
        return JSONResponse({"ok": False, "error": f"ZIM '{name}' not found"}, status_code=404)

    zim_retrieval.write_config_zims()
    return JSONResponse({"ok": True})


# ── Article embedding ─────────────────────────────────────────────────────────

@router.post("/api/embed/{zim_name}")
async def embed_article(zim_name: str, request: Request):
    body  = await request.json()
    path  = body.get("path", "").strip()
    title = body.get("title", "").strip()

    if not path and not title:
        return JSONResponse({"ok": False, "error": "Provide 'path' or 'title'"}, status_code=400)

    if not path and title:
        import sqlite3
        from SearchEngine.zim_retrieval import _handles, _handles_lock
        from indexer.title_index import _title_db_path

        with _handles_lock:
            handle = next((h for h in _handles if h.name == zim_name), None)
        if not handle:
            return JSONResponse({"ok": False, "error": f"Article '{title}' not found in title index"}, status_code=404)

        db_path = _title_db_path(handle.zim_path)
        if db_path.exists():
            try:
                con = sqlite3.connect(str(db_path))
                row = con.execute(
                    "SELECT path FROM titles WHERE LOWER(title) = LOWER(?) LIMIT 1", (title,)
                ).fetchone()
                con.close()
                path = row[0] if row else ""
            except Exception:
                path = ""
        if not path:
            return JSONResponse({"ok": False, "error": f"Article '{title}' not found in title index"}, status_code=404)

    result = zim_retrieval.embed_article(zim_name, path)
    return JSONResponse(result, status_code=200 if result["ok"] else 400)


# ── Title index build ─────────────────────────────────────────────────────────

@router.post("/api/libraries/{zim_name}/build-title-index")
def start_title_index_build(zim_name: str):
    with _builds_lock:
        if _builds.get(zim_name, {}).get("building"):
            return JSONResponse({"ok": False, "error": "Already building"}, status_code=409)
        _builds[zim_name] = {"done": 0, "total": 0, "building": True, "finished": False, "error": None}

    def _run():
        def _progress(done, total):
            with _builds_lock:
                if zim_name in _builds:
                    _builds[zim_name]["done"]  = done
                    _builds[zim_name]["total"] = total

        result = zim_retrieval.build_title_index(zim_name, progress_cb=_progress)
        with _builds_lock:
            _builds[zim_name].update({
                "building": False,
                "finished": True,
                "error":    result.get("error"),
                "indexed":  result.get("indexed", 0),
            })

    threading.Thread(target=_run, daemon=True, name=f"title-index-{zim_name}").start()
    return JSONResponse({"ok": True, "message": "Title index build started"})


@router.get("/api/libraries/{zim_name}/title-index-status")
def title_index_status(zim_name: str):
    from indexer.title_index import title_db_count, title_db_exists
    from SearchEngine.zim_retrieval import _handles, _handles_lock

    with _handles_lock:
        handle = next((h for h in _handles if h.name == zim_name), None)
    if not handle:
        return JSONResponse({"ok": False, "error": "ZIM not found"}, status_code=404)

    with _builds_lock:
        state = _builds.get(zim_name, {})

    ready = title_db_exists(handle.zim_path)
    count = title_db_count(handle.zim_path)
    done  = state.get("done", 0)
    total = state.get("total", 0)
    pct   = round(done / total * 100, 1) if total > 0 else (100.0 if ready else 0.0)

    return JSONResponse({
        "ok":       True,
        "ready":    ready,
        "count":    count,
        "building": state.get("building", False),
        "finished": state.get("finished", False),
        "error":    state.get("error"),
        "progress": {"done": done, "total": total, "pct": pct},
    })


# ── Embed limits ──────────────────────────────────────────────────────────────

@router.get("/api/libraries/{zim_name}/embed-limits")
def get_embed_limits(zim_name: str):
    limits = zim_retrieval.get_embed_limits(zim_name)
    if limits is None:
        return JSONResponse({"ok": False, "error": "ZIM not found"}, status_code=404)
    return JSONResponse({"ok": True, **limits})
