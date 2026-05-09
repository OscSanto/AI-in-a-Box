"""
ZIM library management and on-demand article embedding.

GET  /api/libraries              — list all ZIM handles
POST /api/libraries/rescan       — rescan ZIM directory
PUT  /api/libraries/{name}       — enable/disable or set count
POST /api/embed/{zim_name}       — embed one article into FAISS
GET  /api/libraries/{zim_name}/embed-limits
"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from SearchEngine import zim_retrieval

router = APIRouter()


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
        from SearchEngine.title_index import _title_db_path

        with _handles_lock:
            handle = next((h for h in _handles if h.name == zim_name), None)
        if not handle:
            return JSONResponse({"ok": False, "error": f"Article '{title}' not found"}, status_code=404)

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
            return JSONResponse({"ok": False, "error": f"Article '{title}' not found"}, status_code=404)

    result = zim_retrieval.embed_article(zim_name, path)
    return JSONResponse(result, status_code=200 if result["ok"] else 400)


@router.get("/api/libraries/{zim_name}/embed-limits")
def get_embed_limits(zim_name: str):
    limits = zim_retrieval.get_embed_limits(zim_name)
    if limits is None:
        return JSONResponse({"ok": False, "error": "ZIM not found"}, status_code=404)
    return JSONResponse({"ok": True, **limits})
