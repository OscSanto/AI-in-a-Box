"""
On-demand article embedding and title index management API.

POST /api/embed/{zim_name}
    Body: {"path": "A/Heart_attack"} or {"title": "Heart attack"}
    Embeds the article's chunks and adds them to the ZIM's FAISS index.

POST /api/libraries/{zim_name}/build-title-index
    Triggers building the title index for a ZIM (background thread).
    Returns 409 if already building.

GET  /api/libraries/{zim_name}/title-index-status
    Returns {ready, count, building, progress}.

GET  /api/libraries/{zim_name}/embed-limits
    Returns {embedded_articles, max_articles, embedded_vectors, max_vectors, ...}.
"""
import threading
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from SearchEngine import zim_retrieval

router = APIRouter()

# Track active title-index builds: {zim_name: {done, total, finished, error}}
_builds: dict[str, dict] = {}
_builds_lock = threading.Lock()


# ── Embed a specific article ──────────────────────────────────────────────────

@router.post("/api/embed/{zim_name}")
async def embed_article(zim_name: str, request: Request):
    body  = await request.json()
    path  = body.get("path", "").strip()
    title = body.get("title", "").strip()

    if not path and not title:
        return JSONResponse({"ok": False, "error": "Provide 'path' or 'title'"}, status_code=400)

    # If title given, resolve to path via title index
    if not path and title:
        path = _resolve_title_to_path(zim_name, title)
        if not path:
            return JSONResponse({"ok": False, "error": f"Article '{title}' not found in title index"}, status_code=404)

    result = zim_retrieval.embed_article(zim_name, path)
    status = 200 if result["ok"] else 400
    return JSONResponse(result, status_code=status)


def _resolve_title_to_path(zim_name: str, title: str) -> str | None:
    """Look up a title in the title DB and return its ZIM path."""
    import sqlite3
    from SearchEngine.zim_retrieval import ZIM_INDEX_BASE, _handles, _handles_lock
    from indexer.title_index import _title_db_path

    with _handles_lock:
        handle = next((h for h in _handles if h.name == zim_name), None)
    if not handle:
        return None

    db_path = _title_db_path(handle.zim_path)
    if not db_path.exists():
        return None
    try:
        con = sqlite3.connect(str(db_path))
        row = con.execute(
            "SELECT path FROM titles WHERE LOWER(title) = LOWER(?) LIMIT 1", (title,)
        ).fetchone()
        con.close()
        return row[0] if row else None
    except Exception:
        return None


# ── Title index build ─────────────────────────────────────────────────────────

@router.post("/api/libraries/{zim_name}/build-title-index")
def start_title_index_build(zim_name: str):
    with _builds_lock:
        state = _builds.get(zim_name, {})
        if state.get("building"):
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
            _builds[zim_name]["building"]  = False
            _builds[zim_name]["finished"]  = True
            _builds[zim_name]["error"]     = result.get("error")
            _builds[zim_name]["indexed"]   = result.get("indexed", 0)

    threading.Thread(target=_run, daemon=True, name=f"title-index-{zim_name}").start()
    return JSONResponse({"ok": True, "message": "Title index build started"})


@router.get("/api/libraries/{zim_name}/title-index-status")
def title_index_status(zim_name: str):
    from indexer.title_index import title_db_count, title_db_exists
    from SearchEngine.zim_retrieval import _handles, _handles_lock, ZIM_INDEX_BASE

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
