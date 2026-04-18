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
