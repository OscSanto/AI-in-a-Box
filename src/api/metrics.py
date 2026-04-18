import asyncio
import json
import os
import time

import psutil
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from SearchEngine.cache import db_clear_ai
from SearchEngine.metrics import llm_metrics

router = APIRouter()

_METRICS_HTML = os.path.join(os.path.dirname(__file__), "..", "SearchEngine", "metrics", "metrics_dashboard.html")
_BOOT_TIME    = time.time()
_OWN_PROC     = psutil.Process(os.getpid())


def _safe(fn, fallback=None):
    try:
        return fn()
    except Exception:
        return fallback


def _cpu():
    pcts = psutil.cpu_percent(percpu=True)
    freq = psutil.cpu_freq()
    return {
        "per_core_pct": pcts,
        "avg_pct": round(sum(pcts) / len(pcts), 1) if pcts else 0,
        "count_logical": psutil.cpu_count(logical=True),
        "count_physical": psutil.cpu_count(logical=False),
        "freq_mhz": round(freq.current) if freq else None,
        "freq_max_mhz": round(freq.max) if freq else None,
    }


def _memory():
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "ram_used_mb": round(ram.used / 1024**2),
        "ram_total_mb": round(ram.total / 1024**2),
        "ram_pct": ram.percent,
        "ram_avail_mb": round(ram.available / 1024**2),
        "swap_used_mb": round(swap.used / 1024**2),
        "swap_total_mb": round(swap.total / 1024**2),
        "swap_pct": swap.percent,
    }


def _disk():
    u = psutil.disk_usage(os.getcwd())
    io = psutil.disk_io_counters()
    return {
        "used_gb": round(u.used / 1024**3, 2),
        "total_gb": round(u.total / 1024**3, 2),
        "free_gb": round(u.free / 1024**3, 2),
        "pct": u.percent,
        "read_mb": round(io.read_bytes / 1024**2) if io else None,
        "write_mb": round(io.write_bytes / 1024**2) if io else None,
    }


def _temperatures():
    out = []
    for chip, entries in (psutil.sensors_temperatures() or {}).items():
        for e in entries:
            out.append({
                "chip": chip,
                "label": e.label or chip,
                "current": e.current,
                "high": e.high,
                "critical": e.critical,
            })
    return out


def _process():
    mem = _OWN_PROC.memory_info()
    try:
        open_files = _OWN_PROC.num_fds()
    except Exception:
        open_files = len(_OWN_PROC.open_files())
    return {
        "pid": _OWN_PROC.pid,
        "rss_mb": round(mem.rss / 1024**2, 1),
        "vms_mb": round(mem.vms / 1024**2, 1),
        "cpu_pct": _OWN_PROC.cpu_percent(),
        "threads": _OWN_PROC.num_threads(),
        "open_files": open_files,
    }


def _network():
    out = []
    try:
        io_counters = psutil.net_io_counters(pernic=True)
        addrs = psutil.net_if_addrs()
        for iface, io in io_counters.items():
            ipv4 = None
            for addr in addrs.get(iface, []):
                if addr.family == 2:  # AF_INET
                    ipv4 = addr.address
                    break
            out.append({
                "iface": iface,
                "ipv4": ipv4,
                "sent_mb": round(io.bytes_sent / 1024**2, 1),
                "recv_mb": round(io.bytes_recv / 1024**2, 1),
                "errors_in": io.errin,
                "errors_out": io.errout,
            })
    except Exception:
        pass
    return out


def _gpu():
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = 0
            gpus.append({
                "name": name,
                "vram_used_mb": round(mem_info.used / 1024**2),
                "vram_total_mb": round(mem_info.total / 1024**2),
                "util_pct": util.gpu,
                "temp_c": temp,
            })
        return gpus if gpus else None
    except Exception:
        return None


@router.get("/cache/clear")
def cache_clear():
    return JSONResponse(db_clear_ai())


@router.get("/metrics/dashboard")
def metrics_dashboard():
    with open(_METRICS_HTML, encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), headers={"Cache-Control": "no-store"})


@router.get("/api/metrics")
def api_metrics():
    records = llm_metrics.get_last(100)
    return JSONResponse({
        "uptime_s": round(time.time() - _BOOT_TIME),
        "cpu": _safe(_cpu, {}),
        "memory": _safe(_memory, {}),
        "disk": _safe(_disk, {}),
        "temperatures": _safe(_temperatures, []),
        "network": _safe(_network, []),
        "process": _safe(_process, {}),
        "gpu": _safe(_gpu, None),
        "llm": {"count": len(records), "records": records},
    }, headers={"Cache-Control": "no-store"})


@router.get("/api/metrics/stream")
async def metrics_stream():
    queue: asyncio.Queue = asyncio.Queue()
    llm_metrics.subscribe(queue)

    async def _events():
        try:
            while True:
                record = await queue.get()
                yield f"data: {json.dumps(record, default=str)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            llm_metrics.unsubscribe(queue)

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
