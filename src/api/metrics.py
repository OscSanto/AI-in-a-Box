import os
import time

import psutil
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from SearchEngine.cache import db_clear_ai

router = APIRouter()

_BOOT_TIME = time.time()
_OWN_PROC  = psutil.Process(os.getpid())


def _safe(fn, fallback=None):
    try:
        return fn()
    except Exception:
        return fallback


@router.get("/cache/clear")
def cache_clear():
    return JSONResponse(db_clear_ai())


@router.get("/api/metrics")
def api_metrics():
    pcts     = _safe(lambda: psutil.cpu_percent(percpu=True), [])
    freq     = _safe(psutil.cpu_freq)
    ram      = _safe(psutil.virtual_memory)
    swap     = _safe(psutil.swap_memory)
    disk_u   = _safe(lambda: psutil.disk_usage(os.getcwd()))
    disk_io  = _safe(psutil.disk_io_counters)
    proc_mem = _safe(_OWN_PROC.memory_info)

    temps = []
    for chip, entries in (_safe(psutil.sensors_temperatures) or {}).items():
        for e in entries:
            temps.append({"chip": chip, "label": e.label or chip,
                          "current": e.current, "high": e.high, "critical": e.critical})

    network = []
    try:
        io_counters = psutil.net_io_counters(pernic=True)
        addrs = psutil.net_if_addrs()
        for iface, io in io_counters.items():
            ipv4 = next((a.address for a in addrs.get(iface, []) if a.family == 2), None)
            network.append({
                "iface": iface, "ipv4": ipv4,
                "sent_mb": round(io.bytes_sent / 1024**2, 1),
                "recv_mb": round(io.bytes_recv / 1024**2, 1),
                "errors_in": io.errin, "errors_out": io.errout,
            })
    except Exception:
        pass

    try:
        open_files = _OWN_PROC.num_fds()
    except Exception:
        open_files = len(_OWN_PROC.open_files())

    return JSONResponse({
        "uptime_s": round(time.time() - _BOOT_TIME),
        "cpu": {
            "per_core_pct": pcts,
            "avg_pct": round(sum(pcts) / len(pcts), 1) if pcts else 0,
            "count_logical":  _safe(lambda: psutil.cpu_count(logical=True)),
            "count_physical": _safe(lambda: psutil.cpu_count(logical=False)),
            "freq_mhz":     round(freq.current) if freq else None,
            "freq_max_mhz": round(freq.max)     if freq else None,
        },
        "memory": {} if not ram else {
            "ram_used_mb":  round(ram.used  / 1024**2),
            "ram_total_mb": round(ram.total / 1024**2),
            "ram_pct":      ram.percent,
            "ram_avail_mb": round(ram.available / 1024**2),
            "swap_used_mb":  round(swap.used  / 1024**2) if swap else 0,
            "swap_total_mb": round(swap.total / 1024**2) if swap else 0,
            "swap_pct":      swap.percent               if swap else 0,
        },
        "disk": {} if not disk_u else {
            "used_gb":  round(disk_u.used  / 1024**3, 2),
            "total_gb": round(disk_u.total / 1024**3, 2),
            "free_gb":  round(disk_u.free  / 1024**3, 2),
            "pct":      disk_u.percent,
            "read_mb":  round(disk_io.read_bytes  / 1024**2) if disk_io else None,
            "write_mb": round(disk_io.write_bytes / 1024**2) if disk_io else None,
        },
        "temperatures": temps,
        "network": network,
        "process": {} if not proc_mem else {
            "pid":        _OWN_PROC.pid,
            "rss_mb":     round(proc_mem.rss / 1024**2, 1),
            "vms_mb":     round(proc_mem.vms / 1024**2, 1),
            "cpu_pct":    _safe(_OWN_PROC.cpu_percent, 0),
            "threads":    _safe(_OWN_PROC.num_threads, 0),
            "open_files": open_files,
        },
    }, headers={"Cache-Control": "no-store"})
