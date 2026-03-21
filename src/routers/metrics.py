"""
System metrics endpoint — /api/metrics

Measures:
  CPU     per-core %, aggregate %, current frequency, model name
  Memory  RAM used/total/%, swap used/total/%
  Disk    used/total/% for the partition hosting this process
  GPU     VRAM used/total if nvidia-smi or psutil GPU is available (best-effort)
  Temp    per-sensor readings (CPU package, NVMe, etc.)
  Network bytes sent/recv since boot, per-interface
  Process this process: RSS, CPU%, thread count, open files
  Uptime  system uptime in seconds
"""

import os
import time
import psutil
from fastapi import APIRouter

router = APIRouter()

_BOOT_TIME = psutil.boot_time()
_PID       = os.getpid()
_OWN_PROC  = psutil.Process(_PID)


def _cpu() -> dict:
    percents = psutil.cpu_percent(percpu=True)
    freq     = psutil.cpu_freq()
    return {
        "per_core_pct":  percents,
        "avg_pct":       round(sum(percents) / len(percents), 1) if percents else 0,
        "count_logical": psutil.cpu_count(logical=True),
        "count_physical": psutil.cpu_count(logical=False),
        "freq_mhz":      round(freq.current) if freq else None,
        "freq_max_mhz":  round(freq.max)     if freq else None,
    }


def _memory() -> dict:
    ram  = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "ram_used_mb":   round(ram.used  / 1024**2),
        "ram_total_mb":  round(ram.total / 1024**2),
        "ram_pct":       ram.percent,
        "ram_avail_mb":  round(ram.available / 1024**2),
        "swap_used_mb":  round(swap.used  / 1024**2),
        "swap_total_mb": round(swap.total / 1024**2),
        "swap_pct":      swap.percent,
    }


def _disk() -> dict:
    # Use the partition that holds this process's working directory
    path = os.getcwd()
    usage = psutil.disk_usage(path)
    io    = psutil.disk_io_counters()
    return {
        "path":           path,
        "used_gb":        round(usage.used  / 1024**3, 2),
        "total_gb":       round(usage.total / 1024**3, 2),
        "free_gb":        round(usage.free  / 1024**3, 2),
        "pct":            usage.percent,
        "read_mb":        round(io.read_bytes  / 1024**2) if io else None,
        "write_mb":       round(io.write_bytes / 1024**2) if io else None,
    }


def _temperatures() -> list[dict]:
    """Returns all available sensor readings — CPU, NVMe, battery, etc."""
    try:
        sensors = psutil.sensors_temperatures()
    except Exception:
        return []   # Android / Windows / unsupported platform
    out = []
    for chip, entries in (sensors or {}).items():
        for e in entries:
            try:
                out.append({
                    "chip":     chip,
                    "label":    e.label or chip,
                    "current":  e.current,
                    "high":     e.high,
                    "critical": e.critical,
                })
            except Exception:
                pass
    return out


def _network() -> list[dict]:
    try:
        stats = psutil.net_io_counters(pernic=True) or {}
        addrs = psutil.net_if_addrs() or {}
    except Exception:
        return []
    # AF_INET is always 2; use the integer to avoid enum differences on Android
    AF_INET = 2
    out = []
    for iface, io in stats.items():
        try:
            ipv4 = next(
                (a.address for a in addrs.get(iface, []) if a.family == AF_INET),
                None,
            )
            out.append({
                "iface":        iface,
                "ipv4":         ipv4,
                "sent_mb":      round(io.bytes_sent / 1024**2, 1),
                "recv_mb":      round(io.bytes_recv / 1024**2, 1),
                "packets_sent": io.packets_sent,
                "packets_recv": io.packets_recv,
                "errors_in":    io.errin,
                "errors_out":   io.errout,
            })
        except Exception:
            pass
    # Sort: loopback last, highest-traffic first
    out.sort(key=lambda x: (x["iface"].startswith("lo"), -(x["sent_mb"] + x["recv_mb"])))
    return out


def _process() -> dict:
    try:
        mem  = _OWN_PROC.memory_info()
        cpu  = _OWN_PROC.cpu_percent()
        return {
            "pid":         _PID,
            "rss_mb":      round(mem.rss / 1024**2, 1),
            "vms_mb":      round(mem.vms / 1024**2, 1),
            "cpu_pct":     cpu,
            "threads":     _OWN_PROC.num_threads(),
            "open_files":  len(_OWN_PROC.open_files()),
        }
    except psutil.Error:
        return {"pid": _PID}


def _gpu() -> dict | None:
    """Best-effort NVIDIA GPU stats via nvidia-smi. Returns None if unavailable."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode != 0:
            return None
        rows = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                rows.append({
                    "name":       parts[0],
                    "vram_used_mb": int(parts[1]),
                    "vram_total_mb": int(parts[2]),
                    "util_pct":   int(parts[3]),
                    "temp_c":     int(parts[4]),
                })
        return rows if rows else None
    except Exception:
        return None


def _safe(fn, fallback=None):
    try:
        return fn()
    except Exception as e:
        print(f"[metrics] {fn.__name__} failed: {e}", flush=True)
        return fallback


@router.get("/api/metrics")
def get_metrics():
    return {
        "uptime_s":     round(time.time() - _BOOT_TIME),
        "cpu":          _safe(_cpu, {}),
        "memory":       _safe(_memory, {}),
        "disk":         _safe(_disk, {}),
        "temperatures": _safe(_temperatures, []),
        "network":      _safe(_network, []),
        "process":      _safe(_process, {}),
        "gpu":          _safe(_gpu),
    }
