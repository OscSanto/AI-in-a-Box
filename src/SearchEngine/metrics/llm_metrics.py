"""
LLM metrics recorder.

Records per-query timing data from Ollama for both the utility model
(Stage 2 intent/rewrite) and the main model (Stage 11 answer generation).

Each record is appended as one JSON line to data/<experiment>/llm_metrics.jsonl
and kept in an in-memory deque for fast reads via /api/metrics.

SSE subscribers are notified immediately when a record is added.
"""

import json
import time
import threading
from collections import deque
from pathlib import Path

_MAX_RECORDS   = 200
_records: deque = deque(maxlen=_MAX_RECORDS)
_jsonl_path: Path | None = None
_model_cache: dict = {}
_cache_lock = threading.Lock()

# SSE subscribers — list of (asyncio.Queue, event_loop) tuples
_subscribers: list = []
_sub_lock = threading.Lock()


def init(data_dir: str) -> None:
    """Call once at startup with the experiment data directory."""
    global _jsonl_path
    _jsonl_path = Path(data_dir) / "llm_metrics.jsonl"


def subscribe(queue) -> None:
    import asyncio
    loop = asyncio.get_event_loop()
    with _sub_lock:
        _subscribers.append((queue, loop))


def unsubscribe(queue) -> None:
    with _sub_lock:
        _subscribers[:] = [(q, l) for q, l in _subscribers if q is not queue]


def _notify_subscribers(entry: dict) -> None:
    with _sub_lock:
        dead = []
        for q, loop in _subscribers:
            try:
                loop.call_soon_threadsafe(q.put_nowait, entry)
            except Exception:
                dead.append((q, loop))
        for item in dead:
            _subscribers.remove(item)


def _fetch_model_info(model: str) -> dict:
    """Fetch and cache static model metadata from Ollama /api/show."""
    with _cache_lock:
        if model in _model_cache:
            return _model_cache[model]

    try:
        import ollama
        info = ollama.show(model)
        details    = info.details   or {}
        model_info = info.modelinfo or {}   # NOTE: attribute is "modelinfo" not "model_info"

        arch = model_info.get("general.architecture", "")

        def _mi(key):
            return model_info.get(f"{arch}.{key}") if arch else None

        context_length   = _mi("context_length")
        block_count      = _mi("block_count")
        head_count       = _mi("attention.head_count")
        head_count_kv    = _mi("attention.head_count_kv")
        embedding_length = _mi("embedding_length")
        param_count      = model_info.get("general.parameter_count")

        gqa_ratio = None
        if head_count and head_count_kv:
            gqa_ratio = round(head_count / head_count_kv, 1)

        # KV cache bytes per token — assume F16 (2 bytes per element)
        # 2 (K+V) × layers × kv_heads × head_dim × 2 bytes
        kv_per_token_bytes = None
        if block_count and head_count_kv and head_count and embedding_length:
            head_dim = embedding_length // head_count
            kv_per_token_bytes = 2 * block_count * head_count_kv * head_dim * 2

        result = {
            "param_size":         getattr(details, "parameter_size",    None),
            "param_count":        param_count,
            "quantization":       getattr(details, "quantization_level", None),
            "family":             getattr(details, "family",             None),
            "context_length":     context_length,
            "gqa_ratio":          gqa_ratio,
            "kv_per_token_bytes": kv_per_token_bytes,
        }
    except Exception as e:
        print(f"[llm_metrics] /api/show failed for {model!r}: {e}", flush=True)
        result = {}

    with _cache_lock:
        _model_cache[model] = result
    return result


def _fetch_ps_info(model: str) -> dict:
    """Fetch live model state from Ollama /api/ps."""
    try:
        import ollama
        ps = ollama.ps()
        for m in (ps.models or []):
            name = getattr(m, "model", None) or getattr(m, "name", None)
            if name and (name == model or name.startswith(model)):
                return {
                    "size_bytes":      getattr(m, "size",      None),
                    "size_vram_bytes": getattr(m, "size_vram", None),
                    "expires_at":      str(getattr(m, "expires_at", None)),
                    "digest":          getattr(m, "digest",    None),
                }
    except Exception as e:
        print(f"[llm_metrics] /api/ps failed: {e}", flush=True)
    return {}


def get_model_info(model: str) -> dict:
    """Public accessor for static model info — used by dashboard endpoint."""
    return _fetch_model_info(model)


def record_embed(model: str, response, batch_size: int = 1) -> None:
    """
    Persist one embedder call record.

    model      — Ollama embedding model string
    response   — raw ollama.embed() response object
    batch_size — number of texts in the batch
    """
    total_ns = getattr(response, "total_duration", None) or 0
    load_ns  = getattr(response, "load_duration",  None) or 0
    tokens   = getattr(response, "prompt_eval_count", None) or 0

    embed_ns   = max(total_ns - load_ns, 0)
    embed_s    = round(embed_ns  / 1e9, 3)
    total_s    = round(total_ns  / 1e9, 3)
    load_s     = round(load_ns   / 1e9, 3)
    was_cold   = load_ns > 500_000_000
    embed_tok_s = round(tokens / embed_s, 1) if embed_s > 0 else None

    ps_info = _fetch_ps_info(model)

    entry = {
        "ts":           time.time(),
        "role":         "embed",
        "model":        model,
        "total_s":      total_s,
        "load_s":       load_s,
        "embed_s":      embed_s,
        "tokens":       tokens,
        "batch_size":   batch_size,
        "embed_tok_s":  embed_tok_s,
        "was_cold":     was_cold,
        **ps_info,
    }

    _records.append(entry)
    _notify_subscribers(entry)

    if _jsonl_path is not None:
        try:
            with open(_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            print(f"[llm_metrics] embed write failed (non-fatal): {e}", flush=True)


def record(role: str, model: str, timings: dict, pipeline: dict | None = None) -> None:
    """
    Persist one LLM call record.

    role     — "utility" (Stage 2) or "answer" (Stage 11)
    model    — Ollama model string e.g. "qwen2.5:1.5b-instruct-q4_K_M"
    timings  — dict from llm_client._extract_timings()
    pipeline — optional dict with pipeline-level fields (answer role only):
               { "pipeline_total_s", "cache_hit", "chunk_count" }
    """
    model_info = _fetch_model_info(model)
    ps_info    = _fetch_ps_info(model)

    context_ratio = None
    if model_info.get("context_length") and timings.get("prompt_tokens"):
        context_ratio = round(timings["prompt_tokens"] / model_info["context_length"], 4)

    est_kv_ram_mb = None
    if model_info.get("kv_per_token_bytes") and timings.get("prompt_tokens"):
        est_kv_ram_mb = round(
            model_info["kv_per_token_bytes"] * timings["prompt_tokens"] / 1024 ** 2, 2
        )

    gen_efficiency = None
    if timings.get("total_tokens") and timings.get("total_s") and timings["total_s"] > 0:
        gen_efficiency = round(timings["total_tokens"] / timings["total_s"], 1)

    output_input_ratio = None
    if timings.get("prompt_tokens") and timings.get("gen_tokens"):
        output_input_ratio = round(timings["gen_tokens"] / timings["prompt_tokens"], 3)

    prefill_pct = None
    prefill_s = timings.get("prefill_s") or 0
    gen_s     = timings.get("gen_s")     or 0
    if prefill_s + gen_s > 0:
        prefill_pct = round(prefill_s / (prefill_s + gen_s) * 100, 1)

    entry = {
        "ts":    time.time(),
        "role":  role,
        "model": model,
        **timings,
        "gen_efficiency_tok_s": gen_efficiency,
        "output_input_ratio":   output_input_ratio,
        "prefill_pct":          prefill_pct,
        "context_ratio":        context_ratio,
        "est_kv_ram_mb":        est_kv_ram_mb,
        **model_info,
        **ps_info,
        **(pipeline or {}),
    }

    _records.append(entry)
    _notify_subscribers(entry)

    if _jsonl_path is not None:
        try:
            with open(_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            print(f"[llm_metrics] write failed (non-fatal): {e}", flush=True)


def get_last(n: int = 100) -> list[dict]:
    """Return the last n records, oldest first."""
    records = list(_records)
    return records[-n:] if n < len(records) else records
