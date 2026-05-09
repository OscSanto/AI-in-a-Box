"""
In-process embedding via fastembed (ONNX Runtime).
No HTTP roundtrip, no keep_alive, not evicted by LLM calls.
Embedding model files are cached on disk by fastembed (configurable via FASTEMBED_CACHE_PATH env var).
ONNX Runtime is not safe for concurrent inference on the same model instance,
so all calls are serialized through _embed_lock.

Utilizing FAISS L2-normalized ONNX embeddings
"""

import os
import time
import threading
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding as _FE

from SearchEngine.config import EMBED_MODEL

# Always resolves to ~/.cache/fastembed unless FASTEMBED_CACHE_PATH is explicitly set.
# Works the same whether you start with `python main.py` or `bash start.sh`:
#   - start.sh sets FASTEMBED_CACHE_PATH → os.environ.get reads it
#   - direct python run → falls back to ~/.cache/fastembed, then writes it into the env
#     so fastembed's own internal env reads see the same path.
_FASTEMBED_CACHE = os.environ.get(
    "FASTEMBED_CACHE_PATH",
    str(Path.home() / ".cache" / "fastembed"),
)
os.environ.setdefault("FASTEMBED_CACHE_PATH", _FASTEMBED_CACHE)

# Check whether the model is already cached 
_model_slug = "models--" + EMBED_MODEL.lower().replace("/", "--")
_model_cache_dir = Path(_FASTEMBED_CACHE) / _model_slug # check if model files already exist in cache (fastembed/onnx will create this dir on download)
_is_cached = _model_cache_dir.exists() and any(_model_cache_dir.rglob("*.onnx"))
if _is_cached:
    print(f"✅ Embed model cached — loading from {_model_cache_dir}", flush=True)
else:
    print(f"⬇️  Embed model not found in cache, downloading {EMBED_MODEL!r} ...", flush=True)
    print(f"   Cache: {_FASTEMBED_CACHE}", flush=True)
try:
    _embed_model = _FE(EMBED_MODEL, cache_dir=_FASTEMBED_CACHE, threads=4)
    list(_embed_model.embed(["warmup"]))  # force ONNX JIT compile before first real request
    print("✅ Embed model ready", flush=True)
except Exception as e:
    raise RuntimeError(
        f"Failed to load embedding model {EMBED_MODEL!r}. "
        f"Check internet connection (first run downloads ~100 MB) and that "
        f"FASTEMBED_CACHE_PATH ({_FASTEMBED_CACHE}) is writable.\n  Cause: {e}"
    ) from e

# ONNX Runtime is not thread-safe for concurrent inference on the same model instance, so we serialize all calls through this lock.
_embed_lock = threading.Lock()

# Short-lived cache: same text within _EMBED_CACHE_TTL seconds -> skip re-encode.
# Avoids duplicate query encodes when concurrent WebUI requests arrive.
_EMBED_CACHE_TTL = 10.0   # seconds
_embed_cache: dict[str, tuple[float, np.ndarray]] = {}  # text → (timestamp, vec)


def _encode(texts: list[str]) -> np.ndarray:
    """Encode texts → L2-normalised (n, dim) float32 array.
    Serialized via _embed_lock; batching is key — one lock acquisition per batch.
    Single-text results are cached for _EMBED_CACHE_TTL seconds to avoid
    duplicate encoding when concurrent requests embed the same query.
    """
    # Cache hit path — single text only (batch calls bypass cache)
    if len(texts) == 1:
        entry = _embed_cache.get(texts[0])

        # Condition true if previous embedding exists and is fresh (within TTL). If so, return cached vector.
        if entry and (time.monotonic() - entry[0]) < _EMBED_CACHE_TTL:
            return entry[1][np.newaxis]   # return (1, dim) array

    # Serialize calls via _embed_lock.
    # Thread-safe embedding call 
    # ONNX Runtime is not thread-safe for concurrent calls  
    with _embed_lock:
        # Only one thread here at a time — 
        # re-check cache in case another thread populated it while we waited for the lock.
        if len(texts) == 1:
            entry = _embed_cache.get(texts[0])
            if entry and (time.monotonic() - entry[0]) < _EMBED_CACHE_TTL:
                return entry[1][np.newaxis] 
        
        raw = list(_embed_model.embed(texts))  # generator → list of 1-D arrays
        vecs = np.array(raw, dtype=np.float32)  # stack into (n, dim) matrix

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms

    if len(texts) == 1:
        _embed_cache[texts[0]] = (time.monotonic(), vecs[0])

    return vecs

# HELPER FUNCTIONS 
def embed(text: str) -> np.ndarray:
    return _encode([text])[0]

def embed_batch(texts: list) -> np.ndarray:
    return _encode(texts)


