"""
In-process embedding via fastembed (ONNX Runtime).
No HTTP roundtrip, no keep_alive, not evicted by LLM calls.

ONNX Runtime is not safe for concurrent inference on the same model instance,
so all calls are serialized through _embed_lock.
"""
import time
import threading

import numpy as np
from fastembed import TextEmbedding as _FE

from SearchEngine.config import EMBED_HF_MODEL

print(f"⏳ Loading embed model {EMBED_HF_MODEL!r} ...", flush=True)
_fe_model = _FE(EMBED_HF_MODEL)
list(_fe_model.embed(["warmup"]))   # warm ONNX runtime before first real request
print("✅ Embed model ready", flush=True)

_embed_lock = threading.Lock()

# Short-lived cache: same text within _EMBED_CACHE_TTL seconds → skip re-encode.
# Eliminates duplicate embeds when /ai-mode and /zim-sources arrive simultaneously.
_EMBED_CACHE_TTL = 10.0   # seconds
_embed_cache: dict[str, tuple[float, np.ndarray]] = {}  # text → (timestamp, vec)


def _st_encode(texts: list[str]) -> np.ndarray:
    """Encode texts → L2-normalised (n, dim) float32 array.
    Serialized via _embed_lock; batching is key — one lock acquisition per batch.
    Single-text results are cached for _EMBED_CACHE_TTL seconds to avoid
    duplicate encoding when concurrent requests embed the same query.
    """
    # Cache hit path — single text only (batch calls bypass cache)
    if len(texts) == 1:
        entry = _embed_cache.get(texts[0])
        if entry and (time.monotonic() - entry[0]) < _EMBED_CACHE_TTL:
            return entry[1][np.newaxis]   # return (1, dim) array

    with _embed_lock:
        # Re-check inside lock — another thread may have populated cache
        if len(texts) == 1:
            entry = _embed_cache.get(texts[0])
            if entry and (time.monotonic() - entry[0]) < _EMBED_CACHE_TTL:
                return entry[1][np.newaxis]
        vecs = np.array(list(_fe_model.embed(texts)), dtype=np.float32)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms

    if len(texts) == 1:
        _embed_cache[texts[0]] = (time.monotonic(), vecs[0])

    return vecs


def embed(text: str) -> np.ndarray:
    return _st_encode([text])[0]


def embed_batch(texts: list) -> np.ndarray:
    return _st_encode(texts)


def embed_ai_mode(text: str) -> np.ndarray:
    return _st_encode([text])[0]


def embed_batch_ai_mode(texts: list) -> np.ndarray:
    return _st_encode(texts)


class _Embedder:
    def embed_batch(self, texts): return embed_batch(texts)

class _AiModeEmbedder:
    def embed_batch(self, texts): return embed_batch_ai_mode(texts)

embedder         = _Embedder()
ai_mode_embedder = _AiModeEmbedder()
