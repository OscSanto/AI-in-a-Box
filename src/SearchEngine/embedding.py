"""
In-process embedding via fastembed (ONNX Runtime).
No HTTP roundtrip, no keep_alive, not evicted by LLM calls.

ONNX Runtime is not safe for concurrent inference on the same model instance,
so all calls are serialized through _embed_lock.
"""
import threading

import numpy as np
from fastembed import TextEmbedding as _FE

from SearchEngine.config import EMBED_HF_MODEL

print(f"⏳ Loading embed model {EMBED_HF_MODEL!r} ...", flush=True)
_fe_model = _FE(EMBED_HF_MODEL)
list(_fe_model.embed(["warmup"]))   # warm ONNX runtime before first real request
print("✅ Embed model ready", flush=True)

_embed_lock = threading.Lock()


def _st_encode(texts: list[str]) -> np.ndarray:
    """Encode texts → L2-normalised (n, dim) float32 array.
    Serialized via _embed_lock; batching is key — one lock acquisition per batch.
    """
    with _embed_lock:
        vecs = np.array(list(_fe_model.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


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
