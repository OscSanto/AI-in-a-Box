"""
Embedding via fastembed — sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(384-dim, 50+ languages). L2-normalized output for cosine similarity with IndexFlatIP.
"""
import threading
import numpy as np
from fastembed import TextEmbedding

_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_lock = threading.Lock()
_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        print(f"  Loading embed model {_MODEL_NAME!r} ...", flush=True)
        _model = TextEmbedding(_MODEL_NAME)
        list(_model.embed(["warmup"]))
        print("  Embed model ready", flush=True)
    return _model


def encode(texts: list[str]) -> np.ndarray:
    """
    Encode a batch of texts. Returns L2-normalized (N, 384) float32 array.
    Thread-safe: serialized via lock (ONNX Runtime not safe for concurrent inference).
    """
    with _lock:
        model = _get_model()
        vecs  = np.array(list(model.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms
