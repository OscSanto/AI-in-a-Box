"""
FAISS index management.

Index type is chosen automatically based on corpus size:
  n < _IVF_MIN_N  → IndexFlatIP   (exact search, no training needed)
  n >= _IVF_MIN_N → IndexIVFFlat  (approximate, faster at scale)

Both are wrapped in IndexIDMap so FAISS vector ID == SQLite chunks.id.
"""
import numpy as np
import faiss
from pathlib import Path

_DIM        = 384        # Snowflake/snowflake-arctic-embed-xs
_IVF_MIN_N  = 500_000   # FlatIP is fast enough below this (~5ms for 133k vecs);
                        # IVFFlat only worthwhile for Wikipedia-scale (1M+)
_NPROBE     = 64         # clusters to search at query time (quality vs speed)


def _make_flat() -> faiss.IndexIDMap:
    return faiss.IndexIDMap(faiss.IndexFlatIP(_DIM))


def make_ivf(n_total: int, train_vecs: np.ndarray) -> faiss.IndexIDMap:
    """
    Create and train an IVFFlat index.
    Must be called before adding any vectors — IVF requires training first.
    n_total: expected final number of vectors (used to size nlist).
    train_vecs: L2-normalised (N, DIM) float32 training sample.
    """
    nlist     = max(4, min(int(np.sqrt(n_total)), 4096))
    quantizer = faiss.IndexFlatIP(_DIM)
    inner     = faiss.IndexIVFFlat(quantizer, _DIM, nlist, faiss.METRIC_INNER_PRODUCT)
    inner.train(train_vecs)
    inner.nprobe = min(nlist, _NPROBE)
    idx = faiss.IndexIDMap(inner)
    print(f"  IVFFlat trained: nlist={nlist} nprobe={inner.nprobe} "
          f"train_n={len(train_vecs):,}", flush=True)
    return idx


def load_or_create(index_path: Path) -> faiss.IndexIDMap:
    if index_path.exists() and index_path.stat().st_size > 100:
        idx = faiss.read_index(str(index_path))
        # Restore nprobe if this is an IVF index
        inner = faiss.downcast_index(idx.index)
        if hasattr(inner, "nprobe"):
            inner.nprobe = _NPROBE
        print(f"  Loaded FAISS index: {idx.ntotal:,} vectors from {index_path}",
              flush=True)
        return idx
    print(f"  Creating new FAISS index (dim={_DIM}, FlatIP)", flush=True)
    return _make_flat()


def save(index: faiss.IndexIDMap, index_path: Path):
    # Write to a temp file first, then rename — prevents a partial/corrupt
    # index if the process is killed mid-write (OOM or Ctrl-C).
    tmp = index_path.with_suffix(".tmp")
    faiss.write_index(index, str(tmp))
    tmp.rename(index_path)


def add_vectors(index: faiss.IndexIDMap, chunk_ids: list[int], vectors: np.ndarray):
    """vectors: (N, DIM) float32, L2-normalised. chunk_ids: SQLite chunk IDs."""
    index.add_with_ids(vectors, np.array(chunk_ids, dtype=np.int64))


def search(index: faiss.IndexIDMap, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
    """
    query_vec: (DIM,) float32, L2-normalised.
    Returns [(chunk_id, cosine_score), ...] sorted descending.
    """
    q = query_vec.reshape(1, -1).astype(np.float32)
    scores, ids = index.search(q, top_k)
    return [
        (int(cid), float(score))
        for cid, score in zip(ids[0], scores[0])
        if cid != -1
    ]
