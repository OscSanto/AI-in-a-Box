"""
SQLite-backed cache for RAG answers.
DB_PATH is set once at startup by init_db() called from lifespan() in main.py.
"""
import sqlite3
import time
from contextlib import contextmanager

import numpy as np

from SearchEngine.config import CFG, AI_MODE_SIM_THRESH

# Set once by init_db() at startup. Empty until then — do not call db functions before init.
DB_PATH: str = ""
_CACHE_TTL = CFG.get("cache_ttl", 60 * 60 * 24 * 7)  # 7 days in seconds


@contextmanager
def _connect():
    if not DB_PATH:
        raise RuntimeError("cache.init_db() has not been called — DB_PATH is not set")
    con = sqlite3.connect(DB_PATH)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db(db_path: str):
    global DB_PATH
    DB_PATH = db_path
    with _connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS ai_mode_cache (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                query     TEXT    NOT NULL,
                query_vec BLOB    NOT NULL,
                answer    TEXT    NOT NULL,
                timestamp REAL    NOT NULL,
                mode      TEXT    NOT NULL DEFAULT 'balanced',
                zim_name  TEXT    NOT NULL DEFAULT 'all'
            )
        """)


def db_ai_mode_lookup(query_vec: np.ndarray, mode: str = "balanced",
                      zim_name: str = "all", verbose: bool = True) -> str | None:
    """Semantic similarity cache lookup for RAG answers.
    Finds the closest stored query vector by cosine similarity.
    Returns the cached answer if similarity >= AI_MODE_SIM_THRESH, else None.
    """
    cutoff = time.time() - _CACHE_TTL
    with _connect() as con:
        rows = con.execute(
            """
            SELECT query_vec, answer FROM ai_mode_cache
            WHERE mode = ? AND zim_name = ? AND timestamp > ?
            ORDER BY timestamp DESC
            """,
            (mode, zim_name, cutoff),
        ).fetchall()
    if not rows:
        return None
    best_sim    = -1.0
    best_answer = None
    for vec_bytes, answer in rows:
        stored_vec = np.frombuffer(vec_bytes, dtype=np.float32)
        sim = float(np.dot(query_vec, stored_vec))
        if sim > best_sim:
            best_sim    = sim
            best_answer = answer
    if best_sim >= AI_MODE_SIM_THRESH:
        if verbose:
            print(f"Cache hit (sim={best_sim:.4f})", flush=True)
        return best_answer
    if verbose:
        print(f"Cache miss (best sim={best_sim:.4f})", flush=True)
    return None


def db_set_ai_mode(query: str, query_vec: np.ndarray, answer: str,
                   mode: str = "balanced", zim_name: str = "all"):
    with _connect() as con:
        con.execute(
            """
            INSERT INTO ai_mode_cache (query, query_vec, answer, timestamp, mode, zim_name)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (query, query_vec.tobytes(), answer, time.time(), mode, zim_name),
        )


def db_clear_ai() -> dict:
    """Delete all cached answers."""
    with _connect() as con:
        count = con.execute("SELECT COUNT(*) FROM ai_mode_cache").fetchone()[0]
        con.execute("DELETE FROM ai_mode_cache")
    return {"deleted": {"ai_mode_cache": count}}
