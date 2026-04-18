"""
SQLite-backed caches:
  ai_mode_cache — WebUI RAG answers with query vectors for semantic lookup
"""
import sqlite3
import time

import numpy as np

from SearchEngine.config import CFG, AI_MODE_SIM_THRESH

_DB_PATH   = CFG.get("db_path", "cache.db")
_CACHE_TTL = CFG.get("cache_ttl", 60 * 60 * 24 * 7)


def init_db(db_path: str | None = None):
    global _DB_PATH
    if db_path is not None:
        _DB_PATH = db_path
    con = sqlite3.connect(_DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ai_mode_cache (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            query     TEXT    NOT NULL,
            query_vec BLOB    NOT NULL,
            answer    TEXT    NOT NULL,
            timestamp REAL    NOT NULL
        )
    """)
    cols = {row[1] for row in con.execute("PRAGMA table_info(ai_mode_cache)").fetchall()}
    if "mode" not in cols:
        con.execute("ALTER TABLE ai_mode_cache ADD COLUMN mode TEXT NOT NULL DEFAULT 'balanced'")
    if "zim_name" not in cols:
        con.execute("ALTER TABLE ai_mode_cache ADD COLUMN zim_name TEXT NOT NULL DEFAULT 'all'")
    con.commit()
    con.close()


def db_ai_mode_lookup(query_vec: np.ndarray, mode: str = "balanced",
                      zim_name: str = "all", verbose: bool = True) -> str | None:
    """Semantic similarity cache lookup for WebUI RAG answers.
    Loads all stored query vectors, finds the closest by cosine similarity.
    Returns the cached answer if similarity >= AI_MODE_SIM_THRESH, else None.
    """
    con = sqlite3.connect(_DB_PATH)
    rows = con.execute(
        """
        SELECT query_vec, answer, timestamp
        FROM ai_mode_cache
        WHERE mode = ? AND zim_name = ?
        ORDER BY timestamp DESC
        """,
        (mode, zim_name),
    ).fetchall()
    con.close()
    if not rows:
        return None
    best_sim    = -1.0
    best_answer = None
    for vec_bytes, answer, ts in rows:
        if (time.time() - ts) >= _CACHE_TTL:
            continue
        stored_vec = np.frombuffer(vec_bytes, dtype=np.float32)
        sim = float(np.dot(query_vec, stored_vec))
        if sim > best_sim:
            best_sim    = sim
            best_answer = answer
    if best_sim >= AI_MODE_SIM_THRESH:
        if verbose:
            print(f"AI Mode semantic cache hit (sim={best_sim:.4f})", flush=True)
        return best_answer
    if verbose:
        print(f"AI Mode cache miss (best sim={best_sim:.4f})", flush=True)
    return None


def db_set_ai_mode(query: str, query_vec: np.ndarray, answer: str,
                   mode: str = "balanced", zim_name: str = "all"):
    con = sqlite3.connect(_DB_PATH)
    con.execute(
        """
        INSERT INTO ai_mode_cache (query, query_vec, answer, timestamp, mode, zim_name)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (query, query_vec.tobytes(), answer, time.time(), mode, zim_name),
    )
    con.commit()
    con.close()


def db_clear_ai() -> dict:
    """Delete all cached generated answers."""
    con = sqlite3.connect(_DB_PATH)
    ai_mode_count = con.execute("SELECT COUNT(*) FROM ai_mode_cache").fetchone()[0]
    con.execute("DELETE FROM ai_mode_cache")
    con.commit()
    con.close()
    return {"deleted": {"ai_mode_cache": ai_mode_count}}
