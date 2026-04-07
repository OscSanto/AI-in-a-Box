"""
SQLite-backed caches:
  search_cache   — raw Kiwix results by query string (TTL-keyed)
  ai_cache       — /ai LLM answers by query string (TTL-keyed)
  ai_mode_cache  — /ai-mode LLM answers with query vector for semantic lookup
"""
import json
import sqlite3
import time

import numpy as np

from SearchEngine.config import CFG, AI_MODE_SIM_THRESH

_DB_PATH   = CFG.get("db_path", "cache.db")
_CACHE_TTL = CFG.get("cache_ttl", 60 * 60 * 24 * 7)


def init_db():
    con = sqlite3.connect(_DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS search_cache (
            query TEXT PRIMARY KEY, results TEXT NOT NULL, timestamp REAL NOT NULL
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_query ON search_cache(query)")
    con.execute("""
        CREATE TABLE IF NOT EXISTS ai_cache (
            query TEXT PRIMARY KEY, answer TEXT NOT NULL, timestamp REAL NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ai_mode_cache (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            query     TEXT    NOT NULL,
            query_vec BLOB    NOT NULL,
            answer    TEXT    NOT NULL,
            timestamp REAL    NOT NULL
        )
    """)
    con.commit()
    con.close()


def db_get_results(q: str):
    con = sqlite3.connect(_DB_PATH)
    row = con.execute(
        "SELECT results, timestamp FROM search_cache WHERE query = ?", (q,)
    ).fetchone()
    con.close()
    if row and (time.time() - row[1]) < _CACHE_TTL:
        return json.loads(row[0])
    return None


def db_set_results(q: str, ranked: list):
    con = sqlite3.connect(_DB_PATH)
    con.execute(
        "INSERT OR REPLACE INTO search_cache (query, results, timestamp) VALUES (?, ?, ?)",
        (q, json.dumps(ranked), time.time()),
    )
    con.commit()
    con.close()


def db_get_ai(q: str):
    con = sqlite3.connect(_DB_PATH)
    row = con.execute(
        "SELECT answer, timestamp FROM ai_cache WHERE query = ?", (q,)
    ).fetchone()
    con.close()
    if row and (time.time() - row[1]) < _CACHE_TTL:
        return row[0]
    return None


def db_set_ai(q: str, answer: str):
    con = sqlite3.connect(_DB_PATH)
    con.execute(
        "INSERT OR REPLACE INTO ai_cache (query, answer, timestamp) VALUES (?, ?, ?)",
        (q, answer, time.time()),
    )
    con.commit()
    con.close()


def db_suggest(q: str, limit: int = 8) -> list:
    con = sqlite3.connect(_DB_PATH)
    rows = con.execute(
        "SELECT query FROM search_cache WHERE query LIKE ? ORDER BY timestamp DESC LIMIT ?",
        (q + "%", limit),
    ).fetchall()
    con.close()
    return [r[0] for r in rows]


def db_ai_mode_lookup(query_vec: np.ndarray) -> str | None:
    """Semantic similarity cache lookup for AI Mode answers.
    Loads all stored query vectors, finds the closest by cosine similarity.
    Returns the cached answer if similarity >= AI_MODE_SIM_THRESH, else None.
    """
    con = sqlite3.connect(_DB_PATH)
    rows = con.execute(
        "SELECT query_vec, answer, timestamp FROM ai_mode_cache ORDER BY timestamp DESC"
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
        print(f"✅ AI Mode semantic cache hit (sim={best_sim:.4f})", flush=True)
        return best_answer
    print(f"  AI Mode cache miss (best sim={best_sim:.4f})", flush=True)
    return None


def db_set_ai_mode(query: str, query_vec: np.ndarray, answer: str):
    con = sqlite3.connect(_DB_PATH)
    con.execute(
        "INSERT INTO ai_mode_cache (query, query_vec, answer, timestamp) VALUES (?, ?, ?, ?)",
        (query, query_vec.tobytes(), answer, time.time()),
    )
    con.commit()
    con.close()


def db_clear_ai() -> dict:
    """Delete all AI-generated answers. Search cache is kept."""
    con = sqlite3.connect(_DB_PATH)
    ai_count      = con.execute("SELECT COUNT(*) FROM ai_cache").fetchone()[0]
    ai_mode_count = con.execute("SELECT COUNT(*) FROM ai_mode_cache").fetchone()[0]
    con.execute("DELETE FROM ai_cache")
    con.execute("DELETE FROM ai_mode_cache")
    con.commit()
    con.close()
    return {"deleted": {"ai_cache": ai_count, "ai_mode_cache": ai_mode_count}}
