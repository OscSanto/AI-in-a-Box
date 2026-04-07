"""
SQLite helpers for the ZIM indexer.
"""
import sqlite3
import time
from pathlib import Path


def open_db(db_path: Path) -> sqlite3.Connection:
    schema = Path(__file__).parent / "schema.sql"
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.executescript(schema.read_text())
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.commit()
    return con


def article_exists(con: sqlite3.Connection, title: str) -> bool:
    return con.execute(
        "SELECT 1 FROM articles WHERE title = ?", (title,)
    ).fetchone() is not None


def insert_article(con: sqlite3.Connection, title: str, url: str, zim_path: str) -> int:
    cur = con.execute(
        "INSERT INTO articles (title, url, zim_path, indexed_at) VALUES (?, ?, ?, ?)",
        (title, url, zim_path, time.time()),
    )
    con.commit()
    return cur.lastrowid


def insert_chunks(con: sqlite3.Connection, article_id: int, chunks: list[dict]) -> list[int]:
    """
    Insert all chunks for an article. Returns list of inserted chunk IDs.
    chunks: [{"section_title": str, "chunk_index": int, "text": str}, ...]
    """
    ids = []
    for c in chunks:
        cur = con.execute(
            "INSERT INTO chunks (article_id, section_title, chunk_index, text, embedded) "
            "VALUES (?, ?, ?, ?, 0)",
            (article_id, c["section_title"], c["chunk_index"], c["text"]),
        )
        ids.append(cur.lastrowid)
    con.commit()
    return ids


def get_unembedded_chunks(con: sqlite3.Connection, limit: int = 500) -> list[tuple]:
    """Returns (id, text) rows for chunks not yet embedded."""
    return con.execute(
        "SELECT id, text FROM chunks WHERE embedded = 0 ORDER BY id LIMIT ?",
        (limit,),
    ).fetchall()


def mark_embedded(con: sqlite3.Connection, chunk_ids: list[int]):
    con.executemany(
        "UPDATE chunks SET embedded = 1 WHERE id = ?",
        [(cid,) for cid in chunk_ids],
    )
    con.commit()


def get_chunk_by_id(con: sqlite3.Connection, chunk_id: int) -> dict | None:
    row = con.execute(
        """SELECT c.id, c.article_id, c.section_title, c.chunk_index, c.text,
                  a.title, a.url
           FROM chunks c JOIN articles a ON c.article_id = a.id
           WHERE c.id = ?""",
        (chunk_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "chunk_id":      row[0],
        "article_id":    row[1],
        "section_title": row[2],
        "chunk_index":   row[3],
        "text":          row[4],
        "title":         row[5],
        "url":           row[6],
    }


def init_fts(con: sqlite3.Connection):
    """Build FTS5 title index from articles table. Safe to call multiple times."""
    try:
        con.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
                title, tokenize='porter unicode61'
            )
        """)
    except Exception:
        con.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
                title, tokenize='unicode61'
            )
        """)
    # Insert any articles not yet in FTS (rowid == articles.id)
    con.execute("""
        INSERT INTO articles_fts(rowid, title)
        SELECT id, title FROM articles
        WHERE id NOT IN (SELECT rowid FROM articles_fts)
    """)
    con.commit()


def title_search(con: sqlite3.Connection, query: str, limit: int = 30) -> list[tuple[int, int]]:
    """
    FTS5 BM25 search on article titles.
    Returns [(article_id, rank_position), ...] ordered best-first.
    """
    # Quote each word to avoid FTS5 special character errors
    words = [f'"{w}"' for w in query.replace("_", " ").split() if len(w) >= 2]
    if not words:
        return []
    try:
        rows = con.execute(
            "SELECT rowid FROM articles_fts WHERE articles_fts MATCH ? ORDER BY rank LIMIT ?",
            (" OR ".join(words), limit),
        ).fetchall()
        return [(row[0], i) for i, row in enumerate(rows)]
    except Exception:
        return []


def get_chunks_for_article(con: sqlite3.Connection, article_id: int,
                            max_chunk_index: int = 3) -> list[tuple[int, int]]:
    """Returns [(chunk_id, chunk_index)] for embedded Phase 1 chunks of an article."""
    rows = con.execute(
        "SELECT id, chunk_index FROM chunks "
        "WHERE article_id = ? AND embedded = 1 AND chunk_index < ? ORDER BY chunk_index",
        (article_id, max_chunk_index),
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def stats(con: sqlite3.Connection) -> dict:
    articles  = con.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    chunks    = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    embedded  = con.execute("SELECT COUNT(*) FROM chunks WHERE embedded = 1").fetchone()[0]
    return {"articles": articles, "chunks": chunks, "embedded": embedded}
