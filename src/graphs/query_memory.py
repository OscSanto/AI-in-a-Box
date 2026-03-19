import json
import os
import sqlite3
from datetime import datetime

import faiss
import numpy as np



SCHEMA_QUERIES = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  
    raw_query TEXT, --Users Raw query
    normalized_query TEXT, --LLM rewritten version (needs to be changed)
    faiss_id INTEGER,  --Position in FAISS index (slot in FAISS vector index holding query's embedding)
    entity_list TEXT,  -- spaCy NER hits
    answer_text TEXT,  -- Full LLM answer we generated
    article_titles TEXT,  -- JSON: Which wikipedia articles were used
    chunk_ids TEXT,  --JSON: which chunks 
    llm_model TEXT,  --Model ollama generated the answer
    zim_ids TEXT,  --JSON: which ZIM file was active
    timestamp TEXT  
);
"""

SCHEMA_EDGES = """
CREATE TABLE IF NOT EXISTS query_edges (
    src INTEGER,
    dst INTEGER,
    edge_type TEXT,
    similarity REAL,
    UNIQUE(src, dst, edge_type)
);
"""


class QueryMemory:
    def __init__(self, config, embedder):
        self.config = config
        self.embedder = embedder
        self.dim = embedder.dim  # use probed dimension, not config (config may be 0 for auto)
        data_dir = config.data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.db_path = os.path.join(data_dir, "query_memory.db")
        self.faiss_path = os.path.join(data_dir, "query_memory.faiss")
        self.index = None
        self._id_map = []  # faiss position -> db row id
        self._init_db()
        self.load()

    def _init_db(self):
        con = sqlite3.connect(self.db_path)
        con.execute(SCHEMA_QUERIES)
        con.execute(SCHEMA_EDGES)
        # Migrate old schema — add column if missing
        try:
            con.execute("ALTER TABLE queries ADD COLUMN zim_ids TEXT")
        except Exception:
            pass
        con.commit()
        con.close()

    def load(self):
        if os.path.exists(self.faiss_path):
            loaded = faiss.read_index(self.faiss_path)
            if loaded.d == self.dim:
                self.index = loaded
            else:
                print(f"[query_memory] FAISS dim mismatch (stored={loaded.d}, expected={self.dim}) — rebuilding", flush=True)
                self.index = faiss.IndexFlatIP(self.dim)
        else:
            self.index = faiss.IndexFlatIP(self.dim)  # inner product for cosine on normalized vecs

        # Rebuild id_map from DB ordering
        con = sqlite3.connect(self.db_path)
        rows = con.execute("SELECT id, faiss_id FROM queries ORDER BY faiss_id ASC").fetchall()
        con.close()
        self._id_map = [None] * (self.index.ntotal)
        for db_id, fid in rows:
            if fid is not None and 0 <= fid < len(self._id_map):
                self._id_map[fid] = db_id

    def lookup(self, query_vec: np.ndarray, topk: int) -> list:
        if self.index is None or self.index.ntotal == 0:
            return []
        k = min(topk, self.index.ntotal)
        try:
            vec = np.ascontiguousarray(np.array([query_vec], dtype=np.float32))
            scores, faiss_ids = self.index.search(vec, k)
        except Exception as e:
            print(f"[query_memory] FAISS search error: {type(e).__name__}: {e!r}", flush=True)
            return []

        con = sqlite3.connect(self.db_path)
        results = []
        for score, fid in zip(scores[0], faiss_ids[0]):
            if fid < 0 or fid >= len(self._id_map):
                continue
            db_id = self._id_map[fid]
            if db_id is None:
                continue
            row = con.execute(
                "SELECT id, raw_query, normalized_query, answer_text, entity_list, article_titles, chunk_ids, llm_model, zim_ids, timestamp FROM queries WHERE id=?",
                (db_id,),
            ).fetchone()
            if row:
                results.append({
                    "id": row[0],
                    "raw_query": row[1],
                    "normalized_query": row[2],
                    "answer_text": row[3],
                    "entity_list": json.loads(row[4] or "[]"),
                    "article_titles": json.loads(row[5] or "[]"),
                    "chunk_ids": json.loads(row[6] or "[]"),
                    "llm_model": row[7],
                    "zim_ids": json.loads(row[8] or "[]"),
                    "timestamp": row[9],
                    "similarity": float(score),
                })
        con.close()
        return results

    def store(self, raw_query: str, normalized_query: str, query_vec: np.ndarray,
              entities: list, answer: str, article_titles: list, chunk_ids: list, llm_model: str,
              zim_ids: list = None):
        fid = self.index.ntotal
        try:
            vec = np.ascontiguousarray(np.array([query_vec], dtype=np.float32))
            self.index.add(vec)
        except Exception as e:
            print(f"[query_memory] FAISS add error: {type(e).__name__}: {e!r}", flush=True)
            fid = -1

        self._init_db()  # ensure tables exist (guards against post-reset-cache race)
        con = sqlite3.connect(self.db_path)
        cur = con.execute(
            """INSERT INTO queries
               (raw_query, normalized_query, faiss_id, entity_list, answer_text, article_titles, chunk_ids, llm_model, zim_ids, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                raw_query,
                normalized_query,
                fid,
                json.dumps(entities),
                answer,
                json.dumps(article_titles),
                json.dumps(chunk_ids),
                llm_model,
                json.dumps(zim_ids or []),
                datetime.utcnow().isoformat(),
            ),
        )
        new_db_id = cur.lastrowid

        # Link to similar past queries
        if fid >= 0:
            self._id_map.append(new_db_id)
            # Find similar queries above 0.75 threshold
            if self.index.ntotal > 1:
                k = min(10, self.index.ntotal - 1)
                scores, faiss_ids = self.index.search(np.array([query_vec]), k + 1)
                for score, other_fid in zip(scores[0], faiss_ids[0]):
                    if other_fid == fid or other_fid < 0:
                        continue
                    if float(score) > 0.75 and other_fid < len(self._id_map):
                        other_db_id = self._id_map[other_fid]
                        if other_db_id and other_db_id != new_db_id:
                            try:
                                con.execute(
                                    "INSERT OR IGNORE INTO query_edges (src, dst, edge_type, similarity) VALUES (?, ?, ?, ?)",
                                    (new_db_id, other_db_id, "query_to_similar_query", float(score)),
                                )
                            except Exception:
                                pass

        con.commit()
        con.close()

        # Persist FAISS
        try:
            faiss.write_index(self.index, self.faiss_path)
        except Exception as e:
            print(f"[query_memory] FAISS save error: {e}")
