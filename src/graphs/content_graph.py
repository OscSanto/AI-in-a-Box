import json
import os
import sqlite3
from collections import deque
from datetime import datetime

import faiss
import numpy as np

"""
TODO: Ensure ZIM_FILE_NAME is saved on each chunk + aritcle + metadata etc.
At the moment, entites will be saved regardless of zim location found. 
 -Problem: Conflicting entites from opposite zim spectrums
 -Consideration: Langauges to consider (en, fr, etc). How will these entites live side by side?
 -Cluster by langugage? Geo? file dir?

TODO: Data migratino. Save old schema? New schema may cause crashes on old schema
TODO: ZIM metadata availability may not be consistent, same cross-zim. 
 -Fall backs on missing data

 -
"""

SCHEMA_NODES = """
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT,
    entity_type TEXT,
    title TEXT,
    content TEXT,
    faiss_id INTEGER,
    zim_id TEXT,
    zim_label TEXT,
    created_at TEXT
);
"""

SCHEMA_EDGES = """
CREATE TABLE IF NOT EXISTS edges (
    src INTEGER,
    dst INTEGER,
    edge_type TEXT,
    confidence REAL DEFAULT 1.0,
    UNIQUE(src, dst, edge_type)
);
"""


class ContentGraph:
    def __init__(self, config, embedder):
        self.config = config
        self.embedder = embedder
        self.dim = config.embedding_dim
        data_dir = config.data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.db_path = os.path.join(data_dir, "content_graph.db")
        self.faiss_path = os.path.join(data_dir, "content_graph.faiss")
        self.index = None
        self._faiss_id_counter = 0
        self._init_db()
        self._load_faiss()

    def _init_db(self):
        con = sqlite3.connect(self.db_path)
        con.execute(SCHEMA_NODES)
        con.execute(SCHEMA_EDGES)
        # Migrate old schema — add columns if missing
        for col, col_type in [("zim_id", "TEXT"), ("zim_label", "TEXT"), ("entity_type", "TEXT")]:
            try:
                con.execute(f"ALTER TABLE nodes ADD COLUMN {col} {col_type}")
            except Exception:
                pass
        try:
            con.execute("ALTER TABLE edges ADD COLUMN confidence REAL DEFAULT 1.0")
        except Exception:
            pass
        con.commit()
        con.close()

    def _load_faiss(self):
        if os.path.exists(self.faiss_path):
            self.index = faiss.read_index(self.faiss_path)
            self._faiss_id_counter = self.index.ntotal
        else:
            self.index = faiss.IndexFlatL2(self.dim)

    def _save_faiss(self):
        faiss.write_index(self.index, self.faiss_path)

    def is_built(self) -> bool:
        if not os.path.exists(self.db_path):
            return False
        con = sqlite3.connect(self.db_path)
        row = con.execute("SELECT COUNT(*) FROM nodes").fetchone()
        con.close()
        return row[0] > 0

    def _insert_node(self, con, node_type: str, title: str, content: str, faiss_id: int = -1,
                     zim_id: str = None, zim_label: str = None, entity_type: str = None) -> int:
        cur = con.execute(
            "INSERT INTO nodes (type, entity_type, title, content, faiss_id, zim_id, zim_label, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (node_type, entity_type, title, content, faiss_id, zim_id, zim_label, datetime.utcnow().isoformat()),
        )
        return cur.lastrowid

    def _insert_edge(self, con, src: int, dst: int, edge_type: str, confidence: float = 1.0):
        try:
            con.execute(
                "INSERT OR IGNORE INTO edges (src, dst, edge_type, confidence) VALUES (?, ?, ?, ?)",
                (src, dst, edge_type, confidence),
            )
        except Exception:
            pass

    def build(self, articles: list, zim_meta: dict = None):
        zim_id = zim_meta.get("zim_id") if zim_meta else None
        zim_label = zim_meta.get("zim_label") if zim_meta else None
        con = sqlite3.connect(self.db_path)
        entity_map = {}  # lower(word) -> node_id

        for art in articles:
            title = art.get("title", "")
            content = art.get("content", "")
            chunks = art.get("chunks", [])

            # Skip articles already in the graph to avoid duplicate nodes/embeddings
            existing = con.execute(
                "SELECT id FROM nodes WHERE type='article' AND title=? AND zim_id=?", (title, zim_id)
            ).fetchone()
            if existing:
                continue

            art_id = self._insert_node(con, "article", title, content[:500], zim_id=zim_id, zim_label=zim_label)

            # Entity candidates: words > 4 chars from title
            for word in [w for w in title.split() if len(w) > 4]:
                word_lower = word.lower()
                if word_lower not in entity_map:
                    eid = self._insert_node(con, "entity", word, "", zim_id=zim_id, zim_label=zim_label)
                    entity_map[word_lower] = eid
                else:
                    eid = entity_map[word_lower]
                self._insert_edge(con, eid, art_id, "entity_to_article")

            # Chunk nodes — stored as text only, no embedding at ingest time.
            for chunk_text in chunks:
                chunk_id = self._insert_node(con, "chunk", title, chunk_text[:1000], faiss_id=-1, zim_id=zim_id, zim_label=zim_label)
                self._insert_edge(con, art_id, chunk_id, "article_to_chunk")

        con.commit()
        con.close()
        self._save_faiss()

    def get_neighbors(self, node_id: int, max_hops: int, max_neighbors: int) -> list:
        con = sqlite3.connect(self.db_path)
        visited = set()
        queue = deque([(node_id, 0)])
        results = []

        while queue and len(results) < max_neighbors:
            nid, hop = queue.popleft()
            if nid in visited or hop > max_hops:
                continue
            visited.add(nid)
            if nid != node_id:
                row = con.execute(
                    "SELECT id, type, title, content, faiss_id FROM nodes WHERE id=?", (nid,)
                ).fetchone()
                if row:
                    results.append({
                        "id": row[0], "type": row[1], "title": row[2],
                        "content": row[3], "faiss_id": row[4]
                    })
            if hop < max_hops:
                neighbors = con.execute(
                    "SELECT dst FROM edges WHERE src=?", (nid,)
                ).fetchall()
                for (dst,) in neighbors:
                    if dst not in visited:
                        queue.append((dst, hop + 1))

        con.close()
        return results[:max_neighbors]

    def find_similar_chunks(self, query_vec: np.ndarray, topk: int) -> list:
        if self.index is None or self.index.ntotal == 0:
            return []
        k = min(topk, self.index.ntotal)
        try:
            distances, faiss_ids = self.index.search(np.array([query_vec]), k)
        except Exception as e:
            print(f"[content_graph] FAISS search error: {e}")
            return []

        con = sqlite3.connect(self.db_path)
        results = []
        for fid in faiss_ids[0]:
            if fid < 0:
                continue
            row = con.execute(
                "SELECT id, type, title, content, faiss_id FROM nodes WHERE faiss_id=? AND type='chunk'",
                (int(fid),),
            ).fetchone()
            if row:
                results.append({
                    "id": row[0], "type": row[1], "title": row[2],
                    "content": row[3], "faiss_id": row[4]
                })
        con.close()
        return results

    # Controlled relation vocabulary — keeps the graph queryable and consistent.
    VALID_RELATIONS = {"participated_in", "part_of", "created_by", "caused_by", "located_in", "related_to"}
    CONFIDENCE_THRESHOLD = 0.6

    def store_relations(self, triples: list, zim_meta: dict = None):
        """
        Store entity→relation→entity triples extracted by LLM.
        Expected triple schema:
          {"from": str, "from_type": str, "relation": str, "to": str, "to_type": str, "confidence": float}
        Only stores triples whose relation is in VALID_RELATIONS and confidence >= CONFIDENCE_THRESHOLD.
        """
        if not triples:
            return
        zim_id = zim_meta.get("zim_id") if zim_meta else None
        zim_label = zim_meta.get("zim_label") if zim_meta else None
        con = sqlite3.connect(self.db_path)
        cache = {}  # lower(name) → node_id

        def get_or_create(name: str, etype: str = None) -> int:
            key = name.lower().strip()
            if key in cache:
                return cache[key]
            row = con.execute(
                "SELECT id FROM nodes WHERE type='entity' AND lower(title)=?", (key,)
            ).fetchone()
            if row:
                cache[key] = row[0]
                return row[0]
            eid = self._insert_node(con, "entity", name.strip(), "", zim_id=zim_id, zim_label=zim_label, entity_type=etype)
            cache[key] = eid
            return eid

        stored = skipped = 0
        for t in triples:
            frm = t.get("from", "").strip()
            rel = t.get("relation", "").strip().lower().replace(" ", "_")
            to  = t.get("to",  "").strip()
            confidence = float(t.get("confidence", 1.0))
            if not frm or not rel or not to:
                skipped += 1
                continue
            if rel not in self.VALID_RELATIONS:
                skipped += 1
                continue
            if confidence < self.CONFIDENCE_THRESHOLD:
                skipped += 1
                continue
            src_id = get_or_create(frm, t.get("from_type"))
            dst_id = get_or_create(to,  t.get("to_type"))
            self._insert_edge(con, src_id, dst_id, rel, confidence)
            stored += 1

        con.commit()
        con.close()
        print(f"[content_graph] stored {stored} relation triples ({skipped} skipped — low confidence or invalid relation)", flush=True)

    def find_article_node(self, title: str) -> dict:
        con = sqlite3.connect(self.db_path)
        row = con.execute(
            "SELECT id, type, title, content, faiss_id FROM nodes WHERE type='article' AND title=?",
            (title,),
        ).fetchone()
        con.close()
        if row:
            return {"id": row[0], "type": row[1], "title": row[2], "content": row[3], "faiss_id": row[4]}
        return {}
