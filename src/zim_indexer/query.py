"""
Query the ZIM index with RRF hybrid retrieval (FAISS semantic + title BM25).

Search:
  python -m zim_indexer.query <output_dir> "your question" [--top 10]

Expand article (embed remaining sections):
  python -m zim_indexer.query <output_dir> --expand "Article Title"

Status:
  python -m zim_indexer.query <output_dir> --status
"""

import sys
import argparse
from pathlib import Path

import faiss
from zim_indexer import embed
from zim_indexer import index as faiss_index
from zim_indexer.db import (
    open_db, stats, init_fts, title_search,
    get_chunks_for_article, get_chunk_by_id, mark_embedded,
)

_RRF_K = 60   # RRF constant — higher = less rank-sensitive


def search(out_dir: Path, query: str, top_k: int = 10,
           threshold: float = 0.0) -> list[dict]:
    """
    Hybrid retrieval: FAISS cosine + FTS5 title BM25 → RRF fusion.

    FAISS finds semantically similar chunks.
    Title BM25 finds articles whose titles match query keywords.
    RRF merges both ranked lists without requiring score normalisation.
    """
    db_path  = out_dir / "data.db"
    idx_path = out_dir / "faiss.index"

    if not db_path.exists():
        print(f"No index found at {out_dir}", file=sys.stderr)
        sys.exit(1)

    con       = open_db(db_path)
    init_fts(con)
    idx       = faiss_index.load_or_create(idx_path)
    query_vec = embed.encode([query])[0]

    # ── FAISS semantic search ─────────────────────────────────────────────────
    faiss_results  = faiss_index.search(idx, query_vec, top_k=top_k * 4)
    faiss_rank_map = {chunk_id: rank for rank, (chunk_id, _) in enumerate(faiss_results)}
    faiss_score_map = {chunk_id: score for chunk_id, score in faiss_results}

    # ── Title BM25 search → expand article → chunk IDs ───────────────────────
    bm25_articles  = title_search(con, query, limit=top_k * 3)
    bm25_rank_map: dict[int, int] = {}   # chunk_id → best bm25 rank
    for article_id, article_rank in bm25_articles:
        for chunk_id, _ in get_chunks_for_article(con, article_id):
            if chunk_id not in bm25_rank_map:
                bm25_rank_map[chunk_id] = article_rank

    # ── RRF fusion ────────────────────────────────────────────────────────────
    all_chunk_ids = set(faiss_rank_map) | set(bm25_rank_map)
    n             = len(all_chunk_ids)

    rrf_scores = {
        cid: (1 / (_RRF_K + faiss_rank_map.get(cid, n)) +
              1 / (_RRF_K + bm25_rank_map.get(cid, n)))
        for cid in all_chunk_ids
    }

    top_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k * 2]

    # ── Fetch chunk details (dedup by article_id + section_title) ─────────────
    hits = []
    seen: set[tuple] = set()
    for cid in top_ids:
        if len(hits) >= top_k:
            break
        chunk = get_chunk_by_id(con, cid)
        if not chunk:
            continue
        score = faiss_score_map.get(cid, 0.0)
        if score < threshold:
            continue
        key = (chunk["title"], chunk["section_title"])
        if key in seen:
            continue
        seen.add(key)
        hits.append({
            **chunk,
            "rrf_score":   round(rrf_scores[cid], 6),
            "faiss_score": round(score, 4),
            "in_faiss":    cid in faiss_rank_map,
            "in_bm25":     cid in bm25_rank_map,
        })

    con.close()
    return hits


def expand_article(out_dir: Path, title: str):
    """Embed all remaining (unembedded) sections for a specific article."""
    db_path  = out_dir / "data.db"
    idx_path = out_dir / "faiss.index"

    con = open_db(db_path)
    idx = faiss_index.load_or_create(idx_path)

    row = con.execute(
        "SELECT id FROM articles WHERE title = ?", (title,)
    ).fetchone()
    if not row:
        print(f"Article not found: {title!r}", file=sys.stderr)
        con.close()
        return

    article_id = row[0]
    pending    = con.execute(
        "SELECT id, text FROM chunks WHERE article_id = ? AND embedded = 0 ORDER BY chunk_index",
        (article_id,),
    ).fetchall()

    if not pending:
        print(f"No unembedded chunks for {title!r} — already fully expanded.")
        con.close()
        return

    print(f"Expanding {len(pending)} chunks for {title!r}...", flush=True)
    chunk_ids = [r[0] for r in pending]
    texts     = [r[1] for r in pending]
    vecs      = embed.encode(texts)
    faiss_index.add_vectors(idx, chunk_ids, vecs)
    mark_embedded(con, chunk_ids)
    faiss_index.save(idx, idx_path)
    con.close()
    print(f"Done. {len(chunk_ids)} new vectors added.", flush=True)


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def status(out_dir: Path):
    db_path  = out_dir / "data.db"
    idx_path = out_dir / "faiss.index"

    if not db_path.exists():
        print(f"No index found at {out_dir}")
        return

    # ── File sizes ────────────────────────────────────────────────────────────
    db_size  = db_path.stat().st_size  if db_path.exists()  else 0
    idx_size = idx_path.stat().st_size if idx_path.exists() else 0
    print(f"\n{'─'*50}")
    print(f"  Directory : {out_dir}")
    print(f"  data.db   : {_fmt_bytes(db_size)}")
    print(f"  faiss.idx : {_fmt_bytes(idx_size)  if idx_size else 'not found'}")
    print(f"  Total     : {_fmt_bytes(db_size + idx_size)}")

    # ── SQLite stats ──────────────────────────────────────────────────────────
    con = open_db(db_path)
    s   = stats(con)

    embedded_pct = (s["embedded"] / s["chunks"] * 100) if s["chunks"] else 0
    pending      = s["chunks"] - s["embedded"]
    avg_per_art  = s["chunks"] / s["articles"] if s["articles"] else 0

    # Chunk text length stats
    lens = con.execute(
        "SELECT MIN(LENGTH(text)), MAX(LENGTH(text)), AVG(LENGTH(text)) FROM chunks"
    ).fetchone()
    min_len, max_len, avg_len = lens

    # Section distribution — how many articles have N chunks
    phase1_full = con.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT article_id FROM chunks WHERE chunk_index < 3"
        "  GROUP BY article_id HAVING COUNT(*) >= 3"
        ")"
    ).fetchone()[0]

    # Per-article total text size stats
    art_lens = con.execute(
        "SELECT MIN(art_len), MAX(art_len), AVG(art_len) FROM ("
        "  SELECT SUM(LENGTH(text)) as art_len FROM chunks GROUP BY article_id"
        ")"
    ).fetchone()
    art_min, art_max, art_avg = art_lens

    # Top 5 most-chunked articles (with total text size)
    top_articles = con.execute(
        "SELECT a.title, COUNT(c.id) as n, SUM(LENGTH(c.text)) as total_chars "
        "FROM chunks c JOIN articles a ON c.article_id = a.id "
        "GROUP BY c.article_id ORDER BY n DESC LIMIT 5"
    ).fetchall()

    con.close()

    print(f"\n  {'─'*20} Articles {'─'*20}")
    print(f"  Total articles     : {s['articles']:>8,}")
    print(f"  Avg chunks/article : {avg_per_art:>8.1f}")
    print(f"  Articles w/ ≥3 chunks (Phase 1 complete) : {phase1_full:,}")
    print(f"  Text per article   :  min={art_min:,}  avg={int(art_avg):,}  max={art_max:,} chars")

    print(f"\n  {'─'*20} Chunks {'─'*22}")
    print(f"  Total chunks       : {s['chunks']:>8,}  (all extracted sections)")
    print(f"  Embedded (in FAISS): {s['embedded']:>8,}  ({embedded_pct:.1f}%)")
    print(f"  Pending (Phase 2)  : {pending:>8,}  (text stored, no vector yet)")
    print(f"  Chunk text length  :  min={min_len:,}  avg={int(avg_len):,}  max={max_len:,} chars")

    print(f"\n  {'─'*20} FAISS {'─'*23}")
    if idx_path.exists() and idx_size > 100:
        idx         = faiss_index.load_or_create(idx_path)
        inner       = faiss.downcast_index(idx.index)
        idx_type    = type(inner).__name__
        nprobe_info = f"  nprobe={inner.nprobe}" if hasattr(inner, "nprobe") else ""
        print(f"  Vectors            : {idx.ntotal:>8,}")
        print(f"  Index type         : {idx_type}{nprobe_info}")
        print(f"  Dims               : {inner.d}")
        bytes_per_vec = idx_size / idx.ntotal if idx.ntotal else 0
        print(f"  Bytes/vector       : {bytes_per_vec:.1f}")
    else:
        print(f"  (no FAISS index found)")

    if top_articles:
        print(f"\n  {'─'*20} Top articles by chunk count {'─'*5}")
        for title, n, total_chars in top_articles:
            print(f"  {n:>4} chunks  {total_chars:>8,} chars  {title}")


def _print_results(hits: list, verbose: bool = False):
    if not hits:
        print("No results.")
        return
    for i, h in enumerate(hits, 1):
        sources = []
        if h["in_faiss"]: sources.append(f"faiss={h['faiss_score']:.3f}")
        if h["in_bm25"]:  sources.append("bm25")
        print(f"\n[{i}] rrf={h['rrf_score']:.5f} ({', '.join(sources)})  "
              f"{h['title']!r}  §{h['section_title']!r}")
        print(f"    url: {h['url']}")
        if verbose:
            print(f"    {h['text'][:300]}{'...' if len(h['text']) > 300 else ''}")


def main():
    parser = argparse.ArgumentParser(description="Query a ZIM FAISS index")
    parser.add_argument("out_dir", help="Output directory (contains data.db + faiss.index)")
    parser.add_argument("query",   nargs="?", help="Search query")
    parser.add_argument("--top",       type=int,   default=10,  help="Number of results")
    parser.add_argument("--threshold", type=float, default=0.0, help="Min FAISS cosine score")
    parser.add_argument("--verbose",   action="store_true",     help="Show chunk text")
    parser.add_argument("--expand",    metavar="TITLE",         help="Embed remaining sections")
    parser.add_argument("--status",    action="store_true",     help="Show indexing progress")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.status:
        status(out_dir)
    elif args.expand:
        expand_article(out_dir, args.expand)
    elif args.query:
        hits = search(out_dir, args.query, top_k=args.top, threshold=args.threshold)
        _print_results(hits, verbose=args.verbose)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
