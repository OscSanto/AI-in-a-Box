"""
ZIM indexing pipeline.

Two separate steps to keep memory bounded on low-RAM devices:

  # Step 1 — extract articles from ZIM into SQLite (no embedding model loaded)
  python -m zim_indexer.pipeline --extract /path/to/file.zim

  # Step 2 — embed pending chunks into FAISS (fresh process, no ZIM in RAM)
  python -m zim_indexer.pipeline --embed /path/to/file.zim

  # Or run both steps sequentially in one command (uses more peak RAM)
  python -m zim_indexer.pipeline /path/to/file.zim

Resume: already-indexed articles are skipped in --extract.
        Already-embedded chunks are skipped in --embed.
"""

import gc
import re
import time
from pathlib import Path

from libzim.reader import Archive

from zim_indexer import db, extract
from zim_indexer import index as faiss_index

# ── Config ────────────────────────────────────────────────────────────────────

_SKIP_RE = re.compile(
    r"^(Category:|Template:|Portal:|File:|Help:|Special:|Talk:)"
    r"|\(disambiguation\)",
    re.I,
)
_SKIP_NS = {"Category", "Template", "Portal", "File", "Help", "Special",
             "Talk", "Wikipedia", "User", "MediaWiki", "Module"}

_PRIORITY_LEAD    = 3   # lead units placed first in the embedding queue
_PRIORITY_INFOBOX = 10  # infobox rows placed early, independent of prose count
_PRIORITY_PROSE   = 12  # prose units placed early after lead/infobox chunks
_EMBED_BATCH   = 16   # small batches — keeps ONNX + vector RAM low
_SAVE_EVERY    = 128  # write FAISS to disk every N chunks (lose less on crash)


def _output_dir(zim_path: Path) -> Path:
    d = zim_path.parent / zim_path.stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_chunk_text(title: str, section_title: str, text: str) -> str:
    return f"Article: {title}\nSection: {section_title}\nText: {text}"


def _build_infobox_text(title: str, header: str, label: str, value: str) -> str:
    return f"Article: {title}\nInfobox: {header}\nFact: {label} = {value}"


def _is_redirect(entry) -> bool:
    try:
        return entry.is_redirect
    except Exception:
        return False


# ── Step 1: Extract ───────────────────────────────────────────────────────────

def run_extract(zim_path: Path):
    """
    Iterate ZIM, extract article text, store in SQLite.
    Does NOT load the embedding model — keeps RAM low during extraction.
    """
    zim_path = zim_path.resolve()
    out_dir  = _output_dir(zim_path)
    db_path  = out_dir / "data.db"

    print(f"\n{'='*60}", flush=True)
    print(f"[extract] ZIM:    {zim_path}", flush=True)
    print(f"[extract] Output: {db_path}", flush=True)
    print(f"{'='*60}\n", flush=True)

    con     = db.open_db(db_path)
    archive = Archive(str(zim_path))

    total   = archive.entry_count
    print(f"ZIM entries: {total:,}", flush=True)

    t0           = time.time()
    seen         = 0
    skipped      = 0
    already_done = 0
    new_articles = 0
    total_chunks = 0

    for i in range(total):
        seen += 1

        if seen % 50_000 == 0:
            elapsed = time.time() - t0
            print(f"  [scan] {seen:,}/{total:,} entries scanned | "
                  f"new={new_articles:,} skip={skipped:,} | {seen/elapsed:.0f} entries/s",
                  flush=True)

        try:
            entry = archive._get_entry_by_id(i)
        except Exception:
            skipped += 1
            continue

        title = entry.title.strip()
        path  = str(entry.path)

        if not title or _is_redirect(entry):
            skipped += 1
            continue
        if _SKIP_RE.search(title):
            skipped += 1
            continue
        ns = path.split("/")[0] if "/" in path else ""
        if ns in _SKIP_NS:
            skipped += 1
            continue

        if db.article_exists(con, title):
            already_done += 1
            if already_done % 10_000 == 0:
                print(f"  [resume] {already_done:,} already indexed...", flush=True)
            continue

        try:
            item   = entry.get_item()
            html   = bytes(item.content).decode("utf-8", errors="replace")
            parsed = extract.extract(html)
        except Exception:
            skipped += 1
            continue

        if not parsed or not parsed["lead"]:
            skipped += 1
            continue

        prose_chars  = len(parsed["lead"]) + sum(
            len(p) for sec in parsed["sections"] for p in sec["paragraphs"]
        )
        infobox_rows = len(parsed["infobox"]["rows"]) if parsed.get("infobox") else 0
        if prose_chars < 200 and infobox_rows < 3:
            skipped += 1
            continue

        url = path if path.startswith("A/") else f"A/{path}"

        # Build lead / infobox / prose lists before assigning chunk_index,
        # so each content type gets early coverage if embedding is interrupted.
        lead_units = [
            {
                "section_title": "Lead",
                "text": _build_chunk_text(title, "Lead", para),
            }
            for para in (parsed.get("lead_paragraphs") or [parsed["lead"]])
            if para and para.strip()
        ]

        infobox_chunks = []
        if parsed.get("infobox"):
            ib     = parsed["infobox"]
            header = ib["header"]
            for row in ib["rows"]:
                infobox_chunks.append({
                    "section_title": f"Infobox: {header}",
                    "text":          _build_infobox_text(title, header,
                                                         row["label"], row["value"]),
                })

        # Round-robin across sections: take 1 para from each section before
        # taking a 2nd from any section. This gives many sections early FAISS
        # coverage even if the full embedding job is interrupted.
        _sec_lists = [
            {"title": sec["title"], "paras": sec["paragraphs"]}
            for sec in parsed["sections"]
            if sec["paragraphs"]
        ]
        prose_chunks = []
        max_depth = max((len(s["paras"]) for s in _sec_lists), default=0)
        for _round in range(max_depth):
            for _sec in _sec_lists:
                if _round < len(_sec["paras"]):
                    prose_chunks.append({
                        "section_title": _sec["title"],
                        "text":          _build_chunk_text(title, _sec["title"],
                                                           _sec["paras"][_round]),
                    })

        # Ordering: priority lead | priority infobox | priority prose
        #           | remaining lead | remaining prose | remaining infobox
        # The --embed step still embeds every chunk. This order only controls
        # which chunks are embedded first during resumable/incremental runs.
        ordered = (
            lead_units[:_PRIORITY_LEAD]
            + infobox_chunks[:_PRIORITY_INFOBOX]
            + prose_chunks[:_PRIORITY_PROSE]
            + lead_units[_PRIORITY_LEAD:]
            + prose_chunks[_PRIORITY_PROSE:]
            + infobox_chunks[_PRIORITY_INFOBOX:]
        )

        all_chunks = [
            {**c, "chunk_index": i} for i, c in enumerate(ordered)
        ]

        article_id = db.insert_article(con, title, url, str(zim_path))
        db.insert_chunks(con, article_id, all_chunks)
        new_articles += 1
        total_chunks += len(all_chunks)

        # Release BeautifulSoup tree — it holds a lot of C-level memory
        del html, parsed, all_chunks

        # Batch commit every 500 articles — much faster than per-article commits
        if new_articles % 500 == 0:
            con.commit()

        if new_articles % 1000 == 0:
            gc.collect()
            elapsed = time.time() - t0
            print(f"  [{seen:>8,}/{total:,}] new={new_articles:,} skip={skipped:,} "
                  f"resume={already_done:,} | chunks={total_chunks:,} "
                  f"| {new_articles/elapsed:.1f} art/s", flush=True)

    con.commit()  # final commit for any remaining buffered articles
    con.close()
    elapsed = time.time() - t0
    print(f"\n[extract] Done in {elapsed/60:.1f} min — "
          f"{new_articles:,} articles, {total_chunks:,} chunks", flush=True)


# ── Step 2: Embed ─────────────────────────────────────────────────────────────

def run_embed(zim_path: Path):
    """
    Embed every pending chunk into FAISS.
    Runs as a separate process — starts with a clean memory slate after extraction.
    Fetches _EMBED_BATCH rows at a time so RAM stays bounded.
    """
    # Import embed here so the ONNX model only loads in the embed step
    from zim_indexer import embed

    zim_path = zim_path.resolve()
    out_dir  = _output_dir(zim_path)
    db_path  = out_dir / "data.db"
    idx_path = out_dir / "faiss.index"

    print(f"\n{'='*60}", flush=True)
    print(f"[embed] DB:    {db_path}", flush=True)
    print(f"[embed] FAISS: {idx_path}", flush=True)
    print(f"{'='*60}\n", flush=True)

    con = db.open_db(db_path)
    idx = faiss_index.load_or_create(idx_path)

    total_pending = con.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedded = 0",
    ).fetchone()[0]

    if not total_pending:
        print("  No pending chunks — already fully embedded.", flush=True)
        con.close()
        return

    print(f"  {total_pending:,} pending chunks to embed (batch={_EMBED_BATCH})...", flush=True)
    t0   = time.time()
    done = 0

    # ── IVFFlat training pre-pass ─────────────────────────────────────────────
    # IVFFlat requires a training sample before any vectors can be added.
    # We embed enough chunks to train, add them, mark them done, then
    # continue the main loop for the rest. Each chunk is only embedded once.
    if idx.ntotal == 0 and total_pending >= faiss_index._IVF_MIN_N:
        import math
        import numpy as np
        nlist    = max(4, min(int(math.sqrt(total_pending)), 4096))
        train_n  = min(max(nlist * 39, 5_000), total_pending)
        print(f"  Training IVFFlat — fetching {train_n:,} vectors "
              f"(nlist={nlist})...", flush=True)

        # Stream-encode training chunks in small batches to avoid
        # holding all text strings + vectors in RAM simultaneously.
        train_ids       = []
        train_vecs_list = []
        cur = con.execute(
            "SELECT id, text FROM chunks WHERE embedded = 0 "
            "ORDER BY id LIMIT ?",
            (train_n,),
        )
        while True:
            rows = cur.fetchmany(_EMBED_BATCH)
            if not rows:
                break
            ids   = [r[0] for r in rows]
            texts = [r[1] for r in rows]
            train_ids.extend(ids)
            train_vecs_list.append(embed.encode(texts))
            if len(train_ids) % (_EMBED_BATCH * 25) == 0 or len(train_ids) >= train_n:
                elapsed = time.time() - t0
                rate = len(train_ids) / elapsed if elapsed > 0 else 0
                print(
                    f"  Training vectors {len(train_ids):,}/{train_n:,} "
                    f"({rate:.0f} chunks/s)",
                    flush=True,
                )
            del ids, texts, rows

        train_vecs = np.vstack(train_vecs_list)
        del train_vecs_list
        gc.collect()

        idx = faiss_index.make_ivf(total_pending, train_vecs)
        faiss_index.add_vectors(idx, train_ids, train_vecs)
        db.mark_embedded(con, train_ids)
        faiss_index.save(idx, idx_path)
        done += len(train_ids)

        del train_vecs, train_ids
        gc.collect()
        print(f"  Training done. Continuing with remaining chunks...", flush=True)

    while True:
        batch = con.execute(
            "SELECT id, text FROM chunks WHERE embedded = 0 "
            "ORDER BY id LIMIT ?",
            (_EMBED_BATCH,),
        ).fetchall()

        if not batch:
            break

        chunk_ids = [r[0] for r in batch]
        texts     = [r[1] for r in batch]

        vecs = embed.encode(texts)                  # (N, 384) normalized float32
        faiss_index.add_vectors(idx, chunk_ids, vecs)
        db.mark_embedded(con, chunk_ids)

        del texts, vecs                             # free immediately after use

        done += len(batch)

        if done % (_EMBED_BATCH * 10) == 0 or done >= total_pending:
            elapsed = time.time() - t0
            rate    = done / elapsed if elapsed > 0 else 0
            print(f"  Embedded {done:,}/{total_pending:,} ({rate:.0f} chunks/s)", flush=True)

        if done % _SAVE_EVERY == 0:
            faiss_index.save(idx, idx_path)
            gc.collect()

    faiss_index.save(idx, idx_path)

    # Build FTS5 title index so queries can use hybrid RRF retrieval immediately
    print("  Building FTS5 title index...", flush=True)
    from zim_indexer.db import init_fts
    init_fts(con)
    print("  FTS5 ready.", flush=True)

    s = db.stats(con)
    con.close()
    elapsed = time.time() - t0
    print(f"\n[embed] Done in {elapsed/60:.1f} min — "
          f"{s['embedded']:,}/{s['chunks']:,} chunks embedded | "
          f"{idx.ntotal:,} FAISS vectors", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ZIM indexer")
    parser.add_argument("zim", help="Path to .zim file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--extract", action="store_true", help="Extract only (no embedding)")
    mode.add_argument("--embed",   action="store_true", help="Embed only (no ZIM parsing)")
    args = parser.parse_args()

    zim_path = Path(args.zim)

    if args.extract:
        run_extract(zim_path)
    elif args.embed:
        run_embed(zim_path)
    else:
        # Both steps sequentially — more RAM than running separately
        run_extract(zim_path)
        gc.collect()
        run_embed(zim_path)


if __name__ == "__main__":
    main()
