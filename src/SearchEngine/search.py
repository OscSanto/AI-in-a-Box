"""
BM25 title ranking, full search pipeline, and pipeline deduplication.

_get_or_run_pipeline guarantees the scrape+rank pipeline runs at most once per
query at a time — /search and /ai fire simultaneously on every page load, so
without this both would hit a cache miss together and scrape all ZIMs twice.
"""
import threading

import numpy as np
from rank_bm25 import BM25Okapi

from SearchEngine.config import SEARCH_MAX_RESULTS, TITLE_WEIGHT_FLOOR
from SearchEngine.cache import db_get_results, db_set_results
from SearchEngine.kiwix import scrape_all_zims
from SearchEngine.keywords import extract_keywords
from SearchEngine.utils import make_marker

_inflight_lock = threading.Lock()
_inflight: dict[str, threading.Event] = {}


def rank_by_title(query: str, results: list, mark=None) -> list:
    """
    BM25 ranking over title + snippet text — zero embedding cost, runs in milliseconds.
    Including the snippet gives richer signal for topical queries.
    Returns up to SEARCH_MAX_RESULTS results sorted by score descending.
    """
    if not results:
        return []
    tokenized_docs  = [(r["title"] + " " + r.get("snippet", "")).lower().split() for r in results]
    tokenized_query = query.lower().split()
    scores = BM25Okapi(tokenized_docs).get_scores(tokenized_query)

    # Exact-title boost: push the article whose title == query to the front.
    # Prevents "Banana (disambiguation)" outranking "Banana".
    q_lower = query.lower().strip()
    for i, r in enumerate(results):
        if r["title"].lower().strip() == q_lower:
            scores[i] = float(max(scores)) * 1.5

    ranked_idx = np.argsort(scores)[::-1][:SEARCH_MAX_RESULTS]
    ranked     = [results[i] for i in ranked_idx]

    # Normalise scores to [TITLE_WEIGHT_FLOOR, 1.0] so low-ranked articles' chunks
    # still get considered (title_w=0 would zero out the final chunk score).
    max_score = float(max(scores)) if max(scores) > 0 else 1.0
    for i, r in enumerate(results):
        normalized = max(0.0, float(scores[i])) / max_score
        r["_title_score"] = TITLE_WEIGHT_FLOOR + (1.0 - TITLE_WEIGHT_FLOOR) * normalized

    if mark:
        mark("2_bm25_rank", f"{len(ranked)} results ranked")
    for rank, i in enumerate(ranked_idx):
        print(f"  [{rank+1:02d}] {scores[i]:.4f}  {results[i]['title']!r:<40}  ({results[i]['source_zim']})", flush=True)
    return ranked


def full_search_pipeline(q: str) -> list:
    mark     = make_marker("search", q)
    keywords = extract_keywords(q)
    results, zim_counts = scrape_all_zims(keywords)
    mark("1_scrape_zims", f"{len(results)} total | {q!r} → {keywords!r}")
    for zim, count in sorted(zim_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  → {zim:<45} {count} hits", flush=True)
    return rank_by_title(keywords, results, mark=mark)


def get_or_run_pipeline(q: str) -> list:
    """
    First caller runs the pipeline and saves to SQLite.
    Any concurrent caller for the same query blocks on the event, then reads
    the result from SQLite once the first caller is done.
    """
    cached = db_get_results(q)
    if cached:
        return cached

    with _inflight_lock:
        if q in _inflight:
            ev       = _inflight[q]
            is_first = False
        else:
            ev            = threading.Event()
            _inflight[q]  = ev
            is_first      = True

    if not is_first:
        ev.wait(timeout=120)          # 120s safety timeout — never block forever
        return db_get_results(q) or []

    ranked = []
    try:
        ranked = full_search_pipeline(q)
        if ranked:
            db_set_results(q, ranked)  # save synchronously so waiters can read
    finally:
        with _inflight_lock:
            _inflight.pop(q, None)
        ev.set()                       # wake all waiters regardless of success or failure

    return ranked
