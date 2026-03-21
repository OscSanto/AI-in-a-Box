import re
import numpy as np
from rank_bm25 import BM25Okapi
from ingest.article_cleaner import JUNK_SECTIONS as _JUNK_SECTIONS

"""
    Rank chunks in three stages and return the top_k best.
         final = cosine(chunk, query) * (1 + ALPHA * cosine(section, query))

    A  BM25 keyword filter  →  candidate pool (fast, lexical)
    B  Cosine rerank        →  semantic similarity to query_vec
    C  Section bias         →  scale score by section-name relevance
         Alpha controls how much the section name's relevance impacts the final score:
         score = cosine(chunk, query) * (1.0 + ALPHA * cosine(section, query))
         If section is junk (references, trivia…) → multiplier = 0.3 regardless of cosine similarity.
            Otherwise, multiplier boosts chunks from sections semantically close to the query, and reduces those from

        After BM25 narrows down to a candidate pool, we compute cosine similarity of each chunk to the query vector.
         final = _cosine(chunk, query) * _section_multiplier(section, query)
         _section_multiplier = 1.0 + ALPHA * cosine(section, query)

Why this formula:
Using this formula: section bias scales proportionally to the chunk's own relevance. 
High relevant chunks (cos = 0.9) gets bigger absolute boost for a good section than a low relevant chunk (cos = 0.5).


How it works:
Every chunk is prefixed with its section in the format "[Article Title | Section]\nContent…"
Section names are extracted and embedded separately to compute a relevance-based multiplier for each chunk's score.

1. BM25 is a fast lexical filter to narrow down the pool before expensive embedding.
2. Cosine similarity of chunk and query embeddings captures semantic relevance.
   We L2-normalise vectors and compute dot product, so scores are in [-1, 1] and comparable across chunks.
3. Section bias boosts chunks from sections semantically close to the query, and reduces those from unrelated sections.
   Junk sections (references, trivia…) get a fixed low multiplier.
   A cache avoids re-embedding duplicate section names.
This approach balances speed and relevance, using BM25 to filter out clearly irrelevant chunks before the more expensive embedding-based reranking.
4. The final score combines semantic similarity and section relevance, improving the chances of selecting chunks that are not only close to the query but also from pertinent sections of the article.

Multiplier = 1.0 + ALPHA * cosine(section_name, query)
- If section_name = junk section ("references", "see also", etc ) -> multiplier = 0.3. Heavily penalize chunks from these sections, regardless of query relevance.
- If section_name is semantically close to the query -> multiplier > 1.0, boost these chunks.
- If section_name is semantically unrelated to the query -> multiplier < 1.0, reduce these chunks.

Example:
Query: "What are the symptoms of diabetes?"
- A chunk from the "Symptoms" section with high cosine similarity to the query might get a score of 0.8 (cosine) * 1.2 (section multiplier) = 0.96
- A chunk from the "Treatment" section with similar cosine similarity might get a score of 0.8 (cosine) * 0.9 (section multiplier) = 0.72
- A chunk from the "References" section, even if it has some lexical overlap, would get a score of 0.5 (cosine) * 0.3 (junk multiplier) = 0.15, likely pushing it out of the top results.

Example:
Query: Tell me about Boston's history.
[Boston | History] "history" Section cosine with query = ~0.8 → multiplier = 1.0 + 0.3 * 0.8 = 1.24. BOOSTED
[Boston | Introduction] "introduction" Section cosine with query = ~0.5 → multiplier = 1.0 + 0.3 * 0.5 = 1.15. BOOSTED but less than "history"
[Boston | References] "references" is a junk section → multiplier = 0.3 regardless of cosine similarity. CRUSHED
"""

# https://github.com/ev2900/BM25_Search_Example

# Penalty multiplier applied to chunks from junk sections
_JUNK_PENALTY  = 0.3   # replaces the multiplier for junk sections

# Deduplication: chunks whose body text overlaps above this Jaccard threshold
# with an already-selected chunk are dropped.  0.6 = 60% token overlap.
_DEDUP_THRESHOLD = 0.6

# Section bias controls how much the section name's relevance scales the score.
#   final_score = cosine(chunk, query) * (1.0 + ALPHA * cosine(section, query))
#   ALPHA = 0.3  →  multiplier range [0.7, 1.3]
_SECTION_ALPHA = 0.3   # tune this to increase/decrease the impact of section relevance on the final score

_HEADER_RE = re.compile(r"^\[([^\|]+)\|([^\]]+)\]")


def _get_title(chunk: str) -> str:
    """Extract article title from '[Title | Section]' chunk header."""
    m = _HEADER_RE.match(chunk)
    return m.group(1).strip() if m else ""


def _get_section(chunk: str) -> str:
    """Extract section name from '[Title | Section]' chunk header."""
    m = _HEADER_RE.match(chunk)
    return m.group(2).strip() if m else ""


def _body(chunk: str) -> str:
    """Chunk text with the '[Title | Section]' header line stripped."""
    return chunk.split("\n", 1)[1] if "\n" in chunk else chunk

# Measures degree of overlap between two texts.
# Deduplication filter to skip chunks too similar.
def _chunk_overlap(a: str, b: str) -> float:
    """Neat! Jaccard!
    Measures similarity between two sets.
    J(A,B) = |A ∩ B| / |A ∪ B|
    Range [0, 1], where 0 means no overlap and 1 means identical sets.
    """
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _bm25_pass(query: str, chunks: list, top_k: int) -> list[str]:
    """
    Stage A — BM25 keyword filter.
    We use a weak BM25 filter to quickly narrow down to a pool of candidates for more expensive semantic reranking.
    Return a pool of size max(top_k * 5, 20)

    This will be reranked semantically in Stage B, 
    but BM25 is a good first pass to filter out completely irrelevant sections and speed up the embedding stage.
    """
    bm25        = BM25Okapi([c.split() for c in chunks])
    scores      = bm25.get_scores(query.split())
    pool_size   = min(len(chunks), max(top_k * 5, 20)) #
    ranked      = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in ranked[:pool_size]]


def _cosineScorePass(chunkPool: list[str], query_vec: np.ndarray, embedder) -> np.ndarray:
    """
    Stage B — Cosine similarity.
    Batch-embeds all chunks in one Ollama call, then computes dot product against query_vec.
    """
    chunk_vecs = embedder.embed_batch([c[:500] for c in chunkPool])  # (pool_size, dim), already L2-normalised
    return chunk_vecs @ query_vec          # shape (pool_size,)


def _section_multipliers(chunkPool: list[str], query_vec: np.ndarray, embedder) -> list[float]:
    """
    Stage C — Section bias multipliers.
    Batch-embeds all unique non-junk section names in one Ollama call.
    """
    sections = [_get_section(c).lower() for c in chunkPool]

    # Collect unique sections that need embedding
    to_embed = [s for s in dict.fromkeys(sections)
                if s and not any(j in s for j in _JUNK_SECTIONS)]

    cache: dict[str, float] = {}
    if to_embed:
        vecs = embedder.embed_batch(to_embed)   # (n_unique, dim), already normalised
        for sec, vec in zip(to_embed, vecs):
            cache[sec] = float(vec @ query_vec)

    multipliers: list[float] = []
    for section in sections:
        if not section:
            multipliers.append(1.0)
        elif any(j in section for j in _JUNK_SECTIONS):
            multipliers.append(_JUNK_PENALTY)
        else:
            multipliers.append(1.0 + _SECTION_ALPHA * cache[section])

    return multipliers


def rerank_by_title(results: list, queries: list, entities: list, rewritten: str) -> list:
    """
    Sort Kiwix article results by token-overlap score against query terms.
    Adds _title_score to each result dict. Returns sorted list (descending).
    """
    if len(results) <= 1:
        return results

    def _tok(text: str) -> set:
        return set(re.sub(r"[^\w\s]", " ", text.lower()).split())

    ref_tokens: set = set()
    for t in list(queries):
        ref_tokens.update(_tok(t))

    normalized_rewritten = " ".join(re.sub(r"[^\w\s]", " ", rewritten.lower()).split())

    """"
    score = (overlap / len(ref_tokens)) - (0.4 * missing / len(ref_tokens)) - (0.2 * extra / len(title_tokens)) + (0.25 if title matches rewritten else 0)
    where:
    - overlap = number of tokens in both title and reference
    - missing = number of tokens in reference but not in title
    - extra   = number of tokens in title but not in reference
    - len(ref_tokens) = total number of tokens in reference (for normalization)
    - len(title_tokens) = total number of tokens in title (for normalization)
    - The final term adds a small boost if the title matches the rewritten query, as an

    exact match is a strong signal of relevance.
    """
    def _score(title: str) -> float:
        title_tokens = _tok(title)
        overlap  = ref_tokens & title_tokens
        missing  = ref_tokens - title_tokens
        extra    = title_tokens - ref_tokens
        base     = len(overlap) / max(1, len(ref_tokens))
        penalty  = (0.4 * len(missing) / max(1, len(ref_tokens))
                  + 0.2 * len(extra)   / max(1, len(title_tokens)))
        bonus    = 0.25 if re.sub(r"[^\w\s]", "", title.lower()).strip() == normalized_rewritten else 0.0
        return base - penalty + bonus

    results = list(results)
    results.sort(key=lambda r: _score(r["title"]), reverse=True)
    for r in results:
        r["_title_score"] = _score(r["title"])
        print(f"  [title] {r['_title_score']:.3f}  {r['title']!r}", flush=True)
    return results


def rank_chunks(query: str, chunks: list, top_k: int, embedder=None, query_vec=None,
                title_scores: dict = None) -> list:
    
    if not chunks:
        print("[ranker] no chunks to rank", flush=True)
        return []

    # Stage A: BM25 filter 
    pool = _bm25_pass(query, chunks, top_k)
    if embedder is None or query_vec is None:
        return pool[:top_k]

    # Stage B + C: Cosine rerank with section bias
    # B calculates cosine similarity of each chunk to the query vector
    # C calculates a multiplier based on the chunk's section relevance to the query
    try:
        cos    = _cosineScorePass(pool, query_vec, embedder) # param: list of chunks after BM25, returns cosine similarity scores
        mults  = _section_multipliers(pool, query_vec, embedder) # param: same list of chunks, returns multipliers based on section relevance
        title_w = [title_scores.get(_get_title(c), 1.0) for c in pool] if title_scores else [1.0] * len(pool)
        scores = [float(cos[i]) * mults[i] * title_w[i] for i in range(len(pool))]

        ranked = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)

        # Stage D: deduplication
        # skip body text overlaps too heavily with an already selected chunk.
        selected        = []
        selected_bodies = []
        for i in ranked:
            body = _body(pool[i])
            # If this chunk's body text overlaps too much with any already-selected chunk, skip it.
            if any(_chunk_overlap(body, sb) >= _DEDUP_THRESHOLD for sb in selected_bodies):
                print(f"  [ranker skip] dedup  score={scores[i]:.4f}  section='{_get_section(pool[i])}'", flush=True)
                continue

            selected.append(pool[i])
            selected_bodies.append(body)
            print(f"  [ranker {len(selected):02d}] score={scores[i]:.4f}  section='{_get_section(pool[i])}'", flush=True)
            if len(selected) >= top_k:
                break

        return selected

    except Exception as e:
        print(f"[ranker] cosine rerank failed, falling back to BM25: {e}", flush=True)
        return pool[:top_k]
