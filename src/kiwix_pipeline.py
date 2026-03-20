import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Iterator

import numpy as np

import retrieval.kiwix_client as kiwix
import retrieval.chunk_ranker as ranker

from retrieval import intent_classifier as ic
from retrieval.intent_classifier import classify_intent
from retrieval.kiwix_client import is_kiwix_url, lookup_article_by_title, parallel_search, _is_junk_title
from retrieval.entity_store import (
    DynamicEntityStore,
    detect_subtopic_label,
    expand_entity_list, learn_from_articles,
)

"""
https://en.wikipedia.org/wiki/Entity_linking?
User input layer
-user query
-converstaion history
-profile/preferences/admin yaml
-permissoin/guardrails

Intent classifier
-is this just QA?
-are tools needed?
-multi step?
-summarization: PDF, large text, QA
-retreival: URL link (outside of kiwix)

Plan
1.rewrite & preprocess query
2.search retreival sources
3.rank evidence
4.produce grounded answer
"""

class Pipeline:
    """
    RAG pipeline stages 1-10: query rewrite, embedding, cache lookup,
    entity expansion, Kiwix search, title ranking, chunking, and context assembly.

    Instantiate once at startup with stable dependencies; call build_context()
    per request with only the query and per-request metadata.
    """

    def __init__(self, config, embedder, llm, content_graph, query_memory):
        self.config = config
        self.embedder = embedder
        self.llm = llm
        self.content_graph = content_graph
        self.query_memory = query_memory

    def build_context(self, query: str, 
                      zim_meta: dict = None,
                      mode: str = "kiwix", 
                      on_sources=None) -> dict:
        """
        Runs pipeline stages 1-10 (RAG only, no LLM generation).

        Returns a dict:
          cache_hit=True  => {cache_hit, cached_answer, stages, t0}
          cache_hit=False => {cache_hit, query, context, messages, query_vec (list),
                             entity_title_candidates, rewritten, kiwix_results,
                             article_chunks, top_chunks, stages, t0}

        query_vec is returned as a plain Python list so it can be JSON-serialised
        and sent to client for WebGPU generation + later POSTed back to /remember.
        """
        config        = self.config
        embedder      = self.embedder
        llm           = self.llm
        content_graph = self.content_graph
        query_memory  = self.query_memory

        t0 = time.time()
        t_last = [t0]
        stages = {}
        def mark(stage: str, start: float = None, end: float = None):
            now  = end if end is not None else time.time()
            took = round(now - (start if start is not None else t_last[0]), 3)
            stages[stage] = took
            t_last[0] = now
            print(f"[pipeline] {stage} | took={took}s | total={round(now - t0, 3)}s", flush=True)

        print(f"\n[pipeline] ===== query: {query} =====", flush=True)
        # =========== Stage 2 + 3: query_rewrite and query_embedding in parallel
        # Embedding uses a different Ollama endpoint (/api/embed vs /api/chat)
        # so both can be in-flight simultaneously.
        _t_parallel = time.time()
        with ThreadPoolExecutor(max_workers=2) as _ex:
            _f_rewrite = _ex.submit(ic.classify_intent, query.strip(), llm, config, mode)
            _f_embed   = _ex.submit(embedder.embed, query.strip())

            intent_result   = _f_rewrite.result()
            _t_rewrite_done = time.time()
            query_vec       = _f_embed.result()
            _t_embed_done   = time.time()

        rewritten = intent_result.rewritten
        queries   = intent_result.queries
        print(f"[pipeline]  mode={mode}  rewritten={rewritten!r}  queries={list(queries)}", flush=True)
        mark("2_query_rewrite",   start=_t_parallel, end=_t_rewrite_done)
        mark("3_query_embedding", start=_t_parallel, end=_t_embed_done)

        # ============ Stage 4: query_memory_lookup & CACHE HIT detection
        # Check cache before entity detection — on a hit we skip to retreival of previous generated query
        if config.query_memory_enabled:
            """QueryMemory is a cache of past queries (with embeddings + generated answers)
            When new query comes in, we embed it and look for similar past queries in QueryMemory.
              If we find a past query with high similarity, we consider it a cache hit and return the cached answer immediately,
              skipping all subsequent stages of the pipeline.
              This is an optimization to save time on repeated or very similar queries.
            Cache hit detection considers only past queries from the same ZIM (if zim_meta is provided) to improve relevance.

            ON CACHE HIT: return immediately and skip all subsequent stages. Include cache_hit=True and the cached_answer in the return dict.

            ISSUES: If the user asks the same question twice but with different wording, we might miss the cache hit due to embedding differences. 
                Future improvement could include a more robust similarity check or query normalization.
            ISSUE: If answer of a past query is wrong and we get a cache hit on it, we will return the wrong answer. 
                Future improvement could include a freshness mechanism to expire old cache entries or a way for users to flag incorrect answers.
            """
            # Considering the current zim
            current_zim = zim_meta.get("zim_id")

            # Filter to only cache hits from the current ZIM
            # -Most similar vectors to current raw query embeddings
            # -Don't want cached answer from wikipedia_espaNol
            # TODO: consideration on multi-cross zim querry embedding look ups.
            past = query_memory.lookup(query_vec, config.qm_topk) # TODO: Currently: FAISS lookup only. Inlcude Google's SCANN
            past = [p for p in past if current_zim in p.get("zim_ids", [])]

            if past and past[0]["similarity"] >= config.qm_high_sim: # threshold based on config.qm_high_similarty
                hit = past[0]
                mark(f"4_query_memory — cache hit (sim={hit['similarity']:.4f}, zim={hit.get('zim_ids')})")

                return {"cache_hit": True, "cached_answer": hit["answer_text"], "stages": stages, "t0": t0}
            top_sim = past[0]["similarity"] if past else 0
            mark(f"4_query_memory — no hit (top sim={top_sim:.4f})")

        # =====Stage: Entity store => load and expand entity/search lists with learned aliases
        #TODO: This should overall be overhauled and replaced with just disambiguation page scraping and alias extraction from article intros, 
            # rather than the current mix of alias+subtopic extraction from past queries vs article intros.
        """EntityStore is a simple database of entity names and their known aliases, learned from past queries and article intros.
        MLK -> Machine Learning Kit or Martin Luther King, NYC -> New York City, etc.
        When we get a new query, we check if it contains any known entities or aliases from the EntityStore.
        If it does, we expand our search terms to include all known aliases for those entities, improving our chances of finding relevant articles in Kiwix.

        ISSUES: The EntityStore is currently in-memory and will be lost on server restart. Future improvement could include persisting it to disk or a database.
        TODO: Consider giving higher priority to aliases that have been seen more frequently or more recently in past queries, as they might be more relevant.
        TODO: Consider asking user for clarification when we detect a known ambiguous alias (e.g. MLK) in the query, to improve accuracy.
        TODO: Expand entity store with aliases learned from article intros (e.g. "also known as" phrases), not just past queries.
        TODO: Expand entity store using disambiguation pages — if we detect a disambiguation page for a term, we can add all the listed candidates as aliases for that term.
        """
        entity_store    = DynamicEntityStore(config.data_dir)
        query_subtopic  = detect_subtopic_label(query)
        expanded_entities = expand_entity_list(entity_store, query, rewritten, intent_result.entities)
        parent_hint     = rewritten if rewritten else (queries[0] if queries else query)
        store_subtopics = entity_store.subtopic_candidates(parent_hint, topk=3)
        mark(f"5a_entity_expand — {len(expanded_entities)} entities")

        fast_path_hit: dict | None = None 

        # ============ Stage 5: kiwix_search
        if fast_path_hit:
            kiwix_results = [fast_path_hit]
            seen_titles   = {fast_path_hit["title"].strip()}
            mark("5_kiwix_search — fast-path 1 result")
        else:
            entity_list = expanded_entities[:config.kiwix_max_entities + 3]
            all_terms   = [rewritten]
            extra       = store_subtopics if query_subtopic else []
            for q in list(queries) + extra:
                if q and q not in all_terms:
                    all_terms.append(q)

            kiwix_results, seen_titles = parallel_search(
                config.kiwix_endpoint, config.zim_content_id,
                entity_list, all_terms, config.kiwix_result_limit,
            )
            mark(f"5_kiwix_search — {len(kiwix_results)} results | entities={expanded_entities}")

        # ── Stage 6a: article title rerank
        kiwix_results = ranker.rerank_by_title(kiwix_results, queries, intent_result.entities, rewritten)
        mark(f"6_title_rerank — {len(kiwix_results)} articles scored")

        # ── Stage 6b: fetch sections + entity store learning
        # Only fetch articles with a positive title score — negative-scored articles
        # are unrelated and won't yield useful alias/disambiguation data.
        # Always keep at least max_articles_to_chunk candidates so the selection
        # filter below has something to work with.
        _fetch_candidates = [r for r in kiwix_results if r.get("_title_score", 1.0) >= 0.0]
        if len(_fetch_candidates) < config.max_articles_to_chunk:
            _fetch_candidates = kiwix_results[:config.max_articles_to_chunk]

        def _fetch_sections(r):
            try:
                return r, kiwix.fetch_article_sections(r["url"], config.max_chunk_chars)
            except Exception:
                return r, []

        with ThreadPoolExecutor(max_workers=8) as _ex:
            fetched = list(_ex.map(_fetch_sections, _fetch_candidates))

        sections_by_title = learn_from_articles(entity_store, fetched)
        mark(f"6b_entity_store_learn — {len(sections_by_title)} articles observed")

        # ── Stage 6: article selection (threshold filter + disambiguation filter + slice)
        _scored = kiwix_results   # sorted by title_score; save for fallback below
        kiwix_results = [
            r for r in _scored
            if r.get("_title_score", 1.0) >= config.title_score_threshold
            and not r.get("_is_disambiguation", False)
        ][:config.max_articles_to_chunk]

        # Score-threshold fallback: when all non-disambiguation results are below threshold
        # (common on multi-hop queries), keep the top scored non-disambiguation articles.
        _non_disambig = [r for r in _scored if not r.get("_is_disambiguation", False)]
        if not kiwix_results and _non_disambig:
            kiwix_results = _non_disambig[:config.max_articles_to_chunk]
            print(f"  [title-fallback] all below threshold — keeping top {len(kiwix_results)} by score", flush=True)

        # Disambiguation retry: if all results were filtered as disambiguation pages,
        # use candidates the entity_store just learned to search for the real article.
        if not kiwix_results:
            retry_candidates = entity_store.alias_candidates(rewritten, topk=3)
            if retry_candidates:
                print(f"  [disambig-retry] all articles filtered — retrying with {retry_candidates}", flush=True)
                for cand in retry_candidates[:2]:
                    for r in kiwix.search_kiwix(config.kiwix_endpoint, config.zim_content_id, cand, 3):
                        title = r["title"].strip()
                        if (title not in seen_titles and len(title) > 2
                                and any(c.isalpha() for c in title) and not _is_junk_title(title)):
                            kiwix_results.append(r)
                            seen_titles.add(title)
                    if kiwix_results:
                        break
                kiwix_results = kiwix_results[:config.max_articles_to_chunk]

        for r in kiwix_results:
            print(f"  [article] {r['title']!r}  (score={r.get('_title_score', 1.0):.3f})", flush=True)
        mark(f"7_article_select — {len(kiwix_results)} articles")

        # ── Stage 7: candidate_chunk_collection
        # Section-aware chunking with sentence-level overlap.
        # Each chunk: last sentence of previous paragraph + this paragraph + first sentence of next.
        # Sentence overlap respects boundaries by construction — no mid-sentence cutoff.
        # Disabled when overlap_tokens = 0 in config.
        USE_OVERLAP = config.chunk_overlap_tokens > 0

        _SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

        def _last_sentence(text: str) -> str:
            """Last sentence of a paragraph."""
            parts = _SENT_SPLIT.split(text.strip())
            return parts[-1].strip() if parts else ""

        def _first_sentence(text: str) -> str:
            """First sentence of a paragraph."""
            parts = _SENT_SPLIT.split(text.strip())
            return parts[0].strip() if parts else ""

        # Proportional chunk budget: top-scoring article gets max_chunks_per_article,
        # others scale down relative to it. Floor of 2 so every article contributes
        # enough text for the LLM to extract a specific fact (multi-hop queries need this).
        if len(kiwix_results) > 1:
            max_score = max(r.get("_title_score", 1.0) for r in kiwix_results)
            for art in kiwix_results:
                art["_chunk_budget"] = max(2, round(
                    art.get("_title_score", 1.0) / max_score * config.max_chunks_per_article
                ))
        else:
            for art in kiwix_results:
                art["_chunk_budget"] = config.max_chunks_per_article

        article_chunks = []
        graph_articles = []
        for art in kiwix_results:
            # Reuse sections fetched during Stage 6b — avoids a duplicate HTTP call
            sections = sections_by_title.get(art["title"]) or kiwix.fetch_article_sections(art["url"], config.max_chunk_chars)
            if not sections:
                continue
            title = art["title"]

            # Flatten to (section, paragraph) pairs for overlap windowing
            all_paras = []
            for sec in sections:
                for p in sec["paragraphs"]:
                    all_paras.append((sec["section"], p))

            art_chunk_count = 0
            for i, (section, para) in enumerate(all_paras):
                if art_chunk_count >= art["_chunk_budget"]:
                    break
                left  = (_last_sentence(all_paras[i - 1][1]) + " ") if i > 0 and USE_OVERLAP else ""
                right = (" " + _first_sentence(all_paras[i + 1][1])) if i < len(all_paras) - 1 and USE_OVERLAP else ""
                chunk = f"[{title} | {section}]\n{left}{para}{right}"
                if len(chunk) >= config.min_paragraph_chars:
                    article_chunks.append(chunk)
                    art_chunk_count += 1

            # Collect for graph ingestion (all raw paragraphs, no overlap)
            raw_paras = [p for sec in sections for p in sec["paragraphs"]]
            graph_articles.append({
                "title": title,
                "content": " ".join(raw_paras)[:500],
                "chunks": raw_paras,
            })

        for i, chunk in enumerate(article_chunks):
            print(f"  [chunk {i+1:02d}] {chunk[:120].replace(chr(10), ' ')!r}", flush=True)
        mark(f"8_chunk_collection — {len(article_chunks)} chunks from {len(kiwix_results)} articles")

        # ── Stage 7b: content_graph_build (incremental)
        # Ingest articles into the graph for relation extraction (Stage 8.5) and future traversal.
        if config.content_graph_enabled and graph_articles:
            try:
                content_graph.build(graph_articles, zim_meta)
                print(f"[pipeline] graph_build — {len(graph_articles)} articles", flush=True)
            except Exception as e:
                print(f"[pipeline] graph_build failed (non-fatal): {e}", flush=True)

        #Stage 8.5 deprecated
        llm_relations = []
        top_chunks = []

        # ====== Stage 9: chunk_ranking
        # Multi-hop queries (>1 entity): rank against the original query — it encodes all hops.
        # Single-entity queries: rank against rewritten (focused Wikipedia title).
        if article_chunks:
            rank_query = query if len(intent_result.entities) > 1 else rewritten
            top_chunks = ranker.rank_chunks(rank_query, article_chunks, config.topk_chunks, embedder, query_vec)
            # Restore document order so context reads coherently (ranking selects, order presents)
            order_map = {c: i for i, c in enumerate(article_chunks)}
            top_chunks.sort(key=lambda c: order_map.get(c, 0))

        for i, chunk in enumerate(top_chunks):
            print(f"  [ranked {i+1:02d}] ({len(chunk)} chars)\n{chunk}", flush=True)

        mark(f"9_chunk_ranking — {len(top_chunks)} top chunks from {len(article_chunks)} total")

        # ====== Stage 10: context_assembly
        # Chunks already carry [Title | Section] headers — no extra [Source N] wrapper needed.
        # Double-labeling confuses small models into repeating structure instead of synthesizing.
        context = "\n\n".join(top_chunks) if top_chunks else "No relevant content found."
        mark("10_context_assembly")  # 11_llm_generation follows in Runner.run()

        messages = [
            {"role": "system", "content": (
                "Answer the question using only the context below. "
                "Be concise and write in your own words — do not copy the source text. "
                "After each fact, cite its source as [Article | Section]."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]

        entity_store.save()

        # Build article snippets from sections cache (first paragraph of each article)
        article_snippets: dict = {}
        for title, secs in sections_by_title.items():
            if secs:
                paras = secs[0].get("paragraphs") or []
                if paras:
                    article_snippets[title] = paras[0][:500]

        # Fire on_sources callback for callers that want to expose retrieved articles
        if on_sources:
            on_sources([
                {"title": r["title"], "url": r["url"],
                 "snippet": article_snippets.get(r["title"], "")}
                for r in kiwix_results
            ])

        return {
            "cache_hit": False,
            "cached_answer": None,
            "intent": mode,
            "query": query,
            "context": context,
            "messages": messages,
            "query_vec": query_vec.tolist(),  # list[float] — JSON-serialisable, sent to client
            "entity_title_candidates": expanded_entities,
            "rewritten": rewritten,
            "kiwix_results": kiwix_results,
            "article_chunks": article_chunks,
            "top_chunks": top_chunks,
            "llm_relations": llm_relations,   # [] when server extracted, populated for client mode
            "stages": stages,
            "t0": t0,
        }
