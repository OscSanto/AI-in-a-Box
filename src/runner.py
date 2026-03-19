import time
from typing import Iterator

import numpy as np

from kiwix_pipeline import Pipeline


class Runner:
    """
    Mode router and LLM generation (stages 11-12).

    Owns the Pipeline instance. Instantiate once at startup.
    Call run() per request with only query, mode, and request-scoped metadata.

    query_memory is read from pipeline.query_memory so replacing
    pipeline.query_memory (e.g. cache reset) is automatically reflected here.
    """

    def __init__(self, config, embedder, llm, content_graph, query_memory):
        self._pipeline = Pipeline(config, embedder, llm, content_graph, query_memory)
        self.config = config
        self.embedder = embedder
        self.llm = llm

    def build_context(self, query: str, zim_meta: dict = None, mode: str = "kiwix", on_sources=None) -> dict:
        """Delegate for /context endpoint (client-side inference)."""
        return self._pipeline.build_context(query, zim_meta=zim_meta, mode=mode, on_sources=on_sources)

    @property
    def query_memory(self):
        return self._pipeline.query_memory

    def reset_query_memory(self, new_query_memory) -> None:
        """Replace pipeline's query memory (used by /admin/reset-cache)."""
        self._pipeline.query_memory = new_query_memory

    # ====== Mode router ======
    def run(self, query: str, mode: str = "kiwix", zim_meta: dict = None, on_sources=None) -> Iterator[str]:
        if mode == "chat":
            yield from self._run_chat(query, zim_meta)

        elif mode == "summarize":
            yield from self._run_summarize(query, zim_meta)

        elif mode == "wiki_url":
            yield from self._run_wiki_url(query, zim_meta)

        else:
            yield from self._run_kiwix(query, zim_meta, on_sources)

    # ====== Mode handlers 
    def _run_summarize(self, query: str, zim_meta: dict) -> Iterator[str]:
        print(f"\n[pipeline] mode=summarize — stub (not implemented): {query!r}", flush=True)
        yield from ()

    def _run_wiki_url(self, query: str, zim_meta: dict) -> Iterator[str]:
        print(f"\n[pipeline] mode=wiki_url — stub (not implemented): {query!r}", flush=True)
        yield from ()

    def _run_chat(self, query: str, zim_meta: dict) -> Iterator[str]:
        t0 = time.time()
        print(f"\n[pipeline] ===== query: {query!r} =====", flush=True)
        print(f"  mode=chat — direct LLM (no retrieval)", flush=True)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": query.strip()},
        ]
        t1 = time.time()
        full_answer = ""
        for text_chunk in self.llm.generate_stream(
            self.config.llm_model, messages, self.config.llm_temperature, self.config.llm_max_tokens
        ):
            full_answer += text_chunk
            yield text_chunk
        print(f"[pipeline] 11_llm_generation | took={round(time.time()-t1,3)}s | total={round(time.time()-t0,3)}s", flush=True)
        print(f"[pipeline] --- answer ---\n{full_answer}", flush=True)

        # Embed after generation so it doesn't delay first token
        query_vec = self.embedder.embed(query.strip())
        self._store_memory(query=query, 
            normalized_query=query,
            query_vec=query_vec, 
            answer=full_answer, 
            zim_meta=zim_meta)
        
        print(f"[pipeline] done | total={round(time.time()-t0,3)}s", flush=True)

    def _run_kiwix(self, query: str, zim_meta: dict, on_sources) -> Iterator[str]:
        result = self._pipeline.build_context(query, zim_meta=zim_meta, mode="kiwix", on_sources=on_sources)

        stages = result["stages"]
        t0     = result["t0"]

        if result["cache_hit"]:
            print(f"[pipeline] cache=hit", flush=True)
            for word in result["cached_answer"].split():
                yield word + " "
            return
        print(f"[pipeline] cache=miss", flush=True)

        # Stage 11: llm_generation
        print(f"\n[pipeline] --- context ---", flush=True)
        print(result["context"], flush=True)

        print(f"[pipeline] 11_llm_generation — starting ({self.config.llm_model})", flush=True)
        t_gen = time.time()
        full_answer = ""
        for text_chunk in self.llm.generate_stream(
            self.config.llm_model, result["messages"], self.config.llm_temperature, self.config.llm_max_tokens
        ):
            full_answer += text_chunk
            yield text_chunk

        took11 = round(time.time() - t_gen, 3)
        stages["11_llm_generation"] = took11
        print(f"[pipeline] 11_llm_generation | took={took11}s | total={round(time.time()-t0,3)}s", flush=True)

        self._store_memory(
            query=query,
            normalized_query=result["rewritten"],
            query_vec=np.array(result["query_vec"], dtype="float32"),
            answer=full_answer,
            zim_meta=zim_meta,
            entities=result["entity_title_candidates"],
            article_titles=[r["title"] for r in result["kiwix_results"][:5]],
        )
        
        print(f"[pipeline] done | total={round(time.time()-t0,3)}s", flush=True)



    # ── Stage 12: query memory store ─────────────────────────────────────────

    def _store_memory(self, *, query: str, normalized_query: str, query_vec,
                      answer: str, zim_meta: dict,
                      entities: list = None, article_titles: list = None):
        """Persist query + answer to query memory (Stage 12). No-op if disabled or no answer."""
        if not self.config.query_memory_enabled or not answer:
            return
        zim_ids = [zim_meta["zim_id"]] if zim_meta and zim_meta.get("zim_id") else []
        t = time.time()
        try:
            self._pipeline.query_memory.store(
                raw_query=query,
                normalized_query=normalized_query,
                query_vec=query_vec,
                entities=entities or [],
                answer=answer,
                article_titles=article_titles or [],
                chunk_ids=[],
                llm_model=self.config.llm_model,
                zim_ids=zim_ids,
            )
        except Exception as e:
            print(f"[pipeline] 12_query_memory_store failed (non-fatal): {e}", flush=True)
            return
        print(f"[pipeline] 12_query_memory_store | took={round(time.time()-t,3)}s", flush=True)
