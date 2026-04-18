# IIAB WebUI RAG Module

Offline retrieval-augmented generation for Internet in a Box using indexed Kiwix
ZIM content, a local embedding model, FAISS, SQLite/FTS5, and an Ollama LLM.

The active runtime is a single FastAPI process in `src/main.py`. It serves the
built Svelte WebUI from `webui/` and implements the OpenAI-compatible
`/v1/chat/completions` endpoint directly. The previous standalone SearchEngine
HTTP app and graph-query pipeline have been removed or archived.

## Current Pipeline

1. The WebUI sends chat requests to `/v1/chat/completions`.
2. `main.py` embeds the user query with `SearchEngine/embedding.py`.
3. `SearchEngine/zim_retrieval.py` searches indexed ZIM chunks using hybrid
   FAISS semantic search plus SQLite/FTS5 BM25 title and paragraph search.
4. Candidates are fused with reciprocal rank fusion, then reranked with a
   lightweight heuristic reranker using semantic score, BM25 score, title and
   section overlap, query coverage, structure priors, and section-aware aspect
   boosting.
5. The selected chunks are compacted into the LLM context and streamed through
   Ollama.
6. Source cards and LLM metrics are emitted through the same SSE stream used by
   the WebUI.

## Active Source Layout

```text
src/main.py                         FastAPI WebUI + RAG server
src/SearchEngine/config.py          runtime config loader
src/SearchEngine/config.yaml        active RAG/ZIM/model config
src/SearchEngine/embedding.py       fastembed/ONNX embedding wrapper
src/SearchEngine/zim_retrieval.py   hybrid retrieval, fusion, reranking
src/SearchEngine/cache.py           semantic answer cache
src/SearchEngine/keywords.py        BM25 keyword extraction
src/SearchEngine/metrics/           LLM metrics and dashboard assets
src/SearchEngine/modes/             fast/balanced/complex mode configs
src/SearchEngine/prompts/           active ai_mode prompts
src/zim_indexer/                    ZIM extraction, SQLite/FTS5, FAISS indexer
webui-src/                          Svelte WebUI source
webui/                              built WebUI served by FastAPI
scripts/                            launch, nginx, and build helpers
src/archive/                        archived legacy experiments/reference code
```

## Running

```bash
bash scripts/start.sh
```

By default this starts Uvicorn on `0.0.0.0:5050`. Override with environment
variables if needed:

```bash
HOST=127.0.0.1 PORT=5050 CONDA_ENV=sync4 bash scripts/start.sh
```

## WebUI Build

After editing `webui-src/`, rebuild the static WebUI:

```bash
bash scripts/build-webui.sh
```

## Nginx

For IIAB-style deployment, nginx can proxy to the FastAPI process:

```bash
bash scripts/nginx_conf.sh
```

The current nginx config proxies to `http://127.0.0.1:5050/` and disables
buffering so token streaming works. Before production IIAB integration, review
`scripts/aiiab.conf`; `location /` can shadow other IIAB services if this app is
not meant to own the root route.

## Runtime Data

Runtime databases, metrics logs, generated indexes, and run output are ignored
by git. ZIM indexes are expected under the configured `zim_index_base`, usually:

```text
/library/zims/content/<zim-name>/data.db
/library/zims/content/<zim-name>/faiss.index
```
