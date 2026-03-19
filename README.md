# IIAB RAG Module

A retrieval-augmented generation (RAG) system for [Internet in a Box (IIAB)](https://internet-in-a-box.org/). Answers questions offline using Kiwix ZIM files (Wikipedia, etc.) and a local Ollama LLM — no internet connection required after setup.

---

## How It Works

User queries pass through an 11-stage pipeline:

| Stage | Name | Description |
|-------|------|-------------|
| 1 | **Query Preprocess** | Normalize whitespace, lowercase the raw query |
| 2 | **Entity Detection** | spaCy NER + noun chunks extract named entities (people, places, orgs). Results become focused Kiwix search terms alongside the full query. |
| 3 | **Query Rewrite** | Fast LLM (`llm_utility_model`) rewrites the query into 3 varied Wikipedia search terms to improve recall. Model is evicted from RAM immediately after. |
| 4 | **Query Embedding** | Embed the rewritten query via Ollama (`all-minilm:22m`, 384-dim) |
| 5 | **Query Memory Lookup** | FAISS cosine search over past queries. If similarity ≥ `high_similarity` threshold, return cached answer immediately (skips stages 6–11) |
| 6 | **Kiwix Search** | Full-text search against the configured ZIM file via the Kiwix HTTP API. Falls back to direct HEAD-request probing on Android IIAB where the search API is unavailable. |
| 6b | **Article Reranking** | BM25 on article titles + entity overlap boost → keep top `max_articles_to_chunk` articles |
| 7 | **Chunk Collection** | Fetch each article's HTML, parse section headings (h2/h3/h4), merge paragraphs up to `max_chunk_chars`. Each chunk is prefixed `[Article \| Section]` for context. Paragraph overlap is appended from adjacent chunks. |
| 7b | **Graph Build** | Incrementally ingest fetched articles (title, entities, chunks) into the SQLite content graph |
| 8 | **Graph Traversal** | BFS from article nodes up to `max_hops` to collect related chunks. Also queries FAISS for semantically similar chunks already in the graph. |
| 8.5 | **Relation Extraction** | Fast LLM extracts entity→relation→entity triples from top chunks and stores them in the content graph (vocabulary: `participated_in`, `part_of`, `created_by`, `caused_by`, `located_in`, `related_to`) |
| 9 | **Chunk Ranking** | Three-stage hybrid: BM25 prefilter (top 20 candidates) → cosine rerank using query embedding → section-aware bias (section name is embedded and compared to query; junk sections like References/See Also are penalized). Returns top `topk_chunks`. |
| 10 | **Context Assembly** | Concatenate top chunks into numbered `[Source N]` blocks with a factual system prompt |
| 11 | **LLM Generation** | Stream answer from `llm_model` (Ollama). Answer is saved to query memory for future cache hits. |

---

## Installation

### 1. System Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally (`ollama serve`)
- [IIAB](https://internet-in-a-box.org/) with Kiwix serving ZIM files at `http://127.0.0.1/kiwix/`
- Nginx (optional, for reverse-proxy access)

### 2. Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 4. Ollama Models

Pull the required models:

```bash
# Embedding model (384-dim)
ollama pull all-minilm:22m

# Generation model (~15 tok/s)
ollama pull granite3.1-moe:1b-instruct-fp16

# Utility model — query rewrite + relation extraction (~80 tok/s)
ollama pull qwen2.5:0.5b
```

Verify with `ollama list`.

### 5. Nginx (optional)

To expose the module through IIAB's nginx:

```bash
bash scripts/nginx_conf.sh
```

This copies `scripts/aiiab.conf` to `/etc/nginx/conf.d/` and reloads nginx.

---

## Running

```bash
python src/main.py --config configs/exp_01_mini_query_memory_graph.yaml
```

The server starts on `0.0.0.0:5050` by default (configurable in YAML). Visit `http://<server-ip>:5050/exp01/`.

---

## Configuration

All pipeline parameters are set in a YAML config file. The default config is at [configs/exp_01_mini_query_memory_graph.yaml](configs/exp_01_mini_query_memory_graph.yaml).

Key settings:

```yaml
dataset:
  zim_content_id: "wikipedia_en_all_maxi_2025-08"   # ZIM file to search

kiwix:
  endpoint: "http://127.0.0.1/kiwix/search"
  result_limit: 15                # Articles returned by Kiwix before reranking

query_embedding:
  model: "all-minilm:22m"
  dimension: 384
  keep_alive: "20m"               # Keep embedding model in Ollama RAM

chunking:
  max_articles_to_chunk: 5        # Articles fetched and chunked per query
  max_chunks_per_article: 10      # Hard cap per article
  max_chunk_chars: 1200           # ~200 words per chunk
  overlap_tokens: 50              # Context overlap between adjacent chunks

chunk_ranking:
  topk_chunks: 7                  # Final chunks passed to LLM

llm:
  model: "granite3.1-moe:1b-instruct-fp16"   # Generation model
  utility_model: "qwen2.5:0.5b"              # Fast model for rewrite + relations
  temperature: 0.2
  max_new_tokens: 512

query_memory_graph:
  enabled: true
  similarity_search:
    thresholds:
      high_similarity: 0.97       # Cache hit threshold

inference:
  default_mode: "server"          # "server" (Ollama) or "client" (WebLLM in browser)
  allow_user_toggle: true
```

---

## Client-Side Inference (WebGPU)

The UI supports an optional **client mode** where LLM inference runs entirely in the browser via WebLLM (WebGPU). The server still handles RAG (search, chunking, ranking) and sends the assembled context to the browser, which runs generation locally.

To enable:
1. Run `bash scripts/download_webllm.sh` once while online to download the WebLLM bundle and a model.
2. Set `inference.default_mode: "client"` in the config, or let users toggle in the UI.

Client mode requires a WebGPU-capable browser (Chrome 113+ or Edge 113+).

---

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/nginx_conf.sh` | Copies `aiiab.conf` to `/etc/nginx/conf.d/` and reloads nginx |
| `scripts/build-webui.sh` | Rebuilds the compiled web UI from `webui-src/`. Run after editing UI source. Requires Node.js + npm. |
| `scripts/download_webllm.sh` | Downloads WebLLM JS bundle and an MLC model for client-side inference. Run once with internet access. Usage: `bash scripts/download_webllm.sh [MODEL_ID]` |

---

## Data Storage

Per-experiment data is stored under `data/<experiment_name>/`:

- `content_graph.db` — SQLite: article, entity, chunk nodes + relation edges
- `content_graph.faiss` — FAISS index for semantic chunk search
- `query_memory.db` — SQLite: past query embeddings + cached answers
- `query_memory.faiss` — FAISS index for query similarity lookup

Run metrics are appended to `runs/exp01/metrics.jsonl`.

To reset the graph and memory (force full rebuild on next query):

```bash
rm data/exp_minimal_kiwix_memory_graph/*.db
rm data/exp_minimal_kiwix_memory_graph/*.faiss
```

---

## Source Layout

```
src/
  main.py              — FastAPI server, API endpoints
  pipeline.py          — 11-stage RAG pipeline (stages 1–11)
  models/
    embedder.py        — Ollama embedding wrapper (single + batch)
    llm_client.py      — Ollama generation wrapper (streaming + non-streaming + model unload)
  retrieval/
    kiwix_client.py    — Kiwix search + Android fallback + section-aware HTML chunker
    chunk_ranker.py    — BM25 → cosine → section-embedding hybrid ranker
    rewriter.py        — LLM query rewrite (3 search term variants)
    entity_detector.py — spaCy NER + noun chunk extraction
    article_cleaner.py — Strips boilerplate sections and citation artifacts
  graphs/
    content_graph.py   — SQLite/FAISS content graph (nodes, edges, relations)
    query_memory.py    — FAISS query cache with ZIM-tagged answers
  utils/
    config.py          — Loads YAML config into typed Config dataclass
configs/
  exp_01_mini_query_memory_graph.yaml   — Default experiment config
scripts/
  nginx_conf.sh        — Install nginx config
  build-webui.sh       — Rebuild compiled web UI
  download_webllm.sh   — Download WebLLM assets for client-side inference
webui/                 — Compiled web UI (served by FastAPI)
webui-src/             — SvelteKit source for the web UI
```

---

## Credits

WebLLM client-side inference uses [@mlc-ai/web-llm](https://github.com/mlc-ai/web-llm).
Web UI includes components from [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT License, Copyright © 2023–2026 ggml authors).
