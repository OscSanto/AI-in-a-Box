"""
Chat completions and RAG pipeline.

Implements POST /v1/chat/completions with two paths:
  - "chat" mode: direct LLM pass-through (no retrieval)
  - "balanced" mode: embed query → FAISS+BM25 search → compact context → LLM stream

Also exposes GET /props (llama.cpp-compatible server info for the WebUI).
"""
import json
import re
import time
import uuid
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from SearchEngine.cache import db_ai_mode_lookup, db_set_ai_mode
from SearchEngine.config import AI_MODE_CFG, AI_MODE_LLM_OPTIONS, LLAMACPP_CFG, LLM_TIMEOUT
from SearchEngine.embedding import embed
import SearchEngine.zim_retrieval as zim_retrieval
from api.models import get_active_model

router = APIRouter()

_TONE_PROMPTS: dict[str, str] = {
    "neutral":  "You are a knowledgeable reference assistant. Answer factually using only what is provided. Do not speculate beyond the sources.",
    "friendly": "You are a friendly, approachable assistant. Use plain language and avoid jargon. Write as if explaining to a curious friend.",
    "socratic": "You are a thoughtful tutor. Answer the question, then ask one short follow-up question to deepen the learner's understanding.",
}

_FORMAT_HINTS: dict[str, str] = {
    "prose":      "Answer in flowing prose — no bullet points or headers.",
    "structured": "Structure your answer with headers and bullet points where it helps clarity.",
    "direct":     "Give a one-sentence answer first, then explain further below.",
}


# ── Short fork-memory helpers ────────────────────────────────────────────────
# The WebUI persists conversation trees in browser IndexedDB and sends the
# active branch path with every request. Build short-memory directly from that
# path so it survives server restarts and always matches the visible branch.

_HISTORY_MAX_TURNS = 3


def _message_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts).strip()
    return ""


def _fork_history_pairs(messages: list, enabled: bool) -> list[dict]:
    """Build last-N user/assistant turns from the submitted conversation path."""
    if not enabled:
        return []
    turns: list[tuple[str, str]] = []
    current_query = ""
    current_answer_parts: list[str] = []

    for msg in messages:
        role = msg.get("role")
        text = _message_text(msg.get("content"))
        if role == "system":
            continue
        if role == "user":
            if current_query and current_answer_parts:
                turns.append((current_query, "\n".join(current_answer_parts).strip()))
            current_query = text
            current_answer_parts = []
            continue
        if role == "assistant" and current_query:
            if text:
                current_answer_parts.append(text)

    if current_query and current_answer_parts:
        turns.append((current_query, "\n".join(current_answer_parts).strip()))

    pairs: list[dict] = []
    for prior_query, prior_answer in turns[-_HISTORY_MAX_TURNS:]:
        if prior_query:
            pairs.append({"role": "user", "content": prior_query})
        if prior_answer:
            pairs.append({"role": "assistant", "content": prior_answer})
    return pairs


KIWIX_VIEWER_BASE = __import__("os").environ.get("KIWIX_VIEWER_BASE", "http://127.0.0.1/kiwix/viewer").rstrip("/")

_LINE_SPLIT_RE = re.compile(r"\n{2,}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _available_zim_names() -> list[str]:
    return [h.name for h in getattr(zim_retrieval, "_handles", [])]


def _latest_user_message(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                return content
            return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _sse(chat_id: str, created: int, text: str, finish: bool = False,
         model: str = "searchengine", backend: str | None = None) -> str:
    delta: dict = {"content": text}
    if finish:
        delta["model"] = model
    payload: dict = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": "stop" if finish else None}],
    }
    if backend:
        payload["backend"] = backend
    return "data: " + json.dumps(payload) + "\n\n"


def _sse_metrics(chat_id: str, created: int, timings: dict, model: str = "") -> str:
    return "data: " + json.dumps({
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        "se_metrics": timings,
    }) + "\n\n"


def _sse_thinking(chat_id: str, created: int, text: str, model: str = "") -> str:
    return "data: " + json.dumps({
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": None}],
    }) + "\n\n"


def _sse_sources(chat_id: str, created: int, sources: list[dict], model: str = "") -> str:
    return "data: " + json.dumps({
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "sources": sources,
        "choices": [{"index": 0, "delta": {"content": ""}}],
    }) + "\n\n"


def _source_cards(hits: list[dict], context_chunks: list[str]) -> list[dict]:
    """Build per-chunk provenance cards for the WebUI source panel."""
    seen: set[tuple[str, str]] = set()
    sources: list[dict] = []
    for h, chunk in zip(hits, context_chunks):
        key = (h.get("title", ""), h.get("section_title", ""))
        if key in seen:
            continue
        seen.add(key)
        zim_name = h.get("zim_name", "")
        article_path = (h.get("url") or h.get("title", "").replace(" ", "_")).lstrip("/")
        section = h.get("section_title", "")
        sources.append({
            "title":        h.get("title", ""),
            "section":      section,
            "url":          f"{KIWIX_VIEWER_BASE}#{quote(zim_name, safe='')}/{quote(article_path, safe='/%')}",
            "snippet":      (h.get("text") or "")[:240],
            "context":      chunk,
            "rerank_score": h.get("rerank_score"),
            "faiss_score":  h.get("faiss_score"),
            "zim_name":     zim_name,
        })
    return sources


def _format_infobox_facts(text: str) -> str:
    rows: list[tuple[str, str]] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line.startswith("Fact: "):
            continue
        fact = line.removeprefix("Fact: ").strip()
        if " = " not in fact:
            continue
        label, value = fact.split(" = ", 1)
        rows.append((label.strip(), value.strip()))
    if not rows:
        return text.strip()
    out = ["| Field | Value |", "| --- | --- |"]
    for label, value in rows:
        out.append(f"| {label.replace('|', '/')} | {value.replace('|', '/')} |")
    return "\n".join(out)


def _format_hit_context(hit: dict) -> str:
    """Format one retrieved chunk for LLM consumption, stripping stored prefix and adding infobox table."""
    title   = hit.get("title",         "").strip()
    section = hit.get("section_title", "").strip() or "Unknown"
    raw     = (hit.get("text") or "").strip()
    clean   = re.sub(
        r"^Article:.*?\nSection:.*?\nText:\s*", "", raw, count=1, flags=re.DOTALL,
    ).strip()
    parts = [f"Article: {title}", f"Section: {section}"]
    infobox_text = (hit.get("infobox_text") or "").strip()
    if infobox_text:
        parts.append(f"Infobox facts:\n{_format_infobox_facts(infobox_text)}")
    parts.append(f"Text: {clean}")
    return "\n".join(parts)


def _compact_chunk_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    parts = [p.strip() for p in _LINE_SPLIT_RE.split(text) if p.strip()]
    kept: list[str] = []
    total = 0
    for part in parts:
        add = len(part) + (2 if kept else 0)
        if kept and total + add > max_chars:
            break
        if not kept and len(part) > max_chars:
            return part[:max_chars].rsplit(" ", 1)[0] + " ..."
        kept.append(part)
        total += add
    compact = "\n\n".join(kept).strip()
    if len(compact) > max_chars:
        compact = compact[:max_chars].rsplit(" ", 1)[0] + " ..."
    return compact


def _compact_context(chunks: list[str], num_ctx: int) -> tuple[list[str], int]:
    if not chunks:
        return [], 0
    total_budget = min(3200, max(1400, int(num_ctx * 3.0)))
    per_chunk = max(500, total_budget // max(1, len(chunks)))
    compacted = [_compact_chunk_text(c, per_chunk) for c in chunks]
    context = "\n\n".join(compacted)
    if len(context) <= total_budget:
        return compacted, len(context)
    trimmed: list[str] = []
    used = 0
    for c in compacted:
        add = len(c) + (2 if trimmed else 0)
        if trimmed and used + add > total_budget:
            break
        if not trimmed and len(c) > total_budget:
            one = _compact_chunk_text(c, total_budget)
            return [one], len(one)
        trimmed.append(c)
        used += add
    return trimmed, used


def _print_llm_verbose(t: dict) -> None:
    """Print per-query LLM timing to terminal in ollama --verbose style."""
    def _ms(s):  return f"{s * 1000:.2f}ms" if s is not None and s < 1 else (f"{s:.3f}s" if s is not None else "—")
    def _toks(n): return f"{n} token(s)" if n is not None else "—"
    def _rate(r):  return f"{r:.2f} tokens/s" if r is not None else "—"
    cold = " (cold load)" if t.get("was_cold") else ""
    limit = " ⚠ hit token limit" if t.get("hit_token_limit") else ""
    print(
        f"\n[llm] total duration:        {_ms(t.get('total_s'))}{cold}\n"
        f"[llm] load duration:         {_ms(t.get('load_s'))}\n"
        f"[llm] prompt eval count:     {_toks(t.get('prompt_tokens'))}\n"
        f"[llm] prompt eval duration:  {_ms(t.get('prefill_s'))}\n"
        f"[llm] prompt eval rate:      {_rate(t.get('prefill_tok_s'))}\n"
        f"[llm] eval count:            {_toks(t.get('gen_tokens'))}{limit}\n"
        f"[llm] eval duration:         {_ms(t.get('gen_s'))}\n"
        f"[llm] eval rate:             {_rate(t.get('gen_tok_s'))}\n",
        flush=True,
    )


def _stream_llm(messages: list, llm_options: dict, chat_id: str,
                          created: int, model: str):
    """Sync generator — streams llama-server SSE. Yields (sse_str, None, None) per token,
    then (None, full_answer, timings_dict) once done."""
    url = LLAMACPP_CFG.get("url", "http://localhost:8080")
    payload: dict = {
        "model":          model,
        "messages":       messages,
        "stream":         True,
        "stream_options": {"include_usage": True},
        "temperature":    llm_options.get("temperature", 0.1),
        "max_tokens":     llm_options.get("num_predict", -1),
    }
    if llm_options.get("top_p"):  payload["top_p"]  = llm_options["top_p"]
    if llm_options.get("top_k"):  payload["top_k"]  = llm_options["top_k"]
    if llm_options.get("min_p"):  payload["min_p"]  = llm_options["min_p"]

    full_answer  = ""
    timings: dict = {}
    finish_reason = ""
    try:
        with httpx.Client(timeout=LLM_TIMEOUT) as c:
            with c.stream("POST", f"{url}/v1/chat/completions", json=payload) as r:
                if r.status_code == 503:
                    yield _sse(chat_id, created, "Model is still loading — please wait a moment and try again.", model=model), None, None
                    yield None, "", {}
                    return
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                    except Exception:
                        continue
                    # llama-server puts timings in the last content chunk
                    if chunk.get("timings"):
                        t = chunk["timings"]
                        prompt_ms    = t.get("prompt_ms", 0)
                        predicted_ms = t.get("predicted_ms", 0)
                        prompt_n     = t.get("prompt_n", 0)
                        predicted_n  = t.get("predicted_n", 0)
                        prefill_s    = prompt_ms    / 1000
                        gen_s        = predicted_ms / 1000
                        timings = {
                            "prefill_s":       round(prefill_s,  3),
                            "gen_s":           round(gen_s,      3),
                            "prompt_tokens":   prompt_n,
                            "gen_tokens":      predicted_n,
                            "prefill_tok_s":   round(prompt_n    / prefill_s,    1) if prefill_s    > 0 else None,
                            "gen_tok_s":       round(predicted_n / gen_s,        1) if gen_s        > 0 else None,
                            "was_cold":        False,
                            "hit_token_limit": finish_reason == "length",
                        }
                    choice = (chunk.get("choices") or [{}])[0]
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    token = choice.get("delta", {}).get("content", "")
                    if token:
                        full_answer += token
                        yield _sse(chat_id, created, token, model=model), None, None
    except Exception as e:
        yield _sse(chat_id, created, f"llama-server error: {e}", model=model), None, None
    yield None, full_answer, timings


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/props")
def props():
    """llama.cpp-compatible server info endpoint used by the WebUI to read the active model."""
    model = get_active_model()
    return {
        "role": "model",
        "model_path": model,
        "model_alias": model,
        "total_slots": 1,
        "modalities": {"vision": False, "audio": False},
        "build_info": "aiiab",
        "bos_token": "",
        "eos_token": "",
        "chat_template": "",
        "default_generation_settings": {},
        "backend": "llamacpp",
    }


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Main RAG chat endpoint. Streams SSE tokens; sources injected before first token."""
    body = await request.json()
    messages = body.get("messages", [])
    mode = body.get("mode", "balanced")
    if mode not in ("balanced", "chat"):
        mode = "balanced"
    allowed_zims = set(_available_zim_names())
    raw_active = body.get("active_zims")
    if isinstance(raw_active, list) and raw_active:
        invalid = [z for z in raw_active if z not in allowed_zims]
        if invalid:
            return JSONResponse(
                {"error": f"Unknown ZIMs: {invalid}. Available: {sorted(allowed_zims)}"},
                status_code=400,
            )
        active_zims: list[str] | None = [z for z in raw_active if z in allowed_zims]
    else:
        active_zims = None  # None = all ZIMs
    bypass_cache = bool(body.get("bypass_cache", False))
    conv_id  = str(body.get("conv_id") or "").strip()
    fork     = bool(body.get("fork", False)) and bool(conv_id)
    think = body.get("think", None)
    if body.get("reasoning_format") == "none" and think is None:
        think = False  # llama.cpp "none" → Ollama think=False; /think pill takes priority
    log_level = str(body.get("log_level", "full")).lower()
    if log_level not in ("off", "summary", "full"):
        log_level = "off"

    query = _latest_user_message(messages).strip()
    if not query:
        return Response("data: [DONE]\n\n", media_type="text/event-stream")

    _mode_cfg = zim_retrieval._MODE_CFGS.get(mode, {})
    _retrieval_cfg = _mode_cfg.get("retrieval", {})
    _current_active_model = get_active_model()
    if not _current_active_model:
        return Response("data: [DONE]\n\n", media_type="text/event-stream",
                        headers={"X-Error": "No model selected — pick one from the model selector"})
    _llm_model = _mode_cfg.get("llm_model", _current_active_model)
    _llm_options = dict(_mode_cfg.get("llm_options", AI_MODE_LLM_OPTIONS))

    _user_llm_opts = body.get("user_llm_options", {})
    if _user_llm_opts and isinstance(_user_llm_opts, dict):
        _caps = _mode_cfg.get("caps", {})
        for _k, _v in _user_llm_opts.items():
            if isinstance(_v, (int, float)):
                _cap = _caps.get(_k)
                if _cap is not None:
                    _v = min(_v, _cap)
                _llm_options[_k] = _v

    _top_k = _retrieval_cfg.get("top_k", AI_MODE_CFG.get("top_chunks", 3))
    _answer_filter = bool(_retrieval_cfg.get("answer_filter", mode == "balanced"))
    _system_messages = [m for m in messages if m.get("role") == "system"]
    _has_explicit_system = bool(_system_messages)
    _custom_system = ""
    if _has_explicit_system:
        _custom_system = _message_text(_system_messages[0].get("content"))
    _tone = str(body.get("tone") or "neutral").lower()
    _format = str(body.get("format") or "prose").lower()
    if _tone not in _TONE_PROMPTS:
        _tone = "neutral"
    if _format not in _FORMAT_HINTS:
        _format = "prose"
    _style_prompt = f"{_TONE_PROMPTS[_tone]}\n\n{_FORMAT_HINTS[_format]}"
    _system_prompt = f"{_style_prompt}\n\n{_custom_system}" if _custom_system else _style_prompt
    _mode_think = _mode_cfg.get("think", None)
    if think is None and _mode_think is not None:
        think = bool(_mode_think)

    _backend_name = "Pi Local (llama.cpp)"

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def _generate():
        t0 = time.time()
        summary_logs = log_level in ("summary", "full")
        full_logs = log_level == "full"

        if summary_logs:
            print(
                f"\n[webui-rag] ===== q={query!r} mode={mode} zim={active_zims or 'all'!r} "
                f"cache={'bypass' if bypass_cache else 'on'} log={log_level} =====",
                flush=True,
            )
            print(
                f"[webui-rag] mode_strategy | top_k={_top_k} "
                f"answer_filter={_answer_filter} "
                f"faiss_candidates={_retrieval_cfg.get('faiss_candidates', '?')} "
                f"bm25_candidates={_retrieval_cfg.get('bm25_candidates', '?')}",
                flush=True,
            )
        if full_logs:
            _zim_handles = getattr(zim_retrieval, "_handles", [])
            if not _zim_handles:
                print("[webui-rag] zims=none", flush=True)
            else:
                for _h in _zim_handles:
                    print(f"[webui-rag] zim={_h.name!r} vectors={getattr(_h.idx, 'ntotal', '?')}", flush=True)

        # ── Direct chat (no retrieval) ────────────────────────────────────────
        if mode == "chat":
            t_context = time.time()
            if summary_logs:
                print(f"[webui-rag] 1_chat_llm_start | total={t_context - t0:.3f}s | model={_llm_model}", flush=True)
            full_answer = ""
            _chat_timings: dict = {}
            _history_pairs = _fork_history_pairs(messages, fork)
            _chat_messages = [
                *_history_pairs,
                {"role": "user", "content": query},
            ]
            if _system_prompt:
                _chat_messages.insert(0, {"role": "system", "content": _system_prompt})

            for sse_chunk, answer, t in _stream_llm(_chat_messages, _llm_options, chat_id, created, _llm_model):
                if answer is not None:
                    full_answer = answer
                    if t:
                        _chat_timings = t
                else:
                    yield sse_chunk

            t_done = time.time()
            if summary_logs:
                print(
                    f"[webui-rag] 2_chat_done | took={t_done - t_context:.3f}s | "
                    f"total={t_done - t0:.3f}s | answer_chars={len(full_answer)}",
                    flush=True,
                )
            if summary_logs and _chat_timings:
                _print_llm_verbose(_chat_timings)
            if full_logs:
                print("[webui-rag] --- answer ---", flush=True)
                print(full_answer, flush=True)
                print("[webui-rag] --- end answer ---", flush=True)
            if _chat_timings:
                yield _sse_metrics(chat_id, created, _chat_timings, model=_llm_model)
            yield _sse(chat_id, created, "", finish=True, model=_llm_model, backend=_backend_name)
            yield "data: [DONE]\n\n"
            return

        # ── RAG pipeline ──────────────────────────────────────────────────────
        query_vec = embed(query)
        t_embed = time.time()
        if summary_logs:
            print(f"[webui-rag] 1_embed | took={t_embed - t0:.3f}s | vec={query_vec.shape}", flush=True)

        search_result = zim_retrieval.search(
            query, _top_k, mode, query_vec=query_vec, debug=full_logs,
            active_zims=active_zims,
        )
        if full_logs:
            zim_hits, candidate_pools = search_result
        else:
            zim_hits, candidate_pools = search_result, []
        t_search = time.time()
        if summary_logs:
            print(
                f"[webui-rag] 2_search | took={t_search - t_embed:.3f}s | "
                f"total={t_search - t0:.3f}s | {len(zim_hits)} chunks ({mode})",
                flush=True,
            )
        if full_logs:
            def _preview(text: str, n: int = 220) -> str:
                text = re.sub(r"\s+", " ", (text or "")).strip()
                return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."

            print(f"[webui-rag] candidate_pools={len(candidate_pools)}", flush=True)
            for _pool in candidate_pools:
                print(
                    f"[webui-rag] candidate_pool zim={_pool.get('zim_name', '')!r} "
                    f"faiss={_pool.get('faiss_candidates', 0)} "
                    f"paragraph_bm25={_pool.get('paragraph_bm25_candidates', 0)} "
                    f"title_bm25_articles={_pool.get('title_bm25_articles', 0)} "
                    f"unique={_pool.get('unique_candidate_chunks', 0)} "
                    f"post_filter={_pool.get('post_filter_chunks', 0)} "
                    f"rerank_pool={_pool.get('rerank_pool_chunks', 0)}",
                    flush=True,
                )
                for _i, _ch in enumerate(_pool.get("chunks", []), start=1):
                    _src = ",".join(_ch.get("candidate_sources") or [])
                    print(
                        f"  candidate[{_i:02d}] rerank={_ch.get('rerank_score', 0):.5f} "
                        f"fusion={_ch.get('fusion_score', 0):.5f} "
                        f"fusion_rank={_ch.get('fusion_rank', '?')} "
                        f"faiss={_ch.get('faiss_score', 0):.3f} "
                        f"para_bm25={_ch.get('para_bm25_score', 0):.5f} "
                        f"title_bm25={_ch.get('title_bm25_score', 0):.5f} "
                        f"sources={_src or '-'} "
                        f"title={_ch.get('title', '')!r} "
                        f"section={_ch.get('section_title', '')!r}",
                        flush=True,
                    )
                    print(f"       raw_preview={_preview(_ch.get('text', ''), 420)!r}", flush=True)

            print(f"[webui-rag] retrieved_chunks={len(zim_hits)}", flush=True)
            for _i, _h in enumerate(zim_hits, start=1):
                _src = ",".join(_h.get("candidate_sources") or [])
                print(
                    f"  [{_i:02d}] rerank={_h.get('rerank_score', 0):.5f} "
                    f"fusion={_h.get('fusion_score', 0):.5f} "
                    f"fusion_rank={_h.get('fusion_rank', '?')} "
                    f"faiss={_h.get('faiss_score', 0):.3f} "
                    f"para_bm25={_h.get('para_bm25_score', 0):.5f} "
                    f"title_bm25={_h.get('title_bm25_score', 0):.5f} "
                    f"sources={_src or '-'} "
                    f"zim={_h.get('zim_name', '')!r} "
                    f"title={_h.get('title', '')!r} "
                    f"section={_h.get('section_title', '')!r}",
                    flush=True,
                )
                print(f"       preview={_preview(_h.get('text', ''))!r}", flush=True)
                if _h.get("infobox_text"):
                    print(f"       infobox_preview={_preview(_h.get('infobox_text', ''))!r}", flush=True)

        _zim_cache_key = ",".join(sorted(active_zims)) if active_zims else "all"
        cached = None if bypass_cache else db_ai_mode_lookup(
            query_vec, mode=mode, zim_name=_zim_cache_key, verbose=summary_logs,
        )
        t_cache = time.time()
        if summary_logs:
            print(
                f"[webui-rag] 3_cache_lookup | took={t_cache - t_search:.3f}s | "
                f"total={t_cache - t0:.3f}s | {'bypassed' if bypass_cache else 'checked'}",
                flush=True,
            )
        if cached:
            yield _sse(chat_id, created, cached, model=_llm_model)
            if full_logs:
                print(f"[webui-rag] cached_answer:\n{cached}", flush=True)
            yield _sse(chat_id, created, "", finish=True, model=_llm_model, backend=_backend_name)
            yield "data: [DONE]\n\n"
            return

        if not zim_hits:
            yield _sse(chat_id, created, "No relevant content found.", model=_llm_model)
            yield _sse(chat_id, created, "", finish=True, model=_llm_model, backend=_backend_name)
            yield "data: [DONE]\n\n"
            return

        raw_chunks = [_format_hit_context(h) for h in zim_hits]
        num_ctx = int(_llm_options.get("num_ctx", 1024))
        context_chunks, context_len = _compact_context(raw_chunks, num_ctx)
        sources = _source_cards(zim_hits, context_chunks)
        if sources:
            yield _sse_sources(chat_id, created, sources, model=_llm_model)

        context = "\n\n".join(context_chunks)
        raw_context_len = len("\n\n".join(raw_chunks))
        if summary_logs:
            print(
                f"[webui-rag] 4_context | raw={raw_context_len} chars | compacted={context_len} chars | "
                f"chunks={len(context_chunks)} | sources={len(sources)}",
                flush=True,
            )
        if full_logs:
            print("[webui-rag] --- context ---", flush=True)
            print(context, flush=True)
            print("[webui-rag] --- end context ---", flush=True)
        t_context = time.time()
        if summary_logs:
            print(
                f"[webui-rag] 5_llm_start | took={t_context - t_cache:.3f}s | "
                f"total={t_context - t0:.3f}s | model={_llm_model} | mode={mode}",
                flush=True,
            )

        full_answer = ""
        _history_pairs = _fork_history_pairs(messages, fork)
        _doc_blocks = "\n".join(f"<document>\n{c}\n</document>" for c in context_chunks)
        llm_messages = [
            *_history_pairs,
            {"role": "user", "content": f"{_doc_blocks}\n\nQuestion: {query}"},
        ]
        if _system_prompt:
            llm_messages.insert(0, {"role": "system", "content": _system_prompt})
        _last_timings: dict = {}
        for sse_chunk, answer, t in _stream_llm(llm_messages, _llm_options, chat_id, created, _llm_model):
            if answer is not None:
                full_answer = answer
                if t:
                    _last_timings = t
            else:
                yield sse_chunk

        if full_answer:
            db_set_ai_mode(query, query_vec, full_answer, mode=mode, zim_name=_zim_cache_key)
        t_done = time.time()
        if summary_logs:
            print(
                f"[webui-rag] 6_done | took={t_done - t_context:.3f}s | "
                f"total={t_done - t0:.3f}s | answer_chars={len(full_answer)}",
                flush=True,
            )
        if summary_logs and _last_timings:
            _print_llm_verbose(_last_timings)
        if full_logs:
            print("[webui-rag] --- answer ---", flush=True)
            print(full_answer, flush=True)
            print("[webui-rag] --- end answer ---", flush=True)
        if _last_timings:
            yield _sse_metrics(chat_id, created, _last_timings, model=_llm_model)
        yield _sse(chat_id, created, "", finish=True, model=_llm_model, backend=_backend_name)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
