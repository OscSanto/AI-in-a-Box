from typing import Iterator
import ollama
import numpy as np
"""

SSE may have lightweight overhead.
-Plain text over TCP
-Tokens sent in chunks, not all at once.

With stream = True, we keep 1 http connection open per client.
-ollama sends tokens to client as generated, not one by one but in chunks
-connection open while generating

TODO: store_client_answer needs to be refactored!
"""

#Unload ollama's from GPU/RAM immediately when Keep_alive = 0
def unload(model: str):
    try:
        ollama.generate(model=model, prompt="", keep_alive=0)
        print(f"[llm] unloaded {model}", flush=True)
    except Exception as e:
        print(f"[llm] unload {model} failed (non-fatal): {e}", flush=True)


def _extract_timings(response, model: str) -> dict:
    """
    Extract and derive all timing metrics from an Ollama response object.
    Works on both streaming final chunks and non-streaming responses.
    All durations in seconds; speeds in tokens/sec.
    """
    load_ns          = response.load_duration        or 0
    prompt_eval_ns   = response.prompt_eval_duration or 0
    eval_ns          = response.eval_duration        or 0
    total_ns         = response.total_duration       or 0
    prompt_tokens    = response.prompt_eval_count    or 0
    gen_tokens       = response.eval_count           or 0

    prefill_s  = prompt_eval_ns / 1e9
    gen_s      = eval_ns        / 1e9

    return {
        # raw Ollama fields (seconds)
        "load_s":          round(load_ns     / 1e9, 3),
        "prefill_s":       round(prefill_s,           3),
        "gen_s":           round(gen_s,               3),
        "total_s":         round(total_ns    / 1e9,   3),
        # token counts
        "prompt_tokens":   prompt_tokens,
        "gen_tokens":      gen_tokens,
        "total_tokens":    prompt_tokens + gen_tokens,
        # derived speeds
        "prefill_tok_s":   round(prompt_tokens / prefill_s, 1) if prefill_s > 0 else None,
        "gen_tok_s":       round(gen_tokens    / gen_s,     1) if gen_s      > 0 else None,
        # derived flags
        "ttft_s":          round(prefill_s, 3),
        "was_cold":        load_ns > 500_000_000,  # >0.5s load = model was not in memory
        "done_reason":     response.done_reason,
        "hit_token_limit": response.done_reason == "length",
    }


def _log_timings(role: str, model: str, t: dict) -> None:
    cold = " [COLD START]" if t["was_cold"] else ""
    limit = " [HIT TOKEN LIMIT]" if t["hit_token_limit"] else ""
    print(
        f"[llm:{role}] {model} | "
        f"load={t['load_s']}s  prefill={t['prefill_s']}s ({t['prefill_tok_s']} tok/s, {t['prompt_tokens']} tok)  "
        f"gen={t['gen_s']}s ({t['gen_tok_s']} tok/s, {t['gen_tokens']} tok)  "
        f"total={t['total_s']}s{cold}{limit}",
        flush=True,
    )


# non-streamer function (Wait for full response)
# Returns (content, timings_dict)
def generate(model: str, messages: list, temperature: float, max_tokens: int = 256) -> tuple[str, dict]:
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature, "num_predict": max_tokens},
        stream=False,
    )
    timings = _extract_timings(response, model)
    _log_timings("utility", model, timings)
    return response.message.content, timings


# yields tokens as produced; populates timings_out list with one dict on completion
def generate_stream(model: str, messages: list, temperature: float, max_tokens: int = 256,
                    timings_out: list | None = None) -> Iterator[str]:
    try:
        stream = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
            stream=True,
        )
    except Exception as e:
        print(f"[llm] generate_stream failed to start ({model}): {e}", flush=True)
        return
    try:
        for chunk in stream:
            text = chunk.message.content
            if text:
                yield text
            if chunk.done and timings_out is not None:
                timings = _extract_timings(chunk, model)
                _log_timings("answer", model, timings)
                timings_out.append(timings)
    except Exception as e:
        print(f"[llm] generate_stream interrupted ({model}): {e}", flush=True)


def store_client_answer(query: str, answer: str, query_vec_list: list, config, query_memory):
    """
    Stores a client-generated (WebGPU) answer into query memory.
    Called by POST /remember so future cache lookups can hit this answer.
    """
    query_vec = np.array(query_vec_list, dtype="float32")
    query_memory.store(
        raw_query=query,
        normalized_query=query.lower().strip(),
        query_vec=query_vec,
        entities=[],
        answer=answer,
        article_titles=[],
        chunk_ids=[],
        llm_model="client-webgpu",
    )
    print(f"[pipeline] stored client answer for: {query!r}", flush=True)
