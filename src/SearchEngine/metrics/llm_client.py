"""
Ollama response utilities.

_extract_timings: parses timing/token fields from any Ollama response object
    (streaming final chunk or non-streaming). Called after each LLM generation.
"""


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
        "load_s":          round(load_ns     / 1e9, 3),
        "prefill_s":       round(prefill_s,           3),
        "gen_s":           round(gen_s,               3),
        "total_s":         round(total_ns    / 1e9,   3),
        "prompt_tokens":   prompt_tokens,
        "gen_tokens":      gen_tokens,
        "total_tokens":    prompt_tokens + gen_tokens,
        "prefill_tok_s":   round(prompt_tokens / prefill_s, 1) if prefill_s > 0 else None,
        "gen_tok_s":       round(gen_tokens    / gen_s,     1) if gen_s      > 0 else None,
        "ttft_s":          round(prefill_s, 3),
        "was_cold":        load_ns > 500_000_000,
        "done_reason":     response.done_reason,
        "hit_token_limit": response.done_reason == "length",
    }
