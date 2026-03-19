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


# non-streamer function (Wait for full response)
def generate(model: str, messages: list, temperature: float, max_tokens: int = 256) -> str:
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature, "num_predict": max_tokens},
        stream=False,
    )
    return response.message.content

# yields tokens as produced
def generate_stream(model: str, messages: list, temperature: float, max_tokens: int = 256) -> Iterator[str]:
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
    except Exception as e:
        print(f"[llm] generate_stream interrupted ({model}): {e}", flush=True)

#def webLLM_submission()


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

