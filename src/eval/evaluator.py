import json
import os
import time

"""
Run a query batch evaluatoin

    - Loads queries from .jsonl file and runs through each.
    - Records latency, estimated cache hit rate
    - Saves results to output_dir

Arg
    Pipeline_fn: callable function that takes a query string and reutnrs Iterator[str]
    - A function passed as param.
    Config: Default none
    queriers_path: path to querires.jsonl
    output_dir: dir to write results.jsonl

Output: resutls.jsonl
    - one line per query with {query, answer, latency, cache_hit}
"""


# Todo: Improve cache hit registration.

# Notes: This measures every query saved


def run_evaluation(pipeline_fn, queries_path: str, output_dir: str, config=None):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "results.jsonl")

    with open(queries_path, "r") as func:
        # read file line by line.
        # Parse the JSON string into python dict. Result is list of dict {"query" :"When was Chernobyl?"}
        queries = [json.loads(line) for line in func if line.strip()]

    results = []
    for item in queries:
        query = item.get("query", "")

        #Records latency of streamed chunks
        t1 = time.time()
        chunks = list(pipeline_fn(query))
        latency = time.time() - t1

        answer = "".join(chunks) #joins chunks. The complete stream.

        # If response returns under 0.5 & has content, probably a cache hit!
        # This is an estimation of course, since real retrieval takes many seconds (kiwix search + embd + LLM)
        # Else, false.
        # Todo: Future improvement to explicity say whether it was a cache hit or not
        cache_hit = latency < 0.5 and len(answer) > 0  # heuristic: very fast = cache

        results.append({
            "query": query,
            "answer": answer,
            "latency_s": round(latency, 3),
            "cache_hit": cache_hit,
        })

    #Checks for results to write. None: return
    total = len(results)
    
    if total == 0:
        print("No results.")
        return

    with open(output_path, "w") as func:
        for result in results:
            func.write(json.dumps(result) + "\n")

    avg_latency = sum(r["latency_s"] for r in results) / total # adds latency values across every query. Finds the mean.
    cache_hit_rate = sum(1 for r in results if r["cache_hit"]) / total # counts how many results had cache_hit. Finds the percentage

    print(f"Evaluated {total} queries")
    print(f"Avg latency: {avg_latency:.3f}s")
    print(f"Cache hit rate: {cache_hit_rate:.1%}")
    print(f"Results saved to {output_path}")
