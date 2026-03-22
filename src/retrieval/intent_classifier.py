"""
This file takes the user query and rewrites it for retrieval, extracting named entities and classifying intent if needed.
Stage 2 — Intent classifier / query rewriter.

classify_intent(query, llm, config, mode) → IntentResult

  mode="kiwix"  → focused rewrite prompt, no mode-routing overhead
  mode=other    → full classification prompt (chat / summarize / wiki_url routing)

Falls back to the raw query on any LLM failure.
"""
import json
import re
from dataclasses import dataclass



@dataclass
class IntentResult:
    mode:      str        # "kiwix" | "chat" | "summarize" | "wiki_url"
    style:     str        # "concise" | "long_form" | "bullet" | "eli5" | "step_by_step"
    queries:   list[str]  # Wikipedia-style search terms (kiwix only)
    rewritten: str        # single best Wikipedia title  (kiwix only)
    entities:  list[str]  # named entities for entity search



def _kiwix_prompt(query: str, max_queries: int, max_entities: int, intents: list) -> str:
    mode_lines = "\n".join(f'    "{i["name"]}" — {i["description"]}' for i in intents)
    mode_names = "|".join(i["name"] for i in intents)
    return f"""Classify the message and rewrite it into Wikipedia search terms.

Return exactly one JSON object:
{{
  "mode": "<{mode_names}>",
  "style": "<concise|long_form|bullet|eli5|step_by_step>",
  "queries": ["term1", "term2"],
  "rewritten": "<primary Wikipedia title>",
  "entities": ["Entity Name"]
}}

- "mode": what kind of request this is
{mode_lines}

- "style": how the user wants the answer delivered
    "concise"      — short factual answer (default for most questions)
    "long_form"    — essay, multiple paragraphs, or pages of detail requested
    "bullet"       — user asked for a list, bullet points, or key points
    "eli5"         — explain simply, like I'm 5, beginner-friendly
    "step_by_step" — how-to, instructions, or process explanation

- "queries": up to {max_queries} short Wikipedia-style search terms (kiwix mode only)
- "rewritten": the single best Wikipedia article title (kiwix mode only)
- "entities": up to {max_entities} exact article titles named in the message

Formatting: noun phrases, 1-3 words, no question words, no verbs, preserve full names.

Example:
User: tell me about new york
{{"mode": "kiwix", "style": "concise", "queries": ["New York"], "rewritten": "New York", "entities": ["New York City", "New York (state)"]}}

Message: {query.strip()}
JSON:"""


#def _routing_prompt(query: str, intents: list, max_queries: int, max_entities: int) -> str:
#    """Full classification prompt — used when intent routing is enabled."""
#    enabled    = [i for i in intents if i.get("enabled", True)]
#    names      = ", ".join(i["name"] for i in enabled)
#    mode_lines = "\n".join(f'- "{i["name"]}": {i["description"]}' for i in enabled)
#
#    return f"""Classify the message and extract retrieval terms.
#
#Return exactly one JSON object:
#{{
#  "mode": "<one of: {names}>",
#  "confidence": <0.0-1.0>,
#  "queries": ["term1", "term2"],
#  "rewritten": "<primary title>",
#  "entities": ["Entity Name"]
#}}
#
#Modes:
#{mode_lines}
#
#For mode="kiwix":
#- "queries": up to {max_queries} short Wikipedia-style search terms
#- "rewritten": the single best Wikipedia-style title
#- "entities": up to {max_entities} exact Wikipedia article titles from the message
#
#Formatting: noun phrases, 1-3 words, no question words, no verbs, preserve full names.
#
#Example:
#User: tell me about new york
#{{"mode": "kiwix", "confidence": 0.90, "queries": ["New York"], "rewritten": "New York", "entities": ["New York City", "New York (state)"]}}
#
#Message: {query.strip()}
#JSON:"""



def _fallback(query: str) -> IntentResult:
    """Used on any LLM failure — treats the raw query as a kiwix lookup."""
    return IntentResult(
        mode="kiwix",
        style="concise",
        queries=[query.strip()],
        rewritten=query.strip(),
        entities=[],
    )


# ===== Entry point 

def classify_intent(query: str, llm, config, mode: str = "kiwix") -> IntentResult:
    """
    Single LLM call — rewrites the query and extracts named entities.

    mode="kiwix"  → _kiwix_prompt    (focused, no mode routing)
    mode=other    → _routing_prompt  (classifies mode, routes chat/summarize/wiki_url)

    To add a new mode:
      1. Add an entry to intent_classifier.intents in the YAML config.
      2. Add a routing branch in runner.py.
    """
    if not config.intent_classifier_enabled:
        return _fallback(query)

    # Only kiwix mode is supported for now. routing to other modes is not yet implemented (see runner.py).
    if mode == "kiwix":
        prompt = _kiwix_prompt(query, config.kiwix_max_queries, config.kiwix_max_entities,
                               config.intent_classifier_intents)

    else: 
        return _fallback(query) # For now, if not kiwix mode, skip intent classification and route directly to fallback (kiwix retrieval with raw query
    #else:
    #    prompt = _routing_prompt(query, config.intent_classifier_intents,
    #                             config.kiwix_max_queries, config.kiwix_max_entities)

    try:
        raw = llm.generate(
            config.llm_utility_model,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )

        raw = re.sub(r"```json|```", "", raw).strip()
        m   = re.search(r"\{.*\}", raw, re.DOTALL) # we expect exactly one JSON object in the response; if we can't find it, treat as failure and fall back to raw query
        if not m:
            raise ValueError("no JSON object in response")
        data = json.loads(m.group())
        _VALID_MODES  = {i["name"] for i in config.intent_classifier_intents}
        _VALID_STYLES = {"concise", "long_form", "bullet", "eli5", "step_by_step"}

        detected_mode = data.get("mode", "kiwix")
        if detected_mode not in _VALID_MODES:
            detected_mode = "kiwix"

        result_style = data.get("style", "concise")
        if result_style not in _VALID_STYLES:
            result_style = "concise"

        # Routing not yet wired — always execute as kiwix regardless of detected mode.
        # detected_mode is logged so we can observe LLM classifications before enabling routing.
        result_mode = "kiwix"

        print(f"[intent] detected=({detected_mode}, {result_style})  routing→{result_mode}", flush=True)
        print(f"[intent] {json.dumps(data)}", flush=True)

        # ===== Queries 
        raw_terms = data.get("queries") or []
        queries = [
            re.sub(r"^\s*(?:[-•*]|\d+\.)\s*", "", t).strip()[:80]
            for t in raw_terms
            if isinstance(t, str) and t.strip()
        ][:config.kiwix_max_queries]

        if result_mode == "kiwix" and not queries:
            queries = [query.strip()]

        # ===== Rewritten 
        rewritten = (data.get("rewritten") or "").strip()
        if not rewritten:
            rewritten = queries[0] if queries else query.strip()
        words = rewritten.split()
        if len(words) > 6:
            rewritten = " ".join(words[:6])

        # ====== Entities 
        entities = [
            e.strip() for e in (data.get("entities") or [])
            if isinstance(e, str) and e.strip()
        ][:config.kiwix_max_entities]

        # ===== Hallucination check 
        # If result shares no content tokens with the original query the model
        # hallucinated — fall back to the raw query.
        _STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "for",
                 "is", "was", "are", "who", "what", "how", "tell", "about", "me"}

        def _tok(text: str) -> set:
            return set(re.sub(r"[^\w\s]", " ", text.lower()).split()) - _STOP

        query_content  = _tok(query)
        result_content = set()
        for t in [rewritten] + queries + entities:
            result_content |= _tok(t)

        if query_content and not (query_content & result_content):
            print(f"[intent] hallucination — no overlap, falling back", flush=True)
            return _fallback(query)

        # If rewritten has no overlap with the query, reset it to the first query term.
        # Catches cases like query="tell me about peter parker" → rewritten="New York"
        # where the LLM correctly includes the entity in queries but picks a related
        # concept (Peter Parker lives in NYC) as the "best Wikipedia title".
        if query_content and rewritten and not (query_content & _tok(rewritten)):
            print(f"[intent] rewritten {rewritten!r} has no query overlap — resetting to queries[0]", flush=True)
            rewritten = queries[0] if queries else query.strip()

        return IntentResult(
            mode=result_mode,
            style=result_style,
            queries=queries,
            rewritten=rewritten,
            entities=entities,
        )

    except Exception as e:
        print(f"[intent] LLM call failed: {e}", flush=True)
        return _fallback(query)
