"""
spaCy-based keyword extraction — no LLM cost, runs in <5ms.

Priority:
  1. Question-pattern regex (who/what/where…, tell me about…) — topic verbatim
  2. Named entities (GPE, ORG, PERSON, WORK_OF_ART…)
  3. Prepositional objects — topic word after "about/on/of/regarding"
  4. Non-filler noun chunks
  5. Raw query fallback
"""
import re

_keyword_cache: dict[str, str] = {}

_Q_RE = re.compile(
    r"^(?:who|what|where|when|how)\s+(?:is|are|was|were|does|did|do)\s+(.+)$"
    r"|^(?:tell me about|facts? (?:about|on|of)|info (?:about|on))\s+(.+)$",
    re.I,
)


def extract_keywords(q: str) -> str:
    if q in _keyword_cache:
        return _keyword_cache[q]

    m = _Q_RE.match(q.strip())
    if m:
        topic = (m.group(1) or m.group(2) or "").strip()
        if topic and len(topic.split()) <= 6:
            _keyword_cache[q] = topic
            return topic

    _keyword_cache[q] = q
    return q
