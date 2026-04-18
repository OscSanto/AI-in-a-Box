"""
Regex-based keyword extraction — no LLM cost, runs in <1ms.

Strips question framing to isolate the topic for BM25 queries:
  1. Question-pattern regex (who/what/where…, tell me about…) — returns topic verbatim
  2. Raw query fallback
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
