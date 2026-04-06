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

import spacy

try:
    _nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy en_core_web_sm loaded", flush=True)
except OSError:
    _nlp = None
    print("⚠️  spaCy model not found — run: python -m spacy download en_core_web_sm", flush=True)

_keyword_cache: dict[str, str] = {}

# Generic nouns that appear in question templates but aren't the topic.
_FILLER_NOUNS = {
    "fact", "facts", "thing", "things", "info", "information",
    "example", "examples", "detail", "details", "aspect", "aspects",
    "question", "questions", "way", "ways", "type", "types", "kind", "kinds",
}

# Question-pattern shortcut — highest priority, strips question boilerplate and
# returns the topic verbatim, preserving numbers.
# "who is elizabeth 2" → "elizabeth 2"
# "tell me about world war 2" → "world war 2"
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

    keywords = q
    if _nlp is not None:
        doc = _nlp(q)

        # 1. Named entities — append immediately-following number so "Elizabeth 2" stays intact
        raw_ents = [e for e in doc.ents if e.label_ in {
            "GPE", "LOC", "ORG", "PERSON", "NORP", "FAC",
            "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
        }]
        if raw_ents:
            parts = []
            for e in raw_ents[:3]:
                text = e.text
                if e.end < len(doc) and doc[e.end].pos_ == "NUM" and doc[e.end].is_digit:
                    text += " " + doc[e.end].text
                parts.append(text)
            keywords = " ".join(parts)
        else:
            # 2. Prepositional objects — catches "boxing" in "facts on boxing",
            #    "Egypt" in "tell me about Egypt"
            pobjs = [
                t.text for t in doc
                if t.dep_ == "pobj"
                and t.pos_ in {"NOUN", "PROPN"}
                and t.lemma_.lower() not in _FILLER_NOUNS
            ]
            if pobjs:
                keywords = " ".join(pobjs[:2])
            else:
                # 3. Noun chunks — skip pronouns and pure filler phrases
                chunks = [
                    c.text for c in doc.noun_chunks
                    if len(c.text.split()) <= 4
                    and not all(t.pos_ == "PRON" for t in c)
                    and not all(t.lemma_.lower() in _FILLER_NOUNS for t in c if t.pos_ == "NOUN")
                ]
                if chunks:
                    keywords = " ".join(chunks[:2])

    _keyword_cache[q] = keywords
    return keywords
