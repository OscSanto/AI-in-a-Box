"""
entity_store.py — Persistent entity knowledge base

Learns from every pipeline run:
  - aliases:          "mlk" → "Martin Luther King Jr."
  - disambiguations:  "mlk" → {mlk song, mlk jr. station, martin luther king jr.}

  - subtopics:        "Boston" → {"History of Boston", "Culture of Boston"}
  - parents:          "History of Boston" → "Boston"

Stored as JSON at config.data_dir/entity_linking_store.json.
Gets better with use. Second query for "MLK" expands entities to include
"Martin Luther King Jr." automatically via alias_candidates()
Double check the metadata, tags on wiki pages

"""
import json
import os
import re
def _normalize(text: str) -> str:
    text = (text or "").strip().lower()
    return re.sub(r"\s+", " ", text)

class DynamicEntityStore:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.path = os.path.join(self.base_dir, "entity_linking_store.json")

        self.data: dict = {
            "aliases": {},          # alias_norm → {entity: weight}
            "reverse_aliases": {},  # entity → {alias_norm: weight}
            "disambiguations": {},  # term_norm → {candidate: weight}
            "subtopics": {},        # parent → {subtopic: weight}
            "parents": {},          # subtopic → {parent: weight}
        }
        self._dirty = False
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for k in self.data:
                    if isinstance(raw.get(k), dict):
                        self.data[k] = raw[k]
        except Exception as e:
            print(f"[entity_store] load failed (non-fatal): {e}", flush=True)

    def save(self):
        if not self._dirty:
            return
        try:
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2, sort_keys=True)
            self._dirty = False
        except Exception as e:
            print(f"[entity_store] save failed (non-fatal): {e}", flush=True)

    def _bump(self, table: str, key: str, value: str, amount: float = 1.0):
        key = key.strip()
        value = value.strip()
        if not key or not value or key == value:
            return
        bucket = self.data[table].setdefault(key, {})
        bucket[value] = round(float(bucket.get(value, 0.0)) + amount, 3)
        self._dirty = True

    def add_alias(self, alias: str, entity: str, weight: float = 1.0):
        alias_n = _normalize(alias)
        if not alias_n or not entity:
            return
        self._bump("aliases", alias_n, entity, weight)
        self._bump("reverse_aliases", entity, alias_n, weight)

    def add_disambiguation(self, term: str, candidate: str, weight: float = 1.0):
        term_n = _normalize(term)
        if not term_n or not candidate:
            return
        self._bump("disambiguations", term_n, candidate, weight)
        self.add_alias(term_n, candidate, weight * 0.5)

    def add_subtopic(self, parent: str, subtopic: str, weight: float = 1.0):
        if not parent or not subtopic:
            return
        self._bump("subtopics", parent, subtopic, weight)
        self._bump("parents", subtopic, parent, weight)

    def alias_candidates(self, term: str, topk: int = 5) -> list[str]:
        """Return the top-K entity candidates for a term, ranked by accumulated weight."""
        term_n = _normalize(term)
        out: list[str] = []
        seen: set[str] = set()
        for table in ("aliases", "disambiguations"):
            bucket = self.data[table].get(term_n, {})
            for cand, _ in sorted(bucket.items(), key=lambda kv: kv[1], reverse=True):
                if cand not in seen:
                    out.append(cand)
                    seen.add(cand)
                    if len(out) >= topk:
                        return out
        return out

    def subtopic_candidates(self, parent: str, topk: int = 5) -> list[str]:
        """Return known subtopic article titles for a parent topic."""
        bucket = self.data["subtopics"].get(parent.strip(), {})
        return [c for c, _ in sorted(bucket.items(), key=lambda kv: kv[1], reverse=True)[:topk]]


# ── Helpers used by pipeline to populate the store ────────────────────────

_SUBTOPIC_PREFIXES = (
    "history of ", "economy of ", "government of ", "culture of ",
    "geography of ", "demographics of ", "environment of ", "law of ",
    "education in ", "education of ", "transport in ", "transport of ",
    "politics of ", "climate of ", "music of ", "sport in ", "sports in ",
)

# WIkipedia has many "junk" sections that are rarely relevant to user queries.
# We can detect these by name and apply a negative multiplier to their relevance score.
_SUBTOPIC_LABELS = [
    "history", "economy", "government", "culture", "geography", "demographics",
    "environment", "law", "education", "transport", "politics", "climate",
    "music", "sports", "early life", "career", "legacy",
]


def detect_subtopic_label(query: str) -> str | None:
    """Return the subtopic keyword if the query asks about one (e.g. 'history of Boston' → 'history')."""
    q = _normalize(query)
    for label in _SUBTOPIC_LABELS:
        if label in q:
            return label
    return None


def extract_parent_from_subtopic(title: str) -> tuple[str | None, str | None]:
    """'History of Boston' → ('Boston', 'history').  Returns (None, None) if not a subtopic title."""
    t = title.strip()
    low = _normalize(t)
    for prefix in _SUBTOPIC_PREFIXES:
        if low.startswith(prefix):
            parent = t[len(prefix):].strip()
            label = prefix.strip().replace(" of", "").replace(" in", "")
            return parent, label
    return None, None


def looks_like_disambiguation(title: str, intro_text: str = "") -> bool:
    """True if the article is a Wikipedia disambiguation page."""
    title_n = _normalize(title)
    intro_n = _normalize(intro_text[:220])
    if title_n.endswith("(disambiguation)"):
        return True
    return any(p in intro_n for p in ("may refer to", "can refer to", "most commonly refers to"))


def extract_aliases_from_intro(title: str, intro_text: str) -> list[str]:
    """Extract aliases and abbreviations mentioned in the opening sentence of an article."""
    text = intro_text[:250]
    aliases: list[str] = []
    patterns = [
        r"\b(?:also known as|known as)\s+([^,.;]+)",
        r"\b(?:often abbreviated as|abbreviated as)\s+([^,.;]+)",
        r"\(([^()]{2,20})\)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            alias = m.group(1).strip(" \"'")
            if alias and len(alias) <= 40 and alias.lower() != title.lower():
                aliases.append(alias)
    # Acronym from title words (e.g. "Martin Luther King" → "MLK")
    words = [w for w in re.findall(r"[A-Z][a-z]+|[A-Z]+", title) if w]
    if len(words) >= 2:
        acronym = "".join(w[0] for w in words if w[0].isalpha())
        if 2 <= len(acronym) <= 8:
            aliases.append(acronym)
    return list(dict.fromkeys(aliases))


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def expand_entity_list(store: "DynamicEntityStore", query: str, rewritten: str,
                       entities: list, topk: int = 3) -> list:
    """Augment classifier entities with aliases learned from past queries."""
    expanded = list(entities)
    for term in [query, rewritten] + list(entities):
        for cand in store.alias_candidates(term, topk=topk):
            if cand not in expanded:
                expanded.append(cand)
                print(f"  [alias] {term!r} → {cand!r}", flush=True)
    return expanded


def learn_from_articles(store: "DynamicEntityStore", fetched: list) -> dict:
    """
    Process (article_dict, sections) pairs from a parallel fetch.

    For each article:
      - Marks disambiguation pages (_is_disambiguation=True on the dict)
      - Learns disambiguation candidates, subtopic links, and aliases into store

    Returns sections_by_title dict for reuse in chunking (avoids re-fetching).
    """
    sections_by_title: dict = {}
    for r, secs in fetched:
        sections_by_title[r["title"]] = secs
        intro = " ".join((secs[0].get("paragraphs") or [])[:2]) if secs else ""

        if looks_like_disambiguation(r["title"], intro):
            r["_is_disambiguation"] = True
            surface = re.sub(r"\s*\(disambiguation\)$", "", r["title"], flags=re.IGNORECASE).strip()
            for sec in secs[:1]:
                for para in sec.get("paragraphs", [])[:3]:
                    for m in re.finditer(r"\b([A-Z][\w''.:‑-]+(?:\s+[A-Z][\w''.:‑-]+){0,5})\b", para):
                        cand = m.group(1).strip()
                        if cand.lower() not in {surface.lower(), r["title"].lower()} and len(cand) > 2:
                            store.add_disambiguation(surface or r["title"], cand, 1.0)

        parent, _ = extract_parent_from_subtopic(r["title"])
        if parent:
            store.add_subtopic(parent, r["title"], 1.0)

        for alias in extract_aliases_from_intro(r["title"], intro):
            store.add_alias(alias, r["title"], 1.0)

    return sections_by_title
