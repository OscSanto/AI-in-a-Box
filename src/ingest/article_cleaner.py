import re

# Section headers that signal junk — stop keeping content after these (used by kiwix_client fetch)
JUNK = {
    "see also", "references", "further reading", "external links",
    "notes", "bibliography", "footnotes", "citations", "sources",
    "contents", "navigation menu", "retrieved from",
}

# Junk sections for chunk scoring — chunks from these sections get a heavy penalty (used by rerank)
JUNK_SECTIONS = {
    "see also", "references", "further reading", "external links",
    "notes", "footnotes", "bibliography", "trivia", "citations",
}

# Line-level noise patterns to drop entirely
NOISE = [
    re.compile(r"^\s*\[\d+\]\s*$"),               # bare citation markers [1]
    re.compile(r"^\s*\d+\s*$"),                    # lone numbers (TOC page nums)
    re.compile(r"^\s*\^.*$"),                      # footnote anchors ^ Smith 2001
    re.compile(r"^\s*edit\s*$", re.I),             # bare "edit" links
    re.compile(r"^\s*v\s*[•·]\s*t\s*[•·]\s*e"),   # navbox "v · t · e"
    re.compile(r"^\s*Retrieved\s+\d{1,2}\s+\w+"),  # "Retrieved 12 January 2020"
    re.compile(r"^\s*This (article|page|section)"), # Wikipedia meta-text
    re.compile(r"^\s*Wikimedia\s", re.I),           # Wikimedia notices
    re.compile(r"^\s*Category\s*:", re.I),          # Category: tags
    re.compile(r"^\s*Coordinates\s*:", re.I),        # Coordinates: 51°N...
]
# Sentence boundary splitter — splits on ". ", "! ", "? " followed by a capital letter
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def last_sentence(text: str) -> str:
    """Last sentence of a paragraph (used for left overlap in chunking)."""
    parts = _SENT_SPLIT.split(text.strip())
    return parts[-1].strip() if parts else ""


def first_sentence(text: str) -> str:
    """First sentence of a paragraph (used for right overlap in chunking)."""
    parts = _SENT_SPLIT.split(text.strip())
    return parts[0].strip() if parts else ""


_JUNK_TITLE_PREFIXES = (
    "list of", "outline of", "index of", "glossary of",
    "portal:", "wikipedia:", "template:", "category:", "help:",
)
_JUNK_TITLE_SUFFIXES = (
    "(disambiguation)",
)

def is_junk_title(title: str) -> bool:
    """True if the article title is a navigation/meta page rather than real content."""
    t = title.lower().strip()
    if any(t.startswith(p) for p in _JUNK_TITLE_PREFIXES):
        return True
    if any(t.endswith(s) for s in _JUNK_TITLE_SUFFIXES):
        return True
    return False


def merge_paragraphs(raw_paras: list, max_chars: int) -> list:
    """Merge consecutive short paragraphs up to max_chars so chunks have enough text for BM25."""
    chunks = []
    buf = ""
    for p in raw_paras:
        if not p:
            continue
        if buf and len(buf) + 1 + len(p) > max_chars:
            chunks.append(buf)
            buf = p
        else:
            buf = (buf + " " + p).strip() if buf else p
    if buf:
        chunks.append(buf)
    return chunks
