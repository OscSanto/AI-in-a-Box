import re

# Section headers that signal junk — stop keeping content after these
JUNK = {
    "see also", "references", "further reading", "external links",
    "notes", "bibliography", "footnotes", "citations", "sources",
    "contents", "navigation menu", "retrieved from",
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


def clean_article(text: str) -> str:
    lines = text.splitlines()  # split article into individual lines
    cleaned = []
    skip_rest = False

    for line in lines:
        stripped = line.strip()

    
        # Handle §§ section markers injected by kiwix_client
        if stripped.startswith("§§ "):
            section_name = stripped[3:].lower().rstrip(":").strip()
            if section_name in JUNK:
                skip_rest = True  # drop everything from this point on
            else:
                cleaned.append(stripped)  # keep useful section markers
            continue

        # Check plain-text junk section headers — skip everything after it
        header = stripped.lower().replace("[edit]", "").rstrip(":").strip()
        if header in JUNK:
            skip_rest = True

        if skip_rest:
            continue

        # Preserve blank lines for paragraph structure
        if not stripped:
            cleaned.append("")
            continue

        # Drop pure noise lines
        if any(p.search(stripped) for p in NOISE):
            continue

        # Remove inline junk from otherwise good lines
        stripped = re.sub(r"\[\d+\]", "", stripped)           # citation refs [1][2]
        stripped = re.sub(r"\[citation needed\]", "", stripped, flags=re.I)
        stripped = re.sub(r"\[edit\]", "", stripped, flags=re.I)
        stripped = stripped.strip()

        if stripped:
            cleaned.append(stripped)

    # Collapse 3+ blank lines into a single blank line
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))
    return result.strip()

def split_paragraphs(text: str, min_chars: int = 50) -> list:
    paragraphs = []
    for p in text.split("\n\n"):
        p = p.strip()
        if len(p) > min_chars:
            paragraphs.append(p)
    return paragraphs