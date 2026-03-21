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


