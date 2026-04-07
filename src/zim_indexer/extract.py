"""
HTML extraction from ZIM article content.

Wikipedia ZIM HTML structure:
  <h1> = article title (always present, not a section boundary)
  <div class="mw-parser-output"> = all article content
  <h2>/<h3> = actual section headings

Returns:
  {
    "lead":     str,
    "sections": [{"title": str, "text": str}, ...],  # ALL sections (for Phase 2)
  }
"""
import re
from bs4 import BeautifulSoup

_JUNK_SECTIONS = {
    "references", "external links", "see also", "notes", "further reading",
    "bibliography", "footnotes", "citations", "sources",
}

_CITE_RE = re.compile(r"\[\w{0,8}\]")


def _clean_text(text: str) -> str:
    text = _CITE_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def extract(html: str) -> dict | None:
    """
    Parse article HTML. Returns None if the article has no usable content.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "table", "sup"]):
        tag.decompose()
    # Use exact class-name matching to avoid "toc" hitting "vector-feature-toc-*" on <html>
    _JUNK_CLASSES = {"navbox", "navbox-styles", "infobox", "reflist",
                     "mw-editsection", "toc", "sistersitebox",
                     "vertical-navbox", "hatnote", "authority-control"}
    for el in soup.find_all(True):
        classes = set(el.attrs.get("class") or []) if el.attrs else set()
        if classes & _JUNK_CLASSES:
            el.decompose()

    # Scope to mw-parser-output — all article content lives here.
    # h1 is the page title outside this div; h2/h3 are section headings inside.
    content = soup.find(class_="mw-parser-output") or soup.find("body") or soup

    # ── Lead: first <p> before any h2/h3 ─────────────────────────────────────
    lead = ""
    for el in content.children:
        if getattr(el, "name", None) in ("h2", "h3"):
            break   # hit first section — stop
        if getattr(el, "name", None) == "p":
            text = _clean_text(el.get_text(" "))
            if len(text) >= 50:
                lead = text
                break

    if not lead:
        return None

    # ── Sections: group <p> content under h2/h3 headings ─────────────────────
    sections      = []
    current_title = None
    current_parts: list[str] = []

    def _flush():
        if current_title is None:
            return
        text = " ".join(current_parts).strip()
        if len(text) >= 50 and current_title.lower() not in _JUNK_SECTIONS:
            sections.append({"title": current_title, "text": text})

    for el in content.find_all(["h2", "h3", "p"], recursive=True):
        if el.name in ("h2", "h3"):
            _flush()
            current_title = _clean_text(el.get_text(" "))
            current_parts = []
        elif el.name == "p":
            text = _clean_text(el.get_text(" "))
            if len(text) >= 30:
                current_parts.append(text)

    _flush()

    return {"lead": lead, "sections": sections}
