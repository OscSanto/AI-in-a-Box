"""
HTML extraction from ZIM article content.

Wikipedia ZIM HTML structure:
  <h1> = article title (always present, not a section boundary)
  <div class="mw-parser-output"> = all article content
  <h2>/<h3> = actual section headings

Returns:
  {
    "lead":     str,                                       # joined lead paragraphs
    "infobox":  {"header": str, "rows": [{"label": str, "value": str}]} | None,
    "sections": [{"title": str, "paragraphs": [str]}, ...],  # paragraph-level splits
  }
"""
import re
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_JUNK_SECTIONS = {
    "references", "external links", "see also", "notes", "further reading",
    "bibliography", "footnotes", "citations", "sources",
}

_CITE_RE = re.compile(r"\[\w{0,8}\]")


def _clean_text(text: str) -> str:
    text = _CITE_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_infobox(soup) -> dict | None:
    """
    Extract the first infobox table as a header + list of label/value rows.
    Called BEFORE tables are removed from the DOM.
    """
    infobox = None
    for table in soup.find_all("table"):
        classes = " ".join(table.get("class") or [])
        if "infobox" in classes:
            infobox = table
            break

    if not infobox:
        return None

    header = ""
    rows   = []

    for tr in infobox.find_all("tr"):
        th = tr.find("th")
        td = tr.find("td")

        # Header row — th with no td sibling, or infobox-title class
        if th and not td:
            text = _clean_text(th.get_text(" "))
            if text and len(text) < 100:
                header = text
            continue

        # Data row — th label + td value
        if th and td:
            label = _clean_text(th.get_text(" "))
            value = _clean_text(td.get_text(" "))
            # Skip empty, overly long values (images, nested tables), or pure numbers with no label
            if label and value and len(label) < 80 and 2 < len(value) < 250:
                rows.append({"label": label, "value": value})

    if not rows:
        return None

    return {"header": header or "Facts", "rows": rows}


def extract(html: str) -> dict | None:
    """
    Parse article HTML. Returns None if the article has no usable content.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise that doesn't affect infobox or main content
    for tag in soup(["script", "style", "sup"]):
        tag.decompose()

    # ── Extract infobox BEFORE removing tables ────────────────────────────────
    infobox_data = _extract_infobox(soup)

    # Now remove tables and junk classes
    for tag in soup(["table"]):
        tag.decompose()

    _JUNK_CLASSES = {"navbox", "navbox-styles", "infobox", "reflist",
                     "mw-editsection", "toc", "sistersitebox",
                     "vertical-navbox", "hatnote", "authority-control"}
    for el in soup.find_all(True):
        classes = set(el.attrs.get("class") or []) if el.attrs else set()
        if classes & _JUNK_CLASSES:
            el.decompose()

    # Scope to mw-parser-output — all article content lives here.
    content = soup.find(class_="mw-parser-output") or soup.find("body") or soup

    # ── Lead: all <p> before any h2/h3 ───────────────────────────────────────
    lead_parts: list[str] = []
    for el in content.children:
        if getattr(el, "name", None) in ("h2", "h3"):
            break
        if getattr(el, "name", None) == "p":
            text = _clean_text(el.get_text(" "))
            if len(text) >= 50:
                lead_parts.append(text)
    lead = " ".join(lead_parts)

    if not lead:
        return None

    # ── Sections: each <p> is its own paragraph entry ────────────────────────
    sections:       list[dict]  = []
    current_title:  str | None  = None
    current_paras:  list[str]   = []

    def _flush():
        if current_title is None or not current_paras:
            return
        if current_title.lower() not in _JUNK_SECTIONS:
            sections.append({"title": current_title, "paragraphs": current_paras})

    for el in content.find_all(["h2", "h3", "p"], recursive=True):
        if el.name in ("h2", "h3"):
            _flush()
            current_title = _clean_text(el.get_text(" "))
            current_paras = []
        elif el.name == "p":
            text = _clean_text(el.get_text(" "))
            if len(text) >= 30:
                current_paras.append(text)

    _flush()

    return {"lead": lead, "infobox": infobox_data, "sections": sections}
