import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, quote
from bs4 import BeautifulSoup

from ingest.article_cleaner import JUNK, merge_paragraphs, is_junk_title


"""
Recall we are using Kiwix search engine to retrieve aritcles based on Zim file, entity, aritlce-limit
It is their search algo that determines what our aritcles will be

TODO: A retreival (based on kiwix engine) improved via metadata, title, etc.
TODO: Zim compilation seach engine (Multi zim search traversal)
    -At the moment, we are doing singuler Zim searching

TODO: Multi zim search order/priority/weight by query? 
TODO: def prompt_zim_selection(kiwix_endpoint: str, current_zim_id: str) -> dict | None:


"""


def probe_kiwix_endpoint(configured: str, zim_id: str) -> str:
    """Try configured endpoint first, then known fallbacks. Returns first that responds."""
    candidates = [configured]
    if ":8085" not in configured:
        candidates.append(configured.replace("127.0.0.1", "127.0.0.1:8085"))
    else:
        candidates.append(configured.replace(":8085", ""))

    for endpoint in candidates:
        try:
            url = f"{endpoint}?pattern=test&books.name={zim_id}&pageLength=1"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print(f"[Kiwix] endpoint probe OK: {endpoint}", flush=True)
                return endpoint
        except Exception:
            pass
    print(f"[Kiwix] all endpoints failed — using configured: {configured}", flush=True)
    return configured


def _get(url: str, timeout: int):
    # HTTP fetch
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response
    except Exception as e:
        print(f"[Kiwix] Kiwix search services are unreachable @ {url}: {e}")
        return None


def get_zim_metadata(kiwix_endpoint: str, zim_id: str) -> dict:
    fallback = {"zim_id": zim_id, "zim_label": zim_id}

    try:
        parsed = urlparse(kiwix_endpoint)

        # Example:
        # http://127.0.0.1:8085/kiwix/search
        # http://127.0.0.1:8085/kiwix
        path_parts = parsed.path.rstrip("/").rsplit("/", 1)
        kiwix_root = f"{parsed.scheme}://{parsed.netloc}{path_parts[0]}"
        url = f"{kiwix_root}/catalog/v2/entries?name={zim_id}"

        resp = requests.get(url, timeout=5)
        if resp.status_code == 404:
            print(f"[Kiwix] catalog API not available on this IIAB server: {url}")
            return fallback

        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, "lxml-xml")
        entry = soup.find("entry")
        if not entry:
            return fallback

        title_tag = entry.find("title")
        label = title_tag.get_text(strip=True) if title_tag else zim_id
        return {"zim_id": zim_id, "zim_label": label}

    except Exception as e:
        print(f"[Kiwix] metadata fetch failed: {e}")
        return fallback

# TODO: Kiwix search URL on Android IIAB conflict with standard IIAB

def search_kiwix(endpoint: str, zim_content_id: str, query: str, limit: int) -> list:
    """
    Search Kiwix for articles matching query.

    Tries the standard search API first.  If it is unreachable (None response from
    _get), falls back to direct title-slug HEAD requests — used on Android IIAB where
    the search API is unavailable.  URL patterns tried for the slug fallback:
      - /kiwix/content/{zim}/{slug}          standard IIAB (nginx proxy)
      - /kiwix/content/{zim}/A/{slug}        ZIM article namespace (Android IIAB)
      - /{zim}/A/{slug}                      direct kiwix-serve (no nginx prefix)
    """
    parsed   = urlparse(endpoint)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    # ===== Standard Kiwix search API 
    url = f"{endpoint}?pattern={query.replace(' ', '+')}&books.name={zim_content_id}&start=0&pageLength={limit}"
    response = _get(url, timeout=10)
    if response is not None:
        soup = BeautifulSoup(response.content, "lxml")
        results = []
        for a in soup.find_all("a"):
            title = a.get_text(strip=True)
            href  = a.get("href", "")
            if not title or not href:
                continue
            full_url = base_url + href if href.startswith("/") else href
            results.append({"title": title, "path": href, "url": full_url})
        return results

    # =====  fallback: search unreachable (e.g. Android IIAB) 
    print(f"[Kiwix] search API unreachable — trying slug fallback for '{query}'", flush=True)
    results = []
    seen = set()
    for term in [query.strip(), query.strip().title()]:
        slug = quote(term.replace(" ", "_"), safe="()/_:-")
        if slug in seen:
            continue
        for full_url, path in [
            (f"{base_url}/kiwix/content/{zim_content_id}/{slug}",   f"/kiwix/content/{zim_content_id}/{slug}"),
            (f"{base_url}/kiwix/content/{zim_content_id}/A/{slug}", f"/kiwix/content/{zim_content_id}/A/{slug}"),
            (f"{base_url}/{zim_content_id}/A/{slug}",               f"/{zim_content_id}/A/{slug}"),
        ]:
            try:
                r = requests.head(full_url, timeout=5, allow_redirects=True)
                if r.status_code == 200:
                    seen.add(slug)
                    results.append({"title": term, "path": path, "url": full_url})
                    print(f"[Kiwix] fallback found: {term} @ {full_url}", flush=True)
                    break
            except Exception:
                pass
    if not results:
        print(f"[Kiwix] fallback: no match for '{query}'", flush=True)
    return results



def fetch_article_sections(url: str, max_chunk_chars: int = 800) -> list:
    """
    Returns article content as a list of sections, preserving HTML heading structure.
    Each section: {"section": str, "paragraphs": [str]}

    Paragraphs within a section are merged up to _MAX_CHUNK_CHARS so chunks have
    enough text for BM25 to work with. Skips junk sections and cleans citations.
    TODO: change this to just a section and text return only. Then do max_chunk_char truncating as a seperate step.
        
    """
    resp = _get(url, timeout=15)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.content, "lxml")
    body = soup.find(class_="mw-parser-output") or soup.find(id="mw-content-text") or soup

    #Start at Article: Introduction. Sequential walk while sections are tracked.
    sections = []
    current_section = "Introduction"
    current_paragraphs = []
    skip_section = False

    #Call on New Heading Encounter. Save paragraphs collected thus far. 
    # (h2: Paragraph), (p: Paragraph), (h3: title), 
    def _flush_section():
        if current_paragraphs and not skip_section:
            # stores copy of current_paragraphs, then clears buffer for next section
            sections.append({"section": current_section, "paragraphs": list(current_paragraphs)}) 

    #iterator: only through headings and paragraphs. Ignore lists, images, etc.
    #TODO: Consider saving images, tables for display

    for el in body.find_all(["h2", "h3", "h4", "p"]):
        tag = el.name

        if tag in ("h2", "h3", "h4"):
            _flush_section()
            heading = re.sub(r"\[edit\]", "", el.get_text(strip=True), flags=re.I).strip()
            skip_section = heading.lower() in JUNK
            current_section = heading
            current_paragraphs.clear()

        elif tag == "p" and not skip_section:
            text = el.get_text(separator=" ", strip=True)
            text = re.sub(r"\[\s*\d+\s*\]", "", text)
            text = re.sub(r"\[citation needed\]", "", text, flags=re.I)
            text = re.sub(r" {2,}", " ", text).strip()
            #if len(text) >= 30:
            #    current_paragraphs.append(text)
            current_paragraphs.append(text)
    _flush_section()

    # Merge short paragraphs within each section into larger chunks
    for sec in sections:
        sec["paragraphs"] = merge_paragraphs(sec["paragraphs"], max_chunk_chars)

    return sections




def parallel_search(endpoint: str, zim_id: str, entity_list: list, all_terms: list,
                    result_limit: int) -> tuple:
    """
    Run entity and broad keyword searches in parallel against Kiwix.

    Entity searches use a tight limit (3) for exact-title lookups.
    Broad searches use result_limit for topical coverage.

    Returns (results, seen_titles):
      results      — deduplicated, junk-filtered list; entities first, then broad terms
      seen_titles  — set of added titles (needed by disambiguation retry in pipeline)
    """
    seen_titles: set = set()
    results: list = []

    def _add(r: dict):
        title = r["title"].strip()
        if (title not in seen_titles
                and len(title) > 2
                and not title.isdigit()
                and any(c.isalpha() for c in title)
                and not is_junk_title(title)):
            results.append(r)
            seen_titles.add(title)

    def _search(term, limit):
        return term, list(search_kiwix(endpoint, zim_id, term, limit))

    entity_results: dict = {}
    broad_results: dict = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        entity_futures = {ex.submit(_search, e, 3): ("entity", e) for e in entity_list}
        broad_futures  = {ex.submit(_search, t, result_limit): ("broad", t) for t in all_terms}
        for fut in as_completed({**entity_futures, **broad_futures}):
            kind = entity_futures.get(fut) or broad_futures.get(fut)
            term, res = fut.result()
            if kind[0] == "entity":
                entity_results[term] = res
            else:
                broad_results[term] = res

    for entity in entity_list:
        before = len(results)
        for r in entity_results.get(entity, []):
            _add(r)
        print(f"  [entity] {entity!r} → {[r['title'] for r in results[before:]]}", flush=True)
    for term in all_terms:
        before = len(results)
        for r in broad_results.get(term, []):
            _add(r)
        print(f"  [search] {term!r} → {[r['title'] for r in results[before:]]}", flush=True)

    return results, seen_titles
