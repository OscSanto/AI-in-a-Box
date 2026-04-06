"""
Kiwix HTTP scraping — query all configured ZIM files in parallel.
"""
import requests
from lxml import html as lxml_html
from concurrent.futures import ThreadPoolExecutor, as_completed

from SearchEngine.config import BASE_URL, ZIMS


def scrape_zim(q: str, zim_name: str, count: int, has_images: bool) -> list:
    url = f"{BASE_URL}/kiwix/search?pattern={q}&books.name={zim_name}&start=0&pageLength={count}"
    try:
        response = requests.get(url, timeout=8)
        tree     = lxml_html.fromstring(response.content)
        results  = []
        for li in tree.xpath('//div[@class="results"]//li'):
            try:
                title     = li.xpath('.//a/text()')[0].strip()
                href      = li.xpath('.//a/@href')[0]
                snippet   = " ".join(li.xpath('.//cite//text()')).strip()
                wordcount = li.xpath('.//div[@class="informations"]/text()')[0].strip()
                results.append({
                    "title":      title,
                    "url":        BASE_URL + href,
                    "snippet":    snippet,
                    "wordcount":  wordcount,
                    "source_zim": zim_name,
                    "has_images": has_images,
                })
            except Exception:
                continue
        return results
    except Exception:
        return []


def scrape_all_zims(q: str) -> tuple[list, dict]:
    """Scrape all ZIMs in parallel. Returns (all_results, {zim_name: hit_count})."""
    all_results = []
    zim_counts  = {}
    with ThreadPoolExecutor(max_workers=len(ZIMS)) as ex:
        futures = {
            ex.submit(scrape_zim, q, z["name"], z["count"], z["has_images"]): z
            for z in ZIMS
        }
        for future in as_completed(futures):
            zim     = futures[future]
            results = future.result()
            zim_counts[zim["name"]] = len(results)
            all_results.extend(results)
    return all_results, zim_counts
