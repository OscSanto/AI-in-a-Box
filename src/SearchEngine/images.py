"""
Image scraping — fetch images from article pages, filter out icons and math.
"""
import requests
from lxml import html as lxml_html
from urllib.parse import urljoin


def scrape_images(rank: int, url: str, title: str) -> list:
    try:
        response = requests.get(url, timeout=8)
        tree     = lxml_html.fromstring(response.content)
        images   = []
        for img in tree.xpath("//img"):
            src    = img.get("src", "")
            alt    = img.get("alt", title)
            width  = img.get("width", "")
            height = img.get("height", "")
            try:
                if int(width) < 50:
                    continue
            except Exception:
                pass
            try:
                if int(height) < 50:
                    continue
            except Exception:
                pass
            if not src or src.endswith(".svg"):
                continue
            if "math" in src.lower() or "formula" in src.lower():
                continue
            if src.startswith("//"):
                src = "http:" + src
            elif not src.startswith("http"):
                src = urljoin(url, src)
            images.append({"src": src, "alt": alt or title, "title": title, "source": url, "rank": rank})
        return images
    except Exception:
        return []
