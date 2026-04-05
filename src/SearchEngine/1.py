import requests
from lxml import html
from urllib.parse import urljoin
import ollama
import faiss
import numpy as np
import time
import sqlite3
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── SQLite Cache Setup ───────────────────────────────────────────────────────

DB_PATH   = "cache.db"
CACHE_TTL = 60 * 60 * 24 * 7  # 7 days in seconds

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS search_cache (
            query       TEXT PRIMARY KEY,
            results     TEXT NOT NULL,
            timestamp   REAL NOT NULL
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_query ON search_cache(query)")
    con.commit()
    con.close()

def db_get(q: str):
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT results, timestamp FROM search_cache WHERE query = ?", (q,)
    ).fetchone()
    con.close()
    if row:
        results, ts = row
        if (time.time() - ts) < CACHE_TTL:
            return json.loads(results)
    return None

def db_set(q: str, ranked: list):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT OR REPLACE INTO search_cache (query, results, timestamp) VALUES (?, ?, ?)",
        (q, json.dumps(ranked), time.time())
    )
    con.commit()
    con.close()

def db_suggest(q: str, limit: int = 8):
    """Return up to `limit` cached queries that start with q (case-insensitive)."""
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT query FROM search_cache WHERE query LIKE ? ORDER BY timestamp DESC LIMIT ?",
        (q + "%", limit)
    ).fetchall()
    con.close()
    return [r[0] for r in rows]

# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("✅ SQLite cache initialised")
    yield
    print("🛑 Server shutting down")

app = FastAPI(lifespan=lifespan)

BASE_URL = "http://box"

ZIMS = [
    {"name": "wikipedia_en_all_maxi_2025-08", "count": 20, "has_images": True},
    {"name": "wikipedia_en_100_2026-01",      "count": 5,  "has_images": True},
    {"name": "devdocs_en_python_2026-02",     "count": 5,  "has_images": False},
    {"name": "devdocs_en_c_2026-01",          "count": 5,  "has_images": False},
    {"name": "devdocs_en_git_2026-01",        "count": 5,  "has_images": False},
    {"name": "devdocs_en_openjdk_2026-02",    "count": 5,  "has_images": False},
    {"name": "wikipedia_en_astronomy_maxi_2025-11",    "count": 20,  "has_images": True}

]

# ─── Core pipeline ────────────────────────────────────────────────────────────

def scrape_kiwix(q, zim_name, count, has_images):
    url = f"{BASE_URL}/kiwix/search?pattern={q}&books.name={zim_name}&start=0&pageLength={count}"
    print(url)
    try:
        response = requests.get(url, timeout=8)
        tree = html.fromstring(response.content)
        results = []
        for li in tree.xpath('//div[@class="results"]//li'):
            try:
                title     = li.xpath('.//a/text()')[0].strip()
                href      = li.xpath('.//a/@href')[0]
                snippet   = ' '.join(li.xpath('.//cite//text()')).strip()
                wordcount = li.xpath('.//div[@class="informations"]/text()')[0].strip()
                results.append({
                    'title':      title,
                    'url':        BASE_URL + href,
                    'snippet':    snippet,
                    'wordcount':  wordcount,
                    'source_zim': zim_name,
                    'has_images': has_images
                })
            except:
                continue
        return results
    except Exception:
        return []

def scrape_all_zims(q):
    all_results = []
    with ThreadPoolExecutor(max_workers=len(ZIMS)) as executor:
        futures = {
            executor.submit(scrape_kiwix, q, z["name"], z["count"], z["has_images"]): z
            for z in ZIMS
        }
        for future in as_completed(futures):
            all_results.extend(future.result())
    return all_results

def embed(text):
    return np.array(ollama.embeddings(model="mxbai-embed-large", prompt=text)["embedding"], dtype=np.float32)

def rank_results(query, results):
    if not results:
        return []
    with ThreadPoolExecutor(max_workers=8) as executor:
        query_future  = executor.submit(embed, query)
        title_futures = [executor.submit(embed, r['title']) for r in results]
        query_embedding  = query_future.result()
        title_embeddings = [f.result() for f in title_futures]

    matrix = np.stack(title_embeddings)
    faiss.normalize_L2(matrix)
    faiss.normalize_L2(query_embedding.reshape(1, -1))
    index = faiss.IndexFlatIP(len(query_embedding))
    index.add(matrix)
    k = min(len(results), 25)
    scores, indices = index.search(query_embedding.reshape(1, -1), k)

    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        print(f"[{rank+1}] {score:.4f} — {results[idx]['title']} ({results[idx]['source_zim']})")

    return [results[i] for i in indices[0]]

def full_pipeline(q: str):
    results = scrape_all_zims(q)
    ranked  = rank_results(q, results)[:25]
    return ranked

def scrape_images_from_url(rank, url, title):
    try:
        response = requests.get(url, timeout=8)
        tree = html.fromstring(response.content)
        images = []
        for img in tree.xpath('//img'):
            src    = img.get('src', '')
            alt    = img.get('alt', title)
            width  = img.get('width',  '')
            height = img.get('height', '')
            try:
                if int(width)  < 50: continue
            except: pass
            try:
                if int(height) < 50: continue
            except: pass
            if not src or src.endswith('.svg'):
                continue
            if 'math' in src.lower() or 'formula' in src.lower():
                continue
            if src.startswith('//'):
                src = 'http:' + src
            elif not src.startswith('http'):
                src = urljoin(url, src)
            images.append({
                'src':    src,
                'alt':    alt or title,
                'title':  title,
                'source': url,
                'rank':   rank
            })
        return images
    except Exception:
        return []

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/suggest")
async def suggest(q: str):
    """Autocomplete from cached queries. Pure SQLite prefix match — very fast."""
    if not q or len(q) < 2:
        return {"suggestions": []}
    suggestions = db_suggest(q.lower())
    return {"suggestions": suggestions}

@app.get("/search")
async def search(q: str, background_tasks: BackgroundTasks):
    cached = db_get(q)
    if cached:
        print(f"✅ Cache hit: {q}")
        return {'query': q, 'keywords': q, 'results': cached, 'cached': True}
    print(f"🔍 Cache miss: {q}")
    ranked = full_pipeline(q)
    if ranked:
        background_tasks.add_task(db_set, q, ranked)
    return {'query': q, 'keywords': q, 'results': ranked, 'cached': False}

@app.get("/images")
async def images(q: str, background_tasks: BackgroundTasks):
    cached = db_get(q)
    if cached:
        ranked = cached
    else:
        ranked = full_pipeline(q)
        if ranked:
            background_tasks.add_task(db_set, q, ranked)

    image_candidates = [r for r in ranked if r.get('has_images')]
    all_images = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(scrape_images_from_url, rank, r['url'], r['title']): rank
            for rank, r in enumerate(image_candidates)
        }
        for future in as_completed(futures):
            all_images.extend(future.result())

    all_images.sort(key=lambda x: x['rank'])
    return {'query': q, 'images': all_images}

@app.get("/ai")
async def ai_answer(q: str):
    def generate():
        stream = ollama.chat(
            model="qwen2.5:0.5b",
            messages=[
                {"role": "system", "content": "answer the question in spanish."},
                {"role": "user",   "content": q}
            ],
            stream=True
        )
        for chunk in stream:
            yield chunk['message']['content']
    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.get("/results")
async def results():
    return FileResponse("static/results.html")

@app.get("/images-page")
async def images_page():
    return FileResponse("static/images.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
