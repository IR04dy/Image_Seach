import os, io, json, sqlite3, requests
from typing import List, Optional, Dict, Any
from PIL import Image
import numpy as np
import faiss, torch, open_clip
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------- Paths (adjust if needed) ----------
ARTIFACTS_DIR = "./artifacts"
INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss.index")
META_PATH = os.path.join(ARTIFACTS_DIR, "meta.json")
SQLITE_PATH = "ebay_products.db"

# ---------- App ----------
app = FastAPI(title="Image Similarity Search API")

# ---------- Globals (loaded at startup) ----------
index = None
CAT_IDS: List[str] = []
CAT_URLS: List[str] = []
DIM = None
model = None
preprocess = None
DEVICE = "cpu"  # set to "cuda" if you have a GPU + CUDA installed
session = requests.Session()

# ---------- Models ----------
class SearchRequest(BaseModel):
    image_url: str
    topk: int = 8

class SearchResult(BaseModel):
    id: str
    url: str
    similarity: float
    db_row: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    query_image: str
    results: List[SearchResult]

# ---------- Helpers ----------
def _open_db():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}

def _load_image_any(url: str) -> Image.Image:
    r = session.get(url, timeout=15)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def _embed_one(pil_img: Image.Image) -> np.ndarray:
    x = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")

def _fetch_rows_by_image_urls(image_urls: List[str], table="products") -> Dict[str, sqlite3.Row]:
    """Exact match on products.image (fast path)."""
    if not image_urls:
        return {}
    conn = _open_db()
    try:
        placeholders = ",".join("?" for _ in image_urls)
        q = f"SELECT * FROM {table} WHERE image IN ({placeholders})"
        rows = conn.execute(q, image_urls).fetchall()
        return {str(r["image"]): r for r in rows if "image" in r.keys()}
    finally:
        conn.close()

# ---------- Startup ----------
@app.on_event("startup")
def startup_event():
    global index, CAT_IDS, CAT_URLS, DIM, model, preprocess, DEVICE

    # Load FAISS + meta
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise RuntimeError("Missing artifacts (faiss.index/meta.json).")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    CAT_IDS = meta["ids"]
    CAT_URLS = meta["urls"]
    DIM = int(meta["dim"])

    # Load CLIP
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(DEVICE).eval()

    # Sanity
    if index.d != DIM:
        raise RuntimeError(f"FAISS index dim {index.d} != meta dim {DIM}")

@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": DEVICE}

# ---------- API ----------
@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    try:
        q_img = _load_image_any(req.image_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")

    try:
        q = _embed_one(q_img)
        sims, idxs = index.search(q, req.topk)
        sims, idxs = sims[0], idxs[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    # Pull DB rows by matching FAISS image URLs -> products.image
    result_urls = [CAT_URLS[i] for i in idxs]
    image_to_row = _fetch_rows_by_image_urls(result_urls)

    results: List[SearchResult] = []
    for sim, idx in zip(sims, idxs):
        r_url = CAT_URLS[idx]
        r_id = str(CAT_IDS[idx])
        row = image_to_row.get(r_url)
        results.append(
            SearchResult(
                id=r_id,
                url=r_url,
                similarity=float(sim),
                db_row=_row_to_dict(row) if row is not None else None,
            )
        )

    return SearchResponse(query_image=req.image_url, results=results)
