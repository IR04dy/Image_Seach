# =============================================================
# Embedding.py  (RUN ONCE)
# -------------------------------------------------------------
# - You have ONLY a list of image LINKS (no product IDs)
# - This script auto‑generates stable IDs (prod_00000, prod_00001, ...)
# - Encodes with OpenCLIP ViT‑L/14
# - Saves: embeddings.npy, meta.json, faiss.index
# =============================================================

# pip install open_clip_torch faiss-cpu pillow numpy torch requests tqdm

import os, io, json, time, pathlib, hashlib
from typing import List, Tuple

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm
import torch
import faiss
import open_clip
import json

# -------------------------- CONFIG ---------------------------
BATCH_SIZE = 16                 # adjust to your CPU/RAM
TIMEOUT_SEC = 15
RETRIES = 2
SLEEP_BETWEEN_RETRIES = 1.0
ARTIFACTS_DIR = "./artifacts"  # where to save outputs
MODEL_NAME = 'ViT-L-14'        # as requested
PRETRAINED_TAG = 'openai'

# Provide your LINKS‑ONLY catalog here: a Python list of URLs
# Example stub (REPLACE with your real ~2400 links)

with open("results_all_categories_20251005_123054.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Catalog setup — can be local paths or URLs
catalog_links = [item['image'] for item in data["data"] if isinstance(item, dict) and "image" in item]



# ------------------------- HELPERS ---------------------------
session = requests.Session()

def fetch_image(url: str) -> Image.Image:
    """Download image from URL with basic retries; returns RGB PIL image."""
    last_err = None
    for _ in range(RETRIES + 1):
        try:
            r = session.get(url, timeout=TIMEOUT_SEC)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as e:
            last_err = e
            time.sleep(SLEEP_BETWEEN_RETRIES)
    raise last_err

# ----------------------- LOAD MODEL --------------------------
print("Loading model…")
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED_TAG)
model.eval()
DEVICE = "cpu"  # keep CPU; one‑time job
model = model.to(DEVICE)

def embed_batch(pil_images: List[Image.Image]) -> np.ndarray:
    batch = torch.stack([preprocess(img) for img in pil_images], dim=0).to(DEVICE)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")  # (B, d)

# -------------------- ENCODE CATALOG -------------------------
assert isinstance(catalog_links, list) and len(catalog_links) > 0, "catalog_links is empty. Fill it with your image URLs."

# Auto‑generate stable IDs (index‑based)
product_ids = [f"prod_{i:05d}" for i in range(len(catalog_links))]

# Prepare output dir
pathlib.Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)

all_embs: List[np.ndarray] = []
kept_ids: List[str] = []
kept_urls: List[str] = []
failed: List[Tuple[str, str, str]] = []  # (auto_id, url, reason)

print(f"Encoding {len(product_ids)} catalog images…")

for start in tqdm(range(0, len(product_ids), BATCH_SIZE), desc="Batches"):
    end = min(start + BATCH_SIZE, len(product_ids))
    ids_chunk = product_ids[start:end]
    urls_chunk = catalog_links[start:end]

    pil_chunk = []
    ok_mask = []
    for pid, url in zip(ids_chunk, urls_chunk):
        try:
            img = fetch_image(url)
            pil_chunk.append(img)
            ok_mask.append(True)
        except Exception as e:
            failed.append((pid, url, str(e)))
            ok_mask.append(False)

    if pil_chunk:
        embs = embed_batch(pil_chunk)
        all_embs.append(embs)
        # keep only successful ids/urls in the same order
        for pid, url, keep in zip(ids_chunk, urls_chunk, ok_mask):
            if keep:
                kept_ids.append(pid)
                kept_urls.append(url)

# Concatenate
if not all_embs:
    raise RuntimeError("No embeddings were produced. Check your links.")
embs = np.vstack(all_embs)
print(f"Embeddings shape: {embs.shape}")

# ------------------------ SAVE -------------------------------
emb_path = os.path.join(ARTIFACTS_DIR, "embeddings.npy")
meta_path = os.path.join(ARTIFACTS_DIR, "meta.json")
index_path = os.path.join(ARTIFACTS_DIR, "faiss.index")

np.save(emb_path, embs)
meta = {
    "model": MODEL_NAME,
    "pretrained": PRETRAINED_TAG,
    "count": int(embs.shape[0]),
    "dim": int(embs.shape[1]),
    "ids": kept_ids,          # auto IDs we generated
    "urls": kept_urls,        # the URLs we successfully embedded
    "failed": failed,         # URLs that failed (with reason)
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

# Build cosine‑sim FAISS index (vectors are already L2‑normalized)
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)
faiss.write_index(index, index_path)

print(f"Saved:\n - {emb_path}\n - {meta_path}\n - {index_path}")
if failed:
    print(f"Warnings: {len(failed)} images failed to download/parse. See meta.json → 'failed'.")
    