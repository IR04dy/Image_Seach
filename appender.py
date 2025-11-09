# =============================================================
# appender.py
# -------------------------------------------------------------
# - Loads existing artifacts (embeddings.npy, meta.json, faiss.index)
# - Encodes NEW image links with the SAME model (ViT‑L/14, 'openai')
# - Skips URLs already present; appends only truly new ones
# - Updates embeddings.npy, meta.json, and faiss.index in place
# =============================================================

# pip install open_clip_torch faiss-cpu pillow numpy torch requests tqdm

import os, io, json, time, pathlib
from typing import List, Tuple

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm
import torch
import faiss
import open_clip

# -------------------------- CONFIG ---------------------------
ARTIFACTS_DIR = "./artifacts"
EMB_PATH = os.path.join(ARTIFACTS_DIR, "embeddings.npy")
META_PATH = os.path.join(ARTIFACTS_DIR, "meta.json")
INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss.index")

MODEL_NAME = 'ViT-L-14'
PRETRAINED_TAG = 'openai'
DEVICE = 'cpu'

BATCH_SIZE = 16
TIMEOUT_SEC = 15
RETRIES = 2
SLEEP_BETWEEN_RETRIES = 1.0

# Provide your NEW links here (list of URLs). Duplicates vs existing will be auto-skipped.
new_links: List[str] = [
    "https://i.ebayimg.com/images/g/y-0AAeSwZjZpCtlx/s-l1600.webp",
    "https://i.ebayimg.com/images/g/yFAAAeSw6-lpCsH1/s-l1600.webp",
    "https://i.ebayimg.com/images/g/AO0AAOSwdn9mrxJs/s-l1600.webp",
    "https://i.ebayimg.com/images/g/nhUAAeSwDqdpCtF1/s-l1600.webp",
    "https://i.ebayimg.com/images/g/w5gAAeSw-BppCsmd/s-l1600.webp",
    "https://i.ebayimg.com/images/g/EPUAAeSwUfxpCqVY/s-l1600.webp"
]

# ------------------------- LOAD STATE ------------------------
assert os.path.exists(META_PATH), f"Missing {META_PATH}. Run the initial build script first."
assert os.path.exists(EMB_PATH), f"Missing {EMB_PATH}. Run the initial build script first."
assert os.path.exists(INDEX_PATH), f"Missing {INDEX_PATH}. Run the initial build script first."

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

existing_ids: List[str] = meta.get("ids", [])
existing_urls: List[str] = meta.get("urls", [])
old_count = len(existing_urls)

print(f"Loaded meta.json with {old_count} items. Embeddings file: {EMB_PATH}")

# ----------------------- DEDUP NEW LINKS ---------------------
# De-dup input list itself
unique_incoming = []
seen = set()
for u in new_links:
    if u not in seen:
        unique_incoming.append(u)
        seen.add(u)

# Skip any URL already in catalog
to_add = [u for u in unique_incoming if u not in set(existing_urls)]

if not to_add:
    print("No truly new URLs to add. Exiting.")
    raise SystemExit(0)

print(f"Incoming links: {len(new_links)} | After dedup vs existing: {len(to_add)}")

# ----------------------- LOAD MODEL --------------------------
print("Loading model…")
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED_TAG)
model.eval()
model = model.to(DEVICE)

# ------------------------- HELPERS ---------------------------
session = requests.Session()

def fetch_image(url: str) -> Image.Image:
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

@torch.no_grad()
def embed_batch(pil_images: List[Image.Image]) -> np.ndarray:
    batch = torch.stack([preprocess(img) for img in pil_images], dim=0).to(DEVICE)
    feats = model.encode_image(batch)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")

# ----------------------- ENCODE NEW --------------------------
new_ids: List[str] = []
new_urls: List[str] = []
failed: List[Tuple[str, str]] = []  # (url, reason)

# Generate new IDs continuing sequence
start_idx = old_count

all_new_embs: List[np.ndarray] = []

print(f"Encoding {len(to_add)} new images…")
for i in tqdm(range(0, len(to_add), BATCH_SIZE), desc="Batches"):
    chunk_urls = to_add[i:i+BATCH_SIZE]
    pil_imgs = []
    ok_flags = []
    for u in chunk_urls:
        try:
            pil_imgs.append(fetch_image(u))
            ok_flags.append(True)
        except Exception as e:
            failed.append((u, str(e)))
            ok_flags.append(False)

    if pil_imgs:
        embs = embed_batch(pil_imgs)
        all_new_embs.append(embs)

        # assign ids for successful ones in order
        for u, ok in zip(chunk_urls, ok_flags):
            if ok:
                new_id = f"prod_{start_idx:05d}"
                new_ids.append(new_id)
                new_urls.append(u)
                start_idx += 1

if not all_new_embs:
    print("No new embeddings were produced (all downloads failed?) — nothing to append.")
    raise SystemExit(0)

new_embs = np.vstack(all_new_embs)
print(f"New embeddings shape: {new_embs.shape}")

# -------------------- UPDATE ARTIFACTS -----------------------
# 1) Update FAISS index in place (fast)
index = faiss.read_index(INDEX_PATH)
index.add(new_embs)
faiss.write_index(index, INDEX_PATH)
print(f"Updated FAISS index written → {INDEX_PATH}")

# 2) Append to embeddings.npy (simple approach: read → vstack → save)
old_embs = np.load(EMB_PATH)
all_embs = np.vstack([old_embs, new_embs])
np.save(EMB_PATH, all_embs)
print(f"Updated embeddings saved → {EMB_PATH}  (shape={all_embs.shape})")

# 3) Update meta.json
meta["ids"].extend(new_ids)
meta["urls"].extend(new_urls)
meta["count"] = len(meta["urls"])
meta["dim"] = int(all_embs.shape[1])
meta.setdefault("failed", [])
for u, reason in failed:
    meta["failed"].append(["append", u, reason])

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"Appended {len(new_ids)} items. Total catalog now: {meta['count']} (failed this run: {len(failed)}).")

# ------------------------ NOTES ------------------------------
# * This keeps the SAME model & normalization as the original build.
# * If you ever switch model or preprocessing, you must rebuild everything.
# * For very large catalogs, consider memory-mapped arrays or Parquet instead of .npy.
