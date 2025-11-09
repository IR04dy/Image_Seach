import os, io, json, sqlite3, requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import faiss, torch, open_clip

# ----------------------- PATHS ------------------------------
ARTIFACTS_DIR = "./artifacts"
INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss.index")
META_PATH = os.path.join(ARTIFACTS_DIR, "meta.json")
SQLITE_PATH = "ebay_products.db"  # your DB file

# ----------------------- LOAD ARTIFACTS ----------------------
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)
CAT_IDS = meta["ids"]        # e.g., ["prod_01902", ...]
CAT_URLS = meta["urls"]
DIM = int(meta["dim"])       # sanity

# ----------------------- LOAD MODEL --------------------------
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
model.eval()
DEVICE = "cpu"
model = model.to(DEVICE)

# ---------------------- HELPERS ------------------------------
session = requests.Session()

def load_image_any(src: str) -> Image.Image:
    if src.startswith("http://") or src.startswith("https://"):
        r = session.get(src, timeout=15)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    else:
        return Image.open(src).convert("RGB")

def embed_one(pil_img: Image.Image) -> np.ndarray:
    x = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")  # (1, d)

# ---------------------- DB HELPERS (schema-aware) ------------
def _open_db(path=SQLITE_PATH):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def _table_columns(conn, table="products"):
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [r["name"] for r in cur.fetchall()]

def _fetch_rows_by_single_column(conn, table, col, values):
    """Generic IN query; values must be a list, empty-safe."""
    if not values:
        return []
    placeholders = ",".join("?" for _ in values)
    q = f"SELECT * FROM {table} WHERE {col} IN ({placeholders})"
    cur = conn.execute(q, list(values))
    return cur.fetchall()

def _fetch_rows_by_ids(conn, table, id_values):
    """
    Try only columns that actually exist in the table.
    Candidate names: id, product_id, pid, sku, external_id.
    """
    candidate_cols = ["id", "product_id", "pid", "sku", "external_id"]
    cols = _table_columns(conn, table)
    usable = [c for c in candidate_cols if c in cols]
    rows = []
    remaining = set(map(str, id_values))  # normalize to str
    if not usable:
        return rows  # no id-like columns in table; caller will try URL fallback

    for col in usable:
        if not remaining:
            break
        got = _fetch_rows_by_single_column(conn, table, col, list(remaining))
        # remove matched
        for r in got:
            # best-effort: pull any id-ish value to match
            for c in ["id", "product_id", "pid", "sku", "external_id"]:
                if c in r.keys() and r[c] is not None:
                    val = str(r[c])
                    if val in remaining:
                        remaining.remove(val)
                    break
        rows.extend(got)
    return rows

def _fetch_rows_by_urls(conn, table, url_values):
    """
    Fallback join by URL if present in your DB.
    Candidate names: url, product_url, page_url, image_url, thumbnail_url.
    """
    candidate_cols = ["url", "product_url", "page_url", "image_url", "thumbnail_url"]
    cols = _table_columns(conn, table)
    usable = [c for c in candidate_cols if c in cols]
    rows = []
    remaining = set(url_values)
    if not usable:
        return rows

    for col in usable:
        if not remaining:
            break
        got = _fetch_rows_by_single_column(conn, table, col, list(remaining))
        for r in got:
            # pick first matching candidate col to remove
            for c in candidate_cols:
                if c in r.keys() and r[c] in remaining:
                    remaining.remove(r[c])
                    break
        rows.extend(got)
    return rows

def fetch_products_for_results(result_indices, table="products"):
    """
    Map FAISS results to DB rows by matching CAT_URLS[*] -> products.image
    Returns: dict rank_index -> sqlite3.Row
    """
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    by_rank = {}
    try:
        # Collect the image URLs in the same order as top-k indices
        urls = [CAT_URLS[i] for i in result_indices]
        if not urls:
            return by_rank

        # Exact match against products.image
        placeholders = ",".join("?" for _ in urls)
        q = f"SELECT * FROM {table} WHERE image IN ({placeholders})"
        rows = conn.execute(q, urls).fetchall()

        # Build a quick lookup map: image_url -> row
        row_by_image = {}
        for r in rows:
            if "image" in r.keys() and r["image"]:
                row_by_image[str(r["image"])] = r

        # Assign back to ranks
        for rank, idx in enumerate(result_indices):
            u = CAT_URLS[idx]
            if u in row_by_image:
                by_rank[rank] = row_by_image[u]

        # (Optional) fallback: if some images use different size variants (s-l140 vs s-l500),
        # try a relaxed match on the filename stem.
        missing = [r for r in range(len(result_indices)) if r not in by_rank]
        if missing:
            import re, os
            def normalize(u):
                # keep the path but strip the trailing size suffix like /s-l140.jpg or /s-l500.jpg
                return re.sub(r"/s-l\d+\.jpg$", "", u)

            want = {rank: normalize(CAT_URLS[result_indices[rank]]) for rank in missing}
            # Pull a few candidates to compare
            cur = conn.execute(f"SELECT image FROM {table} WHERE image LIKE '%i.ebayimg.com/%'")
            candidates = [row[0] for row in cur.fetchall() if row[0]]
            cand_norm = {c: normalize(c) for c in candidates}

            for rank in missing:
                target = want[rank]
                # find first candidate with the same normalized path
                match = next((c for c, n in cand_norm.items() if n == target), None)
                if match:
                    row = conn.execute(f"SELECT * FROM {table} WHERE image = ?", (match,)).fetchone()
                    if row:
                        by_rank[rank] = row

        return by_rank
    finally:
        conn.close()

def _print_sqlite_row(row: sqlite3.Row, header=None):
    if header:
        print(header)
    keys = row.keys()
    width = max((len(k) for k in keys), default=0)
    for k in keys:
        print(f"  {k:<{width}} : {row[k]}")
    print("-" * 50)

# ------------------------ SEARCH + DISPLAY -------------------
def search_and_display(query_src: str, topk: int = 8, table: str = "products"):
    # 1) embed query
    q_img = load_image_any(query_src)
    q = embed_one(q_img)

    # 2) search
    sims, idxs = index.search(q, topk)
    sims, idxs = sims[0], idxs[0]

    # 3) visualize
    cols = topk + 1
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))
    fig.suptitle(f"Query + top-{topk} similar products", fontsize=14)

    # query
    axes[0].imshow(q_img)
    axes[0].set_title("Query")
    axes[0].axis("off")

    # results
    for i, (idx, sim) in enumerate(zip(idxs, sims), start=1):
        url = CAT_URLS[idx]
        pid = CAT_IDS[idx]
        try:
            img = load_image_any(url)
            axes[i].imshow(img)
            axes[i].set_title(f"#{i} — {pid}\nSim={sim:.2f}")
            axes[i].axis("off")
        except Exception:
            axes[i].text(0.5, 0.5, f"Failed\n{pid}", ha='center', va='center')
            axes[i].axis("off")

    plt.tight_layout()
    plt.show()

    # 4) compact list
    print("Top results:")
    for rank, (idx, sim) in enumerate(zip(idxs, sims), start=1):
        print(f"{rank:2d}. id={CAT_IDS[idx]}  sim={sim:.4f}  url={CAT_URLS[idx]}")
    print()

    # 5) DB lookups for details
    print("=== Database rows for top results ===")
    rank_to_row = fetch_products_for_results(idxs, table=table)

    for rank_zero_based, (idx, sim) in enumerate(zip(idxs, sims)):
        header = f"[#{rank_zero_based+1}] id={CAT_IDS[idx]}  sim={sim:.4f}"
        row = rank_to_row.get(rank_zero_based)
        if row is not None:
            _print_sqlite_row(row, header=header)
        else:
            print(header)
            print("  (No matching row found in DB by id or url)")
            print("-" * 50)

# ------------------------ USAGE ------------------------------
if __name__ == "__main__":
    # Use raw string or forward slashes on Windows to avoid \k escape
    search_and_display(r"images\kindle.jpeg", topk=6, table="products")

#search_and_display("https://i.ebayimg.com/images/g/4ngAAeSwx21pAaAI/s-l1600.webp", topk=6)
