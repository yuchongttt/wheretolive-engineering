#!/usr/bin/env python3
"""Combined DINO + SigLIP embedding daemon.

Replaces dino_embed_daemon.py + siglip_embed_daemon.py running side by
side. On HDD sdb the two daemons saturated IOPS by reading the same
image files independently — each photo was opened twice for two
forward passes, so disk became the bottleneck and both daemons stalled
in `D (disk sleep)`.

This daemon:
  - reads each image once from disk
  - shares the PIL.open across both preprocessors (DINO at 512×512,
    SigLIP at 256×256)
  - runs DINO and SigLIP forward passes back-to-back on the same GPU
    batch (sequential on GPU; both fit in 12 GB VRAM with margin)
  - writes 3 image_embeddings rows (dinov3-l16-base, dinov3-l16-v10,
    siglip-lux128) + 1 image_luxury_score_v2 row per image, all in a
    single transaction → one writer, no inter-daemon SQLite contention

Filter rules per model (matches the two old daemons):
  - DINO: photos only, skip sold properties (property_id is UUID with
    4 dashes) — DINO feeds the active-listing FAISS image-search index
  - SigLIP: all photos (active + sold) — luxury score is used by the
    valuation model on sold properties too

pull_batch returns rows where AT LEAST ONE model is missing. Per row,
needs_dino / needs_siglip flags drive which forward(s) run.

Stop-safe: SIGTERM finishes current GPU batch, drops DataLoader, exits.
"""
from __future__ import annotations

import gc, os, signal, sqlite3, sys, threading, time
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel, AutoProcessor

DB         = "/data/ml/dataset/dataset.db"
CACHE      = "/data/ml/hf-cache"

DINO_MODEL_ID    = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_TAG         = "dinov3-l16-base"
DINO_V10_TAG     = "dinov3-l16-v10"
DINO_HEAD_PATH   = "/data/ml/dataset/m1a_head_v10.pt"
DINO_INPUT_SIZE  = 512

SIG_MODEL_ID     = "google/siglip2-base-patch16-256"
SIG_TAG          = "siglip-lux128"
SIG_PCA_PATH     = "/data/ml/dataset/siglip_pca_128.npz"
SIG_INPUT_SIZE   = 256
SCORE_TEMPERATURE = 10.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16

# Single set of workers reads each image once. 4 workers covers both
# preprocessors per image (one disk read → two CPU resize ops). On HDD
# sdb (~250 IOPS) this stays well inside the budget vs 8 workers from
# two separate daemons.
GPU_BATCH    = 32     # DINO's preferred — SigLIP can take more but we
                       # share the outer batch so both step together
LARGE_BATCH  = 256
NUM_WORKERS  = 4
PREFETCH     = 3
DELAY_IDLE_S = 60
# Watermark re-scan overlap. completed_at is compared as a STRING, and
# datetime.isoformat() drops the fractional part when microsecond == 0, so two
# writers inside the same second can emit values that sort <= a watermark we
# already advanced past. Re-scanning a small trailing window closes that race;
# the window is an index range so it costs ~nothing.
WATERMARK_OVERLAP_S = 60
GC_EVERY_N_BATCHES = 50

# FAISS delta (same as old DINO daemon)
INDEX_DIR        = "/opt/wheretolive/dino-indexes"
FLUSH_INTERVAL_S = 300

# DINO image-search retired 2026-06-28: server offline since 2026-05-29 +
# web feature flag off + no live consumers. When DINO_ENABLED=0 the daemon
# skips the heavy DINO model load, stops writing dinov3-l16-* embeddings, and
# skips all FAISS delta work — only the SigLIP / luxury / decor path runs.
# Set DINO_ENABLED=1 (env) to resume DINO embedding generation.
DINO_ENABLED = os.environ.get("DINO_ENABLED", "0") == "1"
# The FAISS delta index exists to feed increments to dino_server's reverse image
# search — that service was retired on 2026-06-28 and has no consumers. Off by
# default: when on, it keeps a GB-scale delta index + millions of metadata rows
# resident in memory, growing with every new image (when DINO embedding was
# re-enabled on 2026-09-01 it pushed the 15G machine into a chain of oomd kills;
# systemd gave up after the restart counter hit 17). When reviving dino_server,
# set FAISS_DELTA_ENABLED=1 (env) to turn it back on.
FAISS_DELTA_ENABLED = os.environ.get("FAISS_DELTA_ENABLED", "0") == "1"

# ---------- prompt sets (from siglip daemon) ----------
LUXURY_PROMPTS = [
    "a luxurious high-end interior with marble surfaces, gold fixtures, crystal chandelier",
    "an expensive designer interior with premium materials and elegant decor",
    "a luxury upscale apartment interior with high-end finishes and chic furniture",
]
BUDGET_PROMPTS = [
    "a worn-down basic interior with old furniture and dated decor",
    "a cheap simple interior with low-quality materials and bare walls",
    "an outdated budget interior with cluttered shabby furniture",
]
INDOOR_PROMPTS  = ["an indoor photograph of a room inside a house",
                   "interior photo of a living space"]
OUTDOOR_PROMPTS = ["an outdoor photograph of a building exterior",
                   "exterior view of a house from the street"]


# ============ signals ============
STOP = False
def _handle_sigterm(signum, frame):  # noqa: ARG001
    global STOP
    STOP = True
    print(f"[{_now()}] signal {signum}; finishing batch then stopping", flush=True)

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ============ load DINO ============
os.environ["HF_HOME"] = CACHE
print(f"[{_now()}] loading DINO {DINO_MODEL_ID}", flush=True)
t0 = time.time()
DINO_PROC = AutoImageProcessor.from_pretrained(
    DINO_MODEL_ID, cache_dir=CACHE,
    size={"height": DINO_INPUT_SIZE, "width": DINO_INPUT_SIZE},
    do_center_crop=False,
)
if DINO_ENABLED:
    DINO_MODEL = AutoModel.from_pretrained(
        DINO_MODEL_ID, torch_dtype=DTYPE, cache_dir=CACHE,
    ).to(DEVICE).eval()
    print(f"[{_now()}] DINO loaded in {time.time()-t0:.1f}s", flush=True)
else:
    DINO_MODEL = None
    print(f"[{_now()}] DINO model NOT loaded (DINO_ENABLED=0); SigLIP/decor only", flush=True)


class _V10Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        return F.normalize(self.fc2(x), dim=-1)

if DINO_ENABLED:
    DINO_HEAD = _V10Head().to(DEVICE).eval()
    try:
        _state = torch.load(DINO_HEAD_PATH, map_location="cpu", weights_only=False)
        DINO_HEAD.load_state_dict(_state["state_dict"])
        HEAD_OK = True
        print(f"[{_now()}] DINO v10 head loaded", flush=True)
    except Exception as _e:
        HEAD_OK = False
        print(f"[{_now()}] WARNING: DINO v10 head load failed: {_e}; base-only", flush=True)
else:
    DINO_HEAD = None
    HEAD_OK = False


# ============ load SigLIP ============
print(f"[{_now()}] loading SigLIP {SIG_MODEL_ID}", flush=True)
t0 = time.time()
SIG_PROC = AutoProcessor.from_pretrained(SIG_MODEL_ID, cache_dir=CACHE)
SIG_MODEL = AutoModel.from_pretrained(
    SIG_MODEL_ID, torch_dtype=DTYPE, cache_dir=CACHE,
).to(DEVICE).eval()
print(f"[{_now()}] SigLIP loaded in {time.time()-t0:.1f}s", flush=True)

# PCA (768→128)
_pca = np.load(SIG_PCA_PATH, allow_pickle=True)
PCA_MEAN = _pca["mean"].astype(np.float32)
PCA_COMP = _pca["components"].astype(np.float32)
print(f"[{_now()}] SigLIP PCA cumulative variance: "
      f"{float(_pca['explained_variance_ratio'].sum())*100:.1f}%", flush=True)


def _project(vecs: np.ndarray) -> np.ndarray:
    return (vecs - PCA_MEAN) @ PCA_COMP.T


def _encode_text(prompts: list[str]) -> torch.Tensor:
    with torch.no_grad():
        txt_in = SIG_PROC(text=prompts, return_tensors="pt", padding="max_length").to(DEVICE)
        out = SIG_MODEL.get_text_features(**txt_in)
        feats = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out
        return F.normalize(feats.float(), dim=-1).cpu().numpy()


TXT_LUX_128 = _project(_encode_text(LUXURY_PROMPTS))
TXT_BUD_128 = _project(_encode_text(BUDGET_PROMPTS))
TXT_IND_128 = _project(_encode_text(INDOOR_PROMPTS))
TXT_OUT_128 = _project(_encode_text(OUTDOOR_PROMPTS))
print(f"[{_now()}] SigLIP prompt features ready", flush=True)

# ---------- decor (renovated-vs-dated) prompt sets ----------
# Verbatim from test_new_prompts_interior.py (NEW_LUX / NEW_BUD / EMPTY_PROMPTS).
# These produce the per-image `new` score whose top-3 mean == new_top3 ==
# decor_zeroshot_score. CRITICAL: scored in the FULL feature space (no PCA),
# exactly as the calibration ρ≈0.33 + £-tier mapping were derived. Do NOT
# reuse the 128-d luxury path (TXT_*_128) — PCA shifts the margin and breaks
# consistency with the 96.9% already in decor_zeroshot_score on the Mac.
DECOR_NEW_LUX = [
    "a meticulously designed modern interior with floor-to-ceiling windows and natural light",
    "a luxurious classical interior with marble surfaces, ornate moldings, chandelier",
    "a high-end minimalist living space with premium hardwood floors and clean lines",
    "an architect-designed loft with exposed brick, polished concrete, designer furniture",
    "an immaculately renovated home with bespoke joinery and luxury appliances",
    "a freshly refurbished open-plan kitchen with stone worktops and integrated appliances",
    "a recently remodeled bathroom with walk-in shower, large-format tiles, frameless glass",
    "a thoughtfully decorated bedroom with quality bedding and tasteful artwork",
]
DECOR_NEW_BUD = [
    "an ordinary plain interior with basic fixtures and dated decor",
    "a tired room with worn furniture and dated finishes",
    "a budget rental-grade interior with minimal furniture",
    "a plain dated kitchen with old laminate worktops and basic appliances",
    "a small simple bathroom with builder-grade tiles and dated fixtures",
    "a sparsely furnished room with worn carpet and plain walls",
    "an outdated 1990s style interior in need of modernization",
    "an unrenovated room with old wallpaper and dated lighting",
]
DECOR_EMPTY = [
    "a completely empty room with no furniture",
    "an unfurnished vacant room with bare walls and bare floor",
    "an empty space without any furnishings or decor",
]
# Full-space (768-d) text features — _encode_text returns pre-PCA features.
DECOR_TEMPERATURE = 20.0  # matches test_new_prompts_interior.py sigmoid(20*raw)
TXT_DNEW_LUX = _encode_text(DECOR_NEW_LUX)
TXT_DNEW_BUD = _encode_text(DECOR_NEW_BUD)
TXT_DEMPTY   = _encode_text(DECOR_EMPTY)
print(f"[{_now()}] decor prompt features ready (full-space)", flush=True)


# ============ DB ============
conn = sqlite3.connect(DB, isolation_level=None)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA busy_timeout=30000")
conn.execute("PRAGMA synchronous=NORMAL")
conn.execute("CREATE INDEX IF NOT EXISTS idx_image_embeddings_model_created ON image_embeddings(model, created_at)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_image_embeddings_model_real ON image_embeddings(model) WHERE url NOT LIKE 'local:%'")
conn.execute("""
    CREATE TABLE IF NOT EXISTS image_luxury_score_v2 (
        url           TEXT    PRIMARY KEY,
        property_id   TEXT    NOT NULL,
        score         REAL    NOT NULL,
        luxury_cos    REAL    NOT NULL,
        budget_cos    REAL    NOT NULL,
        indoor_cos    REAL    NOT NULL,
        outdoor_cos   REAL    NOT NULL,
        is_interior   INTEGER NOT NULL,
        model_tag     TEXT    NOT NULL,
        computed_at   TEXT    NOT NULL,
        decor_score   REAL,
        decor_empty   REAL
    )
""")
conn.execute("CREATE INDEX IF NOT EXISTS idx_lux_v2_pid ON image_luxury_score_v2(property_id)")
# decor_score / decor_empty added 2026-06-01: per-image renovated-vs-dated
# margin (full-feature space, matches test_new_prompts_interior.py new_top3
# calibration → Mac decor_zeroshot_score). Idempotent ALTER for pre-existing
# tables created before these columns.
for _col, _decl in (("decor_score", "REAL"), ("decor_empty", "REAL")):
    try:
        conn.execute(f"ALTER TABLE image_luxury_score_v2 ADD COLUMN {_col} {_decl}")
        print(f"[{_now()}] image_luxury_score_v2: added column {_col}", flush=True)
    except sqlite3.OperationalError:
        pass  # column already present


def pull_batch(n: int, since: str | None = None) -> list[tuple[str, str, str, int, str, int, int]]:
    """Returns (url, local_path, property_id, url_idx, caption_raw,
                needs_dino, needs_siglip).

    media_type taxonomy on Linux `images` table:
      - 'photo'     → active listings, get BOTH DINO + SigLIP
      - 'sold'      → sold-property images pushed from Mac. Skip DINO
                       (no role in image-search FAISS index — sold props
                       aren't searchable inventory) but SigLIP runs
                       since the luxury score feeds the valuation model.
      - 'floorplan' → skip both (line art breaks DINO/SigLIP embedding
                       distributions; planned OCR pipeline lives on a
                       separate host).

    Eligibility for this fetch:
      - status='done' AND local_path NOT NULL
      - media_type IN ('photo', 'sold')   ← 'floorplan' excluded entirely
      - missing at least one of the required model embeddings

    Order: newest scraped_at first (queue-priority for newly pushed images).
    """
    # Bounding the scan by completed_at is what turns this from O(all done
    # rows) into O(new arrivals). Built as a literal fragment, not a
    # `? IS NULL OR ...` predicate, because the latter is not sargable and
    # would silently defeat idx_images_done_completed.
    since_sql = "AND i.completed_at > ?" if since else ""
    since_params = (since,) if since else ()

    if not DINO_ENABLED:
        # DINO retired: only claim rows still missing SigLIP. needs_dino is
        # hard-coded 0 so the DINO forward + delta paths stay empty. Without
        # this branch the original query would keep re-claiming photo rows that
        # are missing a DINO embedding we no longer produce → hot no-op loop.
        return conn.execute(f"""
            SELECT
              i.url, i.local_path, i.property_id, i.url_idx, i.caption_raw,
              0 AS needs_dino,
              1 AS needs_siglip
            FROM images i
            LEFT JOIN image_embeddings e_sig
                   ON e_sig.url = i.url AND e_sig.model = ?
            WHERE i.status = 'done'
              AND i.local_path IS NOT NULL
              AND i.media_type IN ('photo', 'sold')
              AND e_sig.url IS NULL
              {since_sql}
            ORDER BY i.scraped_at DESC
            LIMIT ?
        """, (SIG_TAG, *since_params, n)).fetchall()
    return conn.execute(f"""
        SELECT
          i.url, i.local_path, i.property_id, i.url_idx, i.caption_raw,
          CASE
            WHEN e_dino.url IS NULL AND i.media_type = 'photo'
            THEN 1 ELSE 0
          END AS needs_dino,
          CASE WHEN e_sig.url IS NULL THEN 1 ELSE 0 END AS needs_siglip
        FROM images i
        LEFT JOIN image_embeddings e_dino
               ON e_dino.url = i.url AND e_dino.model = ?
        LEFT JOIN image_embeddings e_sig
               ON e_sig.url   = i.url AND e_sig.model = ?
        WHERE i.status = 'done'
          AND i.local_path IS NOT NULL
          AND i.media_type IN ('photo', 'sold')
          AND (
                (e_dino.url IS NULL AND i.media_type = 'photo')
             OR (e_sig.url IS NULL)
          )
          {since_sql}
        ORDER BY i.scraped_at DESC
        LIMIT ?
    """, (DINO_TAG, SIG_TAG, *since_params, n)).fetchall()


# ============ FAISS delta (DINO only) ============
delta_lock       = threading.Lock()
delta_index_base = None
delta_index_v10  = None
delta_meta = {
    "base": {"urls": [], "pids": [], "url_idxs": [], "captions": []},
    "v10":  {"urls": [], "pids": [], "url_idxs": [], "captions": []},
}
delta_next_id = {"base": 0, "v10": 0}
delta_dirty   = False


def probe_watermark() -> str | None:
    """Highest completed_at among embeddable rows. O(1) via
    idx_images_done_completed (status, completed_at) -- measured 0.000s vs
    5.88s before that index existed."""
    return conn.execute(
        "SELECT MAX(completed_at) FROM images WHERE status = 'done'"
    ).fetchone()[0]


def _watermark_floor(hw: str) -> str:
    """hw minus WATERMARK_OVERLAP_S, in the exact same isoformat the column
    uses, so string comparison stays chronological."""
    return (datetime.fromisoformat(hw)
            - timedelta(seconds=WATERMARK_OVERLAP_S)).isoformat()


def _load_delta_indices() -> None:
    import faiss
    global delta_index_base, delta_index_v10
    os.makedirs(INDEX_DIR, exist_ok=True)

    def _load(tag: str, dim: int):
        path = f"{INDEX_DIR}/dino-{tag}.delta.faiss"
        meta_path = f"{INDEX_DIR}/dino-{tag}.delta.meta.npz"
        if os.path.exists(path) and os.path.exists(meta_path):
            try:
                idx = faiss.read_index(path)
                m = np.load(meta_path, allow_pickle=True)
                slot = "base" if tag == DINO_TAG else "v10"
                delta_meta[slot]["urls"]     = list(m["urls"])
                delta_meta[slot]["pids"]     = list(m["pids"])
                delta_meta[slot]["url_idxs"] = list(m["url_idxs"])
                delta_meta[slot]["captions"] = list(m["captions"])
                delta_next_id[slot] = idx.ntotal
                print(f"[{_now()}] loaded delta {tag}: ntotal={idx.ntotal:,}", flush=True)
                return idx
            except Exception as e:
                print(f"[{_now()}] WARNING: delta {tag} load failed ({e})", flush=True)
        inner = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        inner.hnsw.efConstruction = 200
        return faiss.IndexIDMap2(inner)

    delta_index_base = _load(DINO_TAG, 1024)
    delta_index_v10  = _load(DINO_V10_TAG, 256)


def _delta_add(rows) -> None:
    """rows: list of (url, base_vec, v10_vec_or_None, pid, url_idx, caption).
    Only for DINO embeddings (active listings) — UUID rows are pre-filtered
    out by pull_batch's needs_dino logic.
    """
    global delta_dirty
    if not FAISS_DELTA_ENABLED or not rows:
        return
    with delta_lock:
        base_vecs = np.stack([r[1].astype(np.float32) for r in rows])
        n = base_vecs.shape[0]
        base_ids = np.arange(delta_next_id["base"], delta_next_id["base"] + n, dtype=np.int64)
        delta_index_base.add_with_ids(base_vecs, base_ids)
        for r in rows:
            delta_meta["base"]["urls"].append(r[0])
            delta_meta["base"]["pids"].append(str(r[3]))
            delta_meta["base"]["url_idxs"].append(int(r[4]))
            delta_meta["base"]["captions"].append(r[5] or "")
        delta_next_id["base"] += n

        v10_rows = [r for r in rows if r[2] is not None]
        if v10_rows and delta_index_v10 is not None:
            v10_vecs = np.stack([r[2].astype(np.float32) for r in v10_rows])
            m = v10_vecs.shape[0]
            v10_ids = np.arange(delta_next_id["v10"], delta_next_id["v10"] + m, dtype=np.int64)
            delta_index_v10.add_with_ids(v10_vecs, v10_ids)
            for r in v10_rows:
                delta_meta["v10"]["urls"].append(r[0])
                delta_meta["v10"]["pids"].append(str(r[3]))
                delta_meta["v10"]["url_idxs"].append(int(r[4]))
                delta_meta["v10"]["captions"].append(r[5] or "")
            delta_next_id["v10"] += m

        delta_dirty = True


def _delta_flush() -> None:
    import faiss
    global delta_dirty
    with delta_lock:
        if not delta_dirty:
            return
        snap_base = faiss.clone_index(delta_index_base) if delta_index_base is not None else None
        snap_v10  = faiss.clone_index(delta_index_v10)  if delta_index_v10  is not None else None
        meta_snap = {
            "base": {k: list(v) for k, v in delta_meta["base"].items()},
            "v10":  {k: list(v) for k, v in delta_meta["v10"].items()},
        }
        delta_dirty = False

    def _write(tag: str, idx, m):
        if idx is None:
            return
        base_path = f"{INDEX_DIR}/dino-{tag}.delta.faiss"
        meta_path = f"{INDEX_DIR}/dino-{tag}.delta.meta.npz"
        tmp_faiss = base_path + ".new"
        tmp_meta  = meta_path.replace(".npz", ".new.npz")
        faiss.write_index(idx, tmp_faiss)
        np.savez(
            tmp_meta,
            urls=np.array(m["urls"], dtype=object),
            pids=np.array(m["pids"], dtype=object),
            url_idxs=np.array(m["url_idxs"], dtype=np.int32),
            captions=np.array(m["captions"], dtype=object),
        )
        os.rename(tmp_faiss, base_path)
        os.rename(tmp_meta, meta_path)

    t0 = time.time()
    _write(DINO_TAG,     snap_base, meta_snap["base"])
    _write(DINO_V10_TAG, snap_v10,  meta_snap["v10"])
    base_n = snap_base.ntotal if snap_base is not None else 0
    v10_n  = snap_v10.ntotal  if snap_v10  is not None else 0
    print(f"[{_now()}] delta flush: base={base_n:,} v10={v10_n:,} in {time.time()-t0:.2f}s",
          flush=True)


def _flush_loop() -> None:
    while not STOP:
        for _ in range(FLUSH_INTERVAL_S):
            if STOP: break
            time.sleep(1)
        try:
            _delta_flush()
        except Exception as e:
            print(f"[{_now()}] flush failed: {e}", flush=True)
    try:
        _delta_flush()
    except Exception as e:
        print(f"[{_now()}] final flush failed: {e}", flush=True)


# ============ DataLoader: one disk read → both preprocessors ============
class EmbedDataset(Dataset):
    """Returns (url, dino_pix, sig_pix, ok). One PIL.open per image,
    then two preprocessors (DINO @ 512×512, SigLIP @ 256×256). Zero
    tensors are placeholders when a preprocessor fails (preserves
    batch stacking).
    """
    def __init__(self, rows: list[tuple]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        url, lp = self.rows[i][0], self.rows[i][1]
        try:
            im = Image.open(lp).convert("RGB")
            dino_pix = DINO_PROC(images=im, return_tensors="pt")["pixel_values"][0]
            sig_pix  = SIG_PROC(images=im,  return_tensors="pt")["pixel_values"][0]
            return url, dino_pix, sig_pix, True
        except Exception:
            return (url,
                    torch.zeros(3, DINO_INPUT_SIZE, DINO_INPUT_SIZE),
                    torch.zeros(3, SIG_INPUT_SIZE,  SIG_INPUT_SIZE),
                    False)


def _collate(batch):
    urls, dino_pix, sig_pix, oks = zip(*batch)
    return (list(urls), torch.stack(list(dino_pix)),
            torch.stack(list(sig_pix)), list(oks))


# ============ GPU forwards ============
@torch.no_grad()
def dino_forward(pixels: torch.Tensor):
    """Returns (base_np (N,1024), v10_np (N,256) or None)."""
    pixels = pixels.to(DEVICE, dtype=DTYPE, non_blocking=True)
    out = DINO_MODEL(pixel_values=pixels)
    feat = getattr(out, "pooler_output", None)
    if feat is None:
        feat = out.last_hidden_state[:, 0]
    feat = F.normalize(feat.float(), dim=-1)
    base_np = feat.cpu().numpy()
    v10_np = DINO_HEAD(feat).cpu().numpy() if HEAD_OK else None
    return base_np, v10_np


@torch.no_grad()
def siglip_forward(pixels: torch.Tensor):
    """Returns (vecs_128, full_feats): PCA-projected (N,128) for the embedding +
    luxury path, and the full normalized (N,768) features the decor margin needs
    (decor calibration was derived in full space, not PCA-128)."""
    pixels = pixels.to(DEVICE, dtype=DTYPE, non_blocking=True)
    out = SIG_MODEL.get_image_features(pixel_values=pixels)
    feats = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out
    full = F.normalize(feats.float(), dim=-1).cpu().numpy()
    return _project(full), full


def _score_batch(vecs_128: np.ndarray):
    lux = (vecs_128 @ TXT_LUX_128.T).mean(axis=1)
    bud = (vecs_128 @ TXT_BUD_128.T).mean(axis=1)
    ind = (vecs_128 @ TXT_IND_128.T).mean(axis=1)
    oud = (vecs_128 @ TXT_OUT_128.T).mean(axis=1)
    return lux, bud, ind, oud


def _score_decor(full_feats: np.ndarray):
    """Per-image decor (renovated-vs-dated) score + emptiness, computed in the
    FULL feature space to match new_top3 / decor_zeroshot_score. Mirrors
    score_image(...,NEW_LUX,NEW_BUD,agg='mean'): raw = mean(lux) - mean(bud),
    score = sigmoid(20*raw); empty = max over EMPTY prompts. Returns
    (decor (N,), empty (N,)) float arrays. Per-property top-3 aggregation
    happens later on the Mac drain side."""
    lux = (full_feats @ TXT_DNEW_LUX.T).mean(axis=1)
    bud = (full_feats @ TXT_DNEW_BUD.T).mean(axis=1)
    raw = lux - bud
    decor = 1.0 / (1.0 + np.exp(-raw * DECOR_TEMPERATURE))
    empty = (full_feats @ TXT_DEMPTY.T).max(axis=1)
    return decor.astype(np.float32), empty.astype(np.float32)


# ============ unified writer ============
def write_batch(dino_rows, sig_rows) -> None:
    """One transaction, three table writes:
      - image_embeddings × 3 models (dino base, dino v10, siglip)
      - image_luxury_score_v2 × 1 row per SigLIP image
    Retries on "database is locked" up to 5 times.

    dino_rows: list of (url, base_vec, v10_vec_or_None)
    sig_rows:  list of (url, pid, vec_128, lux_cos, bud_cos, ind_cos, out_cos,
                        decor_score, decor_empty)   ← decor_* may be None
    """
    if not dino_rows and not sig_rows:
        return

    now = datetime.now(timezone.utc).isoformat()
    emb_payload = []
    for url, base_v, v10_v in dino_rows:
        emb_payload.append((url, DINO_TAG, int(base_v.shape[0]), "float32",
                            base_v.astype(np.float32).tobytes(), now))
        if v10_v is not None:
            emb_payload.append((url, DINO_V10_TAG, int(v10_v.shape[0]), "float32",
                                v10_v.astype(np.float32).tobytes(), now))
    lux_payload = []
    for url, pid, vec, lc, bc, ic, oc, dsc, dem in sig_rows:
        emb_payload.append((url, SIG_TAG, 128, "float32",
                            vec.astype(np.float32).tobytes(), now))
        score = 1.0 / (1.0 + np.exp(-(lc - bc) * SCORE_TEMPERATURE))
        is_interior = 1 if ic > oc else 0
        lux_payload.append((url, str(pid), float(score), float(lc), float(bc),
                            float(ic), float(oc), is_interior, SIG_TAG, now,
                            None if dsc is None else float(dsc),
                            None if dem is None else float(dem)))

    for attempt in range(5):
        try:
            conn.execute("BEGIN IMMEDIATE")
            if emb_payload:
                conn.executemany(
                    "INSERT OR IGNORE INTO image_embeddings "
                    "(url, model, dim, dtype, embedding, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    emb_payload,
                )
            if lux_payload:
                conn.executemany(
                    "INSERT OR REPLACE INTO image_luxury_score_v2 "
                    "(url, property_id, score, luxury_cos, budget_cos, indoor_cos, "
                    " outdoor_cos, is_interior, model_tag, computed_at, "
                    " decor_score, decor_empty) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    lux_payload,
                )
            conn.execute("COMMIT")
            return
        except sqlite3.OperationalError as e:
            try: conn.execute("ROLLBACK")
            except sqlite3.OperationalError: pass
            if "locked" in str(e).lower() and attempt < 4:
                wait = 2 ** attempt
                print(f"[{_now()}] write_batch lock contention, retry "
                      f"{attempt+1}/5 in {wait}s ({e})", flush=True)
                time.sleep(wait)
                continue
            if STOP:
                print(f"[{_now()}] shutdown race tolerated; exiting clean", flush=True)
                sys.exit(0)
            raise
        except Exception:
            try: conn.execute("ROLLBACK")
            except sqlite3.OperationalError: pass
            if STOP:
                print(f"[{_now()}] shutdown race tolerated; exiting clean", flush=True)
                sys.exit(0)
            raise


# ============ main loop ============
def main() -> int:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT,  _handle_sigterm)
    if DINO_ENABLED and FAISS_DELTA_ENABLED:
        _load_delta_indices()
        threading.Thread(target=_flush_loop, daemon=True, name="delta-flush").start()

    print(f"[{_now()}] combined daemon start pid={os.getpid()} "
          f"gpu_batch={GPU_BATCH} large_batch={LARGE_BATCH} "
          f"workers={NUM_WORKERS} prefetch={PREFETCH}", flush=True)

    t_start = time.time()
    n_dino = n_sig = n_err = 0
    batch_idx = 0

    # None = watermark not yet established -> first pass does one full scan.
    # Deliberately in-memory, not persisted: a model-tag change (DINO retired
    # 06-28, resumed 09-01) needs a restart anyway, and the restart is what
    # forces the full rescan that makes the backlog visible again.
    watermark: str | None = None

    while not STOP:
        hw = probe_watermark()
        if watermark is not None and hw == watermark:
            # Nothing entered status='done' since the last confirmed-empty
            # scan, so the anti-join cannot have anything to return. Skip it.
            time.sleep(DELAY_IDLE_S)
            continue

        rows = pull_batch(LARGE_BATCH,
                          since=_watermark_floor(watermark) if watermark else None)
        if not rows:
            # Confirmed: everything completed at or before hw is embedded.
            watermark = hw
            time.sleep(DELAY_IDLE_S)
            continue

        # Lookup: url → (pid, url_idx, caption, needs_dino, needs_siglip)
        meta_by_url = {
            r[0]: (r[2], r[3], r[4], r[5], r[6]) for r in rows
        }

        ds = EmbedDataset(rows)
        loader = DataLoader(
            ds, batch_size=GPU_BATCH, num_workers=NUM_WORKERS,
            prefetch_factor=PREFETCH, collate_fn=_collate,
            pin_memory=False, persistent_workers=False,
        )

        try:
            for urls, dino_pixels, sig_pixels, oks in loader:
                if STOP: break

                # Split the batch by what each row needs. Decode failure
                # (ok=False) excludes the row from both models.
                dino_idx = []
                sig_idx  = []
                for j, (u, ok) in enumerate(zip(urls, oks)):
                    if not ok:
                        n_err += 1
                        # Mark the row so we don't keep re-claiming it.
                        # Image is corrupt or truncated; can't be embedded.
                        try:
                            conn.execute(
                                "UPDATE images SET status='embed_unreadable', "
                                "error=COALESCE(error, ?) WHERE url=?",
                                ('PIL: cannot decode', u),
                            )
                        except Exception:
                            pass
                        continue
                    meta = meta_by_url.get(u)
                    if meta is None:
                        continue
                    _, _, _, needs_d, needs_s = meta
                    if needs_d: dino_idx.append(j)
                    if needs_s: sig_idx.append(j)

                dino_rows = []
                sig_rows  = []
                dino_delta_payload = []

                # ----- DINO forward on subset -----
                if dino_idx and DINO_MODEL is not None:
                    pix_sub = dino_pixels[dino_idx]
                    base_np, v10_np = dino_forward(pix_sub)
                    for k, j in enumerate(dino_idx):
                        u = urls[j]
                        v10 = v10_np[k] if v10_np is not None else None
                        dino_rows.append((u, base_np[k], v10))
                        meta = meta_by_url.get(u)
                        if meta is not None:
                            dino_delta_payload.append((u, base_np[k], v10,
                                                       meta[0], meta[1], meta[2]))

                # ----- SigLIP forward on subset -----
                if sig_idx:
                    pix_sub = sig_pixels[sig_idx]
                    vecs_128, full_feats = siglip_forward(pix_sub)
                    lux, bud, ind, oud = _score_batch(vecs_128)
                    # Decor margin is best-effort: any failure here must never
                    # block the embedding / luxury write (guarded → None).
                    try:
                        decor, dempty = _score_decor(full_feats)
                    except Exception as e:
                        print(f"[{_now()}] decor scoring failed (skip): {e}", flush=True)
                        decor = dempty = None
                    for k, j in enumerate(sig_idx):
                        u = urls[j]
                        meta = meta_by_url.get(u)
                        pid = meta[0] if meta else ""
                        sig_rows.append((u, pid, vecs_128[k],
                                         lux[k], bud[k], ind[k], oud[k],
                                         None if decor is None else decor[k],
                                         None if dempty is None else dempty[k]))

                write_batch(dino_rows, sig_rows)

                # FAISS delta — DINO only (active listings)
                try:
                    _delta_add(dino_delta_payload)
                except Exception as e:
                    print(f"[{_now()}] delta_add failed (skip): {e}", flush=True)

                n_dino += len(dino_rows)
                n_sig  += len(sig_rows)
                batch_idx += 1

                if batch_idx % 25 == 0:
                    elapsed = time.time() - t_start
                    total = n_dino + n_sig
                    print(f"[{_now()}] dino={n_dino} siglip={n_sig} err={n_err} "
                          f"elapsed={elapsed:.0f}s rate={total/max(elapsed,1):.1f}/s "
                          f"(combined)", flush=True)
                if batch_idx % GC_EVERY_N_BATCHES == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
        finally:
            del loader, ds

    conn.close()
    print(f"[{_now()}] stopped. dino={n_dino} siglip={n_sig} err={n_err} "
          f"elapsed={time.time()-t_start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
