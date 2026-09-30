#!/usr/bin/env python3
"""Fit a 768→128 PCA projection on SigLIP-2 image features.

Samples N=10000 random images from the production downloader pool
(status='done'), runs SigLIP-2 base to get 768d features, then PCA-fits
a 128-component projection. Saves the mean + components to disk; the
embed daemon loads this once at startup and projects every new image
through it.

The reason for PCA over a trained projection head: SigLIP's text-image
joint space is already well-organised — top principal components
naturally capture the axes that matter (luxury vs budget, indoor vs
outdoor, modern vs traditional). A trained head would need labels we
don't have. PCA at 128d should retain >90% of the variance.

Output: /data/ml/dataset/siglip_pca_128.npz with:
  - mean (768,)        : centering offset
  - components (128, 768): rotation matrix; project as `(x - mean) @ comp.T`
  - explained_variance_ratio (128,): for sanity (sum should be ≥0.9)
  - sample_n, fit_time, sample_seed: provenance

Usage: run once on the GPU box. About 8-15 min on RTX 3060.
"""
from __future__ import annotations

import os, sys, sqlite3, time, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor
from PIL import Image
from sklearn.decomposition import PCA

DB         = "/data/ml/dataset/dataset.db"
CACHE      = "/data/ml/hf-cache"
SIGLIP     = "google/siglip2-base-patch16-256"
DEVICE     = "cuda"
DTYPE      = torch.bfloat16
N_COMPS    = 128
SAMPLE_N   = 10_000
SEED       = 42
BATCH_SIZE = 32
NUM_WORKERS = 4
OUT_PATH   = "/data/ml/dataset/siglip_pca_128.npz"


def sample_image_paths(n: int, seed: int) -> list[tuple[str, str]]:
    """Return [(url, local_path), ...] for n random images. Uses
    OFFSET-style sampling — RANDOM() on 1.5M rows would full-scan but
    we have idx_images_status so this is bounded."""
    conn = sqlite3.connect(DB, timeout=60)
    conn.execute("PRAGMA query_only = 1")
    # `RANDOM()` is fine here; one-off cost amortised over the fit.
    # The seed gives reproducibility — rerun with same seed picks same rows.
    conn.execute(f"SELECT setseed({seed * 1.0 / 1e9})") if False else None
    rows = conn.execute("""
        SELECT url, local_path FROM images
        WHERE status = 'done' AND local_path IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """, (n,)).fetchall()
    conn.close()
    return rows


class _IDS(Dataset):
    def __init__(self, rows, processor):
        self.rows = rows
        self.proc = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        url, lp = self.rows[i]
        try:
            im = Image.open(lp).convert("RGB")
            inp = self.proc(images=im, return_tensors="pt")
            return inp["pixel_values"][0], True
        except Exception:
            return torch.zeros(3, 256, 256), False


def _collate(batch):
    pixels, oks = zip(*batch)
    return torch.stack(list(pixels)), list(oks)


def main() -> int:
    random.seed(SEED)

    print(f"[plan] sampling {SAMPLE_N:,} images (seed={SEED})", flush=True)
    t0 = time.time()
    rows = sample_image_paths(SAMPLE_N, SEED)
    print(f"[plan] sampled {len(rows):,} in {time.time()-t0:.1f}s", flush=True)

    os.environ["HF_HOME"] = CACHE
    print(f"[init] loading {SIGLIP}", flush=True)
    proc = AutoProcessor.from_pretrained(SIGLIP, cache_dir=CACHE)
    model = AutoModel.from_pretrained(SIGLIP, torch_dtype=DTYPE, cache_dir=CACHE).to(DEVICE).eval()

    feats = np.empty((len(rows), 768), dtype=np.float32)
    write_i = 0
    n_ok = 0
    n_err = 0
    t_extract = time.time()

    ds = _IDS(rows, proc)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                        prefetch_factor=2, collate_fn=_collate, persistent_workers=False)

    with torch.no_grad():
        for batch_idx, (pixels, oks) in enumerate(loader):
            pixels = pixels.to(DEVICE, dtype=DTYPE, non_blocking=True)
            out = model.get_image_features(pixel_values=pixels)
            # Some SigLIP wrappers return ImageModelOutput with pooler_output;
            # get_image_features should give the projected features directly.
            vec = out if not hasattr(out, "pooler_output") else (out.pooler_output if out.pooler_output is not None else out)
            vec = F.normalize(vec.float(), dim=-1).cpu().numpy()
            for j, ok in enumerate(oks):
                if ok:
                    feats[write_i] = vec[j]
                    write_i += 1
                    n_ok += 1
                else:
                    n_err += 1
            if (batch_idx + 1) % 25 == 0:
                done = (batch_idx + 1) * BATCH_SIZE
                rate = done / max(time.time() - t_extract, 0.01)
                print(f"  [{done}/{len(rows)}] ok={n_ok} err={n_err} rate={rate:.0f}/s", flush=True)

    feats = feats[:write_i]
    print(f"[extract] got {write_i:,} valid features (err={n_err}) in {time.time()-t_extract:.1f}s", flush=True)

    print(f"[pca] fitting PCA({N_COMPS}) on {feats.shape}", flush=True)
    t_pca = time.time()
    pca = PCA(n_components=N_COMPS, svd_solver="full")
    pca.fit(feats)
    cum_var = float(pca.explained_variance_ratio_.sum())
    print(f"[pca] fit in {time.time()-t_pca:.1f}s · cumulative variance: {cum_var*100:.1f}%", flush=True)
    print(f"[pca] top-10 component variance: {[round(float(x), 3) for x in pca.explained_variance_ratio_[:10]]}", flush=True)

    out = {
        "mean": pca.mean_.astype(np.float32),
        "components": pca.components_.astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
        "sample_n": np.array(write_i, dtype=np.int64),
        "fit_seconds": np.array(time.time() - t0, dtype=np.float32),
        "sample_seed": np.array(SEED, dtype=np.int64),
        "siglip_model": np.array(SIGLIP, dtype=object),
    }
    np.savez(OUT_PATH, **out)
    print(f"[done] saved {OUT_PATH}  ({Path(OUT_PATH).stat().st_size/1024:.0f} KB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
