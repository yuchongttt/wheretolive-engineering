#!/usr/bin/env python3
"""Build a scalar-quantized HNSW index from the existing flat base index.

Reads:   /opt/wheretolive/dino-indexes/dino-dinov3-l16-base.stable.faiss
Writes:  /opt/wheretolive/dino-indexes/dino-dinov3-l16-base.stable.pq.faiss

NOTE on naming: file kept as ".stable.pq.faiss" for backwards compat
with the server flag (DINO_BASE_USE_PQ). Actual quantization is now
8-bit scalar (SQ8) not Product Quantization — HNSWPQ does not
correctly support METRIC_INNER_PRODUCT on L2-normalized DINO vectors
(recall collapsed to 13 % in eval). SQ8 supports IP natively and
still gives ~4x compression (5.7 GB → ~1.5 GB) with near-zero recall
loss.

The meta file (.stable.meta.npz) is shared between Flat and SQ — same
URL/pid/idx ordering, since we reconstruct vectors in-order.

Run on the GPU box as the service user. Idempotent: overwrites PQ file.
"""
import faiss
import numpy as np
import os
import sys
import time

SRC_PATH    = "/opt/wheretolive/dino-indexes/dino-dinov3-l16-base.stable.faiss"
DST_PATH    = "/opt/wheretolive/dino-indexes/dino-dinov3-l16-base.stable.pq.faiss"
META_PATH   = "/opt/wheretolive/dino-indexes/dino-dinov3-l16-base.stable.meta.npz"
SAMPLE_N    = 256_000
HNSW_M      = 32       # graph connectivity (matches existing Flat index)
HNSW_EF_C   = 200      # construction depth (matches existing pipeline)
SEED        = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    if not os.path.exists(SRC_PATH):
        log(f"ERROR: source index not found: {SRC_PATH}")
        return 1
    log(f"loading source: {SRC_PATH}")
    src = faiss.read_index(SRC_PATH)
    n_total = src.ntotal
    d = src.d
    log(f"source ntotal={n_total:,} d={d}")
    if d != 1024:
        log(f"ERROR: expected d=1024, got d={d}")
        return 1

    # Unwrap to inner index for reliable reconstruct_n. Outer is
    # IndexIDMap -> IndexHNSWFlat -> IndexFlatL2. reconstruct_n on the
    # inner HNSWFlat uses internal IDs 0..n-1; the daemon assigns
    # sequential user IDs starting at 0 so internal == user for stable.
    src_inner = src.index if hasattr(src, "index") else src

    log(f"sampling {SAMPLE_N:,} vectors for PQ training")
    rng = np.random.default_rng(SEED)
    n_take = min(SAMPLE_N, n_total)
    sample_ids = rng.choice(n_total, size=n_take, replace=False).astype(np.int64)
    sample_ids.sort()
    t0 = time.time()
    # reconstruct_n in chunks — batched C path, much faster than per-id loop
    sample = np.zeros((n_take, d), dtype=np.float32)
    # contiguous-range chunks for max speed: read CHUNK at a time, keep
    # only the sampled indices within each chunk
    CHUNK = 50_000
    keep_set = set(int(x) for x in sample_ids)
    filled = 0
    for start in range(0, n_total, CHUNK):
        end = min(start + CHUNK, n_total)
        in_chunk = [i for i in range(start, end) if i in keep_set]
        if not in_chunk:
            continue
        chunk_vecs = src_inner.reconstruct_n(start, end - start)
        offsets = [i - start for i in in_chunk]
        sample[filled:filled + len(in_chunk)] = chunk_vecs[offsets]
        filled += len(in_chunk)
    assert filled == n_take, f"sampled {filled}, expected {n_take}"
    log(f"sample materialized in {time.time()-t0:.1f}s; shape={sample.shape}")

    # Build target index: IndexHNSWSQ (8-bit scalar quantization) wrapped
    # in IndexIDMap. SQ8 supports METRIC_INNER_PRODUCT natively, unlike
    # HNSWPQ which silently degrades on L2-normalized + IP queries.
    log(f"building HNSWSQ8 HNSW_M={HNSW_M} (8-bit per dim)")
    sq_index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    sq_index.hnsw.efConstruction = HNSW_EF_C
    t0 = time.time()
    log("training SQ8 quantizer")
    sq_index.train(sample)
    log(f"SQ8 training done in {time.time()-t0:.1f}s")
    idmap = faiss.IndexIDMap(sq_index)

    chunk = 50_000
    log(f"adding {n_total:,} vectors in chunks of {chunk:,}")
    t0 = time.time()
    for start in range(0, n_total, chunk):
        end = min(start + chunk, n_total)
        vecs = src_inner.reconstruct_n(start, end - start)
        ids = np.arange(start, end, dtype=np.int64)
        idmap.add_with_ids(vecs, ids)
        elapsed = time.time() - t0
        rate = (end / elapsed) if elapsed > 0 else 0
        log(f"  added {end:,} / {n_total:,}  ({rate:.0f}/s)")
    log(f"add phase done in {time.time()-t0:.1f}s; final ntotal={idmap.ntotal:,}")

    sq_index.hnsw.efSearch = 64

    log(f"writing SQ8 index -> {DST_PATH}")
    faiss.write_index(idmap, DST_PATH)
    size_mb = os.path.getsize(DST_PATH) / 1024 / 1024
    src_mb  = os.path.getsize(SRC_PATH) / 1024 / 1024
    log(f"done. file size: {size_mb:.1f} MB (source was {src_mb:.0f} MB)")
    log(f"compression ratio: {src_mb / size_mb:.1f}x")
    log(f"meta file shared from {META_PATH} (unchanged)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
