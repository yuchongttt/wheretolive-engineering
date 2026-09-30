#!/usr/bin/env python3
"""Offline recall@K evaluation: compare Flat vs PQ base index.

Loads both index files, runs N random queries against each, reports
overlap rate of top-K results, mean similarity offset, and median
search latency. No FastAPI server needed.

Run on the GPU box. Exit 0 if recall@10 >= 0.90, else 2.
"""
import faiss
import numpy as np
import os
import sys
import time

FLAT_PATH = "/opt/wheretolive/dino-indexes/dino-dinov3-l16-base.stable.faiss"
PQ_PATH   = "/opt/wheretolive/dino-indexes/dino-dinov3-l16-base.stable.pq.faiss"
N_QUERIES = 100
TOP_K     = 10
SEED      = 1337
THRESHOLD = 0.90


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    if not os.path.exists(PQ_PATH):
        log(f"ERROR: PQ index missing: {PQ_PATH}")
        return 1
    log("loading Flat index")
    flat = faiss.read_index(FLAT_PATH)
    log(f"  flat ntotal={flat.ntotal:,}")
    log("loading PQ index")
    pq = faiss.read_index(PQ_PATH)
    log(f"  pq   ntotal={pq.ntotal:,}")

    # Bump efSearch on both to match server's runtime setting (64)
    for idx in (flat, pq):
        inner = idx.index if hasattr(idx, "index") else idx
        if hasattr(inner, "hnsw"):
            inner.hnsw.efSearch = 64

    # Sample N query vectors from the Flat index (ground truth source).
    # Unwrap to inner HNSWFlat — internal IDs are 0..n-1.
    flat_inner = flat.index if hasattr(flat, "index") else flat
    rng = np.random.default_rng(SEED)
    sample_ids = rng.choice(flat.ntotal, size=N_QUERIES, replace=False).astype(np.int64)
    log(f"sampling {N_QUERIES} query vectors")
    queries = np.zeros((N_QUERIES, flat.d), dtype=np.float32)
    for i, vid in enumerate(sample_ids):
        queries[i] = flat_inner.reconstruct(int(vid))

    log("running Flat queries")
    t0 = time.time()
    flat_d, flat_i = flat.search(queries, TOP_K)
    flat_latency_ms = (time.time() - t0) * 1000 / N_QUERIES

    log("running PQ queries")
    t0 = time.time()
    pq_d, pq_i = pq.search(queries, TOP_K)
    pq_latency_ms = (time.time() - t0) * 1000 / N_QUERIES

    overlaps = []
    sim_offsets = []
    for i in range(N_QUERIES):
        flat_set = set(int(x) for x in flat_i[i] if x >= 0)
        pq_set   = set(int(x) for x in pq_i[i]   if x >= 0)
        overlap = len(flat_set & pq_set) / TOP_K
        overlaps.append(overlap)
        sim_offsets.append(abs(float(flat_d[i][0]) - float(pq_d[i][0])))

    recall_at_k = float(np.mean(overlaps))
    median_overlap = float(np.median(overlaps))
    p10_overlap = float(np.percentile(overlaps, 10))
    mean_sim_off = float(np.mean(sim_offsets))

    log("=" * 60)
    log(f"N_QUERIES = {N_QUERIES}, TOP_K = {TOP_K}")
    log(f"recall@{TOP_K} (mean overlap):     {recall_at_k:.3f}")
    log(f"median overlap:                   {median_overlap:.3f}")
    log(f"p10 overlap (worst 10% queries):  {p10_overlap:.3f}")
    log(f"mean sim offset rank-0:           {mean_sim_off:.4f}")
    log(f"flat search latency / query:      {flat_latency_ms:.1f} ms")
    log(f"pq   search latency / query:      {pq_latency_ms:.1f} ms")
    log(f"flat file size:  {os.path.getsize(FLAT_PATH)/1024/1024:.0f} MB")
    log(f"pq   file size:  {os.path.getsize(PQ_PATH)/1024/1024:.0f} MB")
    log(f"compression ratio: {os.path.getsize(FLAT_PATH)/os.path.getsize(PQ_PATH):.1f}x")
    log("=" * 60)

    if recall_at_k >= THRESHOLD:
        log(f"PASS: recall@{TOP_K} {recall_at_k:.3f} >= {THRESHOLD}")
        return 0
    else:
        log(f"FAIL: recall@{TOP_K} {recall_at_k:.3f} < {THRESHOLD}")
        log("Suggested action: rebuild with PQ_M=128 in build_base_pq_index.py")
        return 2


if __name__ == "__main__":
    sys.exit(main())
