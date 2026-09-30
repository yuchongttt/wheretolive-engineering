#!/usr/bin/env python3
"""Run the 50 benchmark queries through base Qwen3:8B locally via Ollama.

This is the plain baseline — Qwen has no MCP tools, no DB access.
Measures: latency, token rate, response shape. Quality is expected to be poor
(hallucinated postcodes, made-up rental yields); it sets the floor that the
tool-using variants are measured against.

Output: data/benchmark/qwen_baseline.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as urlreq

# Root of the production checkout (holds data/benchmark/).
REPO_ROOT = Path(os.environ.get("WTL_ROOT", Path(__file__).resolve().parents[3]))
DATA_DIR = REPO_ROOT / "data" / "benchmark"
QUERIES_PATH = DATA_DIR / "queries_filtered.jsonl"
REVIEW_PATH = DATA_DIR / "queries_review.json"
OUTPUT_PATH = DATA_DIR / "qwen_baseline.jsonl"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.5:9b"

# 10-query "small" set: covers all 5 classes proportionally + property/area
# follow-up split. Used for fast iteration during development; the full 50 is
# reserved for milestone benchmarks.
SMALL_SET_IDS = [
    "curated_01",  # filter / area_search   (£700k 2-bed)
    "curated_03",  # filter / commute       (Canary Wharf commute)
    "curated_02",  # advice / persona       (first-time buyer)
    "curated_05",  # advice / persona       (family with kids)
    "curated_07",  # advice / purpose       (buy-to-let)
    "curated_50",  # advice / general_know  (leasehold)
    "curated_36",  # compare / type_compare (new vs old build)
    "curated_21",  # follow_up / property   (price reasonable)
    "curated_31",  # follow_up / property   (flood risk)
    "curated_40",  # follow_up / area       (neighbourhood safe?)
]

SYSTEM_PROMPT = (
    "You are a UK property assistant for the wheretolive.xyz app. "
    "Your job is to help users find and evaluate properties in London. "
    "Match the user's query language (Chinese ↔ English). "
    "Keep answers under 220 words, use markdown bullet lists for property listings. "
    "When you don't have enough information, say so honestly rather than guessing."
)

# Qwen3.5 has reasoning ("thinking") mode on by default. KEEP IT ON —
# without thinking, Qwen3.5:9b factual accuracy drops sharply (verified
# 2026-05-09: "What is E14?" → "Walthamstow" with thinking off; correct
# Isle of Dogs answer with thinking on). The trade-off is latency: ~60s
# end-to-end with thinking vs ~3s without. For benchmark purposes we
# capture the accurate-but-slow path; latency is reported separately.
NUM_PREDICT = 4096  # roomy enough for full <think> + answer


def call_ollama(query: str, timeout_s: int = 240) -> dict:
    """Call Ollama generate endpoint, return result dict."""
    payload = {
        "model": MODEL,
        "prompt": query,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": NUM_PREDICT,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = urlreq.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urlreq.urlopen(req, timeout=timeout_s) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        wall_s = time.time() - start
        return {
            "ok": True,
            "response": d.get("response", ""),
            "wall_s": round(wall_s, 2),
            "total_ns": d.get("total_duration", 0),
            "load_ns": d.get("load_duration", 0),
            "prompt_eval_count": d.get("prompt_eval_count", 0),
            "prompt_eval_ns": d.get("prompt_eval_duration", 0),
            "eval_count": d.get("eval_count", 0),
            "eval_ns": d.get("eval_duration", 0),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "wall_s": time.time() - start}


def load_kept_queries() -> list[dict]:
    decisions = json.loads(REVIEW_PATH.read_text())
    by_id = {}
    for line in QUERIES_PATH.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        by_id[rec["candidate_id"]] = rec
    out = []
    for cid, dec in decisions.items():
        if dec.get("status") != "kept" or cid not in by_id:
            continue
        out.append({
            "candidate_id": cid,
            "query": by_id[cid]["primary_query"],
            "final_class": dec.get("final_class") or by_id[cid].get("filter_class"),
            "intent_subtype": by_id[cid].get("intent_subtype"),
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--small", action="store_true", help="Run the 10-query small set (~30 min with thinking)")
    p.add_argument("--full", action="store_true", help="Run all 50 queries (~3 h with thinking)")
    args = p.parse_args()

    if not args.small and not args.full:
        print("Pick --small (10 queries, fast iteration) or --full (50 queries, milestone)", file=sys.stderr)
        return 1

    queries = load_kept_queries()
    set_label = "small"
    if args.small:
        queries = [q for q in queries if q["candidate_id"] in SMALL_SET_IDS]
        set_label = "small"
        # Re-order to match SMALL_SET_IDS order
        queries.sort(key=lambda q: SMALL_SET_IDS.index(q["candidate_id"]))
    else:
        set_label = "full"

    out_path = OUTPUT_PATH.parent / f"qwen_baseline_{set_label}.jsonl"
    print(f"Running {len(queries)} queries ({set_label}) through {MODEL} → {out_path.name}")
    out_path.unlink(missing_ok=True)

    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as f:
        for i, q in enumerate(queries, start=1):
            print(f"[{i}/{len(queries)}] {q['candidate_id']} ({q['final_class']}/{q.get('intent_subtype','-')})", flush=True)
            result = call_ollama(q["query"])
            rec = {
                "candidate_id": q["candidate_id"],
                "query": q["query"],
                "query_class": q["final_class"],
                "intent_subtype": q.get("intent_subtype"),
                "model": MODEL,
                "set": set_label,
                "system_prompt": SYSTEM_PROMPT,
                "thinking_enabled": True,
                "num_predict": NUM_PREDICT,
                "ran_at": datetime.now(timezone.utc).isoformat(),
                **result,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            tok_s = (rec.get("eval_count", 0) / max(1e-9, rec.get("eval_ns", 1) / 1e9)) if rec.get("eval_count") else 0
            preview = (result.get("response") or "")[:60].replace("\n", " ")
            print(f"   wall={result.get('wall_s'):.1f}s tokens={result.get('eval_count')} tok/s={tok_s:.1f} → {preview}…", flush=True)

    print(f"\nDone in {time.time()-t0:.0f}s. Output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
