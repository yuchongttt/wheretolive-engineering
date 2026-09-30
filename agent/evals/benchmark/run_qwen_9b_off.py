#!/usr/bin/env python3
"""Run the small (10) or full (50) curated set on Qwen3.5:9b with thinking OFF.

Output:
  --small: data/benchmark/qwen_baseline_small_9b_off.jsonl (10 queries, ~4 min)
  --full:  data/benchmark/qwen_baseline_full_9b_off.jsonl  (50 queries, ~20 min)

This is the candidate production mode the team is considering. Companion
to qwen_baseline_*.jsonl (9b ON) — same prompt, only `think: false` and
num_predict tightened to 800.
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

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.5:9b"

SMALL_SET_IDS = [
    "curated_01", "curated_03", "curated_02", "curated_05", "curated_07",
    "curated_50", "curated_36", "curated_21", "curated_31", "curated_40",
]

SYSTEM_PROMPT = (
    "You are a UK property assistant for the wheretolive.xyz app. "
    "Your job is to help users find and evaluate properties in London. "
    "Match the user's query language (Chinese ↔ English). "
    "Keep answers under 220 words, use markdown bullet lists for property listings. "
    "When you don't have enough information, say so honestly rather than guessing."
)

# Inject prior context for property follow-up queries (matches gold pipeline).
PROPERTY_PRIOR = (
    "[turn 1 context — user shared this listing]\n"
    "<development>, Poplar, London E14 · 2-bed flat · £585,000 · 788 sqft · "  # anchor A1, address redacted
    "leasehold 995y. Riverside, near Canning Town tube.\n\n"
    "[turn 2]\n"
)
AREA_PRIOR = (
    "[turn 1 context — user mentioned this area]\n"
    "User is looking at properties in Walthamstow E17.\n\n"
    "[turn 2]\n"
)


def needs_property_prior(query: str) -> bool:
    return any(kw in query for kw in ("这个房子", "这套房", "这个房产", "this property", "this house"))


def needs_area_prior(query: str) -> bool:
    return any(kw in query for kw in ("这个区", "这片区", "this area", "这个邮编", "this postcode"))


def call(query: str, intent_subtype: str | None) -> dict:
    if intent_subtype == "property":
        prompt = PROPERTY_PRIOR + query
    elif intent_subtype == "area" and needs_area_prior(query):
        prompt = AREA_PRIOR + query
    else:
        prompt = query

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.7, "num_predict": 800},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urlreq.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urlreq.urlopen(req, timeout=120) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        return {
            "ok": True,
            "response": d.get("response", ""),
            "wall_s": round(time.time() - start, 2),
            "eval_count": d.get("eval_count", 0),
            "eval_ns": d.get("eval_duration", 0),
            "prompt_eval_count": d.get("prompt_eval_count", 0),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "wall_s": round(time.time() - start, 2)}


def load_queries(full: bool) -> list[dict]:
    decisions = json.loads(REVIEW_PATH.read_text())
    by_id = {}
    for line in QUERIES_PATH.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        by_id[rec["candidate_id"]] = rec
    out = []
    if full:
        # All kept queries from the curated set, in candidate_id order.
        keep_ids = sorted(
            cid for cid, dec in decisions.items()
            if dec.get("status") == "kept" and cid in by_id
        )
    else:
        keep_ids = SMALL_SET_IDS
    for cid in keep_ids:
        rec = by_id.get(cid)
        dec = decisions.get(cid, {})
        if not rec:
            continue
        out.append({
            "candidate_id": cid,
            "query": rec["primary_query"],
            "query_class": dec.get("final_class") or rec.get("filter_class"),
            "intent_subtype": rec.get("intent_subtype"),
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--small", action="store_true", help="10-query small set (~4 min)")
    p.add_argument("--full", action="store_true", help="full 50-query set (~20 min)")
    args = p.parse_args()
    if not args.small and not args.full:
        print("Pick --small or --full", file=sys.stderr)
        return 1

    set_label = "full" if args.full else "small"
    output_path = DATA_DIR / f"qwen_baseline_{set_label}_9b_off.jsonl"
    queries = load_queries(args.full)
    print(f"Running {len(queries)} queries ({set_label}) through {MODEL} (thinking OFF) → {output_path.name}")
    output_path.unlink(missing_ok=True)

    t0 = time.time()
    with output_path.open("w", encoding="utf-8") as f:
        for i, q in enumerate(queries, start=1):
            print(f"[{i}/{len(queries)}] {q['candidate_id']} ({q['query_class']}/{q.get('intent_subtype','-')})", flush=True)
            r = call(q["query"], q.get("intent_subtype"))
            rec = {
                "candidate_id": q["candidate_id"],
                "query": q["query"],
                "query_class": q["query_class"],
                "intent_subtype": q.get("intent_subtype"),
                "model": MODEL,
                "set": f"{set_label}_9b_off",
                "thinking_enabled": False,
                "ran_at": datetime.now(timezone.utc).isoformat(),
                **r,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            tok_s = (rec.get("eval_count", 0) / max(1e-9, rec.get("eval_ns", 1) / 1e9)) if rec.get("eval_count") else 0
            preview = (r.get("response") or "")[:80].replace("\n", " ")
            print(f"   wall={r.get('wall_s'):.1f}s tokens={r.get('eval_count')} tok/s={tok_s:.1f}", flush=True)
            print(f"   → {preview}…", flush=True)

    print(f"\nDone in {time.time()-t0:.0f}s. Output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
