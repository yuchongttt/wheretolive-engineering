#!/usr/bin/env python3
"""Quick A/B: Qwen3.5:9b thinking ON vs OFF on 3 representative queries.

Output: data/benchmark/qwen_thinking_compare.jsonl (6 records: 3 queries × 2 modes)
Then prints a side-by-side table.
"""

from __future__ import annotations

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
SOURCE_PATH = DATA_DIR / "queries_filtered.jsonl"
OUTPUT_PATH = DATA_DIR / "qwen_thinking_compare_4b.jsonl"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.5:4b"

# 3 queries spanning class diversity
TEST_IDS = [
    "curated_01",  # filter / area_search   (£700k 2-bed)
    "curated_05",  # advice / persona       (family with kids)
    "curated_31",  # follow_up / property   (flood risk on the prior listing)
]

SYSTEM_PROMPT = (
    "You are a UK property assistant for the wheretolive.xyz app. "
    "Match the user's query language (Chinese ↔ English). "
    "Keep answers under 220 words. "
    "When you don't have enough information, say so honestly rather than guessing."
)

# For property follow-up queries, we need to inject the same prior-turn context
# the gold pipeline uses, otherwise Qwen has nothing to refer to.
PROPERTY_PRIOR = (
    "[turn 1 context — user shared this listing]\n"
    "<development>, Poplar, London E14 · 2-bed flat · £585,000 · 788 sqft · "  # anchor A1, address redacted
    "leasehold 995y. Riverside, near Canning Town tube.\n\n"
    "[turn 2]\n"
)


def call(query: str, thinking: bool) -> dict:
    prompt = (PROPERTY_PRIOR + query) if "这个房子" in query else query
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "think": thinking,
        "options": {
            "temperature": 0.7,
            "num_predict": 4096 if thinking else 800,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = urlreq.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    start = time.time()
    with urlreq.urlopen(req, timeout=600) as resp:
        d = json.loads(resp.read().decode("utf-8"))
    return {
        "thinking": thinking,
        "wall_s": round(time.time() - start, 1),
        "response": d.get("response", ""),
        "thinking_text": d.get("thinking", ""),
        "eval_count": d.get("eval_count", 0),
        "eval_ns": d.get("eval_duration", 0),
    }


def main() -> int:
    by_id = {}
    for line in SOURCE_PATH.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        by_id[rec["candidate_id"]] = rec

    OUTPUT_PATH.unlink(missing_ok=True)
    rows = []
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for cid in TEST_IDS:
            q = by_id[cid]
            print(f"\n=== {cid} | class={q['filter_class']} | {q['primary_query'][:60]} ===")
            for mode in [True, False]:
                label = "ON" if mode else "OFF"
                print(f"  thinking {label} ...", flush=True, end=" ")
                r = call(q["primary_query"], mode)
                print(f"{r['wall_s']}s, {r['eval_count']} tok, {len(r['response'])} chars answer")
                rec = {
                    "candidate_id": cid,
                    "query": q["primary_query"],
                    "query_class": q["filter_class"],
                    "model": MODEL,
                    "ran_at": datetime.now(timezone.utc).isoformat(),
                    **r,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                rows.append(rec)

    # Side-by-side print
    print("\n\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    for cid in TEST_IDS:
        on = next(r for r in rows if r["candidate_id"] == cid and r["thinking"])
        off = next(r for r in rows if r["candidate_id"] == cid and not r["thinking"])
        print(f"\n--- {cid}: {on['query'][:80]} ---")
        print(f"\n[thinking ON  | {on['wall_s']:.1f}s | {on['eval_count']} tok]")
        print(on["response"][:600])
        print(f"\n[thinking OFF | {off['wall_s']:.1f}s | {off['eval_count']} tok]")
        print(off["response"][:600])
    return 0


if __name__ == "__main__":
    sys.exit(main())
