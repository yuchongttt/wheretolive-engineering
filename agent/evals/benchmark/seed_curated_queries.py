#!/usr/bin/env python3
"""Seed the 50 hand-curated benchmark queries.

Replaces the previous set mined from public forum posts (which was off-target).
The new queries are written by the project owner to reflect the real product
research direction: area recommendation + property/area follow-up evaluation.
The queries are in Chinese on purpose (the product's main audience); they are
benchmark inputs and are kept verbatim.

Output:
  data/benchmark/queries_filtered.jsonl   (50 records)
  data/benchmark/queries_review.json      (all 50 marked kept with class)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Root of the production checkout (holds data/benchmark/).
REPO_ROOT = Path(os.environ.get("WTL_ROOT", Path(__file__).resolve().parents[3]))
DATA_DIR = REPO_ROOT / "data" / "benchmark"
QUERIES_PATH = DATA_DIR / "queries_filtered.jsonl"
REVIEW_PATH = DATA_DIR / "queries_review.json"


# (id_suffix, query, class, intent_subtype)
# intent_subtype is metadata for prior-context selection in build_gold_answers.py
CURATED = [
    # ─────────── filter (7) — area-finding with concrete constraints ───────────
    ("01",  "预算 £700k，在 London 哪些区域适合买两居室？", "filter", "area_search"),
    ("03",  "哪些区域最适合在 Canary Wharf 上班的人居住？", "filter", "commute"),
    ("08",  "哪些 postcode 的 rental yield 最高？", "filter", "yield_rank"),
    ("11",  "哪些区域最适合骑车通勤？", "filter", "commute"),
    ("16",  "哪些 London 区域通勤 City 最方便？", "filter", "commute"),
    ("17",  "哪些区域适合预算有限但想住 Zone 2 的人？", "filter", "zone_budget"),
    ("20",  "哪些区域最适合购买 Victorian house？", "filter", "property_type"),

    # ─────────── advice (18) — recommendation / persona-based ───────────
    ("02",  "哪些 London 区域适合首次购房者？", "advice", "persona"),
    ("04",  "哪些 London 区域最适合年轻程序员？", "advice", "persona"),
    ("05",  "哪些区域适合有小孩的家庭？", "advice", "persona"),
    ("06",  "哪些 London 区域最适合买来自住？", "advice", "purpose"),
    ("07",  "哪些区域最适合 buy-to-let 投资？", "advice", "purpose"),
    ("09",  "哪些 London 区域未来升值潜力最大？", "advice", "trend"),
    ("10",  "哪些区域最被低估？", "advice", "trend"),
    ("12",  "哪些区域夜生活最好？", "advice", "lifestyle"),
    ("13",  "哪些区域最安静宜居？", "advice", "lifestyle"),
    ("14",  "哪些区域最适合 remote worker？", "advice", "persona"),
    ("15",  "哪些区域最适合单身女生居住？", "advice", "persona"),
    ("18",  "哪些区域更适合长期持有投资？", "advice", "purpose"),
    ("19",  "哪些区域有明显 gentrification 趋势？", "advice", "trend"),
    ("35",  "lease 剩余 85 年值得买吗？", "advice", "general_principle"),
    ("37",  "靠近地铁站的房子更保值吗？", "advice", "general_principle"),
    ("49",  "英国买房流程是什么？", "advice", "general_knowledge"),
    ("50",  "买 leasehold 需要重点注意什么？", "advice", "general_knowledge"),

    # ─────────── compare (1) ───────────
    ("36",  "新房和老房子哪个更值得买？", "compare", "type_compare"),

    # ─────────── follow_up: property-specific (15) ───────────
    ("21",  "这个房子的挂牌价合理吗？", "follow_up", "property_eval"),
    ("22",  "这个 listing 是否 overpriced？", "follow_up", "property_eval"),
    ("23",  "这个房子适合投资吗？", "follow_up", "property_eval"),
    ("24",  "这个房子适合长期自住吗？", "follow_up", "property_eval"),
    ("25",  "这个房子的 rental yield 大概是多少？", "follow_up", "property_eval"),
    ("26",  "这个房子未来转手容易吗？", "follow_up", "property_eval"),
    ("27",  "这个 listing 最大的问题是什么？", "follow_up", "property_eval"),
    ("28",  "这个房子的户型合理吗？", "follow_up", "property_eval"),
    ("29",  "这个房子的采光怎么样？", "follow_up", "property_eval"),
    ("30",  "这个房子会不会很吵？", "follow_up", "property_eval"),
    ("31",  "这个房子有 flood risk 吗？", "follow_up", "property_eval"),
    ("32",  "这个房子会受铁路噪音影响吗？", "follow_up", "property_eval"),
    ("33",  "这个房子的 EPC rating 好吗？", "follow_up", "property_eval"),
    ("34",  "这个 flat 的 service charge 合理吗？", "follow_up", "property_eval"),
    ("38",  "学区会影响这个房子的价格吗？", "follow_up", "property_eval"),

    # ─────────── follow_up: area-specific (9) ───────────
    ("39",  "这个 neighborhood 更适合年轻人还是家庭？", "follow_up", "area_eval"),
    ("40",  "这个 neighborhood 安全吗？", "follow_up", "area_eval"),
    ("41",  "这个区域租房需求强吗？", "follow_up", "area_eval"),
    ("42",  "这个 neighborhood 的社区氛围怎么样？", "follow_up", "area_eval"),
    ("43",  "这个区域未来会有 redevelopment 吗？", "follow_up", "area_eval"),
    ("44",  "这个区域更偏 owner-occupier 还是 renter？", "follow_up", "area_eval"),
    ("45",  "这个区域未来会不会越来越拥挤？", "follow_up", "area_eval"),
    ("46",  "这个区域的房价增长趋势怎么样？", "follow_up", "area_eval"),
    ("47",  "这个区域适合做 HMO 吗？", "follow_up", "area_eval"),
    ("48",  "Article 4 Direction 会影响这个区域吗？", "follow_up", "area_eval"),
]

assert len(CURATED) == 50, f"expected 50 records, got {len(CURATED)}"


def main() -> None:
    # Wipe old artifacts (the forum-mined set was rejected)
    for old in ["queries_filtered.jsonl", "queries_review.json", "forum_candidates.jsonl",
                "gold_answers.jsonl", "gold_edits.json"]:
        p = DATA_DIR / old
        if p.exists():
            p.unlink()
            print(f"removed {p.name}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    # Write queries_filtered.jsonl
    with QUERIES_PATH.open("w", encoding="utf-8") as f:
        for suffix, query, klass, subtype in CURATED:
            cid = f"curated_{suffix}"
            rec = {
                "candidate_id": cid,
                "primary_query": query,
                "title": query,
                "source": "curated",
                "source_url": "",
                "raw_body": "",
                "search_term": "curated",
                "search_class_hint": klass,
                "filter_class": klass,
                "filter_confidence": 1.0,
                "intent_subtype": subtype,
                "lang": "zh",
                "mined_at": now,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write queries_review.json — auto-mark all 50 as kept
    review = {
        f"curated_{suffix}": {
            "status": "kept",
            "final_class": klass,
            "decided_at": now,
        }
        for suffix, _, klass, _ in CURATED
    }
    REVIEW_PATH.write_text(json.dumps(review, indent=2), encoding="utf-8")

    # Print summary
    from collections import Counter
    by_class = Counter(klass for _, _, klass, _ in CURATED)
    by_subtype = Counter(subtype for _, _, _, subtype in CURATED)
    print(f"\nWrote {QUERIES_PATH.name}: 50 records")
    print(f"Wrote {REVIEW_PATH.name}: 50 kept entries")
    print(f"By class: {dict(by_class)}")
    print(f"By subtype: {dict(by_subtype)}")


if __name__ == "__main__":
    main()
