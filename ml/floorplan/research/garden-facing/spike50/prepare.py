#!/usr/bin/env python3
"""Derived garden-facing spike, sampling: 50 floorplans for sale where the north lane has emitted a
reading (emitted=1; theta was 100% on the fifth double-blind batch) and the floorplan VLM flagged an
outdoor space (has_garden/has_terrace/has_balcony); seeded.

Writes sample.tsv (pid, idx, url, north_deg, north_8way, outdoor_flags, self_reported_garden_facing).
The self_reported column is only used afterwards as a one-sided falsifier: it is not ground truth and
plays no part in sampling (the self-reported set is 100% south/west-ish, so using it for acceptance
would systematically overstate accuracy).

Extract note: the source script also downloaded each floorplan into imgs/ through the private
image-fetch layer; that step is removed here and imgs/ must be provided separately.
"""
import random, sqlite3, sys
from pathlib import Path

B = Path(__file__).parent
REPO = B.parents[2]
sys.path.insert(0, str(REPO / "scripts"))

c = sqlite3.connect(f"file:{REPO / 'data/evaluations.db'}?mode=ro", uri=True)
c.row_factory = sqlite3.Row
rows = c.execute("""
    SELECT n.property_id, n.idx, n.url, n.north_deg, n.north_8way,
           v.has_garden, v.has_terrace, v.has_balcony, o.garden_facing
    FROM floorplan_north n
    JOIN rm_sales_overview o ON o.id = n.property_id
    JOIN floorplan_vlm_results v ON v.rm_uuid = n.property_id AND v.ok = 1
    WHERE n.emitted = 1 AND n.model_version = 'cnn-v1' AND o.delisted_date IS NULL
      AND (v.has_garden = 1 OR v.has_terrace = 1 OR v.has_balcony = 1)
    ORDER BY n.property_id, n.idx""").fetchall()

first = {}
for r in rows:
    first.setdefault(r["property_id"], r)      # one emitted floorplan per property (the first)
pool = list(first.values())
random.Random(20260903).shuffle(pool)
pick = pool[:50]
print("pool", len(pool), "picked", len(pick),
      "self-reported facing", sum(1 for r in pick if r["garden_facing"]))

with open(B / "sample.tsv", "w") as f:
    f.write("pid\tidx\turl\tnorth_deg\tnorth_8way\toutdoor\tself_reported\n")
    for r in pick:
        flags = ",".join(k for k, v in (("garden", r["has_garden"]), ("terrace", r["has_terrace"]),
                                        ("balcony", r["has_balcony"])) if v)
        f.write(f"{r['property_id']}\t{r['idx']}\t{r['url']}\t{r['north_deg']:.1f}\t"
                f"{r['north_8way']}\t{flags}\t{r['garden_facing'] or ''}\n")
