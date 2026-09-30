#!/usr/bin/env python3
"""Fifth double-blind batch: sampling + export of the lane's predictions (never shown to labellers).
Usage: .venv/bin/python3 research/compass-orientation/batch5/prepare.py [--n 120]
Outputs: sample.tsv (pid\\tidx\\turl) and predictions_lane.jsonl (floorplan_north rows). The source
script then downloaded the original images into imgs/ for the labellers.
Sampling pool = listings still for sale, already processed by the lane (ok=1, cnn-v1), not in
batches 1-4; seeded shuffle, first n taken (one floorplan per property: the lowest idx).

Extract note: the image-download step went through the private image-fetch layer and is removed here;
imgs/ must be provided separately.
"""
import argparse
import csv
import glob
import json
import random
import sqlite3
import sys
from pathlib import Path

B = Path(__file__).parent
R = B.parent
REPO = R.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from lib import compass_north as cn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=120)
a = ap.parse_args()
used = set()
for f in glob.glob(str(R / "batch[1-4]/sample.tsv")):
    used |= {row[0] for row in csv.reader(open(f), delimiter="\t") if row}
c = sqlite3.connect(f"file:{REPO / 'data/evaluations.db'}?mode=ro", uri=True)
c.row_factory = sqlite3.Row
rows = c.execute("""SELECT n.* FROM floorplan_north n JOIN rm_sales_overview o ON o.id = n.property_id
                    WHERE o.delisted_date IS NULL AND n.ok = 1 AND n.model_version = ?
                    ORDER BY n.property_id, n.idx""", (cn.MODEL_VERSION,)).fetchall()
first = {}
for r in rows:
    if r["property_id"] not in used and r["property_id"] not in first:
        first[r["property_id"]] = r
pool = list(first.values())
random.Random(20260902).shuffle(pool)
pick = pool[: a.n]
print(f"pool {len(pool)} picked {len(pick)}")
with open(B / "sample.tsv", "w") as f:
    for r in pick:
        f.write(f"{r['property_id']}\t{r['idx']}\t{r['url']}\n")
with open(B / "predictions_lane.jsonl", "w") as f:
    for r in pick:
        f.write(json.dumps(dict(r)) + "\n")
# (image download into imgs/ removed in this extract -- see module docstring)
