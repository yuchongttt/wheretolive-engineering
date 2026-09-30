#!/usr/bin/env python3
"""P1 gate (fifth double-blind batch): yield = emitted / floorplans that truly have a compass;
precision = share of emitted readings with |delta| <= 22.5 degrees.
Also reports exact 8-way match (excluding north_8way NULL), emissions on floorplans with no compass
(fp_emit), and emissions on symbols the labeller flagged AMBIG (leaks past a symmetric-symbol abstain gate).
Inputs (same directory): truth_angles.tsv (pid\\tclock\\tstyle\\tevidence, written by independent blind
labelling agents), present.txt (one pid per line for floorplans with a compass), predictions_lane.jsonl
(floorplan_north rows, exported before labelling and hidden from the labellers)."""
import csv
import json
import sys
from pathlib import Path

B = Path(__file__).parent
truth = {}
for row in csv.reader(open(B / "truth_angles.tsv"), delimiter="\t"):
    if not row or row[0] == "pid":
        continue
    try:
        truth[row[0]] = (float(row[1]) * 30 % 360, "AMBIG" in "\t".join(row))
    except ValueError:
        pass
present = set(open(B / "present.txt").read().split())
pred = {}
for line in open(B / "predictions_lane.jsonl"):
    r = json.loads(line)
    pred[r["property_id"]] = r


def cd(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def bin8(a):
    return int(((a + 22.5) % 360) // 45)


emitted = [p for p in present if pred.get(p, {}).get("emitted") == 1]
correct = [p for p in emitted if p in truth and cd(pred[p]["north_deg"], truth[p][0]) <= 22.5]
wrong = [(p, pred[p]["north_deg"], truth.get(p), pred[p]["spread_deg"]) for p in emitted if p not in correct]
fp_emit = [p for p, r in pred.items() if r.get("emitted") == 1 and p not in present]
amb_emit = [p for p in emitted if truth.get(p, (0, False))[1]]   # labeller flagged AMBIG but we emitted -> leaked past the symmetric-symbol abstain gate
det_recall = sum(1 for p in present if pred.get(p, {}).get("compass_present") == 1) / max(1, len(present))
n8 = [p for p in emitted if pred[p].get("north_8way")]
ok8 = sum(1 for p in n8 if p in truth and bin8(pred[p]["north_deg"]) == bin8(truth[p][0]))
yield_ = len(emitted) / max(1, len(present))
prec = len(correct) / max(1, len(emitted))
print(f"present(truth)={len(present)} detect_recall={det_recall:.3f} emitted={len(emitted)} yield={yield_:.3f} "
      f"precision22={prec:.3f} fp_emitted_on_no_compass={len(fp_emit)} 8way_exact={ok8}/{len(n8)}")
print("wrong:", wrong)
print("fp_emit:", fp_emit)
print("emitted_on_AMBIG:", amb_emit)
gate = yield_ >= 0.5 and prec >= 0.95 and not fp_emit
print("P1 GATE:", "PASS" if gate else "FAIL")
sys.exit(0 if gate else 1)
