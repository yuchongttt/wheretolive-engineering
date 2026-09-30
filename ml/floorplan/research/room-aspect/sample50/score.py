#!/usr/bin/env python3
"""P2 spike scoring: predictions (predictions_<route>.jsonl) vs ground truth (truth.jsonl); rooms are
matched, then their "set of window sides" is compared.
Room matching: within the same pid, exact match on the normalised label (lower-case, punctuation and
dimensions removed, numeric suffix kept); failing that, same type and unique.
Metrics (only ground-truth rooms with confidence=high):
  room_recall    = matched truth rooms / truth rooms
  side_precision = correct predicted sides / all predicted sides   <- product precision (every
                   "this room faces south" we publish must be right)
  side_recall    = truth sides that were predicted / all truth sides
  set_exact      = rooms whose side sets match exactly / matched rooms
  yield          = rooms with a non-empty, all-correct prediction / truth rooms with windows
Usage: score.py a   (reads predictions_a.jsonl)"""
import json, re, sys
from collections import Counter
from pathlib import Path
B = Path(__file__).parent
route = sys.argv[1] if len(sys.argv) > 1 else "a"
APERTURES = "--apertures" in sys.argv   # truth = windows ∪ glazed_doors (route H v0 does not separate external windows from doors)
HIGH_ONLY = "--high-only" in sys.argv    # only trust rooms the route itself marks confidence=high (others count as abstentions: empty prediction set)
SIDES = ("top", "right", "bottom", "left")


def norm(label: str) -> str:
    s = (label or "").lower()
    s = re.sub(r"\d+['′\"″.]?\s*\d*\s*[x×]\s*\d+.*$", "", s)          # strip dimensions
    s = re.sub(r"[^a-z0-9 /]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load(path):
    out = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line); out[r["pid"]] = r
    return out


truth = load(B / "truth.jsonl")
if "+" in route:   # ensemble: route X+Y -> intersection of both routes' predicted sides for the same room (emit only on agreement; abstain first)
    parts = [load(B / f"predictions_{r}.jsonl") for r in route.split("+")]
    pred = {}
    for pid in truth:
        rooms_by = []
        for pp in parts:
            rooms_by.append({norm(r.get("label", "")): r for r in pp.get(pid, {"rooms": []})["rooms"] if isinstance(r, dict)})
        merged = []
        for key, r0 in rooms_by[0].items():
            sets = [set(x for x in r0.get("windows", []) if x in SIDES)]
            ok = True
            for other in rooms_by[1:]:
                if key not in other:
                    ok = False; break
                sets.append(set(x for x in other[key].get("windows", []) if x in SIDES))
            if ok:
                merged.append(dict(r0, windows=sorted(set.intersection(*sets))))
        pred[pid] = {"rooms": merged}
else:
    pred = load(B / f"predictions_{route}.jsonl")
tot = Counter(); unmatched = []; wrong_sides = []
for pid, t in truth.items():
    p = pred.get(pid, {"rooms": []})
    prooms = [dict(r, _n=norm(r.get("label", ""))) for r in p["rooms"] if isinstance(r, dict)]
    used = set()
    for tr in t["rooms"]:
        if tr.get("confidence", "high") != "high":
            tot["truth_rooms_low"] += 1; continue
        tot["truth_rooms"] += 1
        tn = norm(tr["label"]); cand = [i for i, r in enumerate(prooms) if i not in used and r["_n"] == tn]
        if not cand:
            same = [i for i, r in enumerate(prooms) if i not in used and r.get("type") == tr.get("type")]
            cand = same if len(same) == 1 else []
        if not cand:
            unmatched.append((pid, tr["label"])); continue
        i = cand[0]; used.add(i); pr = prooms[i]; tot["matched"] += 1
        tw = list(tr.get("windows", [])) + (list(tr.get("glazed_doors", [])) if APERTURES else [])
        ts = set(s for s in tw if s in SIDES); ps = set(s for s in pr.get("windows", []) if s in SIDES)
        if HIGH_ONLY and pr.get("confidence") != "high":
            ps = set()
        tot["truth_sides"] += len(ts); tot["pred_sides"] += len(ps); tot["side_correct"] += len(ts & ps)
        tot["set_exact"] += ts == ps
        if ts: tot["truth_rooms_with_win"] += 1
        if ps and ps <= ts: tot["room_all_correct_nonempty"] += 1
        if ps - ts: wrong_sides.append((pid, tr["label"], sorted(ts), sorted(ps)))
n = max(1, tot["truth_rooms"]); m = max(1, tot["matched"])
print(f"route={route}{' apertures' if APERTURES else ''}{' high-only' if HIGH_ONLY else ''} truth_rooms={tot['truth_rooms']} (low-conf skipped {tot['truth_rooms_low']}) matched={tot['matched']} room_recall={tot['matched']/n:.3f}")
print(f"  side_precision={tot['side_correct']/max(1,tot['pred_sides']):.3f} ({tot['side_correct']}/{tot['pred_sides']})  side_recall={tot['side_correct']/max(1,tot['truth_sides']):.3f}  set_exact={tot['set_exact']/m:.3f}")
print(f"  yield(rooms with non-empty all-correct prediction / truth rooms with windows)={tot['room_all_correct_nonempty']/max(1,tot['truth_rooms_with_win']):.3f} ({tot['room_all_correct_nonempty']}/{tot['truth_rooms_with_win']})")
print("  unmatched truth rooms:", len(unmatched), unmatched[:8])
print("  wrong-side examples:", wrong_sides[:10])
