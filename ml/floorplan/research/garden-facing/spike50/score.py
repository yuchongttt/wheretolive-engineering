#!/usr/bin/env python3
"""Score predictions_g.jsonl.

Definitions:
- precision = share of emitted images whose predicted side is in the ground-truth primary_side set
- yield = emitted images / images with ground truth
- **constant baseline**: 88% of ground truth is `top` (UK floorplans conventionally draw the rear garden
  at the top of the page), so every route must first be compared with "always guess top"; skipping that
  comparison passes class imbalance off as skill.
- **non-top recall**: of the images whose truth does not include top, how many were emitted and
  correct -- the only incremental value this route can add.
Gate variants: eq4 (all four frames agree) / maj3 (>=3 frames agree, take the majority) / raw (unrotated
frame only, no gate).
"""
import json, sys
from collections import Counter
from pathlib import Path

B = Path(__file__).parent
truth = {r["pid"]: r for r in map(json.loads, open(B / "truth.jsonl"))}
preds = {r["pid"]: r for r in map(json.loads, open(B / "predictions_g.jsonl"))}
pids = [p for p in truth if p in preds]


def gated(frames, key, mode):
    vals = [f[key] for f in frames]
    if mode == "raw":
        return vals[0]
    good = [v for v in vals if v]
    if mode == "eq4":
        return good[0] if len(good) == 4 and len(set(good)) == 1 else None
    if mode == "maj3":                                  # >=3 frames give the same value
        if not good:
            return None
        v, n = Counter(good).most_common(1)[0]
        return v if n >= 3 else None
    raise ValueError(mode)


def report(key, mode):
    hit = emitted = 0
    nontop_hit = nontop_emitted = 0
    nontop_total = sum(1 for p in pids if "top" not in truth[p]["primary_side"])
    wrong = []
    for p in pids:
        t = truth[p]["primary_side"]
        s = gated(preds[p]["frames"], key, mode)
        is_nontop = "top" not in t
        if s is None:
            continue
        emitted += 1
        ok = s in t
        hit += ok
        if is_nontop:
            nontop_emitted += 1
            nontop_hit += ok
        if not ok:
            wrong.append((p, s, t))
    prec = hit / emitted if emitted else 0.0
    print(f"  {key} · {mode:5s}  yield {emitted:2d}/{len(pids)} = {emitted/len(pids):4.0%}   "
          f"precision {hit:2d}/{emitted:2d} = {prec:4.0%}   "
          f"non-top {nontop_hit}/{nontop_emitted} emitted (of {nontop_total} images)")
    return wrong


if __name__ == "__main__":
    base = sum(1 for p in pids if "top" in truth[p]["primary_side"])
    print(f"n={len(pids)}  constant baseline 'always guess top' precision {base}/{len(pids)} = {base/len(pids):.0%} @ 100% yield\n")
    allwrong = {}
    for key in ("side_g1", "side_g2"):
        for mode in ("raw", "maj3", "eq4"):
            allwrong[(key, mode)] = report(key, mode)
        print()
    if "-v" in sys.argv:
        print("eq4 errors:")
        for k in ("side_g1", "side_g2"):
            for p, s, t in allwrong[(k, "eq4")]:
                print(f"  {k} {p}: predicted {s} truth {t}")
