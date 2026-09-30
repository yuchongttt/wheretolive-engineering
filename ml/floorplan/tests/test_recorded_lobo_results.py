"""The recorded reader results reproduce the numbers quoted in research/compass-orientation/README.md.

Added for this extract (not in the source repo). Reads only the JSON files in reader/results/, which are
the private repo's recorded outputs with listing IDs replaced by opaque ids; no model, no images."""
import json
import statistics
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "research/compass-orientation/reader/results"


def _pooled_lobo():
    rows = []
    for t in (1, 2, 3, 4):
        rows += json.load(open(RES / f"lobo_test{t}.json"))["results"]
    return rows


def test_pooled_four_fold_lobo_matches_readme():
    rows = _pooled_lobo()
    n = len(rows)
    errs = [r["err"] for r in rows]
    spreads = [r["spread"] for r in rows]
    assert n == 233
    assert round(statistics.median(errs), 1) == 4.6
    assert round(sum(e <= 22.5 for e in errs) / n, 3) == 0.948
    assert sum(e >= 150 for e in errs) == 1                       # one 180-degree flip
    gated10 = [e for e, s in zip(errs, spreads) if s <= 10]
    assert len(gated10) == 145 and all(e <= 22.5 for e in gated10)  # 62.2% yield @ 100%
    gated15 = [e for e, s in zip(errs, spreads) if s <= 15]
    assert len(gated15) == 183 and sum(e <= 22.5 for e in gated15) == 182  # 78.5% @ 99.5%


def test_batch4_holdout_run_matches_readme():
    s = json.load(open(RES / "run0_train123_test4.json"))["summary"]
    assert s["n"] == 64 and s["flips"] == 0
    assert round(s["acc22"], 3) == 0.922
    assert s["gates"]["10"]["n"] == 45 and s["gates"]["10"]["prec22"] == 1.0
    assert s["gates"]["15"]["n"] == 53 and s["gates"]["15"]["prec22"] == 1.0


def test_folds_are_disjoint_and_never_trained_on_their_test_batch():
    seen = set()
    for t in (1, 2, 3, 4):
        d = json.load(open(RES / f"lobo_test{t}.json"))
        assert d["args"]["test"] == str(t) and str(t) not in d["args"]["train"].split(",")
        ids = {r["pid"] for r in d["results"]}
        assert not ids & seen
        seen |= ids
