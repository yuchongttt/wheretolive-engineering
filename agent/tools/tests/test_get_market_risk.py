"""get_market_risk tool: area-level "underwater" market risk (agent-benchmark round-01 Q3 review).

In the Q3 comparison the reference agent offered a market-risk section on "new-
build leasehold flats' fall from peak" — methodology unverifiable (multiple
mirror sources, unclear peak definition). Our DB holds a harder signal of the
same kind: active listings' asking price vs what the current owner paid at Land
Registry (the underwater analysis, caliber of the 2026-08 item-127 audit). This
tool pins it for the chat agent.

Audit caliber (every rule pinned; lesson learned: re-verify numbers with the
audit's ORIGINAL SQL):
  - trust only lr_match_strategy='detail_anchored' (street_only and other fuzzy
    matches inflate the rate by 44%)
  - canonical dedup (a home listed by several agents counts once)
  - active only (delisted_date IS NULL); drop shared_ownership / is_auction
  - purchase year anchored to the 2016-2022 cohort (Simpson guard; the window
    is stated in the output)
  - flats vs houses per radar_vocab's PT_FLAT/PT_HOUSE
  - small samples (n<10) must not quote a rate; say "too small"
"""
import asyncio
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import get_market_risk  # noqa: E402


def _db(tmp_path, rows, boroughs=(("E1", "Tower Hamlets"),)):
    path = tmp_path / "evaluations.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE rm_sales_overview (
               id INTEGER PRIMARY KEY, postcode TEXT, property_type TEXT,
               asking_price INTEGER, lr_prev_sold_price INTEGER,
               lr_prev_sold_date TEXT, lr_match_strategy TEXT,
               delisted_date TEXT, canonical_id TEXT,
               shared_ownership INTEGER DEFAULT 0, is_auction INTEGER DEFAULT 0,
               postcode_norm TEXT GENERATED ALWAYS AS
                   (upper(replace(postcode,' ',''))) VIRTUAL)"""
    )
    conn.execute("CREATE TABLE outcode_borough (outcode TEXT PRIMARY KEY, borough TEXT, share REAL)")
    for oc, b in boroughs:
        conn.execute("INSERT INTO outcode_borough VALUES (?,?,1.0)", (oc, b))
    for i, r in enumerate(rows):
        conn.execute(
            """INSERT INTO rm_sales_overview
               (id, postcode, property_type, asking_price, lr_prev_sold_price,
                lr_prev_sold_date, lr_match_strategy, delisted_date,
                canonical_id, shared_ownership, is_auction)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (r.get("id", 1000 + i), r["postcode"], r.get("property_type", "Flat"),
             r["asking_price"], r["lr_prev_sold_price"],
             r.get("lr_prev_sold_date", "2018-06-01"),
             r.get("lr_match_strategy", "detail_anchored"),
             r.get("delisted_date"), r.get("canonical_id"),
             r.get("shared_ownership", 0), r.get("is_auction", 0)),
        )
    conn.commit()
    conn.close()
    return path


def _call(**args):
    out = asyncio.run(get_market_risk.handle_call_tool("get_market_risk", args))
    return out[0].text


def _flat(pc, ask, prev, **kw):
    return dict(postcode=pc, property_type="Flat", asking_price=ask,
                lr_prev_sold_price=prev, **kw)


def _house(pc, ask, prev, **kw):
    return dict(postcode=pc, property_type="Terraced", asking_price=ask,
                lr_prev_sold_price=prev, **kw)


def _e1_fixture():
    """E1: 10 flats (6 underwater), 10 houses (1 underwater) + six kinds of noise that must be excluded."""
    rows = []
    # flats: 6 under (ask < prev), 4 over
    for i in range(6):
        rows.append(_flat("E1 6QH", 450000, 500000))   # -10%
    for i in range(4):
        rows.append(_flat("E1 6QH", 550000, 500000))
    # houses: 1 under, 9 over
    rows.append(_house("E1 6QL", 630000, 700000))
    for i in range(9):
        rows.append(_house("E1 6QL", 800000, 700000))
    # noise (all underwater — if any slips through, the rate goes up):
    rows.append(_flat("E1 6QH", 100000, 500000, delisted_date="2026-01-01"))
    rows.append(_flat("E1 6QH", 100000, 500000, id=1, canonical_id="999"))
    rows.append(_flat("E1 6QH", 100000, 500000, lr_match_strategy="street_only"))
    rows.append(_flat("E1 6QH", 100000, 500000, shared_ownership=1))
    rows.append(_flat("E1 6QH", 100000, 500000, is_auction=1))
    rows.append(_flat("E1 6QH", 100000, 500000, lr_prev_sold_date="2014-03-01"))
    return rows


def test_outcode_split_applies_all_audit_guardrails(tmp_path, monkeypatch):
    monkeypatch.setattr(get_market_risk, "DB_PATH", _db(tmp_path, _e1_fixture()))

    text = _call(outcode="E1")

    assert "60.0%" in text          # flats 6/10 — any noise getting in changes this number
    assert "10.0%" in text          # houses 1/10
    assert "n=10" in text
    assert "2016" in text and "2022" in text   # the cohort window must be stated
    # median shortfall among the underwater: all 6 are -10%
    assert "10.0% median shortfall" in text


def test_small_sample_refuses_to_quote_a_rate(tmp_path, monkeypatch):
    rows = [_flat("E1 6QH", 450000, 500000) for _ in range(3)]
    monkeypatch.setattr(get_market_risk, "DB_PATH", _db(tmp_path, rows))

    text = _call(outcode="E1")

    assert "too small" in text
    assert "33" not in text and "%" not in text.split("too small")[0].split("Flats")[-1]


def test_london_overview_excludes_non_london_outcodes(tmp_path, monkeypatch):
    rows = _e1_fixture()
    # CM2 is in outcode_borough but not a London borough — it must be excluded
    rows += [_flat("CM2 0AA", 100000, 500000) for _ in range(10)]
    monkeypatch.setattr(get_market_risk, "DB_PATH", _db(
        tmp_path, rows, boroughs=(("E1", "Tower Hamlets"), ("CM2", "Chelmsford"))))

    text = _call()

    assert "60.0%" in text                 # the London flats rate is not inflated by CM2's 10 underwater flats
    assert "CM2" not in text


def test_empty_cohort_is_honest(tmp_path, monkeypatch):
    monkeypatch.setattr(get_market_risk, "DB_PATH", _db(tmp_path, []))

    text = _call(outcode="N14")

    assert "no LR-anchored" in text.lower() or "0" in text
    assert "%" not in text


def test_seller_anchor_capability_pointer_present(tmp_path, monkeypatch):
    """R2-2 Fix S (2026-08-14): in the no-listing area due-diligence round the
    reference agent named "check the seller's purchase price + holding period"
    as a negotiation anchor — that data is in our LR tables (R2-5 had just
    built the detail_anchored labelling and inline unit-level LR), but the
    checklist never connected to it. The fix follows the R2-3/R2-5 lesson:
    put the capability pointer in first-hop tool output rather than betting on
    the model remembering."""
    monkeypatch.setattr(get_market_risk, "DB_PATH", _db(tmp_path, _e1_fixture()))
    text = _call(outcode="E1")
    low = text.lower()
    assert "paid" in low and "negotiation anchor" in low
    assert "fetch_listing" in text
    # Promise-bounded (review #6): only a detail_anchored match is THIS home's
    # sale, so the wording must carry the "verified as this property" qualifier
    # and never unconditionally promise "the seller's purchase price".
    assert "verified as this property" in low
    # The empty-cohort branch (review #5: 30% of outcodes have no cohort rows)
    # must give the pointer too — thin-data areas are exactly where a next step
    # matters most.
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    empty = _db(empty_dir, [])
    monkeypatch.setattr(get_market_risk, "DB_PATH", empty)
    text2 = _call(outcode="E1")
    assert "fetch_listing" in text2 and "paid" in text2.lower()
