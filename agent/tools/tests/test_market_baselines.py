"""Controlled views + a seasonality primitive: make the wrong answer structurally impossible.

The evidence that opened this (2026-08-18): working from raw sales data, the agent wrote for R2-9
  SELECT strftime('%m',date), AVG(price) ... WHERE date>='2019-01-01'
The window contains the three 2021 stamp-duty-holiday deadlines (June 2021 ran
at 3.22x its neighbouring months) and the March 2025 threshold deadline, so it
saw spikes in March/June/September/December and **invented an industry
explanation: "UK law firms batch completions at quarter end"** — the real
cause is that the stamp-duty deadlines happen to fall at quarter ends. Wrong
numbers + invented causation: both red lines crossed.

Conclusion: contamination is **removed**, not **documented** — documentation
relies on the model reading it; removal is structural.
"""
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pipeline"))

import market_baselines as mb  # noqa: E402


def _mkdb(tmp_path: Path, rows) -> Path:
    p = tmp_path / "t.db"
    c = sqlite3.connect(p)
    c.execute("""CREATE TABLE lr_transactions (id INTEGER PRIMARY KEY AUTOINCREMENT,
        postcode TEXT, price INTEGER, date TEXT, property_type TEXT, category TEXT)""")
    c.executemany("INSERT INTO lr_transactions (postcode,price,date,property_type,category)"
                  " VALUES (?,?,?,?,?)", rows)
    c.commit()
    c.close()
    return p


def _flat_series(years, per_month=100, price=500_000, spikes=None):
    """Synthetic series with the same volume and price every month; spikes={'2021-06': 400} overrides a month's count."""
    rows = []
    for y in years:
        for m in range(1, 13):
            ym = f"{y}-{m:02d}"
            n = (spikes or {}).get(ym, per_month)
            for i in range(n):
                rows.append(("SW1 1AA", price, f"{ym}-15", "F", "A"))
    return rows




def test_trend_does_not_leak_into_the_seasonal_index(tmp_path):
    rows = []
    for i, y in enumerate(range(2000, 2020)):
        for m in range(1, 13):
            price = int(300_000 * (1.06 ** (i + m / 12)))
            rows += [("SW1 1AA", price, f"{y}-{m:02d}-15", "F", "A")] * 60
    db = _mkdb(tmp_path, rows)
    mb.refresh(db, min_type_sales=10)
    c = sqlite3.connect(db)
    vals = [r[0] for r in c.execute(
        "SELECT price_index FROM lr_seasonality WHERE property_type='F' ORDER BY month")]
    c.close()
    assert len(vals) == 12
    assert max(vals) - min(vals) < 0.01, f"trend leaked into the seasonal index: {vals}"


def test_real_seasonality_is_still_detected(tmp_path):
    """August is 5% dearer, the rest flat — the index must detect it, not detrend it away."""
    rows = []
    for y in range(2000, 2020):
        for m in range(1, 13):
            price = 525_000 if m == 8 else 500_000
            rows += [("SW1 1AA", price, f"{y}-{m:02d}-15", "F", "A")] * 60
    db = _mkdb(tmp_path, rows)
    mb.refresh(db, min_type_sales=10)
    c = sqlite3.connect(db)
    got = dict(c.execute("SELECT month, price_index FROM lr_seasonality WHERE property_type='F'"))
    c.close()
    assert got[8] > 1.03, got
    assert all(abs(got[m] - 1.0) < 0.02 for m in range(1, 13) if m != 8), got


def test_a_policy_spike_month_is_flagged_and_excluded(tmp_path):
    rows = _flat_series(range(2000, 2020), spikes={"2010-06": 400})
    db = _mkdb(tmp_path, rows)
    mb.refresh(db, min_type_sales=10)
    c = sqlite3.connect(db)
    row = c.execute("SELECT is_distorted, reason FROM lr_month_quality WHERE ym='2010-06'").fetchone()
    assert row[0] == 1 and row[1] == "volume_spike", row
    # that month must not exist in the clean view
    assert c.execute("SELECT COUNT(*) FROM v_lr_clean WHERE date LIKE '2010-06%'").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM lr_transactions WHERE date LIKE '2010-06%'").fetchone()[0] == 400
    c.close()


def test_a_collapse_month_is_flagged_too(tmp_path):
    """A lockdown / the post-deadline hangover month is as toxic as a spike."""
    db = _mkdb(tmp_path, _flat_series(range(2000, 2020), spikes={"2010-04": 20}))
    mb.refresh(db, min_type_sales=10)
    c = sqlite3.connect(db)
    assert c.execute("SELECT reason FROM lr_month_quality WHERE ym='2010-04'").fetchone()[0] == "volume_collapse"
    c.close()


def test_non_market_transfers_never_reach_the_clean_view(tmp_path):
    rows = _flat_series([2010, 2011, 2012])
    rows += [("SW1 1AA", 1, "2011-05-15", "F", "B")] * 30   # repossession / related-party transfer
    db = _mkdb(tmp_path, rows)
    mb.refresh(db, min_type_sales=10)
    c = sqlite3.connect(db)
    assert c.execute("SELECT COUNT(*) FROM v_lr_clean WHERE category='B'").fetchone()[0] == 0
    c.close()


def test_trailing_incomplete_months_are_flagged(tmp_path):
    """HMLR filing lags: the latest months are naturally low, and using them raw reads as a "crash"."""
    rows = _flat_series(range(2000, 2020))
    rows += [("SW1 1AA", 500_000, "2020-01-15", "F", "A")] * 30   # only 30% of the constant
    db = _mkdb(tmp_path, rows)
    mb.refresh(db, min_type_sales=10)
    c = sqlite3.connect(db)
    r = c.execute("SELECT is_distorted, reason FROM lr_month_quality WHERE ym='2020-01'").fetchone()
    assert r == (1, "incomplete_filing"), r
    c.close()


def test_staleness_is_reported_not_silently_served(tmp_path):
    """Anything materialised goes stale — clean_ppsf_premium silently going stale with no schedule already bit us."""
    db = _mkdb(tmp_path, _flat_series(range(2000, 2020)))
    mb.refresh(db, min_type_sales=10)
    assert mb.staleness_days(db) < 1   # just computed
    assert mb.is_stale(db) is False
    c = sqlite3.connect(db)
    c.execute("UPDATE lr_seasonality SET computed_at = '2020-01-01T00:00:00Z'")
    c.commit()
    c.close()
    assert mb.staleness_days(db) > 365
    assert mb.is_stale(db) is True


def test_refresh_is_idempotent(tmp_path):
    db = _mkdb(tmp_path, _flat_series(range(2000, 2020)))
    mb.refresh(db, min_type_sales=10)
    c = sqlite3.connect(db)
    first = c.execute("SELECT COUNT(*) FROM lr_seasonality").fetchone()[0]
    c.close()
    assert first > 0, "an idempotency test must not pass on an empty table"
    mb.refresh(db, min_type_sales=10)
    c = sqlite3.connect(db)
    assert c.execute("SELECT COUNT(*) FROM lr_seasonality").fetchone()[0] == first
    c.close()


def test_known_events_are_named_and_unknown_ones_stay_null(tmp_path):
    """Detection is automatic (the safety net); naming is manual (optional annotation).

    An unnamed anomaly must be NULL — that is the data foundation for the
    prompt rule "if you don't know the cause, say you don't know". Inventing a
    name for an unknown anomaly would import the very fabrication we guard
    against into the DB.
    """
    rows = _flat_series(range(2000, 2020), spikes={"2010-06": 400})
    db = _mkdb(tmp_path, rows)
    mb.refresh(db, min_type_sales=10)
    c = sqlite3.connect(db)
    assert c.execute("SELECT event FROM lr_month_quality WHERE ym='2010-06'").fetchone()[0] is None
    c.close()


def test_the_curated_event_map_only_labels_months_we_actually_flag(tmp_path):
    """The curated map must not label a month that wasn't judged distorted — that would mean "the DB says something happened, yet the view still uses it"."""
    db = _mkdb(tmp_path, _flat_series(range(2000, 2020)))
    mb.refresh(db, min_type_sales=10)
    c = sqlite3.connect(db)
    bad = c.execute("SELECT COUNT(*) FROM lr_month_quality"
                    " WHERE event IS NOT NULL AND is_distorted=0").fetchone()[0]
    c.close()
    assert bad == 0
