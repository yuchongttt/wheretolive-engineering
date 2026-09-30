"""get_price_trend: category-A filtering (Fix AI) + CAGR window semantics pinned (Fix AR).

R3-1 evidence (r3-1_compare.md):
  * `_fetch_residential` didn't filter category → the live median/YoY and the
    segment CAGR were contaminated by category B (repossessions / corporate
    transfers, 18.6% of LR rows in the last 3 years);
  * the model answered a pandemic-cohort question with "E14 flats 5y CAGR
    +2.5%/yr" and said "overall it's up", contradicting the cohort's true
    resale outcome of −3.8% (total). Checking PriceAnalysisEvaluator confirmed
    the window's real definition: **every pair whose SALE year falls in the
    last N calendar years, annualised and weighted by holding period** — a
    pair bought in 2010 and sold in 2023 dominates "3y/5y CAGR" with 13 years
    of weight, and the purchase year is unbounded. So it is neither "the price
    path of the last N years" nor "the outcome for buyers N years ago"; the
    semantics must be written into the output — no model can be expected to
    guess our code's window.

(Extract note: the original file also fed an A→B→A chain to
PriceAnalysisEvaluator — the repeat-sales engine, not part of this extract —
so that case is not included here.)
"""
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import get_price_trend as gpt  # noqa: E402


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Fix AI: _fetch_residential and category
# ---------------------------------------------------------------------------

class TestFetchResidentialCategory:
    """The contract after the R3-1-refix review: _fetch_residential returns
    rows of **all categories**, carrying the category column — dropping B in
    SQL would let the pairing stitch phantom cross-owner holds across the
    removed B legs (10,673 A→B→A chains DB-wide). Medians filter to A inside
    type_split_text (poison-row removal is proven by the exact median
    assertions in test_price_trend_by_type — site repo); pair-level B removal
    happens in PriceAnalysisEvaluator."""

    def _mk(self, conn):
        conn.execute(
            "CREATE TABLE lr_transactions (postcode TEXT, address_key TEXT, "
            "price INT, date TEXT, property_type TEXT, category TEXT)"
        )
        rows = [
            ("E14 9ZZ", "10|TEST|E149ZZ", 500000, "2024-01-01", "F", "A"),
            ("E14 9ZZ", "11|TEST|E149ZZ", 300000, "2024-02-01", "F", "B"),
            ("E14 9ZZ", "12|TEST|E149ZZ", 510000, "2024-03-01", "T", "A"),
            ("E14 9ZZ", "13|TEST|E149ZZ", 999999, "2024-04-01", "D", "B"),
        ]
        conn.executemany("INSERT INTO lr_transactions VALUES (?,?,?,?,?,?)", rows)

    def test_rows_carry_category_for_pair_level_filtering(self):
        conn = _conn()
        self._mk(conn)
        rows = gpt._fetch_residential(conn, "outcode", "E14")
        assert len(rows) == 4  # all categories — the pairing must see the full sequence
        assert {r["category"] for r in rows} == {"A", "B"}


# ---------------------------------------------------------------------------
# Fix AR: the CAGR window definition travels inline with the number
# ---------------------------------------------------------------------------

def _mk_geo_cagr(conn, with_values=True):
    conn.execute(
        "CREATE TABLE geo_cagr (level TEXT, geo TEXT, cagr_3y REAL, "
        "cagr_3y_count INT, cagr_5y REAL, cagr_5y_count INT, cagr_10y REAL, "
        "cagr_10y_count INT, direction TEXT, sample_count INT, built_at TEXT, "
        "PRIMARY KEY(level, geo))"
    )
    if with_values:
        conn.execute(
            "INSERT INTO geo_cagr VALUES ('outcode', 'E14', 2.1, 500, 2.5, "
            "900, 4.0, 1500, 'up', 2000, '2026-08-01T00:00:00Z')"
        )
    else:
        conn.execute(
            "INSERT INTO geo_cagr VALUES ('outcode', 'E14', NULL, NULL, NULL, "
            "NULL, NULL, NULL, 'unknown', 0, '2026-08-01T00:00:00Z')"
        )


class TestCagrWindowNote:
    def test_note_present_when_cagr_quoted(self):
        conn = _conn()
        _mk_geo_cagr(conn)
        text = gpt.area_trend_text(conn, "outcode", "E14")
        assert text is not None
        assert "NOT the outcome of buyers who bought" in text
        assert "get_sold_nearby" in text

    def test_note_absent_when_no_cagr(self):
        """Reverse case: no CAGR figure quoted → no definition attached (avoid noise)."""
        conn = _conn()
        _mk_geo_cagr(conn, with_values=False)
        text = gpt.area_trend_text(conn, "outcode", "E14")
        assert text is not None
        assert "NOT the outcome of buyers who bought" not in text

    def test_note_names_the_actual_window_semantics(self):
        """The definition must be anchored in the real implementation: filtered by sale year, weighted by holding period."""
        note = gpt._CAGR_WINDOW_NOTE
        assert "SOLD" in note or "sale completed" in note.lower()
        assert "weighted" in note.lower()
        assert "bought_after" in note
