"""get_comparables: aggregates drop category B + detail rows are labelled line by line (R3-1 loss fix, Fix AK).

18.6% of LR rows in the last 3 years are category B (repossessions /
mortgage-identifiable BTL / corporate transfers); their bimodal prices (median
£415k / mean £1.18M) skew range/median and "Subject asking vs median". Detail
rows are ammunition, not statistics — keep them, label them (same wording as
lookup_address), and let the model choose by the question's meaning; the
excluded count is always disclosed (even B=0, so "found nothing" ≠ "clean").
"""
import asyncio
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import get_comparables as gc  # noqa: E402


def _mk_db(tmp_path, with_b=True):
    db = tmp_path / "evaluations.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE lr_transactions (postcode TEXT, paon TEXT, saon TEXT, "
        "street TEXT, price INT, date TEXT, property_type TEXT, tenure TEXT, "
        "new_build TEXT, category TEXT)"
    )
    rows = [
        ("N1 9DT", "1", "", "TEST ST", 500000, "2026-01-10", "F", "L", "N", "A"),
        ("N1 9DT", "2", "", "TEST ST", 520000, "2026-02-10", "F", "L", "N", "A"),
        ("N1 9DT", "3", "", "TEST ST", 540000, "2026-03-10", "F", "L", "N", "A"),
    ]
    if with_b:
        rows.append(
            ("N1 9DT", "4", "", "TEST ST", 3000000, "2026-04-10", "F", "L", "N", "B"))
    conn.executemany(
        "INSERT INTO lr_transactions VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.execute("CREATE TABLE rm_sales_overview (property_id TEXT, postcode TEXT,"
                 " bedrooms INT, property_type TEXT, asking_price INT)")
    conn.commit()
    conn.close()
    return db


def _call(db, monkeypatch, **args):
    monkeypatch.setattr(gc, "DB_PATH", db)
    out = asyncio.run(gc.handle_call_tool("get_comparables", args))
    return out[0].text


def test_median_excludes_b_and_discloses(tmp_path, monkeypatch):
    text = _call(_mk_db(tmp_path), monkeypatch, postcode="N1 9DT")
    # clean median = £520k; if B leaked in (4 rows) it would become £540k
    assert "median £520,000" in text
    assert "n=3" in text
    # the excluded count is disclosed
    assert "1 category-B" in text
    # the £3M repossession row is still in the detail, labelled
    assert "£3,000,000" in text
    assert "category B — not a standard open-market sale" in text


def test_zero_b_disclosure_still_reported(tmp_path, monkeypatch):
    """Reverse case: B=0 → report 0, don't drop the line; no row is labelled."""
    text = _call(_mk_db(tmp_path, with_b=False), monkeypatch, postcode="N1 9DT")
    assert "0 category-B" in text
    assert "not a standard open-market sale" not in text


def test_structured_json_carries_category(tmp_path, monkeypatch):
    import json
    text = _call(_mk_db(tmp_path), monkeypatch, postcode="N1 9DT")
    payload = json.loads(text.split("--- structured ---")[1])
    assert payload["stats"]["excluded_category_b"] == 1
    assert "category A" in payload["stats"]["basis"]
    cats = {t["category"] for t in payload["transactions"]}
    assert cats == {"A", "B"}


def test_all_b_corner_quotes_nothing(tmp_path, monkeypatch):
    """review finding 4: in the all-B corner never quote a median, never say "standard sales"."""
    db = tmp_path / "evaluations.db"
    import sqlite3 as s
    conn = s.connect(db)
    conn.execute(
        "CREATE TABLE lr_transactions (postcode TEXT, paon TEXT, saon TEXT, "
        "street TEXT, price INT, date TEXT, property_type TEXT, tenure TEXT, "
        "new_build TEXT, category TEXT)")
    conn.executemany(
        "INSERT INTO lr_transactions VALUES (?,?,?,?,?,?,?,?,?,?)",
        [("N1 9DT", str(i), "", "TEST ST", 400000 + i, "2026-0%d-01" % (i + 1),
          "F", "L", "N", "B") for i in range(4)])
    conn.execute("CREATE TABLE rm_sales_overview (property_id TEXT, "
                 "postcode TEXT, bedrooms INT, property_type TEXT, "
                 "asking_price INT)")
    conn.commit()
    conn.close()
    text = _call(db, monkeypatch, postcode="N1 9DT")
    assert "No standard-sale comparables" in text
    assert "median £" not in text
    assert "standard sales" not in text.split("--- structured ---")[0].replace(
        "No standard-sale comparables", "")
    assert "category B — not a standard open-market sale" in text


def test_b_rows_cannot_crowd_out_standard_comps(tmp_path, monkeypatch):
    """review finding 4: split by category BEFORE the LIMIT — the 6 most recent B rows must not crowd A out of the window."""
    db = tmp_path / "evaluations.db"
    import sqlite3 as s
    conn = s.connect(db)
    conn.execute(
        "CREATE TABLE lr_transactions (postcode TEXT, paon TEXT, saon TEXT, "
        "street TEXT, price INT, date TEXT, property_type TEXT, tenure TEXT, "
        "new_build TEXT, category TEXT)")
    rows = [("N1 9DT", f"a{i}", "", "TEST ST", 500000, "2025-01-01",
             "F", "L", "N", "A") for i in range(12)]
    rows += [("N1 9DT", f"b{i}", "", "TEST ST", 100000, "2026-06-01",
              "F", "L", "N", "B") for i in range(6)]
    conn.executemany(
        "INSERT INTO lr_transactions VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.execute("CREATE TABLE rm_sales_overview (property_id TEXT, "
                 "postcode TEXT, bedrooms INT, property_type TEXT, "
                 "asking_price INT)")
    conn.commit()
    conn.close()
    text = _call(db, monkeypatch, postcode="N1 9DT", limit=10)
    assert "n=10 standard sales" in text
    assert "6 category-B" in text
    assert "median £500,000" in text
