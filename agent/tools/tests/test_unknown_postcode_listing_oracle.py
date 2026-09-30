"""UNKNOWN_POSTCODE_ALERT must never tell the model that an address we are currently listing doesn't exist.

T18b (test_unknown_postcode_alert.py) was right to add the alert: E14 5AB is a
business postcode, and giving it a sector median as a house price is
fabrication. But the check only looked at two tables, postcode_coords and
lr_transactions, and postcode_coords has a 22% hole for active-listing
postcodes — measured consequence:

  1,762 **currently listed** properties (968 postcodes) would trigger the
  alert; of 50 sampled postcodes sent to postcodes.io, 49 were live. So this
  loud "this postcode is in none of our datasets, probably a typo" alert was
  ~98% false positives on the set where it fired.

Fix: add a third local table — rm_sales_overview. We are listing that address
ourselves; that is proof it exists, no network needed. No filter on delisted:
a delisted listing proves the address just as well (E1W 2SG is a postcode
terminated in 2009 whose London Dock homes are still being listed; KT2 7FU is
a new-build development postcodes.io doesn't know yet — both are real, and
only the listings table knows).

Reverse discipline: E14 5AB has zero listings anywhere, so the alert must still fire.
"""
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from geo_radius import unit_postcode_known  # noqa: E402


def _conn(with_listings=True):
    """Schema follows the production PRAGMA shape; column names aren't invented."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE postcode_coords (postcode TEXT PRIMARY KEY, "
                 "latitude REAL, longitude REAL)")
    conn.execute("INSERT INTO postcode_coords VALUES ('N1 9DT', 51.53, -0.11)")
    conn.execute("CREATE TABLE lr_transactions (postcode TEXT, price INT)")
    conn.execute("INSERT INTO lr_transactions VALUES ('SE19 3BA', 400000)")
    if with_listings:
        conn.execute("CREATE TABLE rm_sales_overview (id TEXT PRIMARY KEY, "
                     "postcode TEXT, delisted_date TEXT)")
        conn.executemany(
            "INSERT INTO rm_sales_overview VALUES (?,?,?)",
            [("L1", "KT27FU", None),              # new-build development, stored unspaced
             ("L2", "E1W2SG", "2026-04-22"),        # delisted + postcode terminated in 2009
             ("L3", "SE19 3BA", None)])
    conn.commit()
    return conn


def test_active_listing_proves_the_address_exists():
    """New-build postcode: not in coords, not in LR — but we are listing it."""
    conn = _conn()
    assert unit_postcode_known(conn, "KT2 7FU") is True


def test_delisted_listing_also_proves_existence():
    """Delisted ≠ nonexistent. E1W 2SG was terminated in 2009; the London Dock homes are real."""
    conn = _conn()
    assert unit_postcode_known(conn, "E1W 2SG") is True


def test_listing_postcode_stored_without_space_still_matches():
    """The listings table is mixed-format like coords; normalise both sides."""
    conn = _conn()
    assert unit_postcode_known(conn, "kt2 7fu") is True


def test_business_postcode_with_no_listing_still_alerts():
    """Reverse discipline: T18b's E14 5AB has zero listings anywhere; the alert still fires."""
    conn = _conn()
    assert unit_postcode_known(conn, "E14 5AB") is False


def test_existing_oracles_unchanged():
    conn = _conn()
    assert unit_postcode_known(conn, "N1 9DT") is True     # coords
    assert unit_postcode_known(conn, "SE19 3BA") is True   # lr_transactions
    assert unit_postcode_known(conn, "N1") is None         # outcode: not applicable
    assert unit_postcode_known(conn, "N1 9") is None       # sector: not applicable


def test_missing_listings_table_does_not_break_the_check():
    """When the listings table isn't on this deployment, fall back to the two-table check; never raise."""
    conn = _conn(with_listings=False)
    assert unit_postcode_known(conn, "N1 9DT") is True
    assert unit_postcode_known(conn, "E14 5AB") is False
