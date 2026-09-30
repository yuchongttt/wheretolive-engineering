"""pipeline/lib/ptype_kind.py: flat/house/other bucketing + SQL/Python parity.

(Extract note: the original file also pins desc_contradicts_house against
verbatim listing descriptions hand-checked on 2026-08-27; that class quotes
listing-source text and is not included here.)
"""
import sys, os, sqlite3
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
from lib.ptype_kind import (kind_from_property_type, desc_contradicts_house,
                            effective_kind, ensure_ptype_schema, kind_case_sql)


# ── kind_from_property_type: three-bucket mapping ───────────────────────
class TestKindFromPropertyType:
    def test_flat_bucket(self):
        for pt in ["Flat", "Apartment", "Maisonette", "Penthouse", "Studio",
                   "Duplex", "Triplex", "Ground Flat", "Ground Maisonette"]:
            assert kind_from_property_type(pt) == "flat", pt

    def test_house_bucket(self):
        for pt in ["Terraced", "Semi-Detached", "Detached", "House",
                   "End of Terrace", "Town House", "Link Detached House",
                   "Mews", "Cottage", "Bungalow", "Coach House",
                   "Barn Conversion", "Country House", "Detached Villa"]:
            assert kind_from_property_type(pt) == "house", pt

    def test_other_bucket_non_dwellings(self):
        # These belong in neither the flat nor the house bucket (2026-08-27
        # review: House Boat £1.4m / House Share £900k were mixed into the house P25 corpus)
        for pt in ["House Boat", "House Share", "Off-Plan", "Block of Apartments",
                   "Retirement Property", "Hotel Room", "Serviced Apartments",
                   "Residential Development", "House of Multiple Occupation",
                   "Equestrian Facility", "Not Specified", "", None]:
            assert kind_from_property_type(pt) == "other", pt

    def test_commercial_labels_are_other_not_house(self):
        # Review (2026-09-24): OTHER_TYPES lacked commercial types, so Office / Retail /
        # Mixed Use fell into the house bucket, and get_sold_nearby property_kind=house
        # would have served 14 commercial sales as residential comps.
        for pt in ("Office", "Commercial Property", "Retail Property (high street)",
                   "Retail Property (out of town)", "Mixed Use", "Commercial Development",
                   "Restaurant", "Shop", "Warehouse", "Industrial", "Light Industrial",
                   "Pub", "Cafe", "Hotel", "Guest House", "Showroom", "Workshop",
                   "Business Park", "Storage", "Farm", "Smallholding"):
            assert kind_from_property_type(pt) == "other", pt

    def test_block_of_apartments_not_swallowed_by_flat_keyword(self):
        # 'Block of Apartments' contains the 'Apartment' keyword, but a whole building is not a flat
        assert kind_from_property_type("Block of Apartments") == "other"

    def test_no_drift_against_radar_vocab(self):
        # radar_vocab.py's PT_FLAT/PT_HOUSE are the canonical vocabulary shared by
        # radar + search; this module's buckets must agree. The one deliberate
        # divergence: 'block of apartments' counts as flat in radar search (a user
        # searching flats should find it) but as other for statistics (a whole
        # building isn't a flat and doesn't belong in the flat P25 corpus).
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from radar_vocab import PT_FLAT, PT_HOUSE
        for pt in PT_FLAT:
            expected = "other" if pt == "block of apartments" else "flat"
            assert kind_from_property_type(pt) == expected, pt
        for pt in PT_HOUSE:
            assert kind_from_property_type(pt) == "house", pt


# ── effective_kind: combined verdict ───────────────────────────────────
class TestEffectiveKind:
    def test_house_bucket_with_flat_description_becomes_flat(self):
        t = "1 bedroom apartment located on the 9th floor of the tower."
        assert effective_kind("Terraced", t) == "flat"

    def test_house_bucket_with_house_description_stays_house(self):
        t = "Chain free two double bedroom end of terrace house with a private driveway."
        assert effective_kind("Terraced", t) == "house"

    def test_flat_bucket_ignores_description(self):
        # the reverse direction (flat label + house description) is out of this gate's scope: stays flat
        t = "A lovely terraced house with garden."
        assert effective_kind("Flat", t) == "flat"

    def test_other_bucket_ignores_description(self):
        t = "1 bedroom apartment located on the 9th floor."
        assert effective_kind("House Boat", t) == "other"

    def test_no_description_keeps_property_type_verdict(self):
        assert effective_kind("Terraced", None) == "house"


# ── kind_case_sql: the SQL CASE must not drift from the Python function ─
class TestKindCaseSql:
    # every distinct property_type in the 2026-08 production DB (42 kinds) + edge values
    ALL_TYPES = [
        "Flat", "Apartment", "Terraced", "Semi-Detached", "Detached", "House",
        "End of Terrace", "Maisonette", "Studio", "Ground Flat",
        "Retirement Property", "Penthouse", "Town House", "Not Specified",
        "Duplex", "Link Detached House", "Mews", "Ground Maisonette",
        "Block of Apartments", "Cottage", "Character Property",
        "Barn Conversion", "House Boat", "Chalet", "Equestrian Facility",
        "Serviced Apartments", "Hotel Room", "Bungalow", "House Share",
        "Coach House", "Villa", "Triplex", "Lodge", "Country House",
        "Semi-detached Villa", "Semi-Detached Bungalow",
        "Residential Development", "Off-Plan", "House of Multiple Occupation",
        "Finca", "Detached Villa", "Detached Bungalow", "", None,
    ]

    def test_sql_case_agrees_with_python_on_every_known_type(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (property_type TEXT, ptype_desc_flat INTEGER)")
        conn.executemany("INSERT INTO t VALUES (?, NULL)", [(pt,) for pt in self.ALL_TYPES])
        for (pt, got) in conn.execute(
                f"SELECT property_type, {kind_case_sql()} FROM t"):
            assert got == kind_from_property_type(pt), pt
        conn.close()

    def test_sql_case_respects_desc_flag_on_house_bucket_only(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (property_type TEXT, ptype_desc_flat INTEGER)")
        conn.executemany("INSERT INTO t VALUES (?, ?)",
                         [("Terraced", 1), ("Terraced", 0), ("Terraced", None),
                          ("Flat", 1), ("House Boat", 1)])
        got = [r[0] for r in conn.execute(f"SELECT {kind_case_sql()} FROM t")]
        # house bucket + flag=1 → flat; flag=0/NULL → house; flat/other buckets ignore the flag
        assert got == ["flat", "house", "house", "flat", "other"]
        conn.close()


# ── ensure_ptype_schema: idempotent column add ─────────────────────────
class TestEnsurePtypeSchema:
    def test_adds_columns_and_is_idempotent(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE rm_sales_overview (id INTEGER PRIMARY KEY, property_type TEXT)")
        ensure_ptype_schema(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(rm_sales_overview)")}
        assert {"ptype_desc_flat", "ptype_desc_evidence", "ptype_desc_checked_at"} <= cols
        ensure_ptype_schema(conn)  # a second call must not raise
        conn.close()
