"""radar_vocab.build_where: structured conditions → parameterised SQL.

(Extract note: the original file also AST-checks that search_properties.py
imports the shared predicates, and runs the open-plan-kitchen predicate against
the production schema; neither file/DB is part of this extract.)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import radar_vocab as rv
import pytest


def test_price_and_bedrooms_compile_to_indexed_where():
    where, params, join = rv.build_where([
        {"type": "price", "max": 600000},
        {"type": "bedrooms", "min": 2, "max": 3},
    ])
    assert "o.asking_price <= ?" in where
    assert "o.bedrooms >= ?" in where and "o.bedrooms <= ?" in where
    assert params == [600000, 2, 3]
    assert join.strip() == ""  # no joins needed for these


def test_commute_adds_join_on_sector():
    where, params, join = rv.build_where([
        {"type": "commute", "hub": "Bank", "max_minutes": 40},
    ])
    assert "sector_hub_commute" in join
    assert "shc0.minutes <= ?" in where
    assert "Bank" in params and 40 in params


def test_two_commute_conditions_get_distinct_aliases():
    # "within 30 min of King's Cross and within 40 min of Canary Wharf" → two commute
    # conditions in one spec. With a single shared alias both JOINs were `shc` →
    # "ambiguous column name: shc.minutes" and the
    # reconcile/verify passes crashed, leaving the "verifying" banner stuck.
    import sqlite3
    where, params, join = rv.build_where([
        {"type": "commute", "hub": "King's Cross", "max_minutes": 30},
        {"type": "commute", "hub": "Canary Wharf", "max_minutes": 40},
    ])
    assert "shc0.minutes" in where and "shc1.minutes" in where
    assert join.count("LEFT JOIN sector_hub_commute") == 2
    # must actually EXECUTE: alias collision only explodes at prepare time
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE rm_sales_overview (id TEXT, postcode TEXT)")
    conn.execute("CREATE TABLE sector_hub_commute (sector TEXT, hub TEXT, minutes INT)")
    conn.execute(f"SELECT o.id FROM rm_sales_overview o {join} WHERE {where}", params)


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        rv.build_where([{"type": "vibe", "value": "cosy"}])


def test_canon_hub_resolves_place_aliases_to_nearest_hub():
    # Neighbourhoods / landmarks / employers that are NOT hub names but sit next to
    # one must resolve deterministically (not depend on the LLM snapping). "the City"
    # → Bank, "Clerkenwell" → Farringdon, "UCL" → Euston Square.
    assert rv.canon_hub("Clerkenwell") == "Farringdon"
    assert rv.canon_hub("the City") == "Bank"
    assert rv.canon_hub("UCL") == "Euston Square"
    assert rv.canon_hub("Guy's Hospital") == "London Bridge"  # apostrophe-insensitive
    # v2: employers / venues / districts
    assert rv.canon_hub("Deloitte") == "Farringdon"
    assert rv.canon_hub("HSBC") == "Canary Wharf"
    assert rv.canon_hub("the O2") == "North Greenwich"
    assert rv.canon_hub("Broadgate") == "Liverpool Street"
    # v3: Zone-2 residential tube stations within ~1.5km of a hub
    assert rv.canon_hub("Bethnal Green") == "Whitechapel"
    assert rv.canon_hub("Fulham Broadway") == "Earl's Court"


def test_hub_aliases_all_point_at_a_real_hub():
    # A wrong/typo target would silently LEFT-JOIN to NULL minutes → 0 matches, no
    # warning (the exact failure class this whole change fights). Guard every alias.
    hubs = set(rv.ENUMS["commute.hub"])
    for place, hub in rv.ALIASES.items():
        assert hub in hubs, f"alias {place!r} -> {hub!r} is not a valid hub"


def test_green_space_joins_postcode_green_on_min_score():
    # "green space / near a park / lots of green" → a floor on the postcode's
    # accessible-greenspace score (parks_index green_score, 0-100), materialised
    # per-postcode in postcode_green. NOT the environment facet (greenery is only
    # 25% of it). New listings in a known postcode match immediately via the join.
    where, params, join = rv.build_where([{"type": "green_space", "min": 60}])
    assert "postcode_green" in join
    assert "pg.postcode = o.postcode_norm" in join   # index-seekable, NOT replace() full-scan
    assert "replace(" not in join.lower()
    assert "pg.green_score >= ?" in where
    assert params == [60]


def test_green_space_executes_against_postcode_green():
    # alias + join must actually prepare/execute (mirror the commute exec test).
    import sqlite3
    where, params, join = rv.build_where([{"type": "green_space", "min": 55}])
    conn = sqlite3.connect(":memory:")
    # postcode_norm mirrors scripts/migrate_postcode_norm.py (upper + space-stripped)
    # so a messy listing postcode still index-matches the normalised green key.
    conn.execute("CREATE TABLE rm_sales_overview (id TEXT, postcode TEXT, "
                 "postcode_norm TEXT GENERATED ALWAYS AS (upper(replace(postcode,' ',''))) VIRTUAL)")
    conn.execute("CREATE TABLE postcode_green (postcode TEXT, green_score INT)")
    conn.execute("INSERT INTO rm_sales_overview(id,postcode) VALUES ('a', 'e17 4ab')")  # messy case+space
    conn.execute("INSERT INTO postcode_green VALUES ('E174AB', 70)")  # normalised key
    rows = conn.execute(
        f"SELECT o.id FROM rm_sales_overview o {join} WHERE {where}", params
    ).fetchall()
    assert {r[0] for r in rows} == {"a"}  # 'e17 4ab' → 'E174AB' matches, 70 >= 55


def test_postcode_score_join_is_sargable():
    # area-score join must be index-seekable on postcode_norm (both sides indexed),
    # not replace() on both sides (which forced a full SCAN of postcode_scores).
    where, params, join = rv.build_where(
        [{"type": "postcode_score", "dimension": "total", "op": "gte", "value": 70}])
    assert "ps.postcode_norm = o.postcode_norm" in join
    assert "replace(" not in join.lower()
    assert "ps.total_score >= ?" in where


def test_mortgageable_true_selects_clean_only():
    # mortgage_flags is a JSON array of lender red-flag objects ('[]' = clean).
    # value=true → no flags; value=false → only flagged. Same-table, no JOIN.
    where, params, join = rv.build_where([{"type": "mortgageable", "value": True}])
    assert "json_array_length(o.mortgage_flags)" in where
    assert not where.startswith("(NOT")
    assert params == [] and join.strip() == ""


def test_mortgageable_false_negates():
    where, _, _ = rv.build_where([{"type": "mortgageable", "value": False}])
    assert where.startswith("(NOT")


def test_mortgageable_executes():
    import sqlite3
    where, params, join = rv.build_where([{"type": "mortgageable", "value": True}])
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE rm_sales_overview (id TEXT, mortgage_flags TEXT)")
    conn.execute("INSERT INTO rm_sales_overview VALUES ('clean','[]'), ('flag','[{\"rule\":\"ews1\"}]')")
    rows = conn.execute(f"SELECT o.id FROM rm_sales_overview o {join} WHERE {where}", params).fetchall()
    assert {r[0] for r in rows} == {"clean"}


def test_prior_sale_below_and_within_years():
    # Filter on the property's last Land Registry sale. Requires a known prior sale.
    where, params, join = rv.build_where([{"type": "prior_sale", "below": True, "within_years": 2}])
    assert "o.lr_prev_sold_price IS NOT NULL" in where
    assert "o.asking_price < o.lr_prev_sold_price" in where
    assert "lr_prev_sold_date" in where and params == ["-2 years"]
    assert join.strip() == ""


def test_prior_sale_needs_a_field():
    with pytest.raises(ValueError):
        rv.build_where([{"type": "prior_sale"}])  # neither below nor within_years


def _prior_sale_rows(cond):
    import sqlite3
    where, params, join = rv.build_where([{"type": "prior_sale", **cond}])
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE rm_sales_overview (id TEXT, asking_price INT, lr_prev_sold_price INT,"
                 " lr_prev_sold_date TEXT, lr_match_strategy TEXT)")
    recent = __import__("datetime").date.today().replace(day=1).isoformat()
    rows = [
        ("exact_below", 500_000, 600_000, "2015-06-01", "detail_anchored"),
        ("exact_above", 700_000, 600_000, "2015-06-01", "detail_anchored"),
        ("exact_recent", 700_000, 600_000, recent, "detail_anchored"),
        ("street_below", 500_000, 600_000, recent, "street_only"),
        ("epc_below", 500_000, 600_000, recent, "epc_anchored"),
        ("paon_below", 500_000, 600_000, recent, "paon_street"),
        ("pc_below", 500_000, 600_000, recent, "postcode_only_most_recent"),
    ]
    conn.executemany("INSERT INTO rm_sales_overview VALUES (?,?,?,?,?)", rows)
    got = conn.execute(f"SELECT o.id FROM rm_sales_overview o {join} WHERE {where}", params).fetchall()
    return {r[0] for r in got}


def test_prior_sale_below_ignores_non_exact_matches():
    # A street/postcode/EPC-level LR match is another home's sale: "asking below
    # last sale" against it is noise (46% hit rate vs 20% for exact matches).
    assert _prior_sale_rows({"below": True}) == {"exact_below"}


def test_prior_sale_within_years_ignores_non_exact_matches():
    # street_only rows are "sold within 3y" 80% of the time — the neighbour's sale.
    assert _prior_sale_rows({"within_years": 2}) == {"exact_recent"}


def test_decor_tier_high_uses_percentile_threshold():
    # high tier => decor_score >= p80; builder must JOIN decor_zeroshot_score
    where, params, join = rv.build_where([{"type": "decor_tier", "value": ["high"]}])
    assert "decor_zeroshot_score" in join
    assert "dz.decor_score >= ?" in where  # threshold param resolved at runtime


def test_service_charge_dual_source_excludes_unknown():
    # A-first-B-fallback: the listing's structured field (0 treated as missing) falls back
    # to the description-extracted facts value; unknown-in-both rows must NOT
    # match (COALESCE -> NULL -> comparison is NULL -> excluded).
    where, params, join = rv.build_where([{"type": "service_charge", "max": 3000}])
    assert "NULLIF(o.annual_service_charge, 0)" in where
    assert "service_charge_pa" in where
    assert params == [3000.0]
    assert join.strip() == ""


def test_ground_rent_keeps_zero_as_peppercorn():
    # GR uses plain COALESCE (no NULLIF) — £0 is a real peppercorn value.
    where, params, join = rv.build_where([{"type": "ground_rent", "max": 300}])
    assert "COALESCE(o.annual_ground_rent," in where
    assert "NULLIF(o.annual_ground_rent" not in where
    assert "ground_rent_pa" in where
    assert params == [300.0]


def test_lift_true_has_negation_guard_and_fact_veto():
    where, params, join = rv.build_where([{"type": "lift", "value": True}])
    assert "$.lift.value') = 1" in where      # positive text fact
    assert "no lift" in where                  # description negation guard
    assert "$.lift.value') = 0" in where       # explicit no-lift fact veto
    assert params == []


def test_lift_false_negates_whole_predicate():
    where, _, _ = rv.build_where([{"type": "lift", "value": False}])
    assert where.startswith("(NOT ")


def test_cladding_false_excludes_only_confessed_issues():
    # value=false (the common use) — weak guarantee: only drops listings whose
    # description ADMITS a cladding problem (cladding_issue fact or
    # ews1_status='issue_mentioned'); silence passes.
    where, params, join = rv.build_where([{"type": "cladding", "value": False}])
    assert where.startswith("(NOT EXISTS")
    assert "cladding_issue = 1" in where
    assert "issue_mentioned" in where
    assert params == [] and join.strip() == ""


def test_cladding_true_selects_flagged_listings():
    where, _, _ = rv.build_where([{"type": "cladding", "value": True}])
    assert not where.startswith("(NOT") and "cladding_issue = 1" in where


def test_lease_years_passes_share_of_freehold():
    # 7.4k SOF/freehold flats have NULL lease — the lease floor must not kill
    # the SAFEST tenure. Leasehold-with-unknown-lease still drops.
    where, params, _ = rv.build_where([{"type": "lease_years", "min": 100}])
    assert "freehold" in where and "commonhold" in where
    assert "share_of_freehold" in where  # text-fact source
    assert params == [100]


def test_station_walk_converts_minutes_to_miles():
    where, params, _ = rv.build_where([{"type": "station_walk", "max_minutes": 10}])
    assert "rm_nearest_stations" in where
    assert params == [0.5]  # 10 min / 20 min-per-mile
    with pytest.raises(ValueError):
        rv.build_where([{"type": "station_walk", "max_minutes": 0}])


def test_balcony_includes_guarded_description_source():
    where, _, _ = rv.build_where([{"type": "balcony", "value": True}])
    assert "full_description" in where
    assert "no balcony" in where and "without balcony" in where


def test_balcony_room_area_excludes_ground_level_labels():
    # An elevated-'balcony' size match must drop VLM-mislabelled ground spaces (patio /
    # courtyard / decking / pergola / ground garden) while keeping 'roof garden'. Shared
    # with the picks board + search via scripts/outdoor_space.ground_label_exclusion_sql.
    where, _, _ = rv.build_where([{"type": "room_area", "room": "balcony",
                                   "agg": "max", "min_sqm": 18}])
    low = where.lower()
    for kw in ("patio", "courtyard", "deck", "pergola"):
        assert f"not like '%{kw}%'" in low
    assert "not like '%garden%'" in low and "like '%roof garden%'" in low


def test_property_type_flat_expands_to_family():
    # "flat" is a category: Apartment/Studio/Penthouse/… must match too —
    # exact IN ('flat') silently lost ~45% of true matches (2026-06-10).
    where, params, join = rv.build_where([{"type": "property_type", "value": "flat"}])
    assert "apartment" in params and "studio" in params and "penthouse" in params
    # specific subtype stays exact, no family expansion
    _, p2, _ = rv.build_where([{"type": "property_type", "value": "maisonette"}])
    assert p2 == ["maisonette"]


def test_commute_to_station_widens_screen_buffer():
    # drive/cycle homes can be far from the station, so the sector pre-screen must
    # allow a wider centroid window (ACCESS_SCREEN_BONUS) to survive to verify.
    for mode in ("drive", "cycle"):
        where, params, join = rv.build_where(
            [{"type": "commute", "hub": "Bank", "max_minutes": 40, "to_station": mode}])
        assert (40 + rv.ACCESS_SCREEN_BONUS) in params  # buffer_min=0 here


def test_commute_walk_default_has_no_bonus():
    # neither the absent to_station key nor an explicit "walk" gets the bonus.
    for c in ({"type": "commute", "hub": "Bank", "max_minutes": 40},
              {"type": "commute", "hub": "Bank", "max_minutes": 40, "to_station": "walk"}):
        where, params, join = rv.build_where([c])
        assert 40 in params and (40 + rv.ACCESS_SCREEN_BONUS) not in params


def _mkdb():
    """In-memory DB with rm_sales_overview (incl. date columns) + dp_quality."""
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE rm_sales_overview (
            id TEXT PRIMARY KEY,
            delivery_point_id INTEGER,
            asking_price REAL,
            delisted_date TEXT,
            first_visible_date TEXT,
            date_listed TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE dp_quality (
            delivery_point_id INTEGER PRIMARY KEY,
            is_unit_level INTEGER
        )
    """)
    return conn


def test_relisted_condition_unit_level_and_price_coherence():
    """relisted condition: off-market-gap gate + unit-level dpId + price-coherence.

    Fixtures:
      dpId 1 (unit-level): active 'cur' (first_visible_date='2026-03-01') +
                           delisted 'old' (delisted_date='2026-01-01', price coherent)
                           → MATCH  (prior delist BEFORE current start = genuine relisting)
      dpId 2 (dev-level, is_unit_level=0): active 'dev_a' + delisted 'dev_b' → NO MATCH
      dpId 3 (unit-level): active 'cur3' (500k) + delisted 'old3' (2000k) → NO MATCH (price guard)
      dpId 4 (unit-level): active 'cur4' (first_visible_date='2025-08-01') +
                           delisted 'old4' (delisted_date='2026-06-01', AFTER cur4 started)
                           → NO MATCH  (overlapping concurrent duplicate, not a relisting)
    """
    conn = _mkdb()
    # dpId 1 — unit-level genuine relisting: old delist (Jan) BEFORE cur start (Mar)
    conn.execute("INSERT INTO rm_sales_overview VALUES ('cur',  1, 500000, NULL, '2026-03-01', '2026-03-01')")
    conn.execute("INSERT INTO rm_sales_overview VALUES ('old',  1, 520000, '2026-01-01', '2025-10-01', '2025-10-01')")
    # dpId 2 — dev-level: no match regardless of dates
    conn.execute("INSERT INTO rm_sales_overview VALUES ('dev_a', 2, 500000, NULL, '2026-03-01', '2026-03-01')")
    conn.execute("INSERT INTO rm_sales_overview VALUES ('dev_b', 2, 510000, '2026-01-01', '2025-10-01', '2025-10-01')")
    # dpId 3 — unit-level: wildly different price, no match
    conn.execute("INSERT INTO rm_sales_overview VALUES ('cur3', 3, 500000, NULL, '2026-03-01', '2026-03-01')")
    conn.execute("INSERT INTO rm_sales_overview VALUES ('old3', 3, 2000000, '2026-01-01', '2025-10-01', '2025-10-01')")
    # dpId 4 — unit-level overlapping duplicate: old delist (Jun 2026) AFTER cur4 start (Aug 2025)
    conn.execute("INSERT INTO rm_sales_overview VALUES ('cur4', 4, 500000, NULL, '2025-08-01', '2025-08-01')")
    conn.execute("INSERT INTO rm_sales_overview VALUES ('old4', 4, 510000, '2026-06-01', '2025-08-15', '2025-08-15')")
    conn.execute("INSERT INTO dp_quality VALUES (1, 1)")  # unit-level
    conn.execute("INSERT INTO dp_quality VALUES (2, 0)")  # dev-level
    conn.execute("INSERT INTO dp_quality VALUES (3, 1)")  # unit-level
    conn.execute("INSERT INTO dp_quality VALUES (4, 1)")  # unit-level

    where, params, join = rv.build_where([{"type": "relisted", "value": True}])
    assert join.strip() == ""  # no extra JOINs needed
    sql = f"SELECT o.id FROM rm_sales_overview o {join} WHERE {where} AND o.delisted_date IS NULL"
    rows = conn.execute(sql, params).fetchall()
    matched = {r[0] for r in rows}
    # Only cur (genuine relisting). cur4 is an overlapping dup — must NOT match.
    assert matched == {"cur"}, f"expected {{'cur'}}, got {matched}"


# --- open_plan_kitchen --------------------------------------------------------
# Origin: a radar asking for an "open plan kitchen" compiled everything else, but
# "open plan kitchen" landed in unsupported_json and stayed there —
# the same word-gap class as green_space.
#
# Source is listing text, so the rule is a MEASURED one, not an intuited one
# (London's countless "… Gardens" street names taught us description regexes lie). Measured on
# 2026-09-08 over 72,395 active listings:
#   - naive "open plan" anywhere            → 27k rows, ~8.6% FP in a 35-row read
#     (the FPs are comma/ampersand FEATURE LISTS: "open plan living room, two
#      double bedrooms, an updated kitchen" — the kitchen is a separate item)
#   - forward window + break/room/aspiration guards → 20,060 rows, 0 FP in ~85
#     hand-read rows (30+30 random samples clean, plus an adversarial sweep)
# Recall cost accepted: the SQL reads only the FIRST "open plan" occurrence
# (-3.9% of true hits) and skips hedged prose — misses are fine, FPs are not.
#
# These cases pin the RULE with synthetic text on purpose. Asserting on live
# listing ids would be an alarm clock, not a guardrail: the rows delist and the
# test starts failing for reasons that have nothing to do with the predicate.
OPK_CASES = [
    # (text, should_match, why)
    ("Open Plan Kitchen/Reception Room", True, "canonical key_features form"),
    ("open-plan kitchen, dining and living room", True, "hyphenated + trailing cluster"),
    ("a bright open plan reception and kitchen area", True, "kitchen after cluster words"),
    ("the open-plan living, dining and kitchen areas flow together", True, "kitchen last in cluster"),
    ("OPEN PLAN KITCHEN / FAMILY / DINING ROOM", True, "uppercase"),
    ("open plan living room, two double bedrooms, an updated kitchen and a bathroom",
     False, "feature list — kitchen is a separate item, not part of the open plan"),
    ("excellent potential to redesign into a modern open-plan kitchen/dining space",
     False, "aspirational — the open-plan kitchen does not exist yet"),
    ("with scope to create a larger open-plan kitchen/diner", False, "aspirational"),
    ("Luxury Kitchen & Open Plan Lounge", False, "kitchen precedes; two separate items"),
    ("a bright open-plan living space with a private balcony", False, "no kitchen at all"),
    ("a separate kitchen and a good sized reception room", False, "no open plan at all"),
]


def _opk_matches(conn, text_col="key_features"):
    """Run the built predicate over one-row fixtures, one per OPK_CASES entry."""
    where, params, join = rv.build_where([{"type": "open_plan_kitchen", "value": True}])
    sql = f"SELECT o.id FROM rm_sales_overview o {join} WHERE {where}"
    return {r[0] for r in conn.execute(sql, params)}


def _opk_db():
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE rm_sales_overview (id TEXT PRIMARY KEY,
        key_features TEXT, description TEXT, full_description TEXT)""")
    conn.execute("CREATE TABLE listing_text_facts (rm_uuid TEXT, facts_json TEXT,"
                 " open_tags_json TEXT, ok INT)")
    return conn


@pytest.mark.parametrize("text,should_match,why", OPK_CASES)
def test_open_plan_kitchen_rule(text, should_match, why):
    conn = _opk_db()
    conn.execute("INSERT INTO rm_sales_overview (id,key_features) VALUES ('x',?)", (text,))
    got = "x" in _opk_matches(conn)
    assert got == should_match, f"{why}: {text!r} → matched={got}, want {should_match}"


def test_open_plan_kitchen_reads_full_description_too():
    """key_features is often absent; the long description carries the phrase."""
    conn = _opk_db()
    conn.execute("INSERT INTO rm_sales_overview (id,full_description) VALUES "
                 "('x','The heart of the home is a stunning open-plan kitchen and dining area.')")
    assert "x" in _opk_matches(conn)


def test_open_plan_kitchen_open_tag_is_a_second_source():
    """LLM open_tags carry `open_plan_kitchen` for ~1.5k listings — same
    multi-source shape as _conservatory/_terrace (text broad OR flag secondary)."""
    conn = _opk_db()
    conn.execute("INSERT INTO rm_sales_overview (id,key_features) VALUES ('x','Two bedrooms')")
    conn.execute("INSERT INTO listing_text_facts VALUES ('x','{}','[\"open_plan_kitchen\"]',1)")
    assert "x" in _opk_matches(conn)


def test_open_plan_kitchen_false_excludes_matches():
    conn = _opk_db()
    conn.execute("INSERT INTO rm_sales_overview (id,key_features) VALUES ('yes','Open Plan Kitchen/Diner')")
    conn.execute("INSERT INTO rm_sales_overview (id,key_features) VALUES ('no','Separate kitchen')")
    where, params, join = rv.build_where([{"type": "open_plan_kitchen", "value": False}])
    sql = f"SELECT o.id FROM rm_sales_overview o {join} WHERE {where}"
    assert {r[0] for r in conn.execute(sql, params)} == {"no"}


def test_property_type_freeform_value_does_not_silently_match_nothing():
    # 2026-09-16 simulation: the chat agent searched "ground floor flat" for a
    # user, got 25 listings, then offered "save as Radar" with
    # {"type":"property_type","value":"Ground"} — and that radar matched ZERO
    # listings, forever, silently. Cause: search_properties sends an unknown
    # value through LIKE '%ground%' (Ground Flat / Ground Maisonette both hit),
    # while this builder sent it through exact IN ('ground'), which no stored
    # value equals. The two surfaces import PT_HOUSE/PT_FLAT from here
    # precisely so they cannot drift — the UNKNOWN-value branch had drifted.
    # Same silent-zero class as the "farringdon station" hub (2026-06-05) and
    # the "Ealing" area value.
    where, params, join = rv.build_where([{"type": "property_type", "value": "Ground"}])
    assert "LIKE" in where, "unknown subtype must fall back to substring, like search_properties"
    assert params == ["%ground%"]

    # A phrase no stored value even contains, but which names a family we DO
    # understand, resolves to that family rather than to nothing: the user who
    # writes "ground floor flat" means flats.
    where2, p2, _ = rv.build_where([{"type": "property_type", "value": "ground floor flat"}])
    assert "apartment" in p2 and "studio" in p2, "family word inside the phrase must be honoured"


def test_property_type_freeform_value_executes_and_matches(tmp_path):
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE rm_sales_overview (id TEXT, property_type TEXT, ptype_desc_flat INT)")
    conn.executemany("INSERT INTO rm_sales_overview VALUES (?,?,NULL)",
                     [("1", "Ground Flat"), ("2", "Ground Maisonette"),
                      ("3", "Terraced"), ("4", "Apartment")])
    where, params, join = rv.build_where([{"type": "property_type", "value": "Ground"}])
    got = {r[0] for r in conn.execute(
        f"SELECT o.id FROM rm_sales_overview o {join} WHERE {where}", params)}
    assert got == {"1", "2"}, f"'Ground' must reach Ground Flat/Maisonette, got {got}"

    where2, p2, j2 = rv.build_where([{"type": "property_type", "value": "ground floor flat"}])
    got2 = {r[0] for r in conn.execute(
        f"SELECT o.id FROM rm_sales_overview o {j2} WHERE {where2}", p2)}
    # the flat family includes maisonettes (same umbrella as value="flat"),
    # so Ground Maisonette rides along; the terraced house must not.
    assert got2 == {"1", "2", "4"}, f"'ground floor flat' must reach the flat family, got {got2}"
