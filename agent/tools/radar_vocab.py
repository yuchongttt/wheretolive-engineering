#!/usr/bin/env python3
"""Single source of truth for Radar condition vocabulary.

build_where(conditions) -> (where_sql, params, join_sql) turns a validated spec
into ONE AND-ed SQL predicate over rm_sales_overview (alias `o`), plus any JOINs
its conditions need. The matcher and (via radar_vocab.json) the TS compiler share
this vocabulary so it cannot drift.

decor_tier thresholds + commute/postcode_score JOINs mirror the MCP tools
(screen_decor_value, screen_by_commute, get_postcode_scores) — same data, same
predicates.
"""
import sys
import json
import math
import re
from pathlib import Path

# pipeline/ holds the helpers shared with the site's batch scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline"))
from outdoor_space import ground_label_exclusion_sql  # noqa: E402

VOCAB = json.loads((Path(__file__).resolve().parent / "radar_vocab.json").read_text())
TYPES = VOCAB["types"]
ENUMS = VOCAB["enums"]
# Place / neighbourhood / landmark / employer → nearest canonical hub. Keys are
# lowercase + apostrophe-free (canon_hub deapo's before lookup). A deterministic
# layer under the LLM snapping so common non-hub destinations still resolve.
ALIASES = VOCAB.get("hub_aliases", {})

# Percentile thresholds for decor tiers (match screen_decor_value: p80 high, p20 low).
# Resolved at build time against the live distribution via a placeholder the
# matcher fills; to keep build_where pure + testable, we emit the predicate with
# a sentinel and let the caller pass the resolved threshold. Simpler: build_where
# resolves thresholds itself using a provided conn when decor_tier is present.

# Postcode sector from a stored postcode: outcode + first inward digit, e.g.
# "SW11 2AB" -> "SW11 2". rm_sales_overview.postcode may be stored without a space.
_SECTOR_SQL = (
    "(substr(replace(o.postcode,' ',''),1,length(replace(o.postcode,' ',''))-3)"
    " || ' ' || substr(replace(o.postcode,' ',''),length(replace(o.postcode,' ',''))-2,1))"
)

# FULL postcode re-spaced ("AL100BJ" -> "AL10 0BJ") — for joining the bare
# indexed postcode_scores.postcode PK. Distinct from _SECTOR_SQL (outcode +
# first inward digit). Shared with search_properties (scores filter + near-miss
# probe) so the normalisation cannot drift.
SPACED_POSTCODE_SQL = (
    "(substr(replace(o.postcode,' ',''),1,length(replace(o.postcode,' ',''))-3)"
    " || ' ' || substr(replace(o.postcode,' ',''),length(replace(o.postcode,' ',''))-2))"
)

# Straight-line miles -> walking minutes at ~3 mph. Single source for the
# station_walk conversion (radar builder + search_properties inline literal).
STATION_WALK_MIN_PER_MILE = 20.0

# Sector pre-screen cushion (minutes) added ONLY for drive/cycle commute conditions:
# their centroid time is bus-inclusive + walk-baked and can't be decomposed, so we
# widen the screen so far-from-station homes survive to the door-to-door verify
# (commute_verify) where the real access-leg adjustment happens. ≈ max access-leg
# time a car saves. Heuristic; tune with scripts/radar/commute_verify.py (ACCESS_RATIO).
ACCESS_SCREEN_BONUS = 25


def _area(c):
    vals = c["value"] if isinstance(c.get("value"), list) else [c.get("value")]
    parts, params = [], []
    norm = "replace(o.postcode,' ','')"
    for v in vals:
        oc = "".join(str(v).split()).upper()
        if oc.isalpha():  # area prefix, digit-boundary guard (mirror search_properties)
            parts.append(f"(UPPER({norm}) LIKE ? AND substr(UPPER({norm}),?,1) BETWEEN '0' AND '9')")
            params += [oc + "%", len(oc) + 1]
        else:  # exact outcode
            parts.append(f"UPPER(substr({norm},1,length({norm})-3)) = ?")
            params.append(oc)
    return "(" + " OR ".join(parts) + ")", params, "", []


def _range(col, c):
    where, params = [], []
    if c.get("min") is not None:
        where.append(f"{col} >= ?"); params.append(c["min"])
    if c.get("max") is not None:
        where.append(f"{col} <= ?"); params.append(c["max"])
    return " AND ".join(where), params, "", []


def _in(col, c):
    vals = c["value"] if isinstance(c.get("value"), list) else [c.get("value")]
    ph = ",".join("?" * len(vals))
    return f"LOWER({col}) IN ({ph})", [str(v).lower() for v in vals], "", []


# "house"/"flat" are CATEGORIES, not literal stored values — the data uses
# Terraced/Semi-Detached/… (almost no bare "House") and "Apartment" doesn't
# contain "flat". Expand the umbrella words to their member types. Single
# source shared with search_properties (which imports these constants) so the
# two surfaces cannot drift: before this a flat radar silently lost ~45% of
# true matches (160/289 in SE16/E14/SW8/E20, 2026-06-10).
PT_HOUSE = ('terraced', 'semi-detached', 'detached', 'end of terrace', 'house',
            'town house', 'link detached house', 'mews', 'cottage', 'bungalow',
            'barn conversion', 'semi-detached villa')
PT_FLAT = ('flat', 'apartment', 'maisonette', 'studio', 'ground flat', 'penthouse',
           'duplex', 'triplex', 'ground maisonette', 'block of apartments')


PT_ALL = tuple(dict.fromkeys(PT_HOUSE + PT_FLAT))
# The six umbrella words both branches below understand. Kept as a dict so the
# family fallback resolves the SAME words the top-level branch does — one list
# to keep in step, not two.
PT_UMBRELLA = {"house": PT_HOUSE, "houses": PT_HOUSE, "flat": PT_FLAT,
               "flats": PT_FLAT, "apartment": PT_FLAT, "apartments": PT_FLAT}


def _property_type(c):
    vals = c["value"] if isinstance(c.get("value"), list) else [c.get("value")]
    expanded, likes = [], []
    want_house = want_flat = False
    for v in vals:
        s = str(v).strip().lower()
        if s in ("house", "houses"):
            expanded.extend(PT_HOUSE)
            want_house = True
        elif s in ("flat", "flats", "apartment", "apartments"):
            expanded.extend(PT_FLAT)
            want_flat = True
        elif s in PT_ALL:
            expanded.append(s)  # stored subtype ("maisonette") stays exact
        elif any(s in known for known in PT_ALL):
            # A fragment of a stored type — "ground" is how the listing source
            # spells "Ground Flat"/"Ground Maisonette". search_properties has always
            # sent these through LIKE '%…%'; this builder sent them through
            # exact IN(), so they matched NOTHING and the radar sat at zero
            # forever with no warning (2026-09-16: the chat agent's own
            # "save as Radar" link shipped value="Ground"). Same silent-zero class
            # as an unknown commute hub — match the tool the user just saw.
            likes.append(s)
        else:
            # Not a stored type and not a fragment of one. If the phrase names
            # a family we understand ("ground floor flat", "new build house"),
            # honour the family — the modifier is not filterable but the noun
            # is, and "flats" is what the user asked for. Otherwise fall back
            # to substring, which is still what search_properties would do.
            fam = next((PT_UMBRELLA[w_] for w_ in re.findall(r"[a-z]+", s)
                        if w_ in PT_UMBRELLA), None)
            if fam is None:
                likes.append(s)
            else:
                expanded.extend(fam)
                want_house = want_house or fam is PT_HOUSE
                want_flat = want_flat or fam is PT_FLAT
    expanded = list(dict.fromkeys(expanded))
    parts = []
    params = []
    if expanded:
        parts.append(f"LOWER(o.property_type) IN ({','.join('?' * len(expanded))})")
        params.extend(expanded)
    for s in dict.fromkeys(likes):
        parts.append("LOWER(o.property_type) LIKE ?")
        params.append(f"%{s}%")
    if not parts:  # value was empty/None — never true, never a silent match-all
        return "0", [], "", []
    w = parts[0] if len(parts) == 1 else "(" + " OR ".join(parts) + ")"
    # The agent-entered property_type is unreliable at the cheap end: 17-21% of
    # the "house" bucket are flats/maisonettes in disguise. ptype_desc_flat is a
    # zero-false-positive counter-evidence column (set to 1 only when the
    # description's subject calls itself a flat and there is no house evidence),
    # so using it to drop houses is safe; NULL (new listings not yet stamped)
    # must pass, or a whole batch is wrongly killed. When BOTH families are
    # wanted, don't drop — the user wants both anyway, and an AND on the same
    # IN list would narrow the flats too.
    if want_house and not want_flat:
        w = f"({w} AND COALESCE(o.ptype_desc_flat, 0) = 0)"
    return w, params, "", []


def canon_hub(raw):
    """Map a free-form hub name to its canonical sector_hub_commute spelling.

    The LLM compiler emits things like "farringdon station" / "kings cross";
    the commute table only knows the canonical 20 names (enums["commute.hub"]).
    Before this, an unknown hub silently LEFT-JOINed to NULL minutes => the
    radar matched 0 listings with no warning (2026-06-05, the "extra-large
    balcony" radar case).
    Returns the canonical name, or the input unchanged if nothing matches."""
    hubs = ENUMS.get("commute.hub", [])
    s = str(raw).strip().lower()
    for suffix in (" station", " rail", " underground", " tube", "站"):
        s = s.removesuffix(suffix).strip()
    deapo = lambda x: x.replace("'", "").replace("’", "")  # king's == kings
    for h in hubs:
        if deapo(h.lower()) == deapo(s):
            return h
    # Place / landmark / employer alias → its nearest hub, before fuzzy matching
    # ("Clerkenwell" -> Farringdon, "UCL" -> Euston Square, "the City" -> Bank).
    alias = ALIASES.get(deapo(s))
    if alias:
        return alias
    # Unique prefix/containment match ("liverpool st" -> Liverpool Street).
    cands = [h for h in hubs if deapo(h.lower()).startswith(deapo(s)) or deapo(s).startswith(deapo(h.lower()))]
    if len(cands) == 1:
        return cands[0]
    return raw


def _commute(c, buffer_min=0, alias="shc0"):
    # `alias` MUST be unique per commute condition in one spec — a spec with two
    # commutes (e.g. "King's Cross≤30 and Canary Wharf≤40") produced two JOINs
    # both aliased `shc` → "ambiguous column name: shc.minutes" and the radar's
    # reconcile/verify crashed (banner never cleared). build_where numbers them.
    join = (f" LEFT JOIN sector_hub_commute {alias} ON {alias}.sector = {_SECTOR_SQL}"
            f" AND LOWER({alias}.hub) = LOWER(?)")
    # hub param belongs to the JOIN; build_where threads join params first.
    # buffer_min widens the SECTOR-centroid screen so borderline listings (centroid
    # over the limit but their own postcode under it) survive to the matcher's
    # per-property door-to-door verification (scripts/radar/commute_verify.py).
    # Default 0 keeps search_properties' behaviour unchanged.
    # drive/cycle access widens the screen so far homes reach the verify step.
    bonus = ACCESS_SCREEN_BONUS if c.get("to_station") in ("drive", "cycle") else 0
    return (f"{alias}.minutes <= ?", [c["max_minutes"] + buffer_min + bonus],
            join, [canon_hub(c["hub"])])


def _ppsf(c):
    # price per sqft, gated 300-2500 (mirror screen_decor_value); needs floor_area_sqft
    expr = "(o.asking_price * 1.0 / NULLIF(o.floor_area_sqft,0))"
    where = [f"o.floor_area_sqft > 0", f"{expr} BETWEEN 300 AND 2500"]
    params = []
    if c.get("min") is not None:
        where.append(f"{expr} >= ?"); params.append(c["min"])
    if c.get("max") is not None:
        where.append(f"{expr} <= ?"); params.append(c["max"])
    return " AND ".join(where), params, "", []


def _postcode_score(c):
    dim = c["dimension"]
    if dim not in ENUMS["postcode_score.dimension"]:
        raise ValueError(f"bad postcode_score dimension: {dim}")
    # "total" is the overall area score; the postcode_scores column is total_score.
    # Every other dimension name equals its column name 1:1.
    col = "total_score" if dim == "total" else dim
    op = ">=" if c.get("op", "gte") == "gte" else "<="
    # Index-seekable join on the normalised postcode key (both sides carry an
    # indexed postcode_norm = upper(replace(postcode,' ','')); see
    # scripts/migrate_postcode_norm.py). Replaces the old replace()-on-both-sides
    # form that was non-sargable → full SCAN of postcode_scores.
    join = " LEFT JOIN postcode_scores ps ON ps.postcode_norm = o.postcode_norm"
    return f"ps.{col} {op} ?", [c["value"]], join, []


def _green_space(c):
    # Minimum accessible-greenspace score (0-100) for the listing's POSTCODE.
    # green_score is parks_index.py's area-weighted, edge-distance park score
    # (OS Open Greenspace polygons) materialised per-postcode in postcode_green.
    # Deliberately NOT the postcode_score "environment" facet — greenery is only
    # 25% of that composite (noise/flood/air are the other 75%), so mapping a
    # "near green space" ask onto it would silently mismatch intent. Per-postcode
    # (not per-listing) so a brand-new listing in a known postcode matches at once
    # instead of waiting on an enrich pass. Unknown postcodes (LEFT JOIN → NULL)
    # do not match a floor — a green radar hit must back the "it's green" claim.
    # postcode_green.postcode is already normalised (upper, space-stripped) and is
    # the PK; join it to the indexed o.postcode_norm so this is an index SEARCH,
    # not a full SCAN (see scripts/migrate_postcode_norm.py).
    return ("pg.green_score >= ?", [c["min"]],
            " LEFT JOIN postcode_green pg ON pg.postcode = o.postcode_norm",
            [])


def _mortgageable(c):
    # mortgage_flags is a JSON array of lender red-flag objects on rm_sales_overview
    # ('[]' = clean, ~88% of active). Same-table, no JOIN. value=true → only clean
    # (no red flags: EWS1/cladding-height, non-standard construction, cash-only,
    # uninhabitable, short lease, …); value=false → only flagged (bargain / cash).
    clean = "COALESCE(json_array_length(o.mortgage_flags), 0) = 0"
    if c.get("value") is False:
        return f"NOT ({clean})", [], "", []
    return clean, [], "", []


def _prior_sale(c):
    # Filter on THIS property's last Land Registry sale. Only detail_anchored
    # matches are the same home (postcode + year + price from the listing's
    # own transaction history); street/EPC/postcode-level fallbacks attach a
    # neighbour's sale — measured 2026-09-25: 46% "below last sale" and 80%
    # "sold within 3y" on street_only vs 20% / 5% on detail_anchored.
    # below=true → asking under the last sold price; within_years=N → last
    # sold within N years. AND-ed; ≥1 required.
    clauses = ["o.lr_prev_sold_price IS NOT NULL",
               "o.lr_match_strategy = 'detail_anchored'"]
    params = []
    if c.get("below") is True:
        clauses.append("o.asking_price < o.lr_prev_sold_price")
    if c.get("within_years") is not None:
        clauses.append("o.lr_prev_sold_date IS NOT NULL "
                       "AND date(o.lr_prev_sold_date) >= date('now', ?)")
        params.append(f"-{int(c['within_years'])} years")
    if len(clauses) == 2:
        raise ValueError("prior_sale needs 'below' or 'within_years'")
    return " AND ".join(clauses), params, "", []


def decor_min_to_tier(raw) -> int:
    """Resolve a decor `min` onto the integral 1-10 tier scale.

    The scale is integral but the compiler emits fractions — "decor score
    above 75%" (asked in Chinese) compiles to 7.5 (verified live 2026-07-28).
    This used to be int(), which TRUNCATES, and truncation is wrong in one
    specific direction: it always resolves toward the LOOSER tier, so every
    fractional ask came back weaker than stated. 7.5 became tier 7 (p60) — the top 40% for someone who asked for
    the top 25%, and on live data indistinguishable from having typed 7.

    Rounds half UP rather than using round(), which is banker's rounding:
    round(6.5) == 6 while round(7.5) == 8, so half the fractional asks would
    still drop to the looser tier, inconsistently. floor(x + 0.5) always
    resolves a tie toward what the person asked for.

    float() first so a numeric string ("7", as the old int() accepted) still
    works instead of raising.
    """
    return max(1, min(10, math.floor(float(raw) + 0.5)))


def _decor_tier(c, conn):
    join = " LEFT JOIN decor_zeroshot_score dz ON dz.id = o.id"
    # NEW: min = 1-10 score → decor_score must be at/above the (min-1)*10th
    # percentile of the live distribution. 10 = top 10% (>=p90), 2 = top 90%
    # (>=p10), 1 = no floor. Preferred over the legacy high/mid/low `value`.
    if c.get("min") is not None:
        if conn is None:
            return "dz.decor_score >= ?", [None], join, []
        m = decor_min_to_tier(c["min"])
        thr = _pct(conn, (m - 1) / 10.0)
        return "dz.decor_score >= ?", [thr], join, []
    # LEGACY: value high/mid/low bands (radars created before the 1-10 input).
    vals = c["value"] if isinstance(c.get("value"), list) else [c.get("value")]
    if conn is None:
        # test/build-only path: emit predicate shape with sentinel thresholds
        if "high" in vals and "mid" not in vals and "low" not in vals:
            return "dz.decor_score >= ?", [None], join, []
        return "dz.decor_score IS NOT NULL", [], join, []
    p80 = _pct(conn, 0.80); p20 = _pct(conn, 0.20)
    ors, params = [], []
    if "high" in vals: ors.append("dz.decor_score >= ?"); params.append(p80)
    if "mid" in vals:  ors.append("(dz.decor_score >= ? AND dz.decor_score < ?)"); params += [p20, p80]
    if "low" in vals:  ors.append("dz.decor_score < ?"); params.append(p20)
    return "(" + " OR ".join(ors) + ")", params, join, []


# ── garden: same disease as parking, same cure ────────────────────────────
# o.garden is the listing source's outdoor-space field ("Yes"/"Private garden"/
# "Rear garden"/"Communal garden"/"Terrace"/"Patio"/empty). Count on 2026-09-07:
# **32,391 listings (44.8%) are empty**, the same shape as parking's 44.5%.
# Empty = the field wasn't filled in; the old single-source predicate judged
# all of them as "no garden".
#
# ⚠️ But **no text fallback** — this is the key divergence from parking.
# key_features looks like an obvious second source (753 empty-field rows mention
# a garden), but measured against the floorplan VLM's has_garden as an
# independent judge it is noise:
#     floorplan says "no garden"  28,560 rows → text hit 4,009 (14.0%)
#     floorplan says "has garden"  1,545 rows → text hit   399 (25.8%)
# 4,009 false positives to gain 399 true ones. London is full of streets and
# parks called "… Gardens"; "views over the communal gardens" and "close to
# Kensington Gardens" are not gardens a buyer can use. Narrowing to
# private/rear/own/landscaped and excluding communal brings it down to 0.23% vs
# 7.96%, but precision is still only ~65%, for just 65 extra listings the
# floorplans don't cover — not worth it. Anyone wanting to add text back: rerun
# that measurement first; "the ad says garden, so it has a garden" is exactly
# the intuition the numbers overturned. (Internal note: measure false positives
# before writing detectors.)
GARDEN_STATED_SQL = "(o.garden IS NOT NULL AND o.garden != '')"
# The COALESCE is **defensive hardening, not a fix for a bug that bites today**
# — stated so the next person doesn't think it rescued anything. A bare
# `o.garden = 'Yes' OR lower(o.garden) LIKE …` evaluates to NULL (not FALSE)
# when garden IS NULL, `NOT has` is then NULL too, and those listings would
# vanish from the value=false side as well. It surfaced while writing the
# exclusion-direction tests; but today the DB holds **0 NULLs and 32,391 empty
# strings**, so it has not saved a single row. The column is nullable and the
# ingest could write NULL one day, at which point it would really bite — so
# it stays, just don't count it as a win.
GARDEN_STRUCT_YES_SQL = (
    "(COALESCE(o.garden,'') = 'Yes' OR lower(COALESCE(o.garden,'')) LIKE '%garden%')")
# The only added source: the floorplan reading. Independent, trained for this,
# and already used by _garage/_terrace/_fireplace. Consulted only when the
# listing's own field is silent — so it can never override a "Terrace" the
# listing itself states (a structural guarantee, same as parking).
GARDEN_FLOORPLAN_YES_SQL = (
    "EXISTS (SELECT 1 FROM floorplan_vlm_results f"
    "        WHERE f.rm_uuid = o.id AND f.ok = 1 AND f.has_garden = 1)")
GARDEN_HAS_SQL = (
    f"({GARDEN_STRUCT_YES_SQL}"
    f" OR (NOT {GARDEN_STATED_SQL} AND {GARDEN_FLOORPLAN_YES_SQL}))")


def _garden(c):
    # value=false → the inverse, **deliberately** keeping unknowns: absence can't
    # be proven, and being a bit loose on the "no garden" side won't hurt the
    # person asking. Same asymmetry as _parking, same reason.
    if c.get("value") is False:
        return f"NOT {GARDEN_HAS_SQL}", [], "", []
    return GARDEN_HAS_SQL, [], "", []


# ── garden_facing: garden aspect as stated by the agent (2026-09-01) ────────
# A positive-selection signal: in the active corpus "south facing garden"
# appears 1,000 times vs "north facing garden" 4 times (250:1 disclosure
# asymmetry), so NULL ≠ north-facing and this condition never has exclusion
# semantics. The column is stamped by enrich_garden_facing.py (zero-FP regex,
# scripts/lib/garden_facing.py in the site repo). A cardinal direction includes
# its compounds (south → south/south-east/south-west: a buyer asking for a
# "south-facing garden" wants the sunny family); a compound matches only
# itself exactly. Source of truth for the values = radar_vocab.json
# enums["garden_facing.value"].


def garden_facing_sql(value, col="o.garden_facing"):
    """One direction value → a param-free predicate (the value is checked
    against the enum allowlist first, so inlining it is safe). Shared by
    search_properties and radar so the two SQL surfaces can't drift."""
    allowed = ENUMS["garden_facing.value"]
    if value not in allowed:
        raise ValueError(f"bad garden_facing value: {value!r} (one of {sorted(allowed)})")
    if value in ("south", "north"):
        return f"({col} = '{value}' OR {col} LIKE '{value}-%')"
    if value in ("east", "west"):
        return f"({col} = '{value}' OR {col} LIKE '%-{value}')"
    return f"{col} = '{value}'"


def _garden_sqft(c):
    """Minimum garden area. Shares GARDEN_SQFT_SQL (zero-FP caliber) with search_properties."""
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parent / "pipeline"))
    from lib.garden_area import GARDEN_SQFT_SQL
    v = c.get("min")
    if v is None:
        raise ValueError("garden_sqft needs a min")
    return f"{GARDEN_SQFT_SQL.format(id_col='o.id')} >= {float(v)}", [], "", []


def _garden_facing(c):
    """Stated **or** floorplan-derived. Shares garden_facing_union_sql with
    search_properties so the two surfaces can't drift (2026-09-03: the derived
    side went live and both switched together)."""
    v = c.get("value")
    vals = v if isinstance(v, list) else [v]
    if not vals:
        raise ValueError("garden_facing needs a value")
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parent / "pipeline"))
    from lib.garden_facing_expose import garden_facing_union_sql
    return ("(" + " OR ".join(
        garden_facing_union_sql(x, stated_col="o.garden_facing", id_col="o.id")
        for x in vals) + ")", [], "", [])


# Floorplans label the same space "Reception Room" / "Living Room" / "Lounge"
# interchangeably — a reception-size query must match both VLM types.
_ROOM_AREA_TYPES = {
    "bedroom": ("bedroom",),
    "reception": ("reception", "living"),
    "kitchen": ("kitchen",),
    "dining": ("dining",),
}


def _room_area(c):
    # "master bedroom ≥ 14m²" → {room: bedroom, agg: max, min_sqm: 14}
    # "smallest bedroom ≥ 10m²" → {room: bedroom, agg: min, min_sqm: 10}
    # "balcony ≥ 8m²" → {room: balcony, agg: max, min_sqm: 8}
    # "balcony longest side ≥ 5m" → {room: balcony, agg: max, min_long_edge_m: 5}
    # min_sqm and/or min_long_edge_m. long_edge_m (longest printed side, metres)
    # is a LINEAR measure that survives the bbox-area over-estimate of L/angled
    # balconies — so it still finds big irregular balconies whose area was NULLed.
    # Tier 2/3 areas are dim-derived (±10-20%), so area gets a 10% leniency.
    sqft = float(c["min_sqm"]) * 10.764 * 0.9 if c.get("min_sqm") is not None else None
    edge = float(c["min_long_edge_m"]) if c.get("min_long_edge_m") is not None else None
    if sqft is None and edge is None:
        raise ValueError("room_area: need min_sqm or min_long_edge_m")
    if str(c.get("room")) == "balcony":
        # "balcony" = ELEVATED private outdoor (balcony / roof terrace), matched by
        # room_type balcony/terrace + label balcon/terrace/roof garden. EXCLUDE
        # ground-level outdoor spaces: v6.6b often mis-types a "Patio …" / "Garden …"
        # / "Courtyard …" / "Decking" / "Pergola" label as room_type terrace, so a
        # label naming any of those is dropped (all ≈ ground-level, not a balcony) —
        # but "roof garden" is genuinely elevated, so it is kept.
        base = ("SELECT 1 FROM floorplan_room_areas ra WHERE ra.rm_uuid = o.id"
                " AND (ra.room_type IN ('balcony','terrace')"
                "      OR lower(ra.label) LIKE '%balcon%' OR lower(ra.label) LIKE '%terrace%'"
                "      OR lower(ra.label) LIKE '%roof garden%')"
                + ground_label_exclusion_sql("ra.label"))
        # OR the two metrics: "big balcony" = big AREA *or* long EDGE. AND would
        # wrongly exclude L/angled balconies (area NULLed) that pass only on edge
        # — the whole point of the edge measure. ∃ a balcony big by either rule.
        clauses, params = [], []
        if sqft is not None:
            clauses.append(f"EXISTS ({base} AND ra.area_sqft >= ?)"); params.append(sqft)
        if edge is not None:
            clauses.append(f"EXISTS ({base} AND ra.long_edge_m >= ?)"); params.append(edge)
        return ("(" + " OR ".join(clauses) + ")") if len(clauses) > 1 else clauses[0], params, "", []
    rooms = _ROOM_AREA_TYPES.get(str(c.get("room")))
    if not rooms:
        raise ValueError(f"room_area: unknown room {c.get('room')!r}")
    ph = ",".join("?" * len(rooms))
    base = (f"SELECT 1 FROM floorplan_room_areas ra WHERE ra.rm_uuid = o.id"
            f" AND ra.room_type IN ({ph})")
    clauses, params = [], []
    if sqft is not None:
        if c.get("agg") == "min":
            # Every sized room clears the bar, and at least one exists
            # (NULL-area labels are ignored — unsized rooms can't veto).
            clauses.append(f"EXISTS ({base} AND ra.area_sqft >= ?) AND NOT EXISTS ({base} AND ra.area_sqft < ?)")
            params += list(rooms) + [sqft] + list(rooms) + [sqft]
        else:
            clauses.append(f"EXISTS ({base} AND ra.area_sqft >= ?)"); params += list(rooms) + [sqft]
    if edge is not None:
        clauses.append(f"EXISTS ({base} AND ra.long_edge_m >= ?)"); params += list(rooms) + [edge]
    return " AND ".join(clauses), params, "", []


# "Has a balcony" — four OR'd sources (none alone covers well):
#   1. listing key_features text ("Private balcony", ~18% of active) — richest
#   2. listing garden/outdoor-space field ("Balcony", "Terrace with balcony", …)
#   3. floorplan VLM features.balcony (has_balcony, active coverage grows
#      with the daemon; only listings analysed since the 2026-05-28 cutoff)
#   4. full_description, negation-guarded — 7,872 active listings (2026-06-10
#      audit) mention the balcony ONLY in the description; without this source
#      a balcony=true filter silently drops ~6% of genuinely-matching stock.
# Balcony AREA is intentionally NOT a condition — floorplans almost never
# dimension balconies, so there is no data source.
BALCONY_HAS_SQL = (
    "(lower(COALESCE(o.key_features,'')) LIKE '%balcon%'"
    " OR lower(COALESCE(o.garden,'')) LIKE '%balcon%'"
    " OR EXISTS (SELECT 1 FROM floorplan_vlm_results f"
    "            WHERE f.rm_uuid = o.id AND f.ok = 1 AND f.has_balcony = 1)"
    " OR (lower(COALESCE(o.full_description,'')) LIKE '%balcon%'"
    "     AND lower(COALESCE(o.full_description,'')) NOT LIKE '%no balcony%'"
    "     AND lower(COALESCE(o.full_description,'')) NOT LIKE '%without balcony%'"
    "     AND lower(COALESCE(o.full_description,'')) NOT LIKE '%without a balcony%'))"
)


def _balcony(c):
    if c.get("value") is False:
        return f"NOT {BALCONY_HAS_SQL}", [], "", []
    return BALCONY_HAS_SQL, [], "", []


def _chain_free(c):
    # listing_text_facts.chain_free — LLM-extracted from the description text
    # (generated column over facts_json, quote-gated; see listing_facts/).
    # value=true → only chain-free listings. value=false → exclude listings the
    # text marks chain-free (rarely useful, kept for symmetry). Listings not yet
    # extracted (backlog draining) simply don't match value=true — coverage
    # grows as the scavenger daemon catches up.
    has = ("EXISTS (SELECT 1 FROM listing_text_facts tf"
           "        WHERE tf.rm_uuid = o.id AND tf.ok = 1 AND tf.chain_free = 1)")
    if c.get("value") is False:
        return f"NOT {has}", [], "", []
    return has, [], "", []


# ── parking: if the listing's field says something, trust it; only when it is
#    silent, look at the text ────────────────────────────────────────────────
# o.parking is the listing source's own structured field (Driveway/Garage/
# Permit/Off street/"No parking"/…). Count on 2026-09-07: of 71,875 active
# listings 38,962 have a value, 880 explicitly say "No parking", and **32,033
# (45%) are empty**. Empty = not filled in, NOT "no parking" — the old
# predicate (parking IS NOT NULL AND != '') treated that 45% as failing and
# silently dropped them from parking:true radars: adding "with parking" to a live
# radar evicted listings whose parking field was simply empty.
PARKING_STATED_SQL = "(o.parking IS NOT NULL AND o.parking != '')"
PARKING_STRUCT_YES_SQL = (
    f"({PARKING_STATED_SQL} AND LOWER(o.parking) NOT LIKE '%no parking%')")
# Text evidence is consulted only when the field is silent, so no regex can
# EVER override the 880 explicit "No parking" values — zero false positives
# there is a structural guarantee, not the result of regex tuning.
# Negations are independent: "no allocated parking" does not contain the
# substring "no parking", so it needs its own clause.
_PARKING_KF = "lower(COALESCE(o.key_features,''))"
PARKING_TEXT_YES_SQL = (
    f"(({_PARKING_KF} LIKE '%parking%' OR {_PARKING_KF} LIKE '%driveway%'"
    f"   OR {_PARKING_KF} LIKE '%carport%' OR {_PARKING_KF} LIKE '%car port%')"
    f" AND {_PARKING_KF} NOT LIKE '%no parking%'"
    f" AND {_PARKING_KF} NOT LIKE '%no allocated parking%'"
    f" AND {_PARKING_KF} NOT LIKE '%without parking%'"
    f" AND {_PARKING_KF} NOT LIKE '%no off street parking%'"
    f" AND {_PARKING_KF} NOT LIKE '%no off-street parking%'"
    f" AND {_PARKING_KF} NOT LIKE '%no private parking%')")
# Third source is the same table _garage uses — parking was the last boolean still reading a single column.
PARKING_FLOORPLAN_YES_SQL = (
    "EXISTS (SELECT 1 FROM floorplan_vlm_results f"
    "        WHERE f.rm_uuid = o.id AND f.ok = 1 AND f.has_garage = 1)")
PARKING_HAS_SQL = (
    f"({PARKING_STRUCT_YES_SQL}"
    f" OR (NOT {PARKING_STATED_SQL}"
    f"     AND ({PARKING_TEXT_YES_SQL} OR {PARKING_FLOORPLAN_YES_SQL})))")


def _parking(c):
    # value=false → the inverse. Note the inverse **deliberately** keeps
    # unknowns: someone who "doesn't need parking" isn't hurt by a listing with
    # an unfilled field, and "has no parking" is an equally unprovable absence.
    # Strict on the true side, loose on the false side — the one asymmetric but
    # correct spot in this condition.
    if c.get("value") is False:
        return f"NOT {PARKING_HAS_SQL}", [], "", []
    return PARKING_HAS_SQL, [], "", []


# Tenures that make a lease-length floor moot: the buyer owns (a share of) the
# freehold outright. 13.8k active flats are SHARE_OF_FREEHOLD/FREEHOLD/COMMONHOLD
# and 7.4k of them have NULL lease_years_remaining (2026-06-10 audit) — a plain
# "lease >= N" filter wrongly killed all of them despite being the SAFEST case.
# "freehold" matches FREEHOLD + SHARE_OF_FREEHOLD; "leasehold" contains neither.
TENURE_SECURE_SQL = (
    "(LOWER(COALESCE(o.tenure,'')) LIKE '%freehold%'"
    " OR LOWER(COALESCE(o.tenure,'')) LIKE '%commonhold%'"
    " OR EXISTS (SELECT 1 FROM listing_text_facts tf WHERE tf.rm_uuid = o.id"
    "            AND tf.ok = 1 AND json_extract(tf.facts_json,'$.share_of_freehold.value') = 1))"
)


def _lease_years(c):
    # Minimum remaining lease (years) — OR a freehold-equivalent tenure (see
    # TENURE_SECURE_SQL). Leaseholds with UNKNOWN lease length still drop: a
    # radar hit must be able to back the "lease >= N" claim.
    return (f"(o.lease_years_remaining >= ? OR {TENURE_SECURE_SQL})",
            [int(c["min"])], "", [])


# Service charge / ground rent: dual-source, A-first-B-fallback — the SAME
# precedence fetch_listing uses (A = the listing's structured livingCosts field on
# rm_sales_overview; B = LLM text extraction in listing_text_facts, quote-gated).
# A=0 is treated as missing for SERVICE charge (a flat never truly costs £0/yr;
# 0 is unstated junk) but kept for GROUND rent (£0 = real peppercorn, ~7.4k
# active listings). Listings unknown in BOTH sources do NOT match — a radar hit
# must be able to back the claim "meets your fee cap". Known caveat: both
# sources can over-state (A month-as-year ×12, B ×12 over-annualisation), which
# for a ≤max filter errs on the safe side (drops a compliant listing, never
# admits a violator).
SC_EFF_SQL = ("COALESCE(NULLIF(o.annual_service_charge, 0),"
           " NULLIF((SELECT CAST(json_extract(tf.facts_json,'$.service_charge_pa.value') AS REAL)"
           "         FROM listing_text_facts tf WHERE tf.rm_uuid = o.id AND tf.ok = 1), 0))")
GR_EFF_SQL = ("COALESCE(o.annual_ground_rent,"
           " (SELECT CAST(json_extract(tf.facts_json,'$.ground_rent_pa.value') AS REAL)"
           "  FROM listing_text_facts tf WHERE tf.rm_uuid = o.id AND tf.ok = 1))")


def _service_charge(c):
    # Maximum annual service charge (£/yr). NULL-in-both-sources rows excluded.
    return f"{SC_EFF_SQL} <= ?", [float(c["max"])], "", []


def _ground_rent(c):
    # Maximum annual ground rent (£/yr). £0 (peppercorn) is a real value and matches.
    return f"{GR_EFF_SQL} <= ?", [float(c["max"])], "", []


# "Has a lift" — three OR'd sources + one veto (mirror search_properties EXACTLY):
#   1. listing_text_facts lift fact (LLM quote-gated, high precision)
#   2. key_features — '"lift' catches an element-initial "Lift"/"Lift access"
#      (key_features is a JSON array string), '% lift%' catches "Passenger Lift" /
#      "with Lift"; both boundaries exclude "uplift" (no space/quote before).
#   3. full_description '% lift%' (broadest), guarded against the common
#      negations ("no lift", "without (a) lift") that would otherwise match.
# An EXPLICIT lift=false text fact ("no lift" quote) vetoes all sources.
LIFT_HAS_SQL = (
    "((EXISTS (SELECT 1 FROM listing_text_facts tf WHERE tf.rm_uuid = o.id"
    "          AND tf.ok = 1 AND json_extract(tf.facts_json,'$.lift.value') = 1)"
    "  OR lower(COALESCE(o.key_features,'')) LIKE '%\"lift%'"
    "  OR lower(COALESCE(o.key_features,'')) LIKE '% lift%'"
    "  OR lower(COALESCE(o.key_features,'')) LIKE '%elevator%'"
    "  OR (lower(COALESCE(o.full_description,'')) LIKE '% lift%'"
    "      AND lower(COALESCE(o.full_description,'')) NOT LIKE '%no lift%'"
    "      AND lower(COALESCE(o.full_description,'')) NOT LIKE '%without lift%'"
    "      AND lower(COALESCE(o.full_description,'')) NOT LIKE '%without a lift%'))"
    " AND NOT EXISTS (SELECT 1 FROM listing_text_facts tf WHERE tf.rm_uuid = o.id"
    "                 AND tf.ok = 1 AND json_extract(tf.facts_json,'$.lift.value') = 0))"
)


def _lift(c):
    if c.get("value") is False:
        return f"NOT {LIFT_HAS_SQL}", [], "", []
    return LIFT_HAS_SQL, [], "", []


# Cladding red flag — EXCLUSION is the common use ("no buildings with cladding issues").
# Source: listing_text_facts only (quote-gated LLM extraction). A listing is
# flagged when the description ADMITS a problem: cladding_issue=true, or
# ews1_status='issue_mentioned'. WEAK GUARANTEE by nature: most affected
# buildings never mention it, so value=false only removes the self-confessed
# few — it cannot certify the rest are clean (the /check red card uses the
# same data). value=true (only flagged listings — bargain hunting) kept for
# symmetry.
CLAD_HAS_SQL = (
    "EXISTS (SELECT 1 FROM listing_text_facts tf WHERE tf.rm_uuid = o.id"
    "        AND tf.ok = 1 AND (tf.cladding_issue = 1"
    "        OR json_extract(tf.facts_json,'$.ews1_status.value') = 'issue_mentioned'))"
)


def _cladding(c):
    if c.get("value") is False:
        return f"NOT {CLAD_HAS_SQL}", [], "", []
    return CLAD_HAS_SQL, [], "", []


def _station_walk(c):
    # Maximum walk to the NEAREST station, in MINUTES. rm_nearest_stations stores
    # straight-line miles (99.2% active coverage, 2026-06-10); ~20 min/mile (3 mph)
    # converts — straight-line under-states a real walk, so treat as approximate.
    # Listings with no station rows (~0.8%) don't match when this is set.
    mins = float(c["max_minutes"])
    if not 1 <= mins <= 60:
        raise ValueError(f"bad station_walk max_minutes: {c['max_minutes']}")
    return ("EXISTS (SELECT 1 FROM rm_nearest_stations ns WHERE ns.property_id = o.id"
            " AND ns.distance_miles <= ?)", [mins / STATION_WALK_MIN_PER_MILE], "", [])


def _epc(c):
    # Minimum EPC energy-efficiency SCORE (1-100; A≈92+, B 81-91, C 69-80, D 55-68,
    # E 39-54, F 21-38, G 1-20). Same column search_properties uses.
    return ("o.epc_energy_efficiency IS NOT NULL AND o.epc_energy_efficiency >= ?",
            [float(c["min"])], "", [])


def _council_tax(c):
    # Council-tax band CEILING (A cheapest … H dearest). GLOB '[A-H]' drops noise
    # (TBC/DELETED/I); single letters sort lexically. Mirrors search_properties.
    band = str(c["max"]).strip().upper()
    if band not in "ABCDEFGH" or len(band) != 1:
        raise ValueError(f"bad council_tax band: {c['max']}")
    return "o.council_tax_band GLOB '[A-H]' AND o.council_tax_band <= ?", [band], "", []


def _auction(c):
    # value=false → exclude auction listings (different buying process); the common
    # case. value=true → only auctions. (No NULLs in is_auction; 0/1 only.)
    return ("o.is_auction = 1" if c.get("value") else "o.is_auction = 0"), [], "", []


def _days_on_market(c):
    # "on the market ≥ N days" → first listed on/before today−N. COALESCE so the ~8.5%
    # of listings missing first_visible_date fall back to date_listed.
    n = int(c["min"])
    return ("date(COALESCE(o.first_visible_date, o.date_listed)) <= date('now', ?)",
            [f"-{n} day"], "", [])


def _relisted(c):
    # Active listing whose UNIT-LEVEL dpId also appears in a delisted listing,
    # with the prior listing's price within +/-30% of this one (NULL-tolerant) to
    # avoid false positives on new-build dpIds shared by different sibling flats.
    # Off-market-gap gate: the prior listing must have been delisted BEFORE the
    # current listing started — excludes overlapping concurrent duplicates (a 2nd
    # agent's listing that merely withdrew while this one stayed live), which were
    # 82% of hits (227/277) before this fix.
    where = ("EXISTS (SELECT 1 FROM rm_sales_overview d "
             "JOIN dp_quality q ON q.delivery_point_id = o.delivery_point_id "
             "WHERE d.delivery_point_id = o.delivery_point_id AND d.id != o.id "
             "AND d.delisted_date IS NOT NULL AND q.is_unit_level = 1 "
             "AND date(d.delisted_date) < date(COALESCE(o.first_visible_date, o.date_listed)) "
             "AND (o.asking_price IS NULL OR d.asking_price IS NULL "
             "OR d.asking_price BETWEEN o.asking_price * 0.7 AND o.asking_price * 1.3))")
    return (where, [], "", [])


def _price_reduced(c):
    # value=true → has any reduction (price_history drop OR "Reduced on" text).
    # within_days=N → reduction within N days (price_history date OR parsed text date).
    # min_drop_pct=X → cumulative drop ≥ X% — price_history ONLY (text has no amount);
    #   retro-only listings correctly fall out (the listing source doesn't publish historic amounts).
    clauses, params = [], []
    drop_exists = ("EXISTS (SELECT 1 FROM price_history p1 JOIN price_history p0 "
                   "ON p0.property_id=p1.property_id AND p0.date < p1.date "
                   "WHERE p1.property_id=o.id AND p1.asking_price < p0.asking_price)")
    if c.get("min_drop_pct") is not None:
        x = float(c["min_drop_pct"])
        clauses.append(
            "EXISTS (SELECT 1 FROM price_history p WHERE p.property_id=o.id "
            "GROUP BY p.property_id HAVING "
            "(MAX(CASE WHEN p.date=(SELECT MIN(date) FROM price_history WHERE property_id=o.id) THEN p.asking_price END) "
            "- o.asking_price) * 100.0 / MAX(CASE WHEN p.date=(SELECT MIN(date) FROM price_history WHERE property_id=o.id) THEN p.asking_price END) >= ?)")
        params.append(x)
    if c.get("within_days") is not None:
        n = int(c["within_days"])
        clauses.append(
            "(EXISTS (SELECT 1 FROM price_history p1 JOIN price_history p0 "
            "ON p0.property_id=p1.property_id AND p0.date < p1.date "
            "WHERE p1.property_id=o.id AND p1.asking_price < p0.asking_price "
            "AND date(p1.date) >= date('now', ?)) "
            "OR (o.listing_update_reason LIKE 'Reduced on %' "
            # "Reduced on DD/MM/YYYY": after "Reduced on " (11 chars) DD@12, MM@15, YYYY@18.
            "AND date(substr(o.listing_update_reason,18,4)||'-'||substr(o.listing_update_reason,15,2)||'-'||substr(o.listing_update_reason,12,2)) >= date('now', ?)))")
        params.extend([f"-{n} day", f"-{n} day"])
    if not clauses:
        clauses.append(f"({drop_exists} OR o.listing_update_reason LIKE 'Reduced%')")
    return " AND ".join(clauses), params, "", []


def _shared_ownership(c):
    # value=false → exclude shared-ownership (you'd own only a % share); common case.
    # value=true → only shared-ownership.
    return ("o.shared_ownership = 1" if c.get("value") else "o.shared_ownership = 0"), [], "", []


def _conservatory(c):
    # key_features text (broad) OR floorplan VLM flag (secondary). Same multi-source
    # pattern as _balcony — NOT floorplan-subset-only.
    has = ("(lower(COALESCE(o.key_features,'')) LIKE '%conservatory%'"
           " OR EXISTS (SELECT 1 FROM floorplan_vlm_results f"
           "            WHERE f.rm_uuid = o.id AND f.ok = 1 AND f.has_conservatory = 1))")
    return (f"NOT {has}" if c.get("value") is False else has), [], "", []


def _fireplace(c):
    has = ("(lower(COALESCE(o.key_features,'')) LIKE '%fireplace%'"
           " OR EXISTS (SELECT 1 FROM floorplan_vlm_results f"
           "            WHERE f.rm_uuid = o.id AND f.ok = 1 AND f.has_fireplace = 1))")
    return (f"NOT {has}" if c.get("value") is False else has), [], "", []


def _garage(c):
    # key_features OR the listing's parking field ("Garage") OR floorplan flag. More
    # specific than the generic `parking` condition (which also accepts driveways).
    has = ("(lower(COALESCE(o.key_features,'')) LIKE '%garage%'"
           " OR lower(COALESCE(o.parking,'')) LIKE '%garage%'"
           " OR EXISTS (SELECT 1 FROM floorplan_vlm_results f"
           "            WHERE f.rm_uuid = o.id AND f.ok = 1 AND f.has_garage = 1))")
    return (f"NOT {has}" if c.get("value") is False else has), [], "", []


def _terrace(c):
    # The listing's outdoor field (o.garden = "Terrace") is the clean source; key_features
    # text needs the "terraced" (house TYPE) exclusion to avoid false positives;
    # floorplan flag is the third source. "Has a terrace" — distinct from the
    # balcony room_area condition (which measures terrace SIZE).
    has = ("(lower(COALESCE(o.garden,'')) LIKE '%terrace%'"
           " OR (lower(COALESCE(o.key_features,'')) LIKE '%terrace%'"
           "     AND lower(COALESCE(o.key_features,'')) NOT LIKE '%terraced%')"
           " OR EXISTS (SELECT 1 FROM floorplan_vlm_results f"
           "            WHERE f.rm_uuid = o.id AND f.ok = 1 AND f.has_terrace = 1))")
    return (f"NOT {has}" if c.get("value") is False else has), [], "", []


# --- open_plan_kitchen (2026-09-08) -----------------------------------------
# Origin: a radar asking for an "open plan kitchen" had that half sit in
# unsupported_json. Source is listing prose,
# and prose about kitchens lies in two specific ways a plain LIKE '%open plan%'
# walks straight into. Measured over 72,395 active listings (2026-09-08), ~85
# rows hand-read:
#   1. FEATURE LISTS — "open plan living room, two double bedrooms, an updated
#      kitchen and a bathroom": both phrases present, but the kitchen is its own
#      item. This is ~8.6% of naive hits, the single biggest FP class.
#   2. ASPIRATION — "potential to create a larger open-plan kitchen/diner": the
#      open-plan kitchen is what a builder COULD make, not what is for sale.
# Rule: 'kitchen' within 70 chars AFTER 'open plan', with no list/sentence break
# and no unrelated ROOM word between them, and no potential/scope cue in the 60
# chars before. 20,060 active listings, 0 FP in two 30-row random reads plus an
# adversarial sweep. Two recall costs taken deliberately — only the FIRST
# 'open plan' occurrence is examined (-3.9% of true hits) and hedged prose is
# dropped wholesale. Misses are fine here; false positives are not, because the
# user set this filter precisely to stop seeing galley kitchens.
_OPK_WINDOW = 70        # chars after "open plan" to look for "kitchen"
_OPK_LOOKBACK = 60      # chars before "open plan" to look for aspiration cues
# A list/sentence boundary between the two words = separate features, not one
# room. Comma is deliberately NOT here: "open-plan reception, kitchen and dining
# room" is one room, and the room-word guard below catches the list case.
_OPK_BREAKS = ('"', '.', ';', ' & ', '<br', '</p>')
# A room that is never part of an open-plan kitchen cluster = we are reading a
# feature list. living/dining/reception/lounge/family ARE the cluster and are
# deliberately absent.
_OPK_OTHER_ROOMS = ('bedroom', 'bathroom', 'shower room', 'cloakroom', 'utility',
                    'garden', 'balcony', 'terrace', 'hallway', 'entrance',
                    'suite', 'study', 'garage', 'parking', 'storage')
_OPK_ASPIRE = ('potential to', 'scope to', 'opportunity to', 'could be made',
               'could be turned')


def _opk_text_sql(col):
    """Window predicate over one text column, in PURE SQL.

    build_where's output is executed by the radar matcher AND search_properties
    on connections this module does not own, so a conn.create_function() helper
    would pass in tests and raise "no such function" in production.
    """
    t = (f"replace(replace(lower(COALESCE(o.{col},'')),'open-plan','open plan'),"
         "'openplan','open plan')")
    pos = f"instr({t},'open plan')"
    seg = f"substr({t}, {pos}+9, {_OPK_WINDOW})"     # the window after the phrase
    kpos = f"instr({seg},'kitchen')"
    btw = f"substr({seg}, 1, {kpos}-1)"              # text lying between the two
    pre = (f"substr({t}, CASE WHEN {pos} > {_OPK_LOOKBACK}"
           f" THEN {pos}-{_OPK_LOOKBACK} ELSE 1 END, {_OPK_LOOKBACK})")
    parts = [f"{pos} > 0", f"{kpos} > 0"]
    parts += [f"instr({btw},'{b}') = 0" for b in _OPK_BREAKS + _OPK_OTHER_ROOMS]
    parts += [f"instr({pre},'{a}') = 0" for a in _OPK_ASPIRE]
    return "(" + " AND ".join(parts) + ")"


# Same multi-source shape as _conservatory/_terrace: broad text first, then the
# LLM open_tags as a narrow second source (~1.5k listings carry an
# open_plan_kitchen* tag) so a listing whose prose we cannot parse still lands.
OPK_HAS_SQL = (
    "(" + _opk_text_sql("key_features")
    + " OR " + _opk_text_sql("full_description")
    + " OR EXISTS (SELECT 1 FROM listing_text_facts tf WHERE tf.rm_uuid = o.id"
      " AND tf.ok = 1 AND tf.open_tags_json LIKE '%open_plan_kitchen%'))"
)


def _open_plan_kitchen(c):
    # value=false → the inverse, deliberately keeping unknowns like _parking:
    # someone who "doesn't want an open-plan kitchen" shouldn't be hurt by a
    # listing that doesn't say, and "the kitchen is not open-plan" is equally
    # unprovable.
    if c.get("value") is False:
        return f"NOT {OPK_HAS_SQL}", [], "", []
    return OPK_HAS_SQL, [], "", []


def _ex_council(c):
    # rm_sales_overview.ex_council_verdict ∈ ('likely','possible','unlikely',NULL).
    # The filter acts ONLY on the 'likely' tier (possible/unlikely/NULL are NOT
    # treated as ex-council either way). value=false → exclude likely (keep the
    # rest, including NULL). value=true → only likely.
    if c.get("value") is False:
        return ("(o.ex_council_verdict IS NULL OR o.ex_council_verdict != 'likely')",
                [], "", [])
    return "o.ex_council_verdict = 'likely'", [], "", []


def _pct(conn, pct):
    n = conn.execute("SELECT COUNT(*) FROM decor_zeroshot_score WHERE decor_score IS NOT NULL").fetchone()[0]
    if not n:
        return 0.0
    off = max(0, min(n - 1, int(n * pct)))
    row = conn.execute("SELECT decor_score FROM decor_zeroshot_score WHERE decor_score IS NOT NULL "
                       "ORDER BY decor_score LIMIT 1 OFFSET ?", (off,)).fetchone()
    return row[0] if row else 0.0


def build_where(conditions, conn=None, commute_buffer_min=0):
    """Returns (where_sql, params, join_sql). AND of all conditions.
    Raises ValueError on unknown type or empty/invalid condition.
    `conn` (read-only) is required only when a decor_tier condition is present
    (to resolve live percentile thresholds).
    `commute_buffer_min` widens the commute SECTOR screen (the radar matcher
    passes >0 and then verifies survivors per-property; search_properties leaves
    it 0). See _commute / scripts/radar/commute_verify.py.

    JOIN params precede WHERE params in the returned params list so a composed
    query `SELECT ... FROM o {joins} WHERE {where}` binds correctly. We collect
    join_params and where_params SEPARATELY across all conditions, then concatenate.
    """
    if not conditions:
        raise ValueError("empty conditions")

    # Collect join SQL fragments + params, and where SQL fragments + params
    # SEPARATELY so join params come first in the final binding list.
    all_join_sql = []
    all_join_params = []
    all_where_parts = []
    all_where_params = []
    n_commute = 0  # unique JOIN alias per commute condition (shc0, shc1, …)

    for c in conditions:
        t = c.get("type")
        if t not in TYPES:
            raise ValueError(f"unknown condition type: {t}")

        if t == "area":
            w, wp, j, jp = _area(c)
        elif t == "price":
            w, wp, j, jp = _range("o.asking_price", c)
        elif t == "bedrooms":
            w, wp, j, jp = _range("o.bedrooms", c)
        elif t == "bathrooms":
            w, wp, j, jp = _range("o.bathrooms", {"min": c.get("min")})
        elif t == "sqft":
            w, wp, j, jp = _range("o.floor_area_sqft", c)
        elif t == "property_type":
            w, wp, j, jp = _property_type(c)
        elif t == "tenure":
            w, wp, j, jp = _in("o.tenure", c)
        elif t == "ppsf":
            w, wp, j, jp = _ppsf(c)
        elif t == "postcode_score":
            w, wp, j, jp = _postcode_score(c)
        elif t == "green_space":
            w, wp, j, jp = _green_space(c)
        elif t == "mortgageable":
            w, wp, j, jp = _mortgageable(c)
        elif t == "prior_sale":
            w, wp, j, jp = _prior_sale(c)
        elif t == "commute":
            w, wp, j, jp = _commute(c, buffer_min=commute_buffer_min, alias=f"shc{n_commute}")
            n_commute += 1
        elif t == "decor_tier":
            w, wp, j, jp = _decor_tier(c, conn)
        elif t == "garden":
            w, wp, j, jp = _garden(c)
        elif t == "garden_facing":
            w, wp, j, jp = _garden_facing(c)
        elif t == "garden_sqft":
            w, wp, j, jp = _garden_sqft(c)
        elif t == "balcony":
            w, wp, j, jp = _balcony(c)
        elif t == "room_area":
            w, wp, j, jp = _room_area(c)
        elif t == "chain_free":
            w, wp, j, jp = _chain_free(c)
        elif t == "ex_council":
            w, wp, j, jp = _ex_council(c)
        elif t == "parking":
            w, wp, j, jp = _parking(c)
        elif t == "lease_years":
            w, wp, j, jp = _lease_years(c)
        elif t == "service_charge":
            w, wp, j, jp = _service_charge(c)
        elif t == "ground_rent":
            w, wp, j, jp = _ground_rent(c)
        elif t == "lift":
            w, wp, j, jp = _lift(c)
        elif t == "station_walk":
            w, wp, j, jp = _station_walk(c)
        elif t == "cladding":
            w, wp, j, jp = _cladding(c)
        elif t == "epc":
            w, wp, j, jp = _epc(c)
        elif t == "council_tax":
            w, wp, j, jp = _council_tax(c)
        elif t == "auction":
            w, wp, j, jp = _auction(c)
        elif t == "days_on_market":
            w, wp, j, jp = _days_on_market(c)
        elif t == "price_reduced":
            w, wp, j, jp = _price_reduced(c)
        elif t == "relisted":
            w, wp, j, jp = _relisted(c)
        elif t == "shared_ownership":
            w, wp, j, jp = _shared_ownership(c)
        elif t == "conservatory":
            w, wp, j, jp = _conservatory(c)
        elif t == "fireplace":
            w, wp, j, jp = _fireplace(c)
        elif t == "garage":
            w, wp, j, jp = _garage(c)
        elif t == "terrace":
            w, wp, j, jp = _terrace(c)
        elif t == "open_plan_kitchen":
            w, wp, j, jp = _open_plan_kitchen(c)
        else:
            raise ValueError(f"no builder for type: {t}")

        if not w:
            raise ValueError(f"condition produced empty predicate: {c}")

        if j:
            all_join_sql.append(j)
            all_join_params.extend(jp)
        all_where_parts.append(f"({w})")
        all_where_params.extend(wp)

    # JOIN params precede WHERE params so the composed query binds in order.
    params = all_join_params + all_where_params
    where_sql = " AND ".join(all_where_parts)
    join_sql = " ".join(all_join_sql)

    return where_sql, params, join_sql


# --- Coverage blind spots (2026-08-25) ----------------------------------------
# Two conditions produce, when "we have no data", an empty result identical to
# "nothing on the market":
#   commute — after LEFT JOIN sector_hub_commute, `minutes <= ?`: when the
#             sector has no row, minutes is NULL, the predicate is false, and
#             the listing is dropped before commute_verify ever sees it.
#   area    — outcodes aren't coverage-checked, so a radar outside the London
#             postal areas can be created and will match 0 forever.
# Dropping them is correct (we genuinely can't assert ≤N minutes); dropping
# them SILENTLY is the bug. The two functions below quantify the blind spot so
# create_radar / search_properties can say it out loud — same root as the
# search_properties multi-area fan-out rule "silent truncation makes 'never
# searched' look like 'searched, found nothing'".

# Aligned with LONDON_PREFIXES in web/src/lib/postcode.ts: the postal areas
# where we actually hold listings, area scores, crime and commute data. The two
# live in different language runtimes — change one, change the other.
# Note county borders ≠ postal borders: EN6 is Hertfordshire, KT6 is Surrey,
# DA1/BR6 are Kent — they ARE in this set with full data; AL/SG/HP/LU/TN/ME/
# CT/GU/RH/BN are not.
LONDON_POSTAL_AREAS = frozenset({
    "EC", "WC", "NW", "SE", "SW", "E", "N", "W",
    "BR", "CR", "DA", "EN", "HA", "IG", "KT", "RM", "SM", "TW", "UB", "WD",
})

# Postal-area letter prefix: "WD17"->"WD", "E14"->"E", "SW"->"SW".
_POSTAL_AREA_SQL = (
    "CASE WHEN substr(replace(o.postcode,' ',''),2,1) BETWEEN '0' AND '9'"
    " THEN substr(replace(o.postcode,' ',''),1,1)"
    " ELSE substr(replace(o.postcode,' ',''),1,2) END"
)


def _postal_area(value):
    """Letter prefix of an area value. "WD17" -> "WD", "E14" -> "E"."""
    letters = []
    for ch in "".join(str(value).split()).upper():
        if not ch.isalpha():
            break
        letters.append(ch)
    return "".join(letters)


def out_of_coverage_areas(values):
    """Return the area values outside London postal-area coverage (original
    order, upper-cased, spaces removed).

    Used by create_radar: these outcodes compile fine but are doomed to 0
    matches, so they must be reported to the user as unsupported instead of
    creating a radar that never fires.
    """
    if not isinstance(values, (list, tuple)):
        values = [values]
    return ["".join(str(v).split()).upper() for v in values
            if _postal_area(v) not in LONDON_POSTAL_AREAS]


def commute_coverage_gap(conditions, conn):
    """Quantify listings dropped ONLY because we have no commute data.

    Returns {"hub": canonical hub name, "n": total, "areas": [(postal area,
    count), ...]}, or None when the spec has no commute condition.

    Counts only listings with **no row for that hub** in sector_hub_commute — a
    genuine over-the-limit commute (row present, minutes too high) is not a
    blind spot, it is correct filtering. All other conditions still apply, so an
    over-budget home isn't counted just because it sits in a blind spot.

    With several commute conditions in one spec only the first hub is
    reported: sector_hub_commute is the full sector×hub cross product
    (1048×93 = 97,464 rows), so coverage is all-or-nothing per sector and the
    first hub's blind spot holds for the others too.
    """
    commutes = [c for c in conditions if c.get("type") == "commute"]
    if not commutes:
        return None
    hub = canon_hub(commutes[0]["hub"])
    others = [c for c in conditions if c.get("type") != "commute"]
    if others:
        where, params, join = build_where(others, conn=conn)
    else:
        where, params, join = "1=1", [], ""
    # The blind-spot JOIN goes BEFORE build_where's JOINs, so its ? precedes
    # build_where's parameter sequence (join params first, then where params)
    # and the binding is simply [hub] + params.
    sql = (f"SELECT {_POSTAL_AREA_SQL} AS area, COUNT(*) AS n"
           f" FROM rm_sales_overview o"
           f" LEFT JOIN sector_hub_commute _gap"
           f" ON _gap.sector = {_SECTOR_SQL} AND LOWER(_gap.hub) = LOWER(?)"
           f" {join}"
           f" WHERE ({where}) AND _gap.minutes IS NULL"
           f" AND o.delisted_date IS NULL"
           f" AND (o.canonical_id IS NULL OR o.canonical_id = o.id)"
           f" GROUP BY area ORDER BY n DESC, area")
    rows = conn.execute(sql, [hub] + list(params)).fetchall()
    return {"hub": hub, "n": sum(r[1] for r in rows),
            "areas": [(r[0], r[1]) for r in rows]}
