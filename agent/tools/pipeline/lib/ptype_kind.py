"""flat/house classification + description guard (pure functions, no IO).

Background (2026-08-27 review, prompted by a user questioning a £650k
"house"): when splitting rm_sales_overview.property_type into flat/house,
17-21% of the cheap end of the house bucket are flats/maisonettes mislabelled
by the agent (example: a 9th-floor one-bed in a tower block labelled
"Terraced"). property_type is typed in by the agent; classification keywords
can't protect against the field itself being wrong — only the description
text confesses ("this two-bedroom maisonette...").

Three pure functions:
- kind_from_property_type: property_type → 'flat' | 'house' | 'other'
- desc_contradicts_house:  is the description's subject a flat/maisonette
                           (strict zero-false-positive caliber)
- effective_kind:          combined — house bucket + description confession → 'flat'

Zero-false-positive design (same philosophy as the ex-council and planning-flag
enrichers: only positive evidence changes a verdict; ambiguity changes
nothing): contradicts is returned only when the description matches a
flat-subject pattern AND contains no house-subject evidence AND no
whole-building/annex signal. The cost is letting ambiguous rows through
(of 448 keyword hits, 87+25 were conservatively excluded, a few of them true
mislabels, e.g. the maisonette in "townhouse split into two") — prefer misses
to errors.

Measured false-positive rate (2026-08-27, 22,345 active canonical houses):
448 keyword hits, 336 tagged under the strict caliber, 70 hand-checked with
zero false positives. The false-positive text types (all blocked by the
exclusion rules; original sentences pinned in tests/test_ptype_kind.py):
large house with a self-contained annex flat / new-development ads selling
houses+apartments / whole freehold arranged as N flats / HMO / garage-conversion
annex.

The vocabulary matches PT_FLAT/PT_HOUSE in radar_vocab.py (tests assert
against drift); the one deliberate divergence: 'Block of Apartments' is
'other' here (a whole building doesn't belong in flat statistics) but 'flat'
in radar search (a user searching flats should find it).
"""

import re

# ── property_type: three buckets ────────────────────────────────────────────
# Non-residential / not-a-single-home categories: neither flat nor house.
# (Confirmed in the 2026-08-27 review: House Boat £1.4m / House Share £900k /
#  an Off-Plan flat £1.22m were all mixed into the house P25 corpus.)
OTHER_TYPES = frozenset({
    "house boat", "house share", "flat share", "off-plan",
    "block of apartments", "retirement property", "hotel room",
    "serviced apartments", "residential development",
    "house of multiple occupation", "equestrian facility",
    "park home", "mobile home", "land", "plot", "garages", "parking",
    "not specified", "",
    # Commercial (the listing source's commercial sub-types leak into the sales feed; before
    # 2026-09-24 they fell through to 'house' and get_sold_nearby
    # property_kind=house served an Office as a residential comp).
    "office", "serviced office", "commercial property", "commercial development",
    "retail property (high street)", "retail property (out of town)", "shop",
    "mixed use", "restaurant", "cafe", "takeaway", "pub", "bar / nightclub",
    "hotel", "guest house", "warehouse", "distribution warehouse", "industrial",
    "light industrial", "heavy industrial", "workshop", "showroom", "storage",
    "business park", "trade counter", "healthcare facility", "leisure facility",
    "place of worship", "childcare facility", "petrol station", "post office",
    "hairdresser / barber shop", "convenience store", "data centre",
    "science park", "farm", "smallholding", "campsite", "farm land",
})

# flat-bucket keywords (same source as the site's other SQL CASE expressions and radar_vocab.PT_FLAT)
_FLAT_KEYWORDS = ("flat", "apartment", "maisonette", "penthouse",
                  "studio", "duplex", "triplex")


def kind_from_property_type(property_type):
    """property_type → 'flat' | 'house' | 'other'.

    'other' is checked first (House Boat contains 'House', Block of Apartments
    contains 'Apartment', so they must be decided before the keywords);
    unknown types default to house (consistent with the existing CASE
    expressions).
    """
    if property_type is None:
        return "other"
    pt = property_type.strip().lower()
    if pt in OTHER_TYPES:
        return "other"
    if any(kw in pt for kw in _FLAT_KEYWORDS):
        return "flat"
    return "house"


# ── description guard ─────────────────────────────────────────────────────
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# Flat-subject evidence: the description calls itself a maisonette / X bedroom
# apartment / Xth floor apartment / this apartment. The bare word "flat" is
# deliberately not used (flat roof / flat fee noise).
_APT_RE = re.compile(
    r"maisonette|bedroom apartment|floor apartment|this apartment", re.I)

# House-subject evidence → ambiguous, don't tag.
# terraced/semi-detached/detached/end-of-terrace immediately followed by
# maisonette or garden is not house evidence ("terraced maisonette" is flat
# evidence; "terraced garden" is just a garden).
_HOUSE_SUBJ_RE = re.compile(
    r"\b(?:(?:terraced|semi[- ]detached|detached|end[- ]of[- ]terrace)\s+(?!maisonette|garden)"
    r"|town\s?house|mews house|family home|bedroom house|bedroom home"
    r"|freehold house|coach house|this house\b|cottage\b|bungalow\b)", re.I)

# Whole-building sale / annex signal → ambiguous, don't tag.
# ("self-contained X bedroom apartment" is almost always an annex of a large
#  house; "arranged as three flats" is a whole-building investment — note that
#  "arranged over three floors" is normal wording for a single home and is not
#  in this list.)
_BUILDING_RE = re.compile(
    r"arranged as|converted into|split into"
    r"|comprising\s+(?:two|three|four|five|six|\d+)\s+(?:self[- ]contained\s+)?(?:flats|apartments|maisonettes)"
    r"|freehold (?:building|block|investment)|whole building|self[- ]contained",
    re.I)


def desc_contradicts_house(text):
    """Is the description's subject a flat/maisonette? (Only meaningful for
    rows in the house bucket.)

    Returns the matched flat-evidence snippet (str, storable as evidence), or
    None when there is no match or the text is ambiguous.
    """
    if not text:
        return None
    t = _WS_RE.sub(" ", _TAG_RE.sub(" ", text))
    m = _APT_RE.search(t)
    if not m:
        return None
    if _HOUSE_SUBJ_RE.search(t) or _BUILDING_RE.search(t):
        return None  # ambiguous (annex / new-development ad / whole building / genuine house) — prefer a miss
    start = max(0, m.start() - 40)
    return t[start:m.end() + 40].strip()


def effective_kind(property_type, text):
    """Combined verdict: house bucket + description confessing a flat → 'flat';
    otherwise keep the property_type bucket.

    The reverse direction (flat label + house description) is out of scope for
    this gate — its false-positive rate hasn't been measured, so leave it alone.
    """
    kind = kind_from_property_type(property_type)
    if kind == "house" and desc_contradicts_house(text):
        return "flat"
    return kind


def kind_case_sql(col="property_type", flag_col="ptype_desc_flat"):
    """Generate the SQL CASE expression equivalent to kind_from_property_type
    + the description guard.

    Used by v_property_kind in build_semantic_views.py — the SQL is generated
    from the Python vocabulary rather than hand-copied, so the two sides can
    never drift (the tests compare every type one by one).
    """
    others = ",".join(f"'{t}'" for t in sorted(OTHER_TYPES))
    flat_like = " OR ".join(f"LOWER({col}) LIKE '%{kw}%'" for kw in _FLAT_KEYWORDS)
    return (f"CASE "
            f"WHEN LOWER(TRIM(COALESCE({col},''))) IN ({others}) THEN 'other' "
            f"WHEN {flat_like} THEN 'flat' "
            f"WHEN COALESCE({flag_col},0)=1 THEN 'flat' "
            f"ELSE 'house' END")


# ── schema glue ───────────────────────────────────────────────────────────
def ensure_ptype_schema(conn):
    """Idempotently add 3 columns (shared by the enrich_ptype_kind entrypoint
    and build_semantic_views, so the columns always exist when the view is
    created)."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(rm_sales_overview)")}
    for name, decl in (("ptype_desc_flat", "INTEGER"),
                       ("ptype_desc_evidence", "TEXT"),
                       ("ptype_desc_checked_at", "TEXT")):
        if name not in cols:
            conn.execute(f"ALTER TABLE rm_sales_overview ADD COLUMN {name} {decl}")
    conn.commit()
