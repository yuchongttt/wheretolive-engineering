"""Shared walk-radius postcode helper (R3-1-refix).

Two traps this module owns so callers can't re-trip them:
  * postcode_coords is MIXED-format — 21,395 of 125,767 rows (17%) are stored
    without the space ("AL100BJ", ~10.7k of them London). Any exact-string
    equality against canonical spaced postcodes silently loses those rows
    (review finding 3: the R3-1 walk-radius truth was 88 pairs, not 57).
    Centre lookup tries both forms; results are normalised to spaced.
  * km-per-degree constants: 110.574 for latitude, 111.320·cos(lat) for
    longitude. (The repo's third copy of this math used 111.32 for both —
    ~0.7% radius error; lookup_area_by_landmark.py has the correct pair.)
"""
import math

KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LNG_EQUATOR = 111.320


def normalise_spaced(pc: str) -> str:
    """'e149aa' / 'E14 9AA' → 'E14 9AA' (UK incode is always 3 chars)."""
    s = "".join((pc or "").upper().split())
    if len(s) >= 5:
        return s[:-3] + " " + s[-3:]
    return s


def centre_of(conn, full_pc: str):
    """(lat, lng) for a full postcode from postcode_coords, tolerant of the
    table's unspaced rows; None when the postcode has no coordinates."""
    spaced = normalise_spaced(full_pc)
    row = conn.execute(
        "SELECT latitude, longitude FROM postcode_coords "
        "WHERE postcode IN (?, ?) LIMIT 1",
        (spaced, spaced.replace(" ", "")),
    ).fetchone()
    return (row[0], row[1]) if row else None


def unit_postcode_known(conn, pc: str) -> bool | None:
    """Does this FULL unit postcode exist in our data at all?

    True = seen in postcode_coords (either storage format), lr_transactions,
    or rm_sales_overview — we are (or were) listing a property there, which
    settles the question without a network call.
    False = a full-unit-shaped postcode with zero trace — likely
    non-residential, brand-new, or a typo; tools should say so LOUDLY instead
    of silently answering from sector/outcode aggregates as if the address
    exists (R3-2 T18b: "what is E14 5AB worth?" was answered with sector medians,
    burying that the postcode is a business address).
    None = input is not full-unit-shaped (outcode/sector) — check not
    applicable.

    The listings oracle exists because coords+LR alone made this a ~98%-false
    alarm where it fired: 1,762 CURRENTLY-LISTED properties (968 postcodes)
    tripped it, and 49 of 50 sampled postcodes came back live from
    postcodes.io. postcode_coords is missing 22% of active-listing postcodes,
    and LR only knows postcodes that have ever sold — so new-build developments
    (KT2 7FU) and re-used-but-terminated postcodes (E1W 2SG, London Dock) read
    as "does not exist" while we advertise homes there. Delisted rows count:
    the listing having ended says nothing about the address being real.
    """
    spaced = normalise_spaced(pc)
    compact = spaced.replace(" ", "")
    if len(compact) < 5 or not (compact[-3].isdigit() and compact[-2:].isalpha()):
        return None
    try:
        if conn.execute(
            "SELECT 1 FROM postcode_coords WHERE postcode IN (?, ?) LIMIT 1",
            (spaced, compact),
        ).fetchone():
            return True
        if conn.execute(
            "SELECT 1 FROM lr_transactions WHERE postcode = ? LIMIT 1",
            (spaced,),
        ).fetchone():
            return True
    except Exception:
        return None            # table absent on this deployment — no claim
    try:
        # Raw-column IN (both storage formats) keeps idx_sales_postcode usable;
        # wrapping the column in REPLACE/UPPER would force a full scan here.
        return conn.execute(
            "SELECT 1 FROM rm_sales_overview WHERE postcode IN (?, ?) LIMIT 1",
            (spaced, compact),
        ).fetchone() is not None
    except Exception:
        return False           # no listings table — fall back to the older verdict


UNKNOWN_POSTCODE_ALERT = (
    "⚠ POSTCODE NOT FOUND: {pc} has no trace in ANY of our datasets — no Land "
    "Registry sale, no coordinates, no listing. It is likely non-residential, "
    "brand-new, or a typo. The figures below are the surrounding {scope} "
    "aggregate and say NOTHING about this address. Tell the user, and run ONE "
    "targeted web search to identify what this postcode actually is before "
    "treating the address as real."
)


def postcodes_within(conn, lat: float, lng: float, km: float) -> list[str]:
    """Spaced-normalised postcodes within `km` of (lat, lng).

    Degree-box prefilter (uses idx_pc_lat_lng) + exact planar distance check.
    """
    kx = KM_PER_DEG_LNG_EQUATOR * math.cos(math.radians(lat))
    ky = KM_PER_DEG_LAT
    dlat, dlng = km / ky, km / kx
    rows = conn.execute(
        "SELECT postcode, latitude, longitude FROM postcode_coords "
        "WHERE latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?",
        (lat - dlat, lat + dlat, lng - dlng, lng + dlng),
    ).fetchall()
    return [
        normalise_spaced(r[0]) for r in rows
        if ((r[1] - lat) * ky) ** 2 + ((r[2] - lng) * kx) ** 2 <= km * km
    ]
