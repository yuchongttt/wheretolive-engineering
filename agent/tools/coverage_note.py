#!/usr/bin/env python3
"""A "not found" outside our coverage boundary -> say coverage gap, never state it as fact.

Our LR transactions, active listings, area scores, price trends and commutes
all stop at the London postal areas (radar_vocab.LONDON_POSTAL_AREAS, 20
postal-area prefixes). The three address/price tools used to answer "not
found" for anything outside that boundary:

    Land Registry: no sales recorded in ME7 0ZZ.
    No Land Registry transactions found near ME7 0ZZ in the last 3 years.
    ME7 has no price analysis yet ... suggest /evaluate?postcode=ME7

The model has no second source to tell "genuinely no sales there" from "we
have no data for that whole area", so it relayed the text as-is. A homeowner asking
about a postcode outside our coverage got "**No Land
Registry sale on record** for this address" — which, to a homeowner, reads as
"this home has never sold". We hold zero LR rows for the ME area.

search_properties._area_coverage_note already separated the two cases (on a
zero result it first asks "do we hold stock for this area at all"); this
module extracts that same guard into a shared helper for three more tools.

Deliberately NOT done here:
  - Judging "inside coverage but genuinely empty". search_properties already
    does that, and it needs each tool's own denominator (an outcode with 0
    active listings today is not a data gap) — it doesn't belong in this layer.
  - Unrecognisable postal area (empty string / junk input) -> is_covered
    returns True and stays silent. Better to miss one warning than to brand a
    normal London postcode as a blind spot.
"""
from __future__ import annotations

# The coverage set has ONE definition, in radar_vocab — shared with the radar
# compiler and search_properties. _postal_area is the same parser: a second copy
# would inevitably fork from it after some change.
from radar_vocab import LONDON_POSTAL_AREAS, _postal_area

# Official price-paid data: outside our boundary we can't answer, but the user can look it up.
HMLR_PPD = "https://landregistry.data.gov.uk/app/ppd"


def postal_area(postcode) -> str:
    """'ME7 0ZZ' -> 'ME', 'E14 9AA' -> 'E'. Empty string when unrecognisable."""
    if not postcode:
        return ""
    return _postal_area(postcode)


def is_covered(postcode) -> bool:
    """Is this postcode inside our data coverage? True (silent) when the postal
    area can't be recognised.

    UK postal-area prefixes are at most 2 letters, so a "prefix" longer than
    that is not a postcode at all (free text was passed as the argument) —
    treat it as unrecognisable, don't brand it a blind spot.
    """
    area = postal_area(postcode)
    if not area or len(area) > 2:
        return True
    return area in LONDON_POSTAL_AREAS


def out_of_coverage_note(postcode, *, missing: str,
                         still_valid: str | None = None) -> str | None:
    """None inside coverage; outside it, the ⚠ block to put in the tool output.

    missing      -- what exactly this tool lacks outside the boundary (be
                    specific; don't write "all data").
    still_valid  -- optional: parts of the same output NOT affected by the
                    boundary (lookup_address's EPC / VOA sections come from
                    national registers and still work in ME7).
    """
    area = postal_area(postcode)
    if is_covered(postcode):
        return None
    pc = " ".join(str(postcode).split()).upper()
    note = (
        f"⚠️ NO COVERAGE, NOT ABSENCE: {pc} is in postal area {area}, which is "
        f"OUTSIDE our data footprint. We hold zero {missing} for {area} — so an "
        "empty result here is a hole in OUR data, not a fact about this "
        "property or this market. Do NOT tell the user the property has never "
        "sold, that no comparable sales exist, that nothing is on the market "
        "there, or that prices there cannot be known — we cannot see "
        f"{area} at all. Say we cover London only, and point them at the HM "
        f"Land Registry price-paid search ({HMLR_PPD}) or a property portal for that "
        "area. Our footprint is these postal areas only: "
        + ", ".join(sorted(LONDON_POSTAL_AREAS))
        + ". County borders are NOT postal borders — the covered slice of the "
        "Home Counties (WD Watford + EN6 Potters Bar = Herts, KT = Surrey, "
        "BR/DA = Kent) IS covered, so do not widen this caveat to them."
    )
    if still_valid:
        note += " " + still_valid
    return note
