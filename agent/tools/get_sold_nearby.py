#!/usr/bin/env python3
"""MCP server: get_sold_nearby

Recently SOLD properties near a postcode/outcode — listings we detected as
delisted and matched to a Land Registry sale (rm_delisted_properties). Returns sold price, sold date, beds/type, and the owner's
realised return (capital gain % and annualised %) where we have a confident
match. Distinct from get_comparables (which pulls raw LR comparables for
pricing): this answers "what actually changed hands here lately, and did
owners make money?".

Cohort mode (bought_after / bought_before): answers "did people who bought
in period X make money?" from v_resale_outcomes (Land Registry repeat-sale
pairs, guardrails baked into the view — see scripts/build_semantic_views.py).
Sample ladder: exact postcode detail → ~800m walk radius → sector → outcode,
using the narrowest rung with enough STANDARD-market pairs (both legs LR
category A; HMLR's own HPI is category-A only). Category B pairs
(repossessions / mortgage-identifiable buy-to-lets / corporate transfers)
are excluded from the headline but disclosed, and inlined as a second set of
stats when they materially change the answer — the bias direction flips
locally (R3-1: sector-level B legs were fake losers, walk-radius B legs were
fake auction-entry winners), so neither silent inclusion nor silent removal
is honest.

Read-only, no auth (the chat handler is the trust boundary).
"""

import asyncio
import logging
import os
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Any

from geo_radius import centre_of, postcodes_within
from lr_category import (CATEGORY_B_GLOSS, LR_TYPE, category_label,
                         is_standard)

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s get_sold_nearby %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Data directory holding evaluations.db: $WTL_DATA_DIR, else ./data next to this file.
DATA_DIR = Path(os.environ.get("WTL_DATA_DIR") or Path(__file__).resolve().parent / "data")
DB_PATH = DATA_DIR / "evaluations.db"

# One definition of flat/house/other bucketing (v_property_kind's CASE), so the
# property_kind filter here can never drift from what /map and search use.
# (pipeline/ holds the helpers shared with the site's batch scripts.)
sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline"))
from lib.ptype_kind import kind_case_sql  # noqa: E402

# Min match_score for a trustworthy listing↔LR match (and thus a trustworthy
# realised-return figure). Matches the threshold the rest of the app uses.
MIN_MATCH = 60

server: Server = Server("get-sold-nearby")


def outcode_of(pc: str) -> str:
    return normalise_pc(pc)[0]


def normalise_pc(pc: str) -> tuple[str, str | None]:
    """Return (outcode, sector-or-None), tolerant of missing spaces.

    UK incode is always 3 chars, so 'E149AA' -> ('E14', 'E14 9') just like
    'E14 9AA'. An outcode alone ('E14') -> ('E14', None).
    """
    compact = "".join((pc or "").upper().split())
    if not compact:
        return "", None
    if len(compact) > 4 and compact[-3].isdigit() and compact[-2:].isalpha():
        out, incode = compact[:-3], compact[-3:]
        return out, f"{out} {incode[0]}"
    return compact, None


def dedupe_sales(rows) -> list[dict]:
    """Collapse multi-agent listings of ONE Land Registry sale to one row.

    rm_delisted_properties keeps every listing that matched a sale, so a
    property marketed by two agents (or re-listed) appears twice with the
    same lr_address/lr_date/lr_price (13.8% of confident rows, 2026-09-24;
    HA1's 15-row window held 4 duplicate groups, HA2's had one sale x3).
    Left as-is they eat the LIMIT window and read as two sales. Key is the
    LR side (address fallback), case/space-insensitive; first row wins.
    """
    seen: set[tuple] = set()
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        addr = " ".join(str(d.get("lr_address") or d.get("address") or "").upper().split())
        key = (addr, d.get("lr_date"), d.get("lr_price"))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_sold_nearby",
            description=(
                "Recently sold properties in an outcode — listings "
                "that delisted and matched a Land Registry sale. Returns sold "
                "price, sold date, beds/type, and realised return (capital "
                "gain % and annualised %) where the match is confident. Use "
                "for 'what's sold around here recently' / 'are people making "
                "money in this area' / 'what did similar places actually go "
                "for'. For raw pricing comparables prefer get_comparables. "
                "Cohort mode: pass bought_after/bought_before to ask 'did "
                "people who BOUGHT in period X (e.g. the pandemic) make money "
                "when they resold?' — returns confirmed Land Registry "
                "buy→sell pairs with a summary (n, median return, "
                "losers/gainers). Cohort headline uses standard open-market "
                "sales only (both legs LR category A, the HMLR HPI "
                "convention) at the narrowest adequate rung of ~800m walk "
                "radius → sector → outcode; category B (repossessions / "
                "buy-to-let / corporate) counts are disclosed, with a second "
                "including-B reading when it materially differs, plus a "
                "purchase-year split."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "postcode": {
                        "type": "string",
                        "description": "UK postcode or outcode, e.g. 'SE15' or 'SE15 3AB'. Recent-sales mode matches at outcode level. In cohort mode pass the FULL postcode when you have one — the answer then leads with the ~800m walk-radius sample (what 'nearby' actually means), falling back to sector then outcode only when the standard-sale sample is thin.",
                    },
                    "bedrooms": {
                        "type": "integer",
                        "description": "Optional — restrict to this bedroom count (±1).",
                    },
                    "property_kind": {
                        "type": "string",
                        "enum": ["flat", "house"],
                        "description": "Optional (both modes; cohort mode maps LR type F=flat, T/S/D=house) — restrict to flats (Flat/Apartment/Maisonette/Studio…) or houses (Terraced/Semi/Detached/Bungalow…). Kind comes from the listing's own property_type label, bucketed by the same rule as v_property_kind; a minority of cheap 'house'-labelled listings are really flats, so a flat filter may MISS some flats but never returns a house. Use it for 'what did 3-bed FLATS sell for' — get_comparables cannot filter by kind+bedrooms.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 15, max 40).",
                    },
                    "bought_after": {
                        "type": "string",
                        "description": "Cohort mode — only pairs PURCHASED on/after this date (YYYY-MM-DD), e.g. 2020-03-01 for pandemic buyers.",
                    },
                    "bought_before": {
                        "type": "string",
                        "description": "Cohort mode — only pairs purchased before this date (YYYY-MM-DD).",
                    },
                },
                "required": ["postcode"],
            },
        )
    ]


MIN_COHORT_N = 3  # below this: honest "insufficient data", never stats

# Cohort-mode walk radius (km). ~10 minutes on foot — the denominator a
# "this postcode and nearby" question actually means (R3-1, Fix AN).
RADIUS_KM = 0.8

# |pct_change| at/above this gets a verify-before-quoting flag: auctions and
# lease/share events pass as LR category A (R3-1 measured +67.6% auction
# entry and a -68.1% lease event, both category A), so category filtering
# alone cannot catch them (Fix AQ).
EXTREME_PCT = 50.0

# Column list for pair fetches (LR_TYPE now shared via lr_category).
_PAIR_COLS: tuple[str, ...] = (
    "postcode", "first_date", "last_date", "first_price", "last_price",
    "pct_change", "hold_years", "property_type", "paon", "saon", "street",
    "first_category", "last_category",
)


def _annualised(r) -> float | None:
    hy = r["hold_years"] or 0
    if hy < 0.5 or not r["first_price"]:
        return None
    return ((r["last_price"] / r["first_price"]) ** (1 / hy) - 1) * 100


# London median sale price by year — LR category A, the 20 London postal areas,
# computed 2026-09-27 from lr_transactions. Only an index for what "the market
# did" between two dates, used to judge pre-2005 purchases (see _is_market_era).
_LONDON_MEDIAN_BY_YEAR = {
    1995: 73_200, 1996: 77_500, 1997: 86_000, 1998: 97_000, 1999: 117_500,
    2000: 137_000, 2001: 154_000, 2002: 180_000, 2003: 199_995, 2004: 219_995,
    2005: 228_000, 2006: 242_425, 2007: 255_000, 2008: 250_000, 2009: 250_000,
    2010: 282_500, 2011: 285_000, 2012: 295_000, 2013: 315_000, 2014: 352_000,
    2015: 390_000, 2016: 425_506, 2017: 449_110, 2018: 450_000, 2019: 455_000,
    2020: 481_000, 2021: 497_000, 2022: 512_500, 2023: 515_000, 2024: 520_000,
    2025: 515_000, 2026: 495_000,
}
_ERA_RULES_BEFORE = 2005


def _median_for(year: int) -> float | None:
    if year in _LONDON_MEDIAN_BY_YEAR:
        return _LONDON_MEDIAN_BY_YEAR[year]
    if year > max(_LONDON_MEDIAN_BY_YEAR):
        return _LONDON_MEDIAN_BY_YEAR[max(_LONDON_MEDIAN_BY_YEAR)]
    return None


def _is_market_era(r) -> bool:
    """A pre-2005 purchase judged against the market of its time. The fixed
    rules in _is_market (sub-£50k, doubling in < 8 years, > +20%/yr) are
    today's prices: London's median was £73k in 1995 and rose 2.3x from 1996 to
    2002, so they dropped 40–50% of all 1995–99 London purchases (2026-09-27,
    2.08M repeat pairs) — and the dropped pairs tracked the market to within
    1–3 points a year. Market here: a first price of at least 30% of that
    year's London median (never above the £50k floor), an annualised return
    no more than 12 points above the median's over the same years, and no
    worse than -35%/yr."""
    try:
        fy, ly = int(str(r["first_date"])[:4]), int(str(r["last_date"])[:4])
    except (TypeError, ValueError):
        return False
    m1, m2 = _median_for(fy), _median_for(ly)
    if fy >= _ERA_RULES_BEFORE or not m1 or not m2:
        return False
    if not r["first_price"] or r["first_price"] < min(50_000, 0.3 * m1):
        return False
    hy = r["hold_years"] or 0
    ann = _annualised(r)
    if ann is None or hy < 0.5:
        return False
    market_ann = ((m2 / m1) ** (1 / hy) - 1) * 100
    return ann - market_ann <= 12 and ann >= -35


def _is_market(r) -> bool:
    """Arm's-length market pair. Pre-2005 purchases that the fixed rules below
    drop get a second look against the market of their time (_is_market_era);
    that only ever adds pairs back."""
    return _is_market_fixed(r) or _is_market_era(r)


def _is_market_fixed(r) -> bool:
    # Drop non-arm's-length pairs that pollute the gain/loss split and the
    # listed rows: a <£50k "sale" is a transfer/share (not a market price),
    # and a sustained annualised move beyond ~+20%/yr (or below ~-35%/yr)
    # signals shared-ownership staircasing (bought a cheap % share, later
    # bought/sold the whole) or a related-party price. Genuine long holds
    # stay in (e.g. +294% over 22 yrs = +6%/yr passes).
    if not r["first_price"] or r["first_price"] < 50000:
        return False
    # A doubling in under ~8 years is not a flat's market return (off-plan
    # discount → completion price, staircasing, or a related-party deal).
    if r["pct_change"] is not None and r["pct_change"] > 80 and (r["hold_years"] or 0) < 8:
        return False
    ann = _annualised(r)
    if ann is not None and (ann > 20 or ann < -35):
        return False
    return True


def _is_clean(r) -> bool:
    """Both legs LR category A — a standard open-market buy AND sell.
    NULL/unrecorded legs are NOT clean (never assert cleanliness the data
    doesn't record) — the shared lr_category convention."""
    return is_standard(r["first_category"]) and is_standard(r["last_category"])


def _kind_ok(r, kind: str | None) -> bool:
    """property_kind filter for LR pairs: F = flat, T/S/D = house, O = neither.
    Cohort mode used to drop the argument silently while the playbook tells
    the model to ask for 'flats bought in 2023' (reviewer #3)."""
    if not kind:
        return True
    pt = (r["property_type"] or "").upper()
    return pt == "F" if kind == "flat" else pt in ("T", "S", "D")


def _date_where(bought_after, bought_before, col="first_date"):
    where, params = [], []
    if bought_after:
        where.append(f"{col} >= ?")
        params.append(bought_after)
    if bought_before:
        where.append(f"{col} < ?")
        params.append(bought_before)
    return where, params


def _fetch_geo(conn, geo_col, geo_val, bought_after, bought_before):
    dw, dp = _date_where(bought_after, bought_before)
    sql = (f"SELECT {', '.join(_PAIR_COLS)} FROM v_resale_outcomes "
           f"WHERE {geo_col} = ? AND is_single_hold = 1"
           + "".join(f" AND {w}" for w in dw)
           + " ORDER BY last_date DESC")
    return conn.execute(sql, [geo_val, *dp]).fetchall()


def _fetch_radius(conn, full_pc, bought_after, bought_before):
    """(pairs, unavailable_reason) within RADIUS_KM of full_pc.

    pairs is None exactly when the rung is unavailable, and the reason says
    why — the ladder must be able to tell the model "no coordinates for this
    postcode" apart from "0 pairs within the radius" (review finding 15).
    Coords lookups go through geo_radius, which normalises the mixed
    spaced/unspaced storage of postcode_coords (review finding 3).
    """
    try:
        centre = centre_of(conn, full_pc)
        if centre is None:
            return None, f"no coordinates recorded for {full_pc}"
        pcs = postcodes_within(conn, centre[0], centre[1], RADIUS_KM)
    except sqlite3.OperationalError:
        return None, "postcode_coords absent on this deployment"
    if not pcs:
        return [], None
    dw, dp = _date_where(bought_after, bought_before)
    rows: list[sqlite3.Row] = []
    for i in range(0, len(pcs), 500):
        chunk = pcs[i:i + 500]
        sql = (f"SELECT {', '.join(_PAIR_COLS)} FROM v_resale_outcomes "
               f"WHERE postcode IN ({','.join('?' * len(chunk))}) "
               "AND is_single_hold = 1"
               + "".join(f" AND {w}" for w in dw))
        rows.extend(conn.execute(sql, [*chunk, *dp]).fetchall())
    rows.sort(key=lambda r: r["last_date"], reverse=True)
    return rows, None


def _staleness_note(conn) -> str:
    """Fix AL consumer gate: pairs are a materialized build; when the live LR
    table has moved past the build's source snapshot, recent resales are
    silently missing — say so loudly instead of presenting stale stats."""
    try:
        meta = conn.execute(
            "SELECT source_max_date FROM semantic_meta "
            "WHERE key = 'v_resale_outcomes'"
        ).fetchone()
        if not meta or not meta[0]:
            return ""
        live = conn.execute("SELECT MAX(date) FROM lr_transactions").fetchone()[0]
        if live and live[:7] > meta[0][:7]:
            return (
                f"⚠ STALE PAIRS DATA: Land Registry now runs to {live[:7]} but "
                f"the buy→sell pairs were last built from data to {meta[0][:7]} "
                "— resales completed since then are missing from every figure "
                "below. Tell the user the stats are as of that earlier date "
                "(fix: run scripts/build_semantic_views.py).\n"
            )
    except sqlite3.OperationalError:
        return ""              # older deployment without semantic_meta
    return ""


def _suffixes(r, with_market_guard: bool = False) -> str:
    """Shared row-label suffixes: category, extreme move, and (for rows
    shown OUTSIDE the guarded stats, e.g. the exact-postcode lead rows)
    the non-market-pattern warning — review findings 9/13."""
    out = ""
    if not _is_clean(r):
        worse = (r["first_category"]
                 if not is_standard(r["first_category"])
                 else r["last_category"])
        out += category_label(worse)
    if r["pct_change"] is not None and abs(r["pct_change"]) >= EXTREME_PCT:
        out += (" · ⚠ extreme move — verify legs before quoting "
                "(auctions / lease events / repossessions can pass as "
                "LR category A)")
    if with_market_guard and not _is_market(r):
        out += (" · ⚠ non-market pattern (staircasing / related-party / "
                "sub-£50k) — excluded from the area stats below")
    return out


def _row_line(r) -> str:
    addr = " ".join(x for x in (r["saon"], r["paon"], r["street"]) if x) \
           or r["postcode"]
    rate = _annualised(r)
    ann = f" ({rate:+.1f}%/yr)" if rate is not None else ""
    return (
        f"- {addr.title()} ({r['postcode']}) · bought £{r['first_price']:,} "
        f"{r['first_date'][:7]} → sold £{r['last_price']:,} "
        f"{r['last_date'][:7]} · {r['pct_change']:+.1f}%{ann}"
        + (f" · {LR_TYPE.get(r['property_type'], r['property_type'])}"
           if r["property_type"] else "")
        + _suffixes(r)
    )


def _set_stats(rows):
    pcts = [r["pct_change"] for r in rows]
    losers = sum(1 for p in pcts if p < 0)
    return {
        "n": len(rows),
        "median": statistics.median(pcts),
        "losers": losers,
        "gainers": len(pcts) - losers,
        "loss_share": 100.0 * losers / len(pcts),
        "med_hold": statistics.median(r["hold_years"] for r in rows),
    }


def cohort_outcomes(
    oc: str,
    sector: str | None,
    full_pc: str | None,
    bought_after: str | None,
    bought_before: str | None,
    beds: Any,
    limit: int,
    kind: str | None = None,
) -> list[types.TextContent]:
    """Confirmed buy→sell pairs (LR repeat sales) for a purchase-date cohort.

    Sample ladder (Fix AN): exact-postcode detail is always shown, then the
    headline uses the NARROWEST of [~800m walk radius → sector → outcode]
    whose sample reaches MIN_COHORT_N at the current honesty tier — tier 1
    needs STANDARD-market pairs (both legs category A + market guards, the
    only tier allowed to claim a clean headline); tier 2 falls back to
    all-category market pairs with the headline saying exactly that; tier 3
    is the raw set with guards not applied, also saying so. The widening
    decision always uses the tier's own filtered n (Fix AH).
    """
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        try:
            return _cohort_impl(conn, oc, sector, full_pc,
                                bought_after, bought_before, beds, limit, kind)
        except sqlite3.OperationalError as exc:
            if "v_resale_outcomes" in str(exc) or "first_category" in str(exc):
                return [types.TextContent(
                    type="text",
                    text="Resale-outcome data unavailable on this deployment "
                         "(semantic views not built or predate the category "
                         "columns — run scripts/build_semantic_views.py). "
                         "Fall back to get_price_trend for area-level "
                         "appreciation.",
                )]
            raise
    finally:
        conn.close()


def _cohort_impl(
    conn: sqlite3.Connection,
    oc: str,
    sector: str | None,
    full_pc: str | None,
    bought_after: str | None,
    bought_before: str | None,
    beds: Any,
    limit: int,
    kind: str | None = None,
) -> list[types.TextContent]:
    window = f"bought {bought_after or 'any'} → {bought_before or 'any'}"
    if kind:
        window += f" · property_kind={kind} (LR type {'F' if kind == 'flat' else 'T/S/D'} pairs only)"

    def _kept(rows):
        return None if rows is None else [r for r in rows if _kind_ok(r, kind)]
    # Computed up front and prepended to EVERY return path — a thin/empty
    # sample is the primary symptom of a stale build, so the insufficient
    # path needs the warning most of all (review finding 8).
    stale = _staleness_note(conn)

    exact_note = ""
    if full_pc:
        exact_rows = _kept(_fetch_geo(conn, "postcode", full_pc,
                                      bought_after, bought_before))
        if exact_rows:
            details = "; ".join(
                f"{r['paon']} {r['street']}".strip().title()
                + f" £{r['first_price']:,} {r['first_date'][:7]} → "
                  f"£{r['last_price']:,} {r['last_date'][:7]} "
                  f"({r['pct_change']:+.1f}%)"
                + _suffixes(r, with_market_guard=True)
                for r in exact_rows[:3]
            )
            exact_note = (
                f" Exact postcode {full_pc} itself: {len(exact_rows)} "
                f"pair(s), all categories, unfiltered — {details}. Lead with "
                "these (they ARE the user's postcode) — including their "
                "warning labels — then give the wider-area stats below "
                "as context."
            )
        else:
            # Nothing in-window: say what would make exact-postcode data
            # exist, so the model can answer "what if we widen further?"
            alltime = _kept(_fetch_geo(conn, "postcode", full_pc, None, None))
            if not alltime:
                exact_note = (
                    f" Exact postcode {full_pc} itself: no buy→sell pairs "
                    "at ALL in Land Registry — widening the purchase-date "
                    "window can never surface any; only the wider area "
                    "below (or a specific address via lookup_address) "
                    "helps. Explain this to the user."
                )
            else:
                buys = sorted(r["first_date"] for r in alltime)
                exact_note = (
                    f" Exact postcode {full_pc} itself: 0 pairs in this "
                    f"window. All-time it has {len(alltime)} pairs, with "
                    f"purchase dates {buys[0][:7]} → {buys[-1][:7]}. "
                    "Widening the time window only helps inside that "
                    f"range; buyers after {buys[-1][:7]} haven't resold "
                    "yet, so new exact-postcode data appears only when "
                    "they do. Explain this to the user; the wider-area "
                    "stats below are the honest alternative."
                )

    # --- rung ladder: radius → sector → outcode, all computed so the model
    # can cross-reference wider samples without a second hop --------------
    rung_data: list[dict] = []
    radius_present = False
    ladder_extras: list[str] = []
    if full_pc:
        rad, rad_reason = _fetch_radius(conn, full_pc,
                                        bought_after, bought_before)
        rad = _kept(rad)
        if rad is not None:
            radius_present = True
            rung_data.append({"label": f"within ~800m of {full_pc}",
                              "raw": rad})
        else:
            ladder_extras.append(
                f"(walk-radius rung unavailable — {rad_reason})")
    if sector:
        rung_data.append({"label": f"sector {sector}",
                          "raw": _kept(_fetch_geo(conn, "pc_sector", sector,
                                                  bought_after, bought_before))})
    rung_data.append({"label": f"outcode {oc}",
                      "raw": _kept(_fetch_geo(conn, "pc_outcode", oc,
                                              bought_after, bought_before))})
    for d in rung_data:
        d["market"] = [r for r in d["raw"] if _is_market(r)]
        d["clean"] = [r for r in d["market"] if _is_clean(r)]

    if radius_present:
        ladder_extras.append(
            "(radius counts cover only postcodes with recorded coordinates)")
    ladder_note = "Sample ladder: " + " · ".join(
        f"{d['label']}: {len(d['clean'])} standard / "
        f"{len(d['market'])} incl. category B"
        for d in rung_data
    ) + ". " + " ".join(ladder_extras)

    # Honesty tiers, each scanned NARROWEST-first (the tool's contract; the
    # first version scanned fallbacks widest-first and re-labelled their
    # B-inclusive stats as double-A — review findings 1/15):
    #   clean  → headline may claim "both legs category A";
    #   market → all-category stats, headline says so, no clean claims;
    #   raw    → guards not applied, headline says so, no exclusion claims.
    chosen, tier = None, None
    for t in ("clean", "market", "raw"):
        chosen = next((d for d in rung_data
                       if len(d[t]) >= MIN_COHORT_N), None)
        if chosen is not None:
            tier = t
            break
    if chosen is None:
        widest = rung_data[-1]
        return [types.TextContent(
            type="text",
            text=stale
                 + f"Insufficient data: only {len(widest['raw'])} confirmed "
                   f"buy→sell pair(s) in {widest['label']} for cohort "
                   f"{window}. Not enough to generalise — say so rather "
                   f"than estimating. {ladder_note}{exact_note}",
        )]

    scope = chosen["label"]
    stats_set = chosen[tier]
    n_clean = len(chosen["clean"])
    excluded_b = len(chosen["market"]) - n_clean
    guard_excluded = len(chosen["raw"]) - len(chosen["market"])

    s = _set_stats(stats_set)
    small_note = (" Small sample — treat as indicative, not conclusive."
                  if s["n"] < 10 else "")

    if tier == "clean":
        headline = (
            f"Resale outcomes {scope}, cohort {window}: n={s['n']} "
            "standard-market single-hold pairs (Land Registry repeat sales, "
            "both legs category A — the HMLR house-price-index convention)."
        )
        if guard_excluded:
            headline += (
                f" Excluded {guard_excluded} likely non-market pair(s) "
                "(shared-ownership staircasing / related-party / sub-£50k "
                "transfers)."
            )
        disclosure = (
            f"{excluded_b} non-standard (LR category B) pair(s) excluded "
            f"from the headline — {CATEGORY_B_GLOSS}."
        )
        year_header = "By purchase year (standard sales):"
    elif tier == "market":
        headline = (
            f"Resale outcomes {scope}, cohort {window}: n={s['n']} "
            "single-hold pairs, ALL sale categories — only "
            f"{n_clean} standard-market (category A) pair(s) exist at this "
            "rung, too few for a standard-only headline. Say this to the "
            "user; do NOT present these stats as clean market outcomes."
        )
        disclosure = (
            f"{len(chosen['market']) - n_clean} of these {s['n']} pairs are "
            f"category B ({CATEGORY_B_GLOSS}) — included, NOT excluded."
        )
        year_header = "By purchase year (all categories, includes B):"
    else:
        headline = (
            f"Resale outcomes {scope}, cohort {window}: n={s['n']} raw "
            "pair(s) — too few survive the market-sale guards, so the "
            "guards are NOT applied here; the set may include non-market "
            "transfers and category B sales. Say this to the user."
        )
        disclosure = ""
        year_header = "By purchase year (all categories, unguarded):"

    lines = [
        headline + exact_note,
        f"Summary: median total return {s['median']:+.1f}% · {s['losers']} "
        f"sold at a loss / {s['gainers']} at or above break-even "
        f"({s['loss_share']:.0f}% loss share) · median hold "
        f"{s['med_hold']:.1f} yrs." + small_note,
    ]
    if disclosure:
        lines.append(disclosure)

    # Fix AH: when category B materially changes the answer, inline BOTH
    # readings — the model picks by question semantics (a landlord/BTL
    # question is partly ABOUT category B).
    if tier == "clean" and excluded_b:
        sm = _set_stats(chosen["market"])
        material = (
            excluded_b / len(chosen["market"]) >= 0.05
            or abs(sm["median"] - s["median"]) >= 1.0
            or abs(sm["loss_share"] - s["loss_share"]) >= 2.0
        )
        if material:
            lines.append(
                f"Including category B: n={sm['n']}, median "
                f"{sm['median']:+.1f}%, {sm['loss_share']:.0f}% sold at a "
                "loss. Use this reading only when the question is about "
                "landlord/buy-to-let or distressed sellers — and say it is "
                "partial: LR cannot tag cash landlord purchases, and "
                + (f"{guard_excluded} pair(s) that failed the non-market "
                   "guards (fire-sale/transfer prices) are in NEITHER "
                   "reading." if guard_excluded else
                   "extreme fire-sale prices may fail the non-market "
                   "guards and sit in neither reading.")
            )

    lines.append(ladder_note)

    # Fix AO: purchase-year split — outcome timing is the strongest split
    # variable in cohort questions (2022 buyers fare far worse than 2020).
    years: dict[str, list] = {}
    for r in stats_set:
        years.setdefault(r["first_date"][:4], []).append(r)
    quotable = {y: rs for y, rs in sorted(years.items())
                if len(rs) >= MIN_COHORT_N}
    if len(quotable) >= 2:
        lines.append(year_header)
        for y, rs in quotable.items():
            ys = _set_stats(rs)
            lines.append(
                f"  {y}: n={ys['n']}, median {ys['median']:+.1f}%, "
                f"{ys['loss_share']:.0f}% sold at a loss"
            )
        thin = [y for y in sorted(years) if y not in quotable]
        if thin:
            lines.append(
                "  (" + ", ".join(f"{y}: n={len(years[y])}" for y in thin)
                + " — too few to quote)"
            )

    if beds is not None:
        lines.append("(bedroom filter not available for LR pairs — ignored)")

    listing = chosen["raw"] if tier == "raw" else chosen["market"]
    lines.append(f"Most recent {min(limit, len(listing))} (all categories, "
                 "labelled):")
    for r in listing[:limit]:
        lines.append(_row_line(r))

    lines.append(
        "\n(Single-hold pairs; multi-owner chains excluded. Nominal price "
        "changes only — SDLT, agent & legal fees, and inflation are NOT "
        "deducted; compute_buying_costs can model the net outcome for a "
        "specific purchase.)"
    )
    return [types.TextContent(type="text", text=stale + "\n".join(lines))]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any] | None
) -> list[types.TextContent]:
    if name != "get_sold_nearby":
        raise ValueError(f"Unknown tool: {name}")

    args = arguments or {}
    oc, sector = normalise_pc(args.get("postcode") or "")
    if not oc:
        return [types.TextContent(type="text", text="Missing postcode.")]
    beds = args.get("bedrooms")
    limit = max(1, min(int(args.get("limit", 15)), 40))
    kind = args.get("property_kind")
    if kind is not None and kind not in ("flat", "house"):
        # Reject, never ignore: a silently dropped filter returns houses to a
        # "3-bed flats" question and the model reads them as flats.
        return [types.TextContent(type="text", text="property_kind must be 'flat' or 'house'.")]

    if args.get("bought_after") or args.get("bought_before"):
        compact = "".join((args.get("postcode") or "").upper().split())
        full_pc = f"{oc} {compact[-3:]}" if sector else None
        return cohort_outcomes(
            oc, sector, full_pc,
            args.get("bought_after"), args.get("bought_before"),
            beds, limit, kind,
        )

    where = ["outcode = ?", "match_score >= ?", "lr_price IS NOT NULL"]
    params: list[Any] = [oc, MIN_MATCH]
    if beds is not None:
        where.append("bedrooms BETWEEN ? AND ?")
        params.extend([int(beds) - 1, int(beds) + 1])
    if kind:
        where.append(f"({kind_case_sql('property_type', 'NULL')}) = ?")
        params.append(kind)

    sql = f"""
        SELECT address, postcode, lr_price, lr_date, lr_address, bedrooms,
               property_type, capital_gain_pct, annual_gain_pct, match_confidence
        FROM rm_delisted_properties
        WHERE {' AND '.join(where)}
        ORDER BY lr_date DESC
        LIMIT ?
    """
    # Over-fetch: duplicates of one sale are collapsed below, and the window
    # the model sees must still hold `limit` DISTINCT sales.
    params.append(min(limit * 3, 120))

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    rows = dedupe_sales(rows)[:limit]

    if not rows:
        return [types.TextContent(
            type="text",
            text=f"No confident recent sales found in {oc}"
                 + (f" for {beds}-bed (±1)" if beds is not None else "")
                 + (f" {kind}s" if kind else "") + ".",
        )]

    lines = [f"Recent sales in {oc} ({len(rows)} matched, most recent first; "
             "duplicate agent listings of one sale collapsed):"]
    if kind:
        lines.append(f"({kind} filter: kind from the listing's property_type label — a few cheap "
                     "'house'-labelled listings are really flats, so a flat list can miss "
                     "some flats; it never includes a house)")
    for r in rows:
        addr = r["address"] or r["postcode"]
        # Each line carries ITS OWN postcode: without it the model has merged
        # a sale here with a same-street sale from get_comparables and lifted
        # that row's postcode (a sale in one unit postcode was reported
        # under its neighbour's).
        if r.get("postcode") and r["postcode"] not in addr:
            addr = f"{addr} ({r['postcode']})"
        bits = [f"£{r['lr_price']:,} on {r['lr_date']}"]
        spec = []
        if r["bedrooms"] is not None:
            spec.append(f"{r['bedrooms']} bed")
        if r["property_type"]:
            spec.append(r["property_type"])
        if spec:
            bits.append(" ".join(spec))
        if r["annual_gain_pct"] is not None:
            bits.append(f"{r['annual_gain_pct']:+.1f}%/yr (total {r['capital_gain_pct']:+.0f}%)")
        lines.append(f"- {addr} · " + " · ".join(bits))
    lines.append("\n(realised return shown where a prior sale + confident match exists)")

    return [types.TextContent(type="text", text="\n".join(lines))]


async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="get-sold-nearby",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
