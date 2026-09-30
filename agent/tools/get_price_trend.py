#!/usr/bin/env python3
"""MCP server: get_price_trend

Price-trend / appreciation summary for a postcode, read from the cached
dim_price_analysis (Land Registry-derived): median + average price, price
per sqm, trend direction, and 3/5/10-year CAGR with the sample count behind
each. Answers "are prices rising here?" / "how much have homes appreciated?"
without the agent having to crunch raw transactions.

Read-only, no auth (the chat handler is the trust boundary).
"""

import asyncio
import json
import logging
import os
import sqlite3
import statistics
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from coverage_note import out_of_coverage_note
from geo_radius import UNKNOWN_POSTCODE_ALERT, unit_postcode_known

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s get_price_trend %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# In the full project this is the repo root, where price_analysis_evaluator.py
# (the repeat-sales engine behind by_type=true, see _pe) lives. That module is
# NOT part of this extract, so the by_type path raises ImportError here.
REPO = Path(__file__).resolve().parent.parent
# Data directory holding evaluations.db: $WTL_DATA_DIR, else ./data next to this file.
DATA_DIR = Path(os.environ.get("WTL_DATA_DIR") or Path(__file__).resolve().parent / "data")
DB_PATH = DATA_DIR / "evaluations.db"

server: Server = Server("get-price-trend")


def normalise_postcode(pc: str) -> str:
    cleaned = "".join(pc.split()).upper()
    if len(cleaned) < 5:
        return cleaned
    return cleaned[:-3] + " " + cleaned[-3:]


_OUTCODE_RE = re.compile(r"^([A-Z]{1,2}\d[A-Z\d]?)$")
_SECTOR_RE = re.compile(r"^([A-Z]{1,2}\d[A-Z\d]?)\s+(\d)$")
# Unspaced sector ('SW113'). Only reachable when the string is NOT itself a
# valid outcode — 'N19' stays Archway, never sector N1 9. A full postcode can
# never land here: its inward code is digit + two LETTERS, so it can't end in a
# digit.
_SECTOR_TIGHT_RE = re.compile(r"^([A-Z]{1,2}\d[A-Z\d]?)(\d)$")


def classify_geo(raw: str) -> tuple[str, str]:
    """What granularity did the caller actually ask about?

    'N1 9DT' -> ('unit', 'N1 9DT') · 'N1 9' -> ('sector', 'N1 9') ·
    'n1' -> ('outcode', 'N1').

    Whitespace is load-bearing here and nowhere else in this file: 'N19' is
    Archway's outcode while 'N1 9' is the ninth sector of N1, and both collapse
    to the same string once spaces are stripped. So classification runs on the
    RAW input, before normalise_postcode() flattens it.
    """
    s = " ".join((raw or "").upper().split())
    m = _SECTOR_RE.match(s)
    if m:
        return "sector", f"{m.group(1)} {m.group(2)}"
    if _OUTCODE_RE.match(s):
        return "outcode", s
    m = _SECTOR_TIGHT_RE.match(s.replace(" ", ""))
    if m:
        return "sector", f"{m.group(1)} {m.group(2)}"
    return "unit", normalise_postcode(s)


def area_trend_text(conn, level: str, geo: str) -> str | None:
    """Area-level appreciation from geo_cagr, or None if that area isn't built.

    Deliberately rate-only. Every price LEVEL we hold is per-postcode and each
    one is a median over its own dynamically-sized recent window (5y expanding
    to 30y until it has 5 sales — the window isn't even persisted), so pooling
    them into "N1's median price" would mix incomparable windows and hand the
    agent a number precise enough to quote and wrong enough to matter. The rate
    is the part that survives aggregation: geo_cagr recomputes the SAME
    holding-time-weighted repeat-sales CAGR over every pair in the area.
    """
    try:
        row = conn.execute(
            "SELECT cagr_3y, cagr_3y_count, cagr_5y, cagr_5y_count, cagr_10y, "
            "cagr_10y_count, direction, sample_count, built_at "
            "FROM geo_cagr WHERE level=? AND geo=?",
            (level, geo),
        ).fetchone()
    except sqlite3.OperationalError:      # table absent on an older deployment
        return None
    if not row:
        return None

    lines = [f"Price trend — {geo} ({level} level, whole-{level} aggregate, "
             f"as of {(row['built_at'] or '')[:10]}):"]
    if row["direction"]:
        lines.append(f"  trend: {row['direction']} (from repeat sales in the last 3 years)")
    cagr = [f"{yrs.upper()} {row[f'cagr_{yrs}']:+.2f}%/yr (n={row[f'cagr_{yrs}_count']:,} pairs)"
            for yrs in ("3y", "5y", "10y") if row[f"cagr_{yrs}"] is not None]
    if cagr:
        lines.append("  CAGR: " + " · ".join(cagr))
        lines.append(_CAGR_WINDOW_NOTE)
        stale = _geo_cagr_stale_note(conn)
        if stale:
            lines.append(stale)
    lines.append(f"  built on {row['sample_count']:,} repeat-sale pairs across {geo}")
    lines.append("  (appreciation RATE for the whole area — no price level at this "
                 "granularity. For a price level, ask again with a full postcode.)")
    if level == "london":
        # State the real footprint. It is the LR coverage we hold — Greater
        # London plus commuter fringe (KT/DA/EN/RM…) — not the administrative
        # boundary, and saying "London" without that invites a precision the
        # number doesn't have. The dispersion line matters more than the
        # average: a single city-wide rate is exactly the flattening that makes
        # macro forecasts useless to a specific buyer, and it is the one thing
        # we can say from our own data that a forecast cannot.
        lines.append("  coverage: 19 postcode areas (Greater London + commuter fringe "
                     "such as KT/DA/EN/RM), not the administrative boundary.")
        try:
            d = conn.execute(
                "SELECT COUNT(*) n, MIN(cagr_3y) lo, MAX(cagr_3y) hi "
                "FROM geo_cagr WHERE level='outcode' AND cagr_3y IS NOT NULL"
            ).fetchone()
            if d and d["n"]:
                lines.append(
                    f"  dispersion: across {d['n']} outcodes the 3y rate runs "
                    f"{d['lo']:+.2f}%/yr to {d['hi']:+.2f}%/yr — quote this spread "
                    f"rather than the city-wide average alone; ask for a specific "
                    f"outcode for the figure that actually applies to a buyer."
                )
        except sqlite3.OperationalError:
            pass
    return "\n".join(lines)


# Accepted in the `postcode` field to mean "the whole market", so a market-wide
# question routes to the london row of geo_cagr instead of off to a forecast
# site. Kept as a small literal set rather than fuzzy matching: anything that
# isn't clearly market-wide should fall through to classify_geo and be answered
# at the granularity actually asked for.
_LONDON_ALIASES = frozenset({
    "london", "greater london", "all london", "all of london", "london overall",
    "london wide", "whole of london", "伦敦", "全伦敦", "整个伦敦", "伦敦整体",
})

# A postcode UNIT's 3y appreciation is built on ~1 repeat-sale pair — too thin.
# Below MIN_CAGR_N pairs we widen the RATE only (never the price level) to the
# narrowest geography that has enough pairs, and label which level we used.
MIN_CAGR_N = 8

# Fix AR (R3-1): the model quoted "5y CAGR +2.5%/yr" as "the market is up"
# against a covid cohort whose true repeat-sale outcome was -3.8% total.
# Verified against PriceAnalysisEvaluator._calculate_price_trends: the "Ny"
# window filters pairs by SALE year (last N calendar years) and weights by
# hold length — a 2010→2023 pair dominates "3y CAGR" with 13 years of weight.
# No model can know our window from the label, so the definition ships with
# every number.
def _geo_cagr_stale_note(conn) -> str:
    """Consumer-side freshness gate for geo_cagr (same semantic_meta registry
    as the pairs table — review R3-1-refix: the swap chain can leave geo_cagr
    a vintage behind the pairs with only an easy-to-miss built_at)."""
    try:
        meta = conn.execute(
            "SELECT source_max_date FROM semantic_meta WHERE key = 'geo_cagr'"
        ).fetchone()
        if not meta or not meta[0]:
            return ""
        live = conn.execute(
            "SELECT MAX(date) FROM lr_transactions").fetchone()[0]
        if live and live[:7] > meta[0][:7]:
            return (f"  ⚠ STALE CAGR TABLE: Land Registry runs to {live[:7]} "
                    f"but geo_cagr was built from data to {meta[0][:7]} — "
                    "rebuild scripts/build_geo_cagr.py. Say the figures are "
                    "as of the earlier date.")
    except sqlite3.OperationalError:
        return ""
    return ""


_CAGR_WINDOW_NOTE = (
    "  ⚠ CAGR definition: hold-time-weighted average annualised return of "
    "repeat-sale pairs whose sale COMPLETED (SOLD) in the last N years — "
    "purchases can date back decades, so this is NOT the outcome of buyers "
    "who bought N years ago, and NOT a recent price path. For 'did people "
    "who bought in period X gain or lose?' use get_sold_nearby cohort mode "
    "(bought_after/bought_before)."
)
_PC_SPLIT = re.compile(r"^([A-Z]{1,2}\d[A-Z\d]?)(\d)([A-Z]{2})$")


def split_geo(pc: str):
    """'W11 4JE' -> (sector 'W11 4', outcode 'W11'); (None, None) if unparseable."""
    m = _PC_SPLIT.match((pc or "").upper().replace(" ", ""))
    if not m:
        return None, None
    return f"{m.group(1)} {m.group(2)}", m.group(1)


def _gbp(v: Any) -> str:
    try:
        return f"£{round(float(v)):,}"
    except (TypeError, ValueError):
        return "n/a"


# --- by-type split (flats vs houses) -----------------------------------------
# Median YoY vs repeat-sale CAGR divergence beyond this (percentage points)
# gets a composition warning. R2-3 textbook case: EC2Y flats median YoY +13.3%
# (n=70, unit-mix sensitive) vs 3Y repeat-sale CAGR +4.6%/yr (175 pairs,
# same-home comparisons) — the agent quoted the median as "the market", the
# web-side reference quoted a modelled index saying the opposite; the CAGR was
# the one defensible number and nobody led with it.
COMPOSITION_DIVERGENCE_PP = 8.0


def _composition_warning(yoy_pct: float | None, cagr3_pct: float | None) -> str | None:
    """Warn when the 12m median YoY and 3Y repeat-sale CAGR disagree sharply."""
    if yoy_pct is None or cagr3_pct is None:
        return None
    if abs(yoy_pct - cagr3_pct) <= COMPOSITION_DIVERGENCE_PP:
        return None
    return ("      ⚠ median YoY and repeat-sale CAGR diverge sharply — the 12-month "
            "median is composition-sensitive (unit-mix shifts year to year), NOT a "
            "like-for-like measure. Trust the repeat-sale CAGR for direction; use "
            "the median only as a price level.")


# A quotable 12-month median needs at least this many sales — same floor the
# per-postcode price level uses (its window expands until it has 5 sales).
MIN_TYPE_MEDIAN_N = 5

_PE = None


def _pe():
    """price_analysis_evaluator, imported lazily: only the by_type path pays
    for it (it drags in api_helper), and its repeat-sales methods are the SAME
    ones geo_cagr is built from, so segment CAGRs stay comparable."""
    global _PE
    if _PE is None:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from price_analysis_evaluator import PriceAnalysisEvaluator
        _PE = PriceAnalysisEvaluator
    return _PE


def _fetch_residential(conn, level: str, geo: str):
    # Fix AI (R3-1) + review R3-1-refix: rows come back ALL-category WITH the
    # category column. Medians filter to category A in Python (HMLR HPI
    # convention), but the repeat-sale PAIRING must see the full sequence —
    # a SQL-level category filter here stitched phantom cross-owner holds
    # across removed B sales (10,673 A→B→A chains DB-wide); the pair-level
    # drop now lives in PriceAnalysisEvaluator._identify_repeat_sales.
    base = ("SELECT postcode, address_key, price, date, property_type, "
            "category "
            "FROM lr_transactions WHERE {} AND price > 0 AND date IS NOT NULL "
            "AND property_type IN ('D','S','T','F')")
    if level == "unit":
        return conn.execute(base.format("postcode = ?"), (geo,)).fetchall()
    # Range predicate instead of LIKE so idx_lr2_postcode stays usable: units
    # under outcode 'EN1' all sort in ['EN1 ', 'EN1!') — and 'EN10 …' sorts
    # outside it, so EN10 never bleeds into EN1.
    prefix = geo + " " if level == "outcode" else geo
    hi = prefix[:-1] + chr(ord(prefix[-1]) + 1)
    return conn.execute(base.format("postcode >= ? AND postcode < ?"),
                        (prefix, hi)).fetchall()


def _in_rung(pc: str, level: str, geo: str) -> bool:
    if level == "unit":
        return pc == geo
    if level == "outcode":
        return pc.startswith(geo + " ")
    return pc.startswith(geo)   # sector 'EN1 1' — inward is 3 chars, no false hits


def type_split_text(conn, level: str, geo: str) -> str | None:
    """Flats (F) vs houses (D/S/T) split, computed live from lr_transactions.

    The area path's no-price-level rule bans POOLING per-postcode medians whose
    windows all differ; a direct median over one explicit 12-month window (to
    the geo's latest sale) doesn't have that problem, so each segment gets its
    level + YoY, with the window spelled out. CAGR reuses the exact repeat-
    sales methods behind geo_cagr, restricted to the segment's transactions.

    Sample-size ladder, same spirit as MIN_CAGR_N: each segment walks
    unit → sector → outcode until its last-12m count reaches MIN_TYPE_MEDIAN_N,
    labelling any widening; if even the outcode is too thin, report the count
    and quote nothing.
    """
    if level == "unit":
        sector, outcode = split_geo(geo)
        ladder = [("unit", geo)]
        if sector:
            ladder += [("sector", sector), ("outcode", outcode)]
    elif level == "sector":
        ladder = [("sector", geo), ("outcode", geo.rsplit(" ", 1)[0])]
    else:
        ladder = [("outcode", geo)]

    try:
        rows = _fetch_residential(conn, *ladder[-1])
    except sqlite3.OperationalError as exc:
        if "category" in str(exc):
            # Column missing ≠ no sales: never convert a schema gap into a
            # false "no Land Registry sales" claim (review finding 10).
            return ("  (by-type split unavailable: lr_transactions on this "
                    "deployment predates the category column — re-run the "
                    "LR import. Do NOT tell the user there are no sales.)")
        return None                       # table absent on an older deployment
    if not rows:
        return None

    # Standard sales drive the medians (HMLR HPI convention); the full-
    # category rows still feed the pairing above.
    rows_a = [r for r in rows if (r["category"] or "").strip().upper() == "A"]
    anchor = max(r["date"] for r in (rows_a or rows))[:10]
    try:
        anchor_dt = datetime.strptime(anchor, "%Y-%m-%d")
    except ValueError:
        return None
    t12 = (anchor_dt - timedelta(days=365)).strftime("%Y-%m-%d")
    t24 = (anchor_dt - timedelta(days=730)).strftime("%Y-%m-%d")

    widest_lvl, widest_geo = ladder[-1]
    out = ["  By property type (flats = F, houses = D/S/T; medians are direct "
           "Land Registry medians over the explicit 12-month window shown):"]
    low_pairs = False
    for label, types in (("Flats", ("F",)), ("Houses", ("D", "S", "T"))):
        seg_any = [r for r in rows if r["property_type"] in types]
        seg_all = [r for r in rows_a if r["property_type"] in types]
        if not seg_all:
            if seg_any:
                out.append(
                    f"    {label}: no standard (category A) sales in "
                    f"{widest_lvl} {widest_geo} — {len(seg_any)} non-standard "
                    "(category B) transaction(s) exist but are not quotable "
                    "as market prices.")
            else:
                out.append(f"    {label}: no recorded sales in "
                           f"{widest_lvl} {widest_geo}.")
            continue
        chosen = None
        for lvl, g in ladder:
            seg = [r for r in seg_all if _in_rung(r["postcode"], lvl, g)]
            n12 = sum(1 for r in seg if t12 < r["date"][:10] <= anchor)
            if n12 >= MIN_TYPE_MEDIAN_N:
                chosen = (lvl, g, seg, n12)
                break
        if chosen is None:
            n12 = sum(1 for r in seg_all if t12 < r["date"][:10] <= anchor)
            out.append(
                f"    {label}: only {n12} sale{'s' if n12 != 1 else ''} in the "
                f"last 12m even at {widest_lvl} {widest_geo} level — too few for "
                f"a reliable median (min {MIN_TYPE_MEDIAN_N}); not quoting one.")
            continue
        lvl, g, seg, n12 = chosen
        m12 = statistics.median(
            [r["price"] for r in seg if t12 < r["date"][:10] <= anchor])
        prev = [r["price"] for r in seg if t24 < r["date"][:10] <= t12]
        parts = [f"median {_gbp(m12)} (12m to {anchor[:7]}, n={n12})"]
        yoy_pct = None
        if len(prev) >= MIN_TYPE_MEDIAN_N:
            m_prev = statistics.median(prev)
            if m_prev:
                yoy_pct = (m12 / m_prev - 1) * 100
                parts.append(f"YoY {yoy_pct:+.1f}%")
        else:
            parts.append(f"YoY n/a (prior 12m n={len(prev)})")
        pe = _pe()
        # Pairing sees the FULL sale sequence (all categories) so removed B
        # sales can't stitch phantom holds; the evaluator drops B-legged
        # pairs itself (review R3-1-refix).
        seg_any_rung = [r for r in seg_any if _in_rung(r["postcode"], lvl, g)]
        pairs = pe._identify_repeat_sales(None, [
            {"address": r["address_key"], "price": r["price"],
             "date": r["date"], "category": r["category"]}
            for r in seg_any_rung if r["address_key"]])
        tr = pe._calculate_price_trends(None, None, repeat_sales=pairs)
        cagr = []
        for yrs, key in (("3Y", "3_year"), ("5Y", "5_year"), ("10Y", "10_year")):
            if tr.get(f"{key}_cagr") is not None:
                cnt = tr.get(f"{key}_count") or 0
                cagr.append(f"{yrs} {tr[f'{key}_cagr']:+.1f}%/yr (n={cnt} pairs)")
                if cnt < MIN_CAGR_N:
                    low_pairs = True
        if cagr:
            parts.append("CAGR " + " · ".join(cagr))
        where = "" if lvl == level else f" (widened to {lvl} {g})"
        out.append(f"    {label}{where}: " + " · ".join(parts))
        warn = _composition_warning(yoy_pct, tr.get("3_year_cagr"))
        if warn:
            out.append(warn)
    if low_pairs:
        out.append(f"    (pair counts under {MIN_CAGR_N} are indicative only)")
    return "\n".join(out)


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_price_trend",
            description=(
                "Price trend from Land Registry history, at whichever "
                "granularity you ask for. FULL postcode ('SE15 3AB') → median & "
                "average price, price per sqm, trend direction, 3/5/10yr CAGR. "
                "OUTCODE ('W5') or SECTOR ('W5 2') → the whole area's trend "
                "direction + 3/5/10yr CAGR over every repeat sale in it (no "
                "price level at that granularity). "
                "'London' → the WHOLE-MARKET aggregate (3/5/10yr CAGR over every "
                "repeat-sale pair we hold, plus the spread across outcodes). Call "
                "this for market-wide questions — 'how is the London market "
                "doing', 'is now a good time to buy', 'are prices falling' — "
                "instead of searching the web for a forecast: this is measured "
                "from Land Registry sales that already happened, which is a "
                "different and more defensible claim than a prediction. Web "
                "search is still the right call for things we genuinely cannot "
                "hold, such as the base rate or a house-price forecast. "
                "Use for 'are prices going up "
                "in X' / 'how much have homes appreciated' / 'is this a good "
                "area to invest'. Ask about the area the user asked about — "
                "never substitute a full postcode you made up to stand in for a "
                "district. If it isn't evaluated, says so — don't fabricate. "
                "Optional by_type=true adds a flats-vs-houses split — their "
                "medians and trends often diverge (houses flat while flats "
                "fall), so use it whenever the user cares about a specific "
                "property type."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "postcode": {
                        "type": "string",
                        "description": (
                            "UK postcode ('SE15 3AB'), outcode ('W5'), sector "
                            "('W5 2'), or the literal 'London' for the whole "
                            "market. Auto-normalised. Mind the space: 'N19' is "
                            "Archway's outcode, 'N1 9' is a sector of N1."
                        ),
                    },
                    "by_type": {
                        "type": "boolean",
                        "description": (
                            "true → also split by property type: flats (F) vs "
                            "houses (D/S/T), each with its own 12-month-window "
                            "median, YoY and repeat-sales CAGR (sample sizes "
                            "labelled; thin segments widen to sector/outcode "
                            "and say so). Default false."
                        ),
                    },
                },
                "required": ["postcode"],
            },
        )
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any] | None
) -> list[types.TextContent]:
    if name != "get_price_trend":
        raise ValueError(f"Unknown tool: {name}")

    args = arguments or {}
    raw = (args.get("postcode") or "").strip()
    by_type = bool(args.get("by_type"))
    if not raw:
        return [types.TextContent(type="text", text="Missing postcode.")]

    # An outcode/sector ask ("is W5 going up?") used to fall through to the unit
    # lookup, miss, and come back "W5 has no price analysis yet" — which the
    # agent relayed as "that area has never been evaluated" and then patched
    # over by inventing unit postcodes to query. Answer it at the granularity
    # asked, from the table built for exactly this (geo_cagr).
    # "How is the London market doing?" had no local answer at all before
    # 2026-08-13 — every granularity here needs a postcode, so the agent went
    # to the web and answered a question about OUR market entirely out of
    # forecasts (measured: two WebSearches, zero local tool calls). The whole-
    # London aggregate is the same repeat-sales CAGR over every pair we hold.
    if raw.lower().replace("-", " ") in _LONDON_ALIASES:
        level, geo_key = "london", "London"
    else:
        level, geo_key = classify_geo(raw)
    if level in ("outcode", "sector", "london"):
        area_conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        area_conn.row_factory = sqlite3.Row
        try:
            text = area_trend_text(area_conn, level, geo_key)
            # No whole-London split: it would mean pairing 4.6M transactions
            # live per call. The per-outcode dispersion line already carries
            # the "one number hides the spread" message at this granularity.
            split = (type_split_text(area_conn, level, geo_key)
                     if by_type and text and level != "london" else None)
        finally:
            area_conn.close()
        if text:
            if split:
                text += "\n" + split
            elif by_type:
                text += ("\n  (a flats-vs-houses split isn't computed at whole-London "
                         "granularity — ask about a specific outcode, sector or postcode)"
                         if level == "london" else
                         "\n  (no by-type split: no Land Registry sales found here to segment)")
            return [types.TextContent(type="text", text=text)]
        # Outside the footprint there is nothing to build a trend FROM, and
        # /evaluate would come back just as empty — send the coverage fact
        # instead of a dead end that reads as "this area has no trend".
        gap = out_of_coverage_note(
            geo_key, missing="Land Registry repeat sales or price trends")
        if gap:
            return [types.TextContent(type="text", text=gap)]
        return [types.TextContent(
            type="text",
            text=f"{geo_key} has no price analysis yet (no repeat-sale history built "
                 f"for that {level}). Don't estimate one — ask about a full postcode "
                 f"inside it, or suggest /evaluate?postcode={geo_key.replace(' ', '+')}.")]

    pc = geo_key
    sector, outcode = split_geo(pc)
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    geo: dict[str, Any] = {}
    try:
        row = conn.execute(
            "SELECT data_json, created_at FROM dim_price_analysis "
            "WHERE postcode = ? ORDER BY created_at DESC LIMIT 1",
            (pc,),
        ).fetchone()
        # Wider-geography CAGR for the fallback ladder. Table may be absent on an
        # older deployment that hasn't run scripts/build_geo_cagr.py — degrade
        # gracefully to unit-level only.
        try:
            for lvl, g in (("sector", sector), ("outcode", outcode)):
                if not g:
                    continue
                gr = conn.execute(
                    "SELECT cagr_3y, cagr_3y_count, cagr_5y, cagr_5y_count, "
                    "cagr_10y, cagr_10y_count FROM geo_cagr WHERE level=? AND geo=?",
                    (lvl, g),
                ).fetchone()
                if gr:
                    geo[lvl] = gr
        except sqlite3.OperationalError:
            pass
        # Gate note must be computed while conn is open — the CAGR lines are
        # rendered after close (review R3-1-refix follow-up).
        geo_stale = _geo_cagr_stale_note(conn)
        # R3-2 T18b: distinguish "real postcode, not analysed yet" from
        # "postcode with zero trace anywhere" — /evaluate can't help the latter.
        pc_known = unit_postcode_known(conn, pc)
        split = (type_split_text(conn, "unit", pc)
                 if by_type and row and row["data_json"] else None)
    finally:
        conn.close()

    if not row or not row["data_json"]:
        if pc_known is False:
            return [types.TextContent(
                type="text",
                text=UNKNOWN_POSTCODE_ALERT.format(
                    pc=pc, scope=f"outcode {split_geo(pc)[1] or pc}"),
            )]
        gap = out_of_coverage_note(
            pc, missing="Land Registry repeat sales or price trends")
        if gap:
            return [types.TextContent(type="text", text=gap)]
        return [types.TextContent(
            type="text",
            text=f"{pc} has no price analysis yet. Suggest /evaluate?postcode={pc.replace(' ', '+')}.",
        )]

    try:
        d = json.loads(row["data_json"])
    except json.JSONDecodeError:
        return [types.TextContent(type="text", text=f"{pc}: price data unreadable.")]

    lines = [f"Price trend — {pc} (as of {row['created_at'][:10]}):"]
    lines.append(
        f"  median {_gbp(d.get('median_price'))} · average {_gbp(d.get('average_price'))}"
        + (f" · {_gbp(d.get('price_per_sqm'))}/sqm" if d.get("price_per_sqm") else "")
    )
    # ~5% of rows had too few sales in the unit itself, so the LEVEL above was
    # computed over the surrounding sector/outcode. Unlabelled, the agent
    # repeats it as this postcode's own median — the building-level-sale
    # mistake again, one geography up.
    scope = d.get("area_scope")
    if scope and scope != "unit":
        lines.append(
            f"  ^ that price level is a {scope} figure "
            f"({d.get('area_scope_postcode') or scope}) — this postcode alone had too "
            f"few sales. Attribute it to the {scope}, not to this postcode.")
    if d.get("trend_direction"):
        lines.append(f"  trend: {d['trend_direction']}")
    if d.get("total_transactions"):
        lines.append(f"  based on {d['total_transactions']} transactions over {d.get('years_analyzed', '?')}")

    # Per-window CAGR with a narrowest-enough-sample fallback ladder: use the
    # postcode's own rate when it has >= MIN_CAGR_N repeat-sale pairs, else widen
    # to the sector, then the outcode — labelling whichever level we used and its
    # n. This widens the appreciation RATE only; the median/average price LEVEL
    # above stays postcode-specific. (Rates are spatially smoother than levels.)
    cagr_lines = []
    widened = False
    for yrs in ("3y", "5y", "10y"):
        u_val = d.get(f"cagr_{yrs}")
        u_cnt = d.get(f"cagr_{yrs}_count") or 0
        chosen = None
        if u_val is not None and u_cnt >= MIN_CAGR_N:
            chosen = ("postcode", u_val, u_cnt)
        else:
            for lvl in ("sector", "outcode"):
                gr = geo.get(lvl)
                if gr and gr[f"cagr_{yrs}"] is not None and (gr[f"cagr_{yrs}_count"] or 0) >= MIN_CAGR_N:
                    chosen = (lvl, gr[f"cagr_{yrs}"], gr[f"cagr_{yrs}_count"])
                    widened = True
                    break
            if chosen is None and u_val is not None:   # nothing wider qualified
                chosen = ("postcode", u_val, u_cnt)
        if not chosen:
            continue
        lvl, val, cnt = chosen
        if lvl == "postcode":
            cagr_lines.append(f"{yrs.upper()} {val:+.1f}%/yr (n={cnt})")
        else:
            label = sector if lvl == "sector" else outcode
            cagr_lines.append(f"{yrs.upper()} {val:+.1f}%/yr ({lvl} {label}, n={cnt})")
    if cagr_lines:
        lines.append("  CAGR: " + " · ".join(cagr_lines))
        lines.append(_CAGR_WINDOW_NOTE)
        if geo_stale:
            lines.append(geo_stale)
        if widened:
            # Only claim the LEVEL is postcode-specific when it actually is —
            # area_scope says otherwise on ~5% of rows, and the two notes would
            # contradict each other in the same message.
            lines.append("  (this postcode had too few repeat sales for that window → widened "
                         "to the labelled sector/outcode"
                         + ("; the price level above stays postcode-specific)"
                            if not scope or scope == "unit" else ")"))
        else:
            lines.append("  (low n = few repeat sales; treat that CAGR as indicative only)")

    if by_type:
        lines.append(split if split else
                     "  (no by-type split: no Land Registry sales found here to segment)")

    return [types.TextContent(type="text", text="\n".join(lines))]


async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="get-price-trend",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
