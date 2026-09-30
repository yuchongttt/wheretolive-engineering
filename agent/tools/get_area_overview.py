#!/usr/bin/env python3
"""MCP server: get_area_overview

One-call narrative summary of a UK postcode for chat: 5-dim scores +
active listings + 12-month sale stats + nearest stations + top schools.
Saves the agent from chaining 4 tools when the user just asks "tell me
about N1 9DT".
"""

import asyncio
import json
import logging
import os
import sqlite3
import sys

from area_scope import classify_postcode_scope
from geo_radius import UNKNOWN_POSTCODE_ALERT, unit_postcode_known
from pathlib import Path
from statistics import median
from typing import Any

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s get_area_overview %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Data directory holding evaluations.db + london_outcodes.json:
# $WTL_DATA_DIR, else ./data next to this file.
DATA_DIR = Path(os.environ.get("WTL_DATA_DIR") or Path(__file__).resolve().parent / "data")
DB_PATH = DATA_DIR / "evaluations.db"
OUTCODES_JSON = DATA_DIR / "london_outcodes.json"

# Precomputed per-outcode metrics (incl. rent/yield_by_bed) from the choropleth
# builder — the same rent/yield the /report page shows. Loaded once, cached, so
# chat can answer BTL "what rent / what yield" questions from real data instead
# of saying it has none.
_OUTCODE_METRICS: dict[str, dict] | None = None
# The file's generated_at (YYYY-MM-DD). The rent/yield figures are a SNAPSHOT
# of asking rents as of that day — there is no delist lane for rentals — and
# the header used to say "live rental listings", which a model then repeated
# to a landlord verbatim ("based on N live rental listings").
_OUTCODE_GENERATED_AT: str | None = None


def _load_outcodes() -> None:
    global _OUTCODE_METRICS, _OUTCODE_GENERATED_AT
    if _OUTCODE_METRICS is not None:
        return
    _OUTCODE_METRICS = {}
    try:
        raw = json.loads(OUTCODES_JSON.read_text())
        if isinstance(raw, dict):
            _OUTCODE_GENERATED_AT = (str(raw.get("generated_at") or "")[:10]) or None
        feats = raw.get("outcodes", raw) if isinstance(raw, dict) else raw
        for f in feats:
            p = f.get("properties", f) if isinstance(f, dict) else {}
            code = p.get("outcode")
            if code:
                _OUTCODE_METRICS[code.upper()] = p.get("metrics", {}) or {}
    except Exception:
        _OUTCODE_METRICS = {}


def _outcode_metrics(outcode: str) -> dict:
    _load_outcodes()
    return (_OUTCODE_METRICS or {}).get((outcode or "").upper(), {})


def _precompute_date() -> str:
    """Generation date of the rent/yield precompute, or 'unknown date' —
    never silently blank, the whole point is to date the snapshot."""
    _load_outcodes()
    return _OUTCODE_GENERATED_AT or "unknown date"

server: Server = Server("get-area-overview")

DIMS = [
    ("dim_commute",         "Transport / Commute"),
    ("dim_transit",         "Transport / Transit"),
    ("dim_safety",          "Community / Safety"),
    ("dim_demographics",    "Community / Demographics"),
    ("dim_price_analysis",  "Price"),
    ("dim_schools",         "Schools"),
]


def normalise_postcode(pc: str) -> str:
    cleaned = "".join(pc.split()).upper()
    if len(cleaned) < 5:
        return cleaned
    return cleaned[:-3] + " " + cleaned[-3:]


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_area_overview",
            description=(
                "One-call narrative summary of a UK postcode: livability "
                "scores, active listings (count + median price), 12-month "
                "Land Registry sale stats, nearest stations from "
                "rm_nearest_stations, top schools from dim_schools, and "
                "**rent + gross/net rental yield by bedroom count** (real "
                "figures, with sample sizes). Prefer this over chaining "
                "get_postcode_scores + get_comparables + search_properties "
                "when the user asks an open 'tell me about this area' "
                "question, and use it for any **rent / rental yield / "
                "buy-to-let return** question about an area."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "postcode": {
                        "type": "string",
                        "description": "UK postcode, e.g. 'N1 9DT'.",
                    },
                },
                "required": ["postcode"],
            },
        )
    ]


def _scope_and_outcode(raw: str) -> tuple[str, str, str]:
    """(scope, normalised postcode, outcode) — all three input shapes must
    derive the correct outcode.

    The old `pc.split(" ")[0] if " " in pc else pc[:-3]` trimmed an outcode
    input "SW11" down to "S" (2026-09-03 case: active listings matched 0 while
    SW11 had 1,245, rendered as a false "nothing on the market"). Same root
    cause as compare_postcodes, same fix: delegate to
    area_scope.classify_postcode_scope — it knows the internal space is
    load-bearing, and that unspaced "SW11" and spaced "SW1 1" are different
    places.
    """
    scope, norm = classify_postcode_scope(raw)
    return scope, norm, norm.partition(" ")[0]


def _should_check_unit_known(scope: str) -> bool:
    """unit_postcode_known is only meaningful for a full unit postcode.

    It measures OUR coverage, not existence — asked about the whole WD area it
    returns False, yet those postcodes are all live. Asking it about an area
    input only produces a false alarm.
    """
    return scope == "unit"


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any] | None
) -> list[types.TextContent]:
    if name != "get_area_overview":
        raise ValueError(f"Unknown tool: {name}")

    args = arguments or {}
    raw = (args.get("postcode") or "").strip()
    if not raw:
        return [types.TextContent(type="text", text="Missing postcode parameter.")]
    scope, pc, outcode = _scope_and_outcode(raw)

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    out: dict[str, Any] = {"postcode": pc, "outcode": outcode}
    try:
        # R3-2 T18b: a full unit postcode with zero trace anywhere must be
        # flagged BEFORE sector/outcode aggregates make the address look real.
        if _should_check_unit_known(scope) and unit_postcode_known(conn, pc) is False:
            out["postcode_alert"] = UNKNOWN_POSTCODE_ALERT.format(
                pc=pc, scope=f"outcode {outcode}")
        # Scores
        scores: dict[str, Any] = {}
        for table, label in DIMS:
            row = conn.execute(
                f"SELECT score, data_json FROM {table} WHERE postcode = ? "
                f"ORDER BY created_at DESC LIMIT 1",
                (pc,),
            ).fetchone()
            if row and row["score"] is not None:
                try:
                    data = json.loads(row["data_json"]) if row["data_json"] else {}
                except json.JSONDecodeError:
                    data = {}
                scores[label] = {
                    "score": round(row["score"], 1),
                    "details": {k: v for k, v in data.items() if k not in {"raw", "intermediate", "debug"}},
                }
        out["scores"] = scores

        # Calibrated 5-dimension scores (the real headline numbers) from the
        # precomputed postcode_scores table — dim_*.score above only covers
        # schools + price, so without this the overview drops the other dims.
        try:
            cal = conn.execute(
                "SELECT transport, community, environment, price, schools, "
                "       total_score, data_completeness "
                "FROM postcode_scores WHERE postcode = ?",
                (pc,),
            ).fetchone()
        except sqlite3.OperationalError:
            cal = None
        out["calibrated"] = dict(cal) if cal else None

        # Active listings (postcode)
        # Area = outcode (a single exact postcode is too sparse, and postcodes
        # are stored mostly without spaces). Match the space-stripped outcode
        # (postcode minus the 3-char incode) + dedup multi-agent twins.
        active = conn.execute(
            """SELECT COUNT(*) AS n, AVG(asking_price) AS avg_p,
                      MIN(asking_price) AS lo, MAX(asking_price) AS hi
                 FROM rm_sales_overview
                WHERE UPPER(substr(replace(postcode,' ',''), 1,
                            length(replace(postcode,' ','')) - 3)) = ?
                  AND delisted_date IS NULL
                  AND (canonical_id IS NULL OR canonical_id = id)""",
            (outcode.upper(),),
        ).fetchone()
        out["active_listings"] = {
            "count": active["n"] or 0,
            "avg_price": int(active["avg_p"]) if active["avg_p"] else None,
            "min_price": int(active["lo"]) if active["lo"] else None,
            "max_price": int(active["hi"]) if active["hi"] else None,
            # The caliber travels WITH the number. A model reading the JSON leg
            # has no other way to tell that a unit-postcode question came back
            # with a whole-outcode count.
            "scope": "outcode",
            "scope_key": outcode.upper(),
        }

        # 12-month LR sales (outcode level — postcode is too sparse).
        # LR postcodes carry a space ("N1 9DT"); match 'N1 %' so the outcode
        # "N1" doesn't also pull N10/.../N19 (the substr-prefix form did).
        # Fix AK-b (R3-1) + review R3-1-refix: standard open-market sales
        # only (strict category=='A', NULL is not standard) for the median;
        # PRICED category-B rows counted for the disclosure (unpriced rows
        # were never median candidates, so counting them overstated the
        # exclusion); an all-B window still emits the section with n=0
        # instead of silently deleting stats AND disclosure together.
        try:
            rows = conn.execute(
                """SELECT price, category FROM lr_transactions
                    WHERE postcode LIKE ?
                      AND date >= date('now','-1 year')""",
                (f"{outcode} %",),
            ).fetchall()
        except sqlite3.OperationalError:
            # Deployment predates the category column — degrade to the old
            # unfiltered behaviour rather than failing the whole tool call.
            rows = conn.execute(
                """SELECT price, NULL AS category FROM lr_transactions
                    WHERE postcode LIKE ?
                      AND date >= date('now','-1 year')""",
                (f"{outcode} %",),
            ).fetchall()
            priced = [r["price"] for r in rows if r["price"]]
            if priced:
                out["recent_sales_12mo_outcode"] = {
                    "n": len(priced),
                    "median": int(median(priced)),
                    "min": min(priced),
                    "max": max(priced),
                    "basis": "ALL categories (category column unavailable "
                             "on this deployment)",
                    "excluded_category_b": 0,
                }
            rows = []
        prices, n_b = [], 0
        for r in rows:
            if not r["price"]:
                continue
            if (r["category"] or "").strip().upper() == "A":
                prices.append(r["price"])
            elif (r["category"] or "").strip().upper() == "B":
                n_b += 1
        if prices:
            out["recent_sales_12mo_outcode"] = {
                "n": len(prices),
                "median": int(median(prices)),
                "min": min(prices),
                "max": max(prices),
                "basis": "standard open-market sales only (LR category A)",
                "excluded_category_b": n_b,
            }
        elif n_b:
            out["recent_sales_12mo_outcode"] = {
                "n": 0,
                "excluded_category_b": n_b,
                "basis": "standard open-market sales only (LR category A)",
                "note": (f"all {n_b} priced transaction(s) in the window are "
                         "non-standard (category B) — no standard-sale "
                         "stats; do not say there were no sales"),
            }

        # Nearest stations (best-effort, postcode-keyed)
        stations = []
        try:
            for r in conn.execute(
                """SELECT station_name, distance_m, lines
                     FROM rm_nearest_stations
                    WHERE postcode = ?
                    ORDER BY distance_m ASC LIMIT 5""",
                (pc,),
            ).fetchall():
                stations.append({
                    "name": r["station_name"],
                    "distance_m": r["distance_m"],
                    "lines": r["lines"],
                })
        except sqlite3.OperationalError:
            # Schema may have evolved; non-fatal.
            pass
        if stations:
            out["nearest_stations"] = stations

        # Top schools (best-effort — pulled from dim_schools details if shape supports)
        if "Schools" in scores:
            try:
                school_data = scores["Schools"].get("details", {})
                top = school_data.get("schools") or school_data.get("nearby") or []
                if isinstance(top, list):
                    out["top_schools"] = top[:5]
            except Exception:
                pass
    finally:
        conn.close()

    # Rent & gross/net yield by bedroom count (precomputed per outcode, the same
    # figures /report shows). Best-effort — silent if the outcode isn't cached.
    try:
        yb = _outcode_metrics(outcode).get("yield_by_bed") or {}
        beds_label = {"0": "studio", "1": "1-bed", "2": "2-bed",
                      "3": "3-bed", "4": "4-bed", "5": "5-bed"}
        ry = []
        for k in ("0", "1", "2", "3", "4", "5"):
            d = yb.get(k)
            if not d or not d.get("rent"):
                continue
            ry.append({
                "beds": beds_label.get(k, k),
                "rent_pcm": int(d["rent"]),
                "gross_yield_pct": d.get("gross"),
                "net_yield_pct": d.get("net"),
                "rent_sample": d.get("rent_n"),
            })
        if ry:
            out["rental_yield_by_bed"] = ry
    except Exception:
        pass

    # Format narrative
    lines = [f"# Area overview: {pc} (outcode {outcode})", ""]
    if out.get("postcode_alert"):
        lines.insert(1, out["postcode_alert"])
    cal = out.get("calibrated")
    if cal:
        lines.append("## Scores (0-100)")
        for label, col in (("Transport", "transport"), ("Community", "community"),
                           ("Environment", "environment"), ("Price", "price"),
                           ("Schools", "schools")):
            if cal.get(col) is not None:
                lines.append(f"  - {label}: **{round(cal[col], 1)}**")
        if cal.get("total_score") is not None:
            lines.append(f"  - Overall: **{round(cal['total_score'], 1)}** (completeness {cal.get('data_completeness', '?')})")
        lines.append("")
    elif scores:
        lines.append("## Scores (0-100)")
        for label, v in scores.items():
            lines.append(f"  - {label}: **{v['score']}**")
        lines.append("")
    a = out["active_listings"]
    if a["count"]:
        # Label the SCOPE, not the question. This count is outcode-wide (see
        # the query above); headlining it with the asked-for unit postcode told
        # a user "hundreds of active listings in this postcode" for a postcode
        # that had a handful — and the model then reasoned on top of that caliber.
        lines.append(f"## On the market now (outcode {outcode})")
        lines.append(
            f"  {a['count']} active listing(s), avg £{a['avg_price']:,}, range £{a['min_price']:,}–£{a['max_price']:,}"
        )
        lines.append("")
    if (r12 := out.get("recent_sales_12mo_outcode")) and r12.get("median"):
        lines.append(f"## Last 12 months (outcode {outcode})")
        # Narrative must not present the A-only count as total volume
        # (review finding 5: ~19% of recent rows are category B).
        b_suffix = (f" (+{r12['excluded_category_b']} non-standard "
                    "category-B excluded)"
                    if r12.get("excluded_category_b") else "")
        lines.append(
            f"  {r12['n']} standard sales{b_suffix}, median "
            f"£{r12['median']:,}, range £{r12['min']:,}–£{r12['max']:,}")
        lines.append("")
    if ry := out.get("rental_yield_by_bed"):
        snap = _precompute_date()
        out["rent_snapshot_date"] = snap
        lines.append(f"## Rent & yield by size (outcode {outcode}) — asking-rent SNAPSHOT "
                     f"generated {snap}, NOT live lettings; say the date when you quote it")
        for d in ry:
            yld = " · ".join(x for x in (
                f"{d['gross_yield_pct']}% gross" if d.get("gross_yield_pct") is not None else "",
                f"{d['net_yield_pct']}% net" if d.get("net_yield_pct") is not None else "",
            ) if x)
            samp = f" (n={d['rent_sample']})" if d.get("rent_sample") else ""
            lines.append(f"  - {d['beds']}: ~£{d['rent_pcm']:,}/mo{(' · ' + yld) if yld else ''}{samp}")
        lines.append("")
    else:
        # Silence here cost a real user two wasted tool round-trips: asked what
        # rent a home in an outer outcode would fetch, the model re-asked this tool at outcode
        # level (same silence) before finding get_area_profile. HA1 — like every
        # outcode in BR/CR/DA/EN/HA/IG/KT/RM/SM/TW/UB/WD — is simply absent from
        # the precompute file, and 67 of the 143 that ARE in it carry no
        # yield_by_bed. Say which of the two it is and name the tool that does
        # have an answer. (Filling the precompute is a separate, data-side job.)
        out["rental_yield_by_bed"] = None
        lines.append(f"## Rent & yield by size (outcode {outcode})")
        lines.append(
            f"  Not cached for {outcode}. The rent/yield precompute covers only "
            "the inner-London outcode file (E/EC/N/NW/SE/SW/W/WC — and not every "
            "outcode inside those); it is a gap in OUR precompute, NOT a "
            "statement about how much is let around here. Re-asking this tool at "
            "a different granularity will not help. For a rent figure call "
            "get_area_profile (official ONS borough benchmark, split by bedroom "
            "count) — and do not derive rent from asking prices.")
        lines.append("")
    if stations:
        lines.append("## Nearest stations")
        for s in stations:
            lines.append(f"  - {s['name']} ({s['distance_m']}m){' · ' + s['lines'] if s.get('lines') else ''}")
        lines.append("")
    if not cal and not scores and a["count"] == 0:
        lines.append(
            f"_No cached evaluation for {pc} yet. Suggest the user visit "
            f"/evaluate?postcode={pc.replace(' ', '+')} to trigger one._"
        )
        lines.append("")
    lines.append("--- structured ---")
    lines.append(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return [types.TextContent(type="text", text="\n".join(lines))]


async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(
            read, write,
            InitializationOptions(
                server_name="get-area-overview",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
