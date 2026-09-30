#!/usr/bin/env python3
"""MCP server: get_postcode_scores

Returns 5-dimension livability scores for a UK postcode (Transport,
Community, Environment, Price, Schools). Reads pre-computed cached
scores from dim_* tables in evaluations.db. Read-only.

Phase 0b · Local AI Chat Migration design doc.
"""

import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import asyncio
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s get_postcode_scores %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Data directory holding evaluations.db: $WTL_DATA_DIR, else ./data next to this file.
DATA_DIR = Path(os.environ.get("WTL_DATA_DIR") or Path(__file__).resolve().parent / "data")
DB_PATH = DATA_DIR / "evaluations.db"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from area_scope import classify_postcode_scope, like_pattern, summarise_dimension  # noqa: E402

# Calibrated dimension columns, in the order they are reported.
CAL_DIMS = [
    ("transport", "Transport"), ("community", "Community"),
    ("environment", "Environment"), ("price", "Price"), ("schools", "Schools"),
]

# Score tables and their dimension labels
DIMS = [
    ("dim_commute",      "Transport / Commute"),
    ("dim_transit",      "Transport / Transit"),
    ("dim_safety",       "Community / Safety"),
    ("dim_demographics", "Community / Demographics"),
    ("dim_price_analysis", "Price"),
    ("dim_schools",      "Schools"),
]

server: Server = Server("get-postcode-scores")


def normalise_postcode(pc: str) -> str:
    cleaned = "".join(pc.split()).upper()
    if len(cleaned) < 5:
        return cleaned
    return cleaned[:-3] + " " + cleaned[-3:]


def area_summary(conn, scope: str, norm: str) -> dict | None:
    """Aggregate every scored unit postcode inside an outcode / sector.

    Scores exist per unit postcode; an outcode holds hundreds of them and they
    genuinely disagree (SW11's schools dimension runs 33.6-98.0 over 703 units).
    Returning ONE unit's scores as "the outcode's scores" is what made the same
    question answerable three different ways — so answer the outcode as an
    outcode: a median plus the spread, identical on every run.
    """
    rows = conn.execute(
        "SELECT transport, community, environment, price, schools, total_score, median_price "
        "FROM postcode_scores WHERE postcode LIKE ?",
        (like_pattern(scope, norm),),
    ).fetchall()
    if not rows:
        return None

    dims = {}
    for col, label in CAL_DIMS:
        s = summarise_dimension([r[col] for r in rows])
        if s:
            dims[label] = s
    overall = summarise_dimension([r["total_score"] for r in rows])
    prices = summarise_dimension([r["median_price"] for r in rows])
    return {
        "scope": scope,
        "area": norm,
        "scored_unit_postcodes": len(rows),
        "dimensions": dims,
        "overall": overall,
        "median_price": prices,
    }


# What the five dimensions actually measure. Without this the model fills the
# gap itself, and the gap it filled (2026-09-03) was Price = "value for money"
# — the exact inverse of a score that weights "the more expensive, the
# better" at 35%. Weights mirror
# simple_scorer._score_* ; keep them in step if the scorer changes.
DIMENSION_LEGEND = (
    "What each dimension measures (use these meanings — do not infer your own):\n"
    "- Transport: peak-hour commute time 60% + transit density/line count 40%.\n"
    "- Community: crime rate 40% + deprivation (IMD decile) 60%.\n"
    "- Environment: noise, flood risk, air quality, green space — 25% each.\n"
    "- Schools: quality and proximity of nearby state schools (Ofsted-rated).\n"
    "- Price: an ASSET-QUALITY score, NOT affordability and NOT value for money.\n"
    "  Price level 35% (a HIGHER price scores HIGHER), long-run 10Y CAGR 30%,\n"
    "  3Y momentum 20%, return stability 10%, transaction activity 5%.\n"
    "  An expensive area scores high BECAUSE it is expensive — never present a\n"
    "  high Price score as 性价比 / good value / 物有所值, and never as cheap."
)

def _render_area(s: dict) -> str:
    """Text form of an area aggregate — median first, spread alongside.

    The spread is not decoration: it is the difference between "SW11 scores 67"
    and "SW11 ranges 34-98 depending where you stand", and only the second is
    true. Stating it lets the answer say the area is uneven instead of quietly
    picking a side.
    """
    label = "outcode" if s["scope"] == "outcode" else "sector"
    n = s["scored_unit_postcodes"]
    lines = [
        f"Area scores for {label} {s['area']} — median across {n} scored unit postcodes.",
        "These are AREA-WIDE medians, not one address. Quote the range when it is wide.",
        "",
    ]
    for label_ in ("Transport", "Community", "Environment", "Price", "Schools"):
        d = s["dimensions"].get(label_)
        if d:
            lines.append(f"- {label_}: {d['median']}/100 (range {d['min']}-{d['max']}, n={d['n']})")
    if s.get("overall"):
        o = s["overall"]
        lines.append(f"- Overall: {o['median']}/100 (range {o['min']}-{o['max']}, n={o['n']})")
    if s.get("median_price"):
        p = s["median_price"]
        lines.append(f"- Median sold price: £{int(p['median']):,} (range £{int(p['min']):,}-£{int(p['max']):,})")
    lines.append("")
    lines.append(
        "For a specific address, call this tool again with the full unit postcode "
        "(e.g. 'SW11 3RA') — do not guess which unit postcode represents the area."
    )
    lines.append("")
    lines.append(DIMENSION_LEGEND)
    lines.append("")
    lines.append("--- structured ---")
    lines.append(json.dumps(s, indent=2, ensure_ascii=False))
    return "\n".join(lines)


def _render_unit(out: dict) -> str:
    """Text form of one unit postcode — compact lines + structured trailer.

    Leads with the calibrated 5 dimensions (the real headline scores), and
    carries the same DIMENSION_LEGEND as the area render: a unit-postcode
    answer states the Price score just as bare-facedly as an area one.
    """
    cal = out.get("calibrated")
    lines = [f"Scores for {out['postcode']}:"]
    if cal:
        for label in ("Transport", "Community", "Environment", "Price", "Schools"):
            v = cal.get(label)
            if v is not None:
                lines.append(f"- {label}: {v}/100")
        if cal.get("Overall") is not None:
            lines.append(f"- Overall: {cal['Overall']}/100 (completeness {cal.get('data_completeness', '?')})")
    if "active_listings" in out:
        a = out["active_listings"]
        lines.append(
            f"- Active listings: {a['count']} on market"
            + (f", avg £{a['avg_price']:,}" if a['avg_price'] else "")
        )
    lines.append("")
    lines.append(DIMENSION_LEGEND)
    lines.append("")
    lines.append("--- structured ---")
    lines.append(json.dumps(out, indent=2, ensure_ascii=False))
    return "\n".join(lines)


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_postcode_scores",
            description=(
                "Fetch the 5-dimension livability score for a UK postcode "
                "(Transport, Community, Environment, Price, Schools). "
                "The response defines each dimension — read that legend before "
                "interpreting a score. In particular Price is an asset-quality "
                "score where a HIGHER price scores HIGHER; it is not "
                "affordability and not value for money. "
                "Returns each dimension's raw score (0-100) plus key "
                "underlying metrics. Use this for any 'how good is X postcode' "
                "or 'tell me about X area' question. If postcode hasn't been "
                "scored yet, returns a 'not yet evaluated' message — DO NOT "
                "fabricate scores.\n"
                "ACCEPTS AN OUTCODE OR SECTOR TOO ('SW11', 'SW11 3'), which "
                "returns the MEDIAN across every scored unit postcode inside it "
                "plus the min-max range. For an area-level question ('SW11 值得"
                "买吗', 'what's Battersea like') pass the OUTCODE — never pick a "
                "representative unit postcode yourself: the same question then "
                "gets a different answer each time (SW11's schools dimension "
                "spans 33.6-98.0 across its 703 unit postcodes). Pass a full unit "
                "postcode only when the user named a specific address."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "postcode": {
                        "type": "string",
                        "description": (
                            "UK unit postcode ('N1 9DT', 'EC1V 0HB'), or an outcode / "
                            "sector for an area-wide median ('SW11', 'SW11 3'). "
                            "Auto-normalised for case and spacing."
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
    if name != "get_postcode_scores":
        raise ValueError(f"Unknown tool: {name}")

    args = arguments or {}
    raw = (args.get("postcode") or "").strip()
    if not raw:
        return [types.TextContent(type="text", text="Missing postcode parameter.")]
    scope, pc = classify_postcode_scope(raw)

    # Read-only URI mode — defense in depth. This MCP tool is exposed to the
    # public /chat agent; force the connection to reject any write
    # regardless of what the SELECT statement below evolves to.
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # An outcode / sector has no score row of its own — the per-unit lookups
    # below would all miss and the model would be left to pick a representative
    # postcode itself, differently each time. Answer the area as an area.
    if scope != "unit":
        try:
            summary = area_summary(conn, scope, pc)
        except sqlite3.OperationalError:
            summary = None
        finally:
            conn.close()
        if not summary:
            return [types.TextContent(
                type="text",
                text=(
                    f"No scored postcodes found in {pc}. Do NOT make up scores, "
                    f"and do NOT substitute a nearby postcode without saying so."
                ),
            )]
        return [types.TextContent(type="text", text=_render_area(summary))]

    out: dict[str, Any] = {"postcode": pc, "dimensions": {}}
    try:
        for table, label in DIMS:
            row = conn.execute(
                f"SELECT score, data_json, created_at FROM {table} "
                f"WHERE postcode = ? ORDER BY created_at DESC LIMIT 1",
                (pc,),
            ).fetchone()
            if row:
                try:
                    data = json.loads(row["data_json"]) if row["data_json"] else {}
                except json.JSONDecodeError:
                    data = {}
                # Trim verbose fields the agent doesn't need to see
                trimmed = {
                    k: v for k, v in data.items()
                    if k not in {"raw", "intermediate", "debug"}
                }
                out["dimensions"][label] = {
                    "score": round(row["score"], 1) if row["score"] is not None else None,
                    "as_of": row["created_at"],
                    "details": trimmed,
                }

        # Active listings count — useful context
        listings = conn.execute(
            "SELECT COUNT(*) AS n, AVG(asking_price) AS avg_p FROM rm_sales_overview "
            "WHERE postcode = ? AND delisted_date IS NULL",
            (pc,),
        ).fetchone()
        if listings and listings["n"] > 0:
            out["active_listings"] = {
                "count": listings["n"],
                "avg_price": int(listings["avg_p"]) if listings["avg_p"] else None,
            }

        # Calibrated 5-dimension scores (the headline numbers users see on the
        # site). These come from postcode_scores (precomputed by
        # compute_postcode_scores from simple_scorer) — NOT from dim_*.score,
        # which is only populated for schools + price.
        try:
            cal = conn.execute(
                "SELECT transport, community, environment, price, schools, "
                "       total_score, data_completeness, scored_at "
                "FROM postcode_scores WHERE postcode = ?",
                (pc,),
            ).fetchone()
        except sqlite3.OperationalError:
            cal = None
        if cal:
            out["calibrated"] = {
                "Transport": cal["transport"], "Community": cal["community"],
                "Environment": cal["environment"], "Price": cal["price"],
                "Schools": cal["schools"], "Overall": cal["total_score"],
                "data_completeness": cal["data_completeness"], "as_of": cal["scored_at"],
            }
    finally:
        conn.close()

    cal = out.get("calibrated")
    if not cal and not out["dimensions"]:
        return [types.TextContent(
            type="text",
            text=(
                f"Postcode {pc}: not yet evaluated. The user can trigger a "
                f"first-time evaluation by visiting /evaluate?postcode={pc.replace(' ', '+')} "
                f"on wheretolive.xyz. Do NOT make up scores."
            ),
        )]

    return [types.TextContent(type="text", text=_render_unit(out))]


async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="get-postcode-scores",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
