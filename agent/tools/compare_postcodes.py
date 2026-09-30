#!/usr/bin/env python3
"""MCP server: compare_postcodes

Side-by-side comparison of 2-4 UK postcodes across the calibrated livability
dimensions (Transport / Community / Environment / Price / Schools) + overall
score, plus active-listing count, average asking price and area median.
Reads the precomputed postcode_scores table (populated by
compute_postcode_scores from simple_scorer output), so it shows the real
calibrated dimensions — not just the two that happen to have a dim score
column. Flags the leader on each row.

Read-only, no auth (the chat handler is the trust boundary).
"""

import asyncio
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s compare_postcodes %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Data directory holding evaluations.db: $WTL_DATA_DIR, else ./data next to this file.
DATA_DIR = Path(os.environ.get("WTL_DATA_DIR") or Path(__file__).resolve().parent / "data")
DB_PATH = DATA_DIR / "evaluations.db"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flood_area import flood_area_stats, flood_area_lines  # noqa: E402
from council_tax import rates_for_outcode, council_tax_lines  # noqa: E402
from area_scope import classify_postcode_scope  # noqa: E402
from get_postcode_scores import DIMENSION_LEGEND, area_summary  # noqa: E402

# Calibrated dimension columns in postcode_scores (0-100, higher better).
DIMS = [
    ("transport", "Transport"),
    ("community", "Community"),
    ("environment", "Environment"),
    ("price", "Price"),
    ("schools", "Schools"),
    ("total_score", "Overall"),
]

server: Server = Server("compare-postcodes")


def normalise_postcode(pc: str) -> str:
    cleaned = "".join(pc.split()).upper()
    if len(cleaned) < 5:
        return cleaned
    return cleaned[:-3] + " " + cleaned[-3:]


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="compare_postcodes",
            description=(
                "Compare 2-4 UK postcodes side by side across the calibrated "
                "livability dimensions (Transport, Community, Environment, "
                "Price, Schools) + overall score, plus active-listing count, "
                "the Land Registry median SOLD "
                "price and the mean ASKING price of current stock (two "
                "different markets — never restate one as the other). "
                "Flags which postcode leads on each "
                "row. Use for any 'A vs B' / 'which area is better for <need>' "
                "question. Postcodes not yet scored are reported as such — do "
                "NOT fabricate."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "postcodes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-4 UK postcodes, e.g. ['N1 9DT','E8 1DN']. Auto-normalised.",
                    },
                },
                "required": ["postcodes"],
            },
        )
    ]


def _load(conn: sqlite3.Connection, scope: str, pc: str) -> dict[str, Any]:
    """One compared side. A unit postcode reads its own row; an outcode/sector
    aggregates every scored unit inside it.

    An outcode has no score row of its own, so the old exact-match lookup
    returned nothing and the side rendered as "not yet scored" — a false claim
    of absence over SW11's 747 scored units. Aggregation is delegated to
    get_postcode_scores.area_summary so an area answers identically here and
    there; two calibers for one question is the bug that made the same
    question answerable three ways.
    """
    out: dict[str, Any] = {"postcode": pc, "scope": scope, "scored": False,
                           "scores": {}, "median": None, "area_n": None}
    if scope == "unit":
        row = conn.execute(
            "SELECT transport, community, environment, price, schools, total_score, "
            "       median_price, data_completeness "
            "FROM postcode_scores WHERE postcode = ?",
            (pc,),
        ).fetchone()
        if row:
            out["scored"] = True
            out["completeness"] = row["data_completeness"]
            out["median"] = row["median_price"]
            for col, _label in DIMS:
                out["scores"][col] = row[col]
    else:
        summ = area_summary(conn, scope, pc)
        if summ:
            out["scored"] = True
            out["area_n"] = summ["scored_unit_postcodes"]
            for col, label in DIMS:
                d = summ["overall"] if col == "total_score" else summ["dimensions"].get(label)
                out["scores"][col] = d["median"] if d else None
            out["median"] = summ["median_price"]["median"] if summ.get("median_price") else None

    # Active-listing count. rm_sales_overview stores postcodes mostly unspaced,
    # so match the generated postcode_norm. Area scopes use the same anti-bleed
    # idiom as flood_area (prefix RANGE + exact unit length, sector narrowed by
    # the inward first digit): a plain "SW1%" LIKE would pull SW11 and SW19 into
    # an SW1 answer.
    oc, _, inward = pc.partition(" ")
    if scope == "unit":
        preds, params = ["postcode_norm = ?"], [oc + inward]
    else:
        preds = ["postcode_norm >= ?", "postcode_norm < ?", "length(postcode_norm) = ?"]
        params = [oc, oc[:-1] + chr(ord(oc[-1]) + 1), len(oc) + 3]
        if scope == "sector":
            preds.append("substr(postcode_norm, -3, 1) = ?")
            params.append(inward)
    listings = conn.execute(
        "SELECT COUNT(*) AS n, AVG(asking_price) AS avg_p FROM rm_sales_overview "
        "WHERE delisted_date IS NULL "
        "  AND (canonical_id IS NULL OR canonical_id = id) "
        f"  AND {' AND '.join(preds)}",
        params,
    ).fetchone()
    out["active_listings"] = listings["n"] if listings else 0
    out["avg_asking_price"] = round(listings["avg_p"]) if listings and listings["avg_p"] else None
    return out


def _row(label: str, values: list, fmt, leader_high: bool = True) -> str:
    present = [v for v in values if v is not None]
    best = (max(present) if leader_high else min(present)) if present else None
    cells = []
    for v in values:
        if v is None:
            cells.append("—".ljust(12))
        else:
            mark = "*" if v == best else " "
            cells.append((fmt(v) + mark).ljust(12))
    return label.ljust(24) + "".join(cells)


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any] | None
) -> list[types.TextContent]:
    if name != "compare_postcodes":
        raise ValueError(f"Unknown tool: {name}")

    args = arguments or {}
    raw = args.get("postcodes") or []
    if isinstance(raw, str):
        raw = [p for p in raw.replace(",", " ").split() if p]
    parsed = [classify_postcode_scope(p) for p in raw][:4]
    pcs = [norm for _scope, norm in parsed]
    if len(pcs) < 2:
        return [types.TextContent(type="text", text="Provide at least 2 postcodes to compare.")]

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        data = [_load(conn, scope, norm) for scope, norm in parsed]
    finally:
        conn.close()

    lines: list[str] = [f"Comparing: {', '.join(pcs)}"]
    unscored = [d["postcode"] for d in data if not d["scored"]]
    if unscored:
        lines.append(f"⚠ Not yet scored: {', '.join(unscored)} — suggest /evaluate?postcode=… first.")
    _areas = [d for d in data if d["scope"] != "unit" and d["scored"]]
    if _areas:
        lines.append("; ".join(
            f'{d["postcode"]} = AREA-WIDE median across {d["area_n"]} scored unit postcodes'
            for d in _areas) + " — not one address; units inside an area disagree widely.")
    lines.append("")
    lines.append("Dimension".ljust(24) + "".join(d["postcode"].ljust(12) for d in data))
    for col, label in DIMS:
        lines.append(_row(label, [d["scores"].get(col) for d in data], lambda v: f"{v:.0f}"))
    lines.append(_row("median sold price", [d["median"] for d in data],
                      lambda v: f"£{v/1000:.0f}k", leader_high=False))
    lines.append(_row("avg asking", [d["avg_asking_price"] for d in data],
                      lambda v: f"£{v/1000:.0f}k", leader_high=False))
    lines.append("active listings".ljust(24) + "".join(str(d["active_listings"]).ljust(12) for d in data))
    lines.append("\n(* = leader; for price rows leader = cheapest. Scores 0-100, higher better. Environment may be blank where its inputs aren't cached.)")
    # The two price rows are different markets and were one word apart from
    # being read as the same one: three /chat runs (2026-09-03) rendered
    # "median price" as "on-market median price", when it is Land Registry
    # SOLD. Name the caliber on the row itself, not only here.
    lines.append("median sold price = Land Registry completed sales (what changed hands); "
                 "avg asking = mean asking price of what is on the market right now. "
                 "Different markets — do not present either as the other.")
    lines.append("")
    lines.append(DIMENSION_LEGEND)

    # R2-4 Fix W: area flood aggregate per compared side (outcode level) —
    # cross-area moves are exactly where "one side of the district floods and the
    # other doesn't" changes the decision, and the per-listing stamps already
    # hold the answer.
    # Derive the outcode from the normalised scope, not by slicing: normalise
    # lets short inputs like "TW9" through, and [:-3] would yield "S" and
    # silently drop the block (review). Dedupe by derived outcode so two sides
    # in the same outcode don't print the same line twice.
    _ocs = list(dict.fromkeys(norm.partition(" ")[0] for _scope, norm in parsed))
    _fl_stats = [flood_area_stats(outcode=oc) for oc in _ocs]
    _fl_lines = flood_area_lines(_fl_stats)
    if _fl_lines:
        lines.append("")
        lines.append("Flood zones (EA, area aggregate of active listings):")
        lines.extend("  " + ln for ln in _fl_lines)

    # R2-4 Fix V: same council-tax band costs DIFFERENT money across boroughs
    # (Croydon D £2,600 vs Westminster D £1,050 — £1,550/yr in the test
    # fixture). Cross-borough moves are exactly where this bites.
    _ct_lines = council_tax_lines([rates_for_outcode(oc) for oc in _ocs])
    if _ct_lines:
        lines.append("")
        lines.append("Council tax (official area Band D, gov.uk):")
        lines.extend("  " + ln for ln in _ct_lines)

    return [types.TextContent(type="text", text="\n".join(lines))]


async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="compare-postcodes",
                server_version="0.2.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
