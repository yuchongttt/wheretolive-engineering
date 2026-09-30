#!/usr/bin/env python3
"""MCP server: screen_areas

Cross-postcode SCREENER over the precomputed postcode_scores table: find the
postcodes that satisfy multi-dimension criteria — e.g. "transport >= 70 AND
community >= 60 AND median price <= £600k" — and rank them. The inverse of the
single-postcode lookups: answers "which areas fit what I want?" rather than
"how good is X?".

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
    format="%(asctime)s screen_areas %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Data directory holding evaluations.db: $WTL_DATA_DIR, else ./data next to this file.
DATA_DIR = Path(os.environ.get("WTL_DATA_DIR") or Path(__file__).resolve().parent / "data")
DB_PATH = DATA_DIR / "evaluations.db"

# filter arg → postcode_scores column (all 0-100 score floors)
SCORE_FILTERS = {
    "min_overall": "total_score",
    "min_transport": "transport",
    "min_community": "community",
    "min_environment": "environment",
    "min_schools": "schools",
    "min_price_score": "price",
}
SORT_COLUMNS = {
    "transport": "transport DESC",
    "community": "community DESC",
    "schools": "schools DESC",
    "price_score": "price DESC",
    "overall": "total_score DESC",
    "median_price": "median_price ASC",  # cheapest first
}

server: Server = Server("screen-areas")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="screen_areas",
            description=(
                "Find postcodes matching multi-dimension criteria — a screener "
                "across all scored London postcodes. Filters: min_overall / "
                "min_transport / min_community / min_schools / min_price_score "
                "(0-100 score floors) and max_median_price (£). Ranks by sort_by (default: "
                "overall score). Use for open-ended 'which areas are good for "
                "X' / 'good transport but affordable' / 'best-value family "
                "areas under £Y' questions — NOT for a single named postcode "
                "(use get_postcode_scores / get_area_overview for that)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "min_overall": {"type": "number", "description": "Min OVERALL blended area score 0-100. Use for a generic '区域评分/邮编评分/整体好的区 大于X'."},
                    "min_transport": {"type": "number", "description": "Min transport score 0-100."},
                    "min_community": {"type": "number", "description": "Min community (safety+demographics) score 0-100."},
                    "min_environment": {"type": "number", "description": "Min environment (noise/flood/air/parks) score 0-100."},
                    "min_schools": {"type": "number", "description": "Min schools score 0-100."},
                    "min_price_score": {"type": "number", "description": "Min price/value score 0-100 (higher = better value)."},
                    "max_median_price": {"type": "integer", "description": "Max area median sale price in GBP."},
                    "sort_by": {
                        "type": "string",
                        "description": "transport | community | schools | price_score | overall | median_price. Default overall.",
                    },
                    "limit": {"type": "integer", "description": "Max results (default 15, max 40)."},
                },
                "required": [],
            },
        )
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any] | None
) -> list[types.TextContent]:
    if name != "screen_areas":
        raise ValueError(f"Unknown tool: {name}")

    args = arguments or {}
    limit = max(1, min(int(args.get("limit", 15)), 40))

    where = ["1=1"]
    params: list[Any] = []
    applied = []
    for arg, col in SCORE_FILTERS.items():
        if args.get(arg) is not None:
            where.append(f"{col} >= ?")
            params.append(float(args[arg]))
            applied.append(f"{arg}={args[arg]}")
    if args.get("max_median_price") is not None:
        where.append("median_price IS NOT NULL AND median_price <= ?")
        params.append(int(args["max_median_price"]))
        applied.append(f"max_median_price=£{int(args['max_median_price']):,}")

    sort_by = (args.get("sort_by") or "overall").strip().lower()
    order = SORT_COLUMNS.get(sort_by, "total_score DESC")

    sql = f"""
        SELECT postcode, transport, community, environment, price, schools,
               total_score, median_price
        FROM postcode_scores
        WHERE {' AND '.join(where)}
        ORDER BY {order}
        LIMIT ?
    """
    params.append(limit)

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    crit = ", ".join(applied) if applied else "no filters (top areas overall)"
    if not rows:
        return [types.TextContent(type="text", text=f"No postcodes match: {crit}. Try relaxing a threshold.")]

    lines = [f"Areas matching [{crit}], sorted by {sort_by} ({len(rows)}):"]
    lines.append("postcode    transp  commun  school  price  overall  median")
    for r in rows:
        med = f"£{round(r['median_price']/1000)}k" if r["median_price"] else "—"

        def s(v):
            return f"{v:.0f}" if v is not None else "—"
        lines.append(
            f"{(r['postcode'] or '?'):<11} "
            f"{s(r['transport']):>6}  {s(r['community']):>6}  {s(r['schools']):>6}  "
            f"{s(r['price']):>5}  {s(r['total_score']):>7}  {med:>7}"
        )
    lines.append("\n(scores 0-100, higher better; median = area median sale price)")
    return [types.TextContent(type="text", text="\n".join(lines))]


async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="screen-areas",
                server_version="0.2.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
