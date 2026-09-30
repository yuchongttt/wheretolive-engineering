#!/usr/bin/env python3
"""MCP server: screen_by_commute

Reverse commute screening — "which areas are within N minutes of <hub>".

This is the screening counterpart to `get_commute` (which is point-to-point and
can only VERIFY one origin→destination at a time). It answers the natural buyer
question — "I want to be ≤30 min from Canary Wharf, where should I look?" — with
one instant indexed SELECT against the precomputed `sector_hub_commute` table
(outcode centroid → curated employment hubs, built offline by the site's
scripts/build_sector_hub_commute.py — not part of this extract).

Read-only DB connection (defence-in-depth: kernel refuses writes even if the
SELECT logic ever drifts).
"""

import asyncio
import json
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
    format="%(asctime)s screen_by_commute %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Data directory holding evaluations.db: $WTL_DATA_DIR, else ./data next to this file.
DATA_DIR = Path(os.environ.get("WTL_DATA_DIR") or Path(__file__).resolve().parent / "data")
DB_PATH = DATA_DIR / "evaluations.db"

server: Server = Server("screen-by-commute")


def _open_ro() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _available_hubs(conn: sqlite3.Connection) -> list[str]:
    try:
        return [r[0] for r in conn.execute(
            "SELECT DISTINCT hub FROM sector_hub_commute ORDER BY hub"
        )]
    except sqlite3.OperationalError:
        return []


def _match_hub(requested: str, hubs: list[str]) -> str | None:
    """Case-insensitive exact, then substring (so 'canary' → 'Canary Wharf')."""
    r = requested.strip().lower()
    for h in hubs:
        if h.lower() == r:
            return h
    for h in hubs:
        if r in h.lower() or h.lower() in r:
            return h
    return None


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="screen_by_commute",
            description=(
                "Reverse commute screening: list the postcode SECTORS (e.g. "
                "'SW11 2') within N minutes of a major employment hub, sorted "
                "fastest first. Use for any 'which areas / where should I live "
                "to be within X min of <hub>' question — this is the screener. "
                "(For a single specific postcode→place check, use get_commute.) "
                "Commute is by public transport (incl. National Rail) from each "
                "sector's CENTRE, so treat it as approximate. Covered hubs: "
                "Bank, Liverpool Street, King's Cross, Farringdon, Old Street, "
                "Oxford Circus, Victoria, Waterloo, London Bridge, Paddington, "
                "Canary Wharf, Stratford, Clapham Junction, East Croydon, "
                "Wimbledon, Lewisham, Hammersmith, Ealing Broadway, Heathrow, "
                "Highbury & Islington."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "hub": {
                        "type": "string",
                        "description": "Employment hub, e.g. 'Canary Wharf', 'Farringdon', 'Bank'.",
                    },
                    "max_minutes": {
                        "type": "integer",
                        "description": "Commute-time ceiling in minutes, e.g. 30.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max sectors to return (default 30, max 100).",
                        "default": 30,
                    },
                },
                "required": ["hub", "max_minutes"],
            },
        )
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any] | None
) -> list[types.TextContent]:
    if name != "screen_by_commute":
        raise ValueError(f"Unknown tool: {name}")

    args = arguments or {}
    hub_req = (args.get("hub") or "").strip()
    try:
        max_minutes = int(args.get("max_minutes"))
    except (TypeError, ValueError):
        return [types.TextContent(type="text", text="Need a numeric max_minutes (e.g. 30).")]
    limit = max(1, min(int(args.get("limit", 30) or 30), 100))

    def _run() -> str:
        conn = _open_ro()
        try:
            hubs = _available_hubs(conn)
            if not hubs:
                return ("Commute-screening data is not available yet "
                        "(sector_hub_commute table is empty).")
            hub = _match_hub(hub_req, hubs)
            if not hub:
                return (f"'{hub_req}' is not a covered hub. Available hubs: "
                        f"{', '.join(hubs)}. For an arbitrary destination, "
                        f"use get_commute on specific postcodes instead.")
            rows = conn.execute(
                """
                SELECT sector, minutes
                  FROM sector_hub_commute
                 WHERE hub = ? AND minutes IS NOT NULL AND minutes <= ?
                 ORDER BY minutes ASC, sector ASC
                 LIMIT ?
                """,
                (hub, max_minutes, limit),
            ).fetchall()
            if not rows:
                return (f"No sector is within {max_minutes} min of {hub} "
                        f"(by public transport from the area centre). Try a "
                        f"higher ceiling.")
            lines = [f"Postcode sectors within {max_minutes} min of {hub} "
                     f"(public transport incl. National Rail, departing a weekday "
                     f"at 09:00, from sector centre — approx; "
                     f"{len(rows)} shown, fastest first):"]
            for r in rows:
                lines.append(f"  {r['sector']}: {r['minutes']} min")
            lines.append("")
            lines.append("--- structured ---")
            lines.append(json.dumps({
                "hub": hub,
                "max_minutes": max_minutes,
                "departure": "weekday 09:00",
                "count": len(rows),
                "sectors": [{"sector": r["sector"], "minutes": r["minutes"]} for r in rows],
                "note": "minutes from sector centroid, weekday 09:00 departure; approximate",
            }, ensure_ascii=False))
            return "\n".join(lines)
        finally:
            conn.close()

    text = await asyncio.to_thread(_run)
    return [types.TextContent(type="text", text=text)]


async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="screen-by-commute",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
