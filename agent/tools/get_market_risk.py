#!/usr/bin/env python3
"""MCP server: get_market_risk

Area-level "underwater" market risk: among ACTIVE listings whose prior
purchase is precisely anchored to Land Registry, what share is asking BELOW
what the current owner paid? This is our data-grounded counterpart to the
web narrative "new-build flats are falling from their peak" (agent-benchmark
round-01 Q3: the reference agent offered that section from unverifiable
mirror sources; ours is computed live from LR-anchored listings).

Methodology is pinned to the 2026-08 underwater audit (item 127) — do NOT
"simplify" it (lesson learned: re-verify numbers with the audit's ORIGINAL SQL):
  - lr_match_strategy='detail_anchored' ONLY (fuzzy street matches inflate
    the rate by ~44%)
  - canonical dedup; active only; shared-ownership + auction excluded
  - purchase-year cohort 2016-2022 (Simpson guard; window stated in output)
  - flats vs houses via radar_vocab PT_FLAT / PT_HOUSE
  - n<10 → refuse to quote a rate

Honest framing baked into the output: asking ≠ achieved price; this measures
sellers' asking positions, not realized losses.
"""

import asyncio
import logging
import os
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Any

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

sys.path.insert(0, str(Path(__file__).resolve().parent))
from radar_vocab import PT_FLAT, PT_HOUSE  # noqa: E402

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s get_market_risk %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Data directory holding evaluations.db: $WTL_DATA_DIR, else ./data next to this file.
DATA_DIR = Path(os.environ.get("WTL_DATA_DIR") or Path(__file__).resolve().parent / "data")
DB_PATH = DATA_DIR / "evaluations.db"

COHORT_START, COHORT_END = 2016, 2022

# outcode_borough also carries commuter-belt rows (Chelmsford, Epping…), so
# London scope must whitelist the 32 boroughs + City — never trust the table
# alone (2026-08-13 finding: non-London postcodes present).
LONDON_BOROUGHS = (
    "City of London", "Barking and Dagenham", "Barnet", "Bexley", "Brent",
    "Bromley", "Camden", "Croydon", "Ealing", "Enfield", "Greenwich",
    "Hackney", "Hammersmith and Fulham", "Haringey", "Harrow", "Havering",
    "Hillingdon", "Hounslow", "Islington", "Kensington and Chelsea",
    "Kingston upon Thames", "Lambeth", "Lewisham", "Merton", "Newham",
    "Redbridge", "Richmond upon Thames", "Southwark", "Sutton",
    "Tower Hamlets", "Waltham Forest", "Wandsworth", "Westminster",
)

server: Server = Server("get-market-risk")

# Outcode of a postcode_norm: everything before the final 3-char inward part.
_OC_EXPR = "substr(postcode_norm, 1, length(postcode_norm) - 3)"

# R2-2 Fix S (2026-08-14): the seller's own cost basis is the strongest
# negotiation anchor and it sits in OUR Land Registry data — but in the
# no-listing due-diligence round the checklist never named it (the reference
# answer did: "seller bought at the 2021-22 peak, held <3yrs = strong
# anchoring"). Capability pointers live in first-hop tool output, not in the
# model's memory (R2-3/R2-5 lesson: hints two hops away are a coin flip).
# Emitted on EVERY path incl. the empty cohort (review #5: 30% of outcodes
# have no cohort rows — exactly the thin-data areas where handing the user a
# next step matters most). Wording is promise-bounded (review #6): only a
# detail-anchored LR match is THIS property's sale (43.5% of stock with a
# prior sale); fetch_listing itself labels the weaker matches, so the pointer
# must not promise "the seller's price" unconditionally.
_SELLER_ANCHOR_POINTER = (
    "Given a SPECIFIC listing (id/URL), fetch_listing also shows what the "
    "seller themselves paid and when (Land Registry, when the match is "
    "verified as this property — fetch_listing says so when it is only a "
    "street-level reference). That purchase price + date is the hardest "
    "negotiation anchor (a 2021-22 peak buyer anchors high); name this check "
    "when a user asks what to verify before offering."
)

_BASE_WHERE = """
    delisted_date IS NULL
    AND (canonical_id IS NULL OR canonical_id = id)
    AND lr_match_strategy = 'detail_anchored'
    AND asking_price IS NOT NULL
    AND lr_prev_sold_price IS NOT NULL
    AND lr_prev_sold_date IS NOT NULL
    AND CAST(substr(lr_prev_sold_date, 1, 4) AS INTEGER) BETWEEN ? AND ?
    AND COALESCE(shared_ownership, 0) = 0
    AND COALESCE(is_auction, 0) = 0
"""


def cohort_rows(outcode: str | None) -> list[dict]:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if outcode:
            sql = (f"SELECT property_type, asking_price, lr_prev_sold_price "
                   f"FROM rm_sales_overview WHERE {_BASE_WHERE} AND {_OC_EXPR} = ?")
            params: tuple = (COHORT_START, COHORT_END, outcode)
        else:
            marks = ",".join("?" * len(LONDON_BOROUGHS))
            sql = (f"SELECT property_type, asking_price, lr_prev_sold_price "
                   f"FROM rm_sales_overview WHERE {_BASE_WHERE} AND {_OC_EXPR} IN "
                   f"(SELECT outcode FROM outcode_borough WHERE borough IN ({marks}))")
            params = (COHORT_START, COHORT_END, *LONDON_BOROUGHS)
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def bucket_stats(rows: list[dict]) -> dict:
    """{flats: (n, n_under, median_shortfall_pct|None), houses: (...)}"""
    out = {}
    for label, types_ in (("Flats", PT_FLAT), ("Houses", PT_HOUSE)):
        sub = [r for r in rows if (r["property_type"] or "").lower() in types_]
        under = [r for r in sub if r["asking_price"] < r["lr_prev_sold_price"]]
        med = None
        if under:
            med = statistics.median(
                (r["lr_prev_sold_price"] - r["asking_price"]) / r["lr_prev_sold_price"] * 100
                for r in under)
        out[label] = (len(sub), len(under), med)
    return out


def _bucket_line(label: str, n: int, n_under: int, med: float | None) -> str:
    if n < 10:
        return f"  {label}: n={n} — too small to quote a rate"
    line = f"  {label}: {n_under / n * 100:.1f}% asking BELOW what the owner paid (n={n}"
    if med is not None:
        line += f"; {med:.1f}% median shortfall among those"
    return line + ")"


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_market_risk",
            description=(
                "Area-level 'underwater' market risk: share of ACTIVE listings "
                "asking BELOW the price the current owner paid (Land-Registry-"
                "anchored, 2016-2022 purchase cohort), split flats vs houses, "
                "with median shortfall. Use for 'is this area/flat market "
                "risky', 'are flats here losing value', 'flat vs house as an "
                "investment' questions — it is OUR data-grounded version of "
                "'prices are down from peak' web narratives. Pass outcode "
                "(e.g. 'E1') for one area + London baseline; omit for the "
                "London-wide picture. Asking ≠ achieved price — say so when "
                "quoting it."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "outcode": {
                        "type": "string",
                        "description": "Outcode / postcode district, e.g. 'E1', "
                                       "'N14', 'SW11'. Omit for London-wide.",
                    },
                },
            },
        )
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any] | None
) -> list[types.TextContent]:
    if name != "get_market_risk":
        raise ValueError(f"Unknown tool: {name}")

    args = arguments or {}
    outcode = (args.get("outcode") or "").strip().upper().replace(" ", "") or None

    try:
        rows = await asyncio.to_thread(cohort_rows, outcode)
    except sqlite3.Error as e:
        return [types.TextContent(type="text", text=f"Market-risk data unavailable: {e}")]

    scope = outcode or "London"
    if not rows:
        return [types.TextContent(
            type="text",
            text=(f"No LR-anchored active listings in {scope}'s "
                  f"{COHORT_START}-{COHORT_END} purchase cohort — cannot compute "
                  "an underwater rate. Coverage grows as LR enrichment fills; "
                  "do not substitute a guess.\n" + _SELLER_ANCHOR_POINTER),
        )]

    stats = bucket_stats(rows)
    lines = [
        f"Market risk — {scope} (active listings whose owner bought "
        f"{COHORT_START}-{COHORT_END}, precise Land Registry anchor; "
        f"n={len(rows)}):"
    ]
    for label in ("Flats", "Houses"):
        n, n_under, med = stats[label]
        lines.append(_bucket_line(label, n, n_under, med))

    if outcode:
        try:
            base = bucket_stats(await asyncio.to_thread(cohort_rows, None))
            fl, ho = base["Flats"], base["Houses"]
            if fl[0] >= 10 and ho[0] >= 10:
                lines.append(
                    f"  London baseline: flats {fl[1] / fl[0] * 100:.1f}% / "
                    f"houses {ho[1] / ho[0] * 100:.1f}% underwater")
        except sqlite3.Error:
            pass

    lines.append(
        "Reading: 'underwater' = asking price below the owner's own purchase "
        "price — sellers' asking positions, NOT realized sale losses (asking ≠ "
        "achieved). Excludes shared-ownership/auction and fuzzy LR matches. "
        "Quote the cohort window when citing these numbers."
    )
    lines.append(_SELLER_ANCHOR_POINTER)
    return [types.TextContent(type="text", text="\n".join(lines))]


async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="get-market-risk",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
