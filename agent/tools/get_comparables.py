#!/usr/bin/env python3
"""MCP server: get_comparables

Find comparable sold prices for a property or a postcode. Same data backbone
as /value-property uses: Land Registry transactions matched by outcode +
(optionally) property_type + recency, ranked same-postcode → recency.

IMPORTANT — what Land Registry does and does NOT carry:
  - HAS: price, date, property_type (F/T/S/D/O), tenure, address (paon/street).
  - HAS NOT: bedroom count, floor area. So these comparables can NOT be
    filtered by bedrooms, and there is NO £/m² here. A `beds` arg is accepted
    for backward-compat but is advisory only — it does NOT filter. For a
    bedroom-specific sold figure use get_sold_nearby (listing-matched, has beds);
    for £/m² use get_price_trend (postcode-level). Callers must never present
    a per-bedroom or £/m² claim as if it came from this tool.

Property_type IS available, so we filter flats vs houses when we know the
subject's type (or the caller passes property_type) — otherwise a 2-bed HOUSE
gets a median polluted by cheap flats on the same street.

Usage scenarios:
  - "What have places sold for in E14?" — pass postcode=E14
  - "Is this house's price fair?" — pass postcode + property_type='house'
  - "What's this listing actually worth?" — pass a listing property_id from
    rm_sales_overview; we look up its postcode + type then comparables
"""

import asyncio
import json
import logging
import os
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Any

from coverage_note import out_of_coverage_note
from geo_radius import UNKNOWN_POSTCODE_ALERT, unit_postcode_known
from lr_category import CATEGORY_B_GLOSS, CATEGORY_B_LABEL

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s get_comparables %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Data directory holding evaluations.db: $WTL_DATA_DIR, else ./data next to this file.
DATA_DIR = Path(os.environ.get("WTL_DATA_DIR") or Path(__file__).resolve().parent / "data")
DB_PATH = DATA_DIR / "evaluations.db"

server: Server = Server("get-comparables")


def normalise_postcode(pc: str) -> str:
    cleaned = "".join(pc.split()).upper()
    if len(cleaned) < 5:
        return cleaned
    return cleaned[:-3] + " " + cleaned[-3:]


def outcode_of(pc: str) -> str:
    return pc.strip().upper().split(" ")[0]


# LR property_type single-char codes.
LR_TYPE_NAME = {"F": "Flat", "T": "Terraced", "S": "Semi-detached",
                "D": "Detached", "O": "Other"}
# Which LR codes a free-text/listing property_type maps to. Houses (T/S/D) vs flats
# (F) — the split that stops a 2-bed house's median being dragged down by
# cheap flats on the same street. Returns None when the type is unknown/mixed
# (no filter — report as all-types).
_FLAT_WORDS = ("flat", "apartment", "maisonette", "studio", "penthouse")
_HOUSE_WORDS = ("house", "terrace", "terraced", "semi", "detached", "bungalow",
                "mews", "cottage", "town house", "townhouse", "end of terrace")


def type_codes_for(raw: str | None) -> set[str] | None:
    """Map a subject/free-text property_type to a set of LR codes, or None."""
    if not raw:
        return None
    t = raw.strip().lower()
    if t in {"f", "t", "s", "d", "o"}:          # already an LR code
        return {t.upper()}
    if t in {"flat", "flats"} or any(w in t for w in _FLAT_WORDS):
        return {"F"}
    if t in {"house", "houses"} or any(w in t for w in _HOUSE_WORDS):
        return {"T", "S", "D"}
    return None


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_comparables",
            description=(
                "Recent sold-price comparables from Land Registry (last 3 "
                "years, same outcode, ranked same-postcode > recency). Either "
                "provide a listing property_id (we look up its postcode + "
                "property_type) OR pass postcode (+ optional property_type). "
                "Returns price, date, property_type, tenure and address per "
                "sale, plus the range + median. Use for 'is this price fair' / "
                "'what have similar places sold for' / market context. "
                "IMPORTANT: Land Registry carries NO bedroom count and NO floor "
                "area — these comparables are NOT bedroom-filtered and this tool "
                "returns NO £/m² and NO distance. Do not state a per-bedroom or "
                "£/m² or distance figure from this tool's output. Pass "
                "property_type='house' or 'flat' to avoid mixing houses with "
                "flats (it does split the median). For bedroom-specific sold "
                "data use get_sold_nearby; for £/m² use get_price_trend."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "property_id": {
                        "type": "string",
                        "description": "Listing property_id from rm_sales_overview (optional). We derive its postcode + property_type.",
                    },
                    "postcode": {
                        "type": "string",
                        "description": "UK postcode (used if property_id not given).",
                    },
                    "property_type": {
                        "type": "string",
                        "description": "Optional — 'house' or 'flat' (or an LR code F/T/S/D). Filters comparables to that family so a house's median isn't polluted by flats. Derived from the subject when property_id is given.",
                    },
                    "beds": {
                        "type": "integer",
                        "description": "Advisory ONLY — Land Registry has no bedroom data, so this does NOT filter results. Kept for backward-compat; prefer get_sold_nearby for bedroom-specific sold data.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 10, max 25).",
                        "default": 10,
                    },
                },
                "required": [],
            },
        )
    ]


def fetch_subject(conn: sqlite3.Connection, property_id: str) -> dict[str, Any] | None:
    # Column names must match rm_sales_overview as it actually is: the listing
    # id column is `id` (not `property_id`) and the address column is `address`
    # (not `displayAddress`). Getting these wrong made the whole by-id path —
    # this tool's headline use case — throw OperationalError on every call.
    # tests/test_comparables_subject_lookup.py pins the SELECT to the real
    # schema so a hand-written fixture can't hide the drift again.
    row = conn.execute(
        """SELECT id AS property_id, postcode, bedrooms, bathrooms, asking_price,
                  property_type, address
             FROM rm_sales_overview WHERE id = ?""",
        (property_id,),
    ).fetchone()
    return dict(row) if row else None


def _comp_where(postcode: str, type_codes: set[str] | None,
                category: str | None):
    # Deliberate scope note (review altitude finding): the 3-year window is
    # NOT month-quality-gated (v_lr_clean's distorted-month exclusions).
    # Comps are recency-ranked individual rows — dropping whole months would
    # hide real recent sales; the tradeoff differs from the baseline layer's
    # aggregates. If deadline-month distortion ever matters here, join
    # lr_month_quality instead of re-deriving it.
    outcode = outcode_of(postcode)
    # LR postcodes are stored canonically WITH a space ("N1 9DT"), so match the
    # outcode as 'N1 %' — the trailing space stops "N1" from also matching
    # N10/N12/.../N19 (the substr-prefix form did, surfacing wrong-area comps).
    where = ["t.postcode LIKE ?", "t.date >= date('now', '-3 years')"]
    params: list[Any] = [f"{outcode} %"]
    if type_codes:
        # Filter to the requested type family (e.g. houses = T/S/D) so a house's
        # median isn't dragged down by cheap flats on the same street.
        where.append(f"t.property_type IN ({','.join('?' for _ in type_codes)})")
        params.extend(sorted(type_codes))
    if category:
        # Review R3-1-refix: the category split must live in SQL, BEFORE the
        # LIMIT — post-fetch filtering let B rows crowd standard comps out of
        # the sample window with no back-fill.
        where.append("t.category = ?")
        params.append(category)
    return where, params


def fetch_comparables(
    conn: sqlite3.Connection,
    postcode: str,
    limit: int,
    type_codes: set[str] | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    where, params = _comp_where(postcode, type_codes, category)
    sql = f"""
        SELECT t.price, t.date, t.property_type, t.tenure, t.new_build,
               t.paon, t.street, t.postcode, t.category
          FROM lr_transactions t
         WHERE {' AND '.join(where)}
         ORDER BY
           CASE WHEN t.postcode = ? THEN 0 ELSE 1 END,
           t.date DESC
         LIMIT ?
    """
    params.extend([postcode, limit])
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def count_comparables(conn: sqlite3.Connection, postcode: str,
                      type_codes: set[str] | None, category: str) -> int:
    where, params = _comp_where(postcode, type_codes, category)
    return conn.execute(
        f"SELECT COUNT(*) FROM lr_transactions t WHERE {' AND '.join(where)}",
        params,
    ).fetchone()[0]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any] | None
) -> list[types.TextContent]:
    if name != "get_comparables":
        raise ValueError(f"Unknown tool: {name}")

    args = arguments or {}
    property_id = (args.get("property_id") or "").strip()
    postcode = (args.get("postcode") or "").strip()
    beds = args.get("beds")  # advisory only — LR has no bedroom data
    ptype_arg = (args.get("property_type") or "").strip() or None
    limit = max(1, min(int(args.get("limit", 10) or 10), 25))

    if not property_id and not postcode:
        return [types.TextContent(
            type="text",
            text="Either property_id OR postcode is required.",
        )]

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        subject = None
        if property_id:
            subject = fetch_subject(conn, property_id)
            if not subject:
                return [types.TextContent(
                    type="text",
                    text=f"property_id={property_id} not found in rm_sales_overview.",
                )]
            postcode = subject.get("postcode") or postcode
            if beds is None:
                beds = subject.get("bedrooms")
            # Derive the type family from the subject unless the caller was explicit.
            if ptype_arg is None:
                ptype_arg = subject.get("property_type")

        if not postcode:
            return [types.TextContent(
                type="text",
                text="No postcode resolved (either pass it or use property_id with valid postcode).",
            )]

        pc = normalise_postcode(postcode)
        type_codes = type_codes_for(ptype_arg)
        # R3-2 T18b: comps are outcode-wide by design, so a nonexistent unit
        # postcode would otherwise get sector-flavoured numbers that make the
        # address look real. Flag it before any figures.
        postcode_alert = (
            UNKNOWN_POSTCODE_ALERT.format(pc=pc, scope=f"outcode {outcode_of(pc)}")
            if unit_postcode_known(conn, pc) is False else None
        )
        # Fix AK (R3-1) + review R3-1-refix: STANDARD sales (category A) are
        # fetched in SQL so B rows can't crowd them out of the LIMIT window;
        # category B is disclosed via a window COUNT plus a few labelled
        # example rows (ammunition, not statistics).
        basis_note = None
        try:
            comps = fetch_comparables(conn, pc, limit, type_codes,
                                      category="A")
            # Honest fallback: if the type filter left no STANDARD comps
            # (thin market), widen to all types rather than mislead with
            # "no comparables", and say so. Widening keys on the CLEAN
            # sample — the same filtered-n discipline as get_sold_nearby.
            type_filtered = bool(type_codes) and bool(comps)
            if type_codes and not comps:
                type_codes = None
                comps = fetch_comparables(conn, pc, limit, category="A")
            b_count = count_comparables(conn, pc, type_codes, "B")
            b_examples = (fetch_comparables(conn, pc, 3, type_codes,
                                            category="B")
                          if b_count else [])
        except sqlite3.OperationalError as exc:
            if "category" not in str(exc):
                raise
            # Deployment predates the category column: degrade to the old
            # unfiltered behaviour, loudly — never crash, never claim a
            # clean basis (review finding 5).
            comps = fetch_comparables(conn, pc, limit, type_codes)
            type_filtered = bool(type_codes) and bool(comps)
            if type_codes and not comps:
                type_codes = None
                comps = fetch_comparables(conn, pc, limit)
            b_count, b_examples = 0, []
            basis_note = ("⚠ category column unavailable on this deployment "
                          "— stats may include non-standard (category B) "
                          "sales; re-run the LR import.")
    finally:
        conn.close()

    if not comps and not b_count:
        # Empty because the area has been quiet, or empty because we hold no
        # Land Registry rows for that postal area at all? Only the second one
        # needs saying — and only it makes "no comparable sales exist" a lie.
        gap = out_of_coverage_note(pc, missing="Land Registry transactions")
        empty = f"No Land Registry transactions found near {pc} in the last 3 years."
        return [types.TextContent(
            type="text",
            text=f"{gap}\n\n{empty}" if gap else empty,
        )]

    prices = [c["price"] for c in comps if c.get("price")]
    median = int(statistics.median(prices)) if prices else None
    lo = min(prices) if prices else None
    hi = max(prices) if prices else None

    # Label exactly what the median is over, so it can't be mistaken for a
    # per-bedroom / like-for-like figure.
    if type_filtered:
        fam = "/".join(LR_TYPE_NAME[c] for c in sorted(type_codes))
        scope = f"{fam} only"
    elif ptype_arg and type_codes_for(ptype_arg):  # requested but widened
        scope = "ALL property types (too few of the requested type — widened)"
    else:
        scope = "ALL property types"

    lines = [f"Comparables for {pc} — {scope}, last 3 years:"]
    if postcode_alert:
        lines.insert(0, postcode_alert)
    if basis_note:
        lines.append(basis_note)
    if median:
        lines.append(
            f"Range: £{lo:,} – £{hi:,}  median £{median:,}  "
            f"(n={len(prices)} standard sales; {b_count} category-B "
            "transaction(s) in the same window excluded — "
            f"{CATEGORY_B_GLOSS})"
        )
    elif b_count:
        # All-B corner: no standard-sale stats exist — say so, quote nothing
        # (review finding 4: the old path quoted a repossession median
        # labelled "standard sales" and fed it into the subject delta).
        lines.append(
            f"No standard-sale comparables in the last 3 years — "
            f"{b_count} category-B (non-standard: {CATEGORY_B_GLOSS}) "
            "transaction(s) exist and examples are listed below, but they "
            "are NOT usable as market comps and no median is quoted."
        )
    if subject and subject.get("asking_price") and median:
        ap = subject["asking_price"]
        d = (ap - median) / median * 100
        lines.append(f"Subject asking £{ap:,} ({d:+.1f}% vs median)")
    lines.append("")

    def _comp_line(c):
        addr_bits = [bit for bit in (c.get("paon"), c.get("street")) if bit]
        addr = " ".join(addr_bits) or "(addr unknown)"
        return (
            f"- £{c['price']:,}  {c['date'][:7]}  "
            f"{LR_TYPE_NAME.get(c.get('property_type'), c.get('property_type','?'))}  "
            f"{c.get('tenure','?')}  {addr}, {c.get('postcode','?')}"
            + (CATEGORY_B_LABEL
               if (c.get("category") or "").upper() == "B" else "")
        )

    for c in comps[:limit]:
        lines.append(_comp_line(c))
    for c in b_examples:
        lines.append(_comp_line(c))
    lines.append("")
    lines.append(
        "NOTE: Land Registry has no bedroom count or floor area, so this list "
        "is NOT bedroom-filtered and there is NO £/m² or distance here — do not "
        "state any per-bedroom, £/m² or distance figure from this result. "
        "For bedroom-specific sold data use get_sold_nearby; for £/m² use "
        "get_price_trend."
    )
    lines.append("")
    lines.append("--- structured ---")
    lines.append(json.dumps({
        "postcode": pc,
        "type_scope": scope,
        "subject": subject,
        "stats": {"n": len(prices), "min": lo, "median": median, "max": hi,
                  "basis": ("standard open-market sales only (LR category A)"
                            if not basis_note else
                            "ALL categories (category column unavailable)"),
                  # count of category-B transactions in the SAME window/type
                  # scope (not fetched-set arithmetic — review finding 4)
                  "excluded_category_b": b_count},
        "transactions": comps + b_examples,
    }, indent=2, ensure_ascii=False, default=str))
    return [types.TextContent(type="text", text="\n".join(lines))]


async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(
            read, write,
            InitializationOptions(
                server_name="get-comparables",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
