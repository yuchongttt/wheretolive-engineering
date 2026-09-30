#!/usr/bin/env python3
"""MCP server: compute_buying_costs

Total funds ≠ house price. In an agent-benchmark scenario the user stated a
total budget and our agent used that total as the search price cap — twice.
Stamp duty, transaction fees, renovation and a cash buffer all come out of the
same pot, so the honest search ceiling is well below the stated budget.
The reference agent caught this with live LLM arithmetic; we pin it as a
deterministic tool instead (verified numbers, testable, no re-roll risk).

SDLT rates (England, main residence, in force 2026 — gov.uk; every figure
below was hand-checked against the R2-1 worked example):
  Standard: 0% to £125k · 2% £125k–£250k · 5% £250k–£925k ·
            10% £925k–£1.5M · 12% above
  First-time buyer relief: 0% to £300k · 5% £300k–£500k;
            price ABOVE £500k voids the relief entirely (cliff).
Not covered (say so, don't guess): additional-property surcharge,
non-resident surcharge, shared-ownership elections, Scotland/Wales (LBTT/LTT).
"""

import asyncio
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from council_tax import rates_for_outcode  # noqa: E402

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s compute_buying_costs %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# SDLT rates are LAW, and law changes at Budgets. Before 2026-08-18 the vintage
# lived only in this comment — the model never saw it and the user-facing line
# was a bare "SDLT £X". A Budget could silently turn every stamp-duty figure we
# quote into a wrong one, on a number people actually pay. So the rates now
# carry their own as-of date INTO the output, and the wording escalates once
# they are old enough that a Budget has plausibly moved them.
RATES_AS_OF = "2026-04-06"          # start of the 2026-27 tax year
RATES_SOURCE = "gov.uk stamp-duty-land-tax rates"
RATES_STALE_AFTER_DAYS = 200        # ~ one Budget cycle

# (upper bound of band, rate) — England main-residence SDLT.
STANDARD_BANDS = [(125_000, 0.0), (250_000, 0.02), (925_000, 0.05),
                  (1_500_000, 0.10), (float("inf"), 0.12)]
FTB_BANDS = [(300_000, 0.0), (500_000, 0.05)]
FTB_CEILING = 500_000          # above this the relief is void, not tapered
# Non-residential / mixed-use SDLT (R2-5 Fix Q, 2026-08-14): a mixed-use
# freehold (the R2-5 benchmark case) pays these — often LESS than residential, since
# neither the additional-dwelling (+5%) nor the non-resident (+2%) surcharge
# nor FTB relief exists on this track. Anchor: £850k → £32,000.
NONRES_BANDS = [(150_000, 0.0), (250_000, 0.02), (float("inf"), 0.05)]

DEFAULT_FEES = 6_000           # conveyancing+searches ~£2.5k · L3 survey ~£1.2k
                               # · mortgage fee ~£1.2k · moving ~£1k (typical)
DEFAULT_BUFFER = 15_000

server: Server = Server("compute-buying-costs")


def _banded(price: int, bands) -> int:
    tax, prev = 0.0, 0
    for cap, rate in bands:
        if price > prev:
            tax += (min(price, cap) - prev) * rate
        prev = cap
    return round(tax)


def sdlt(price: int, first_time_buyer: bool) -> int:
    """England main-residence SDLT. FTB relief is a cliff at £500k, not a taper."""
    if first_time_buyer and price <= FTB_CEILING:
        return _banded(price, FTB_BANDS)
    return _banded(price, STANDARD_BANDS)


def sdlt_nonres(price: int) -> int:
    """England non-residential / mixed-use SDLT (no relief, no surcharges)."""
    return _banded(price, NONRES_BANDS)


# R2-7 Fix Y (2026-08-15): the reference answer's most decision-useful artefact
# was a per-month table (mortgage P&I + council tax + service charge). We held
# every input (bands per listing, Fix V borough rates, SC/GR) and no monthly
# arithmetic. Assumptions are LABELLED, never silent: rate defaults to a stated
# assumption and every figure names deposit/rate/term.
ASSUMED_RATE = 5.25   # % annual, typical 2-yr fix ballpark — always labelled
ASSUMED_TERM = 30     # years


def monthly_pi(price: int, deposit_pct: float, rate_pct: float,
               term_years: int) -> int:
    """Standard repayment-mortgage P&I per month."""
    loan = price * (1 - deposit_pct / 100)
    r = rate_pct / 100 / 12
    n = term_years * 12
    if r <= 0:
        return round(loan / n)
    return round(loan * r / (1 - (1 + r) ** -n))


def _monthly_block(price: int, rate, term, deposit_pct,
                   band, outcode, sc_pa, gr_pa) -> list[str]:
    # review A#3: `if rate` swallowed an explicit 0 as "use default" — test for
    # None instead. A#4: deposit domain guard (0-100 is valid, 100 = cash
    # purchase); out of range falls back to the default AND says so; term<=0
    # falls back to the default.
    rate_v = float(rate) if rate is not None else ASSUMED_RATE
    rate_given = rate is not None and float(rate) > 0
    # review B#5: passing 0.0525 (the % written as a fraction) would compute a
    # near-zero-interest payment; a suspicious range falls back to the default.
    if rate_given and not (1.0 <= rate_v <= 15.0):
        rate_given = False
        rate_v = ASSUMED_RATE
    term_v = int(term) if term and int(term) > 0 else ASSUMED_TERM
    dep_given = deposit_pct is not None
    dep_v = float(deposit_pct) if dep_given else 10.0
    lines = [f"Monthly ownership at £{price:,} (assumptions labelled):"]
    if not (0 <= dep_v <= 100):
        lines.append(f"  (deposit_pct {dep_v:.0f} is outside 0-100 — using 10% default)")
        dep_v, dep_given = 10.0, False
    cash = dep_v >= 100
    total = 0.0
    if cash:
        lines.append("  cash purchase (100% deposit) — no mortgage payment")
    else:
        # Assumptions are always labelled (review B#5): the rate must name its
        # source too — the tool cannot tell a user's quote from a rate the model
        # remembered, and neither is OUR quote.
        rate_lbl = (f"{rate_v:.2f}% (caller-supplied, not our quote)" if rate_given
                    else f"{ASSUMED_RATE:.2f}% ASSUMED — confirm a real quote")
        if not rate_given:
            rate_v = ASSUMED_RATE
        pi = monthly_pi(price, dep_v, rate_v, term_v)
        line = (f"  mortgage P&I: £{pi:,}/mo at {dep_v:.0f}% deposit "
                f"(rate {rate_lbl}, {term_v}yr term)")
        if not dep_given:
            line += f" · £{monthly_pi(price, 20, rate_v, term_v):,}/mo at 20%"
        lines.append(line)
        total += pi
    # review A#1: be lenient on outcode/band shapes (full postcode -> outcode;
    # "Band D" -> "D"). When they can't be resolved, say EXPLICITLY that council
    # tax is not included, and list exclusions on the total line — a silently
    # missing item is worse than none.
    tax_included = False
    tax_straddle = False
    if band and outcode:
        oc = str(outcode).strip().upper().replace(" ", "")
        if len(oc) >= 5:
            oc = oc[:-3]
        import re as _re
        bm = _re.search(r"([A-H])\s*$", str(band).strip().upper())
        rt = rates_for_outcode(oc) if oc else None
        amt = rt["bands"].get(bm.group(1)) if (rt and bm) else None
        if amt:
            tax_included = True
            total += amt / 12
            note = (f"  + council tax £{amt / 12:,.0f}/mo (Band {bm.group(1)}, "
                    f"{rt['borough']} {rt['year']}, incl. GLA)")
            if (rt.get("borough_share") or 1.0) < 0.85:
                tax_straddle = True
                note += " ⚠ outcode straddles boroughs — confirm the address's borough"
            lines.append(note)
        else:
            lines.append(f"  + council tax: NOT included — couldn't resolve rates "
                         f"for band={band!r} outcode={outcode!r}; add it on top")
    if sc_pa:
        total += float(sc_pa) / 12
        lines.append(f"  + service charge £{float(sc_pa) / 12:,.0f}/mo")
    if gr_pa:
        total += float(gr_pa) / 12
        lines.append(f"  + ground rent £{float(gr_pa) / 12:,.0f}/mo")
    excl = ["utilities/insurance"]
    if not tax_included:
        excl.insert(0, "council tax")
    tail = f"  ≈ total £{total:,.0f}/mo at {dep_v:.0f}% deposit (excl. {', '.join(excl)})"
    if tax_straddle:
        tail += " — council-tax figure assumes the majority borough, see ⚠"
    lines.append(tail)
    return lines


def max_affordable(total_funds: int, first_time_buyer: bool,
                   renovation: int = 0, buffer: int = DEFAULT_BUFFER,
                   fees: int = DEFAULT_FEES, sdlt_fn=None) -> int:
    """Largest price with price + SDLT + fees + renovation + buffer ≤ funds.
    SDLT is monotonic in price, so binary search is exact enough (±£500).
    sdlt_fn overrides the tax table (review Simp1/Reuse1/Alt5: the non-res
    track once inlined this whole search; one solver, parameterised)."""
    tax = sdlt_fn or (lambda p: sdlt(p, first_time_buyer))
    lo, hi = 0, total_funds
    while hi - lo > 500:
        mid = (lo + hi) // 2
        if mid + tax(mid) + fees + renovation + buffer <= total_funds:
            lo = mid
        else:
            hi = mid
    return lo


def rates_vintage_note(as_of: str | None = None, today: str | None = None) -> str:
    """One line saying which rules these figures are under, louder once stale."""
    as_of = as_of or RATES_AS_OF
    now = date.fromisoformat(today) if today else date.today()
    age = (now - date.fromisoformat(as_of)).days
    if age > RATES_STALE_AFTER_DAYS:
        return (f"[SDLT rules as of {as_of} ({RATES_SOURCE}) — that is {age} days "
                f"old, so a Budget may have changed them: VERIFY on gov.uk before "
                f"the user relies on these figures.]")
    return f"[SDLT rules as of {as_of} — {RATES_SOURCE}.]"


def _cost_lines(price: int, ftb: bool, label: str) -> list[str]:
    tax = sdlt(price, ftb)
    return [f"  {label}: SDLT £{tax:,} → all-in ≈ £{price + tax + DEFAULT_FEES:,} "
            f"(price + SDLT + ~£{DEFAULT_FEES:,} typical fees) "
            f"{rates_vintage_note()}"]


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="compute_buying_costs",
            description=(
                "Deterministic UK buying-cost calculator: SDLT (standard AND "
                "first-time-buyer tracks incl. the £500k relief cliff), typical "
                "transaction fees, and — given total_funds — the MAX AFFORDABLE "
                "PRICE after all costs. **MUST be called before setting any "
                "search price cap whenever the user states a TOTAL budget or "
                "deposit+mortgage structure ('£600k budget, £450k deposit + "
                "£150k mortgage') — total funds ≠ house price.** Then search "
                "with the returned affordable price, not the raw budget. "
                "England main residence only; says so when surcharges may apply. "
                "For a MIXED-USE or commercial property (shop + flats, whole "
                "tenanted building — see the mixed_use mortgage flag) pass "
                "property_class=mixed_use_or_commercial: SDLT then follows "
                "non-residential rates (usually lower, no surcharges). "
                "Costing a specific price (residential track) also returns a MONTHLY OWNERSHIP "
                "block (mortgage P&I with labelled assumptions + official "
                "council tax £/mo when you pass council_tax_band+outcode + "
                "SC/GR) — use it whenever a user weighs two homes or asks "
                "about affordability per month."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "total_funds": {
                        "type": "integer",
                        "description": "Total money available (deposit + mortgage), £. "
                                       "Returns the max affordable price after costs.",
                    },
                    "price": {
                        "type": "integer",
                        "description": "A specific property price to cost out, £.",
                    },
                    "first_time_buyer": {
                        "type": "boolean",
                        "description": "Omit if unknown — both tracks are shown.",
                    },
                    "renovation": {
                        "type": "integer",
                        "description": "Planned renovation budget, £ (default 0).",
                    },
                    "buffer": {
                        "type": "integer",
                        "description": f"Cash buffer to keep, £ (default {DEFAULT_BUFFER:,}).",
                    },
                    "mortgage_rate": {
                        "type": "number",
                        "description": "Annual mortgage rate % for the monthly block; omit → 5.25% labelled ASSUMED.",
                    },
                    "term_years": {"type": "integer", "description": "Mortgage term (default 30)."},
                    "deposit_pct": {"type": "number", "description": "Deposit % (default 10; a 20% variant is also shown)."},
                    "council_tax_band": {
                        "type": "string",
                        "description": "Listing's band A-H (from fetch_listing) — adds official £/mo council tax to the monthly block.",
                    },
                    "outcode": {
                        "type": "string",
                        "description": "Listing's outcode (e.g. 'W5') — resolves the borough's official rates for the band.",
                    },
                    "service_charge_pa": {"type": "number", "description": "Annual service charge £ to fold into £/mo."},
                    "ground_rent_pa": {"type": "number", "description": "Annual ground rent £ to fold into £/mo."},
                    "property_class": {
                        "type": "string",
                        "enum": ["residential", "mixed_use_or_commercial"],
                        "description": (
                            "Default residential. Use mixed_use_or_commercial "
                            "for a mixed-use or commercial property (e.g. shop "
                            "+ flats, whole tenanted building) — SDLT then "
                            "follows NON-residential rates (0/2/5% at "
                            "£150k/£250k): no FTB relief, no additional-"
                            "dwelling or non-resident surcharge."
                        ),
                    },
                },
            },
        )
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any] | None
) -> list[types.TextContent]:
    if name != "compute_buying_costs":
        raise ValueError(f"Unknown tool: {name}")

    args = arguments or {}
    total_funds = args.get("total_funds")
    price = args.get("price")
    ftb = args.get("first_time_buyer")           # None = unknown → show both
    renovation = int(args.get("renovation") or 0)
    buffer = int(args.get("buffer") or DEFAULT_BUFFER)
    nonres = (args.get("property_class") or "residential") == "mixed_use_or_commercial"
    m_rate = args.get("mortgage_rate")
    m_term = args.get("term_years")
    m_dep = args.get("deposit_pct")
    m_band = args.get("council_tax_band")
    m_outcode = args.get("outcode")
    m_sc = args.get("service_charge_pa")
    m_gr = args.get("ground_rent_pa")

    if not total_funds and not price:
        return [types.TextContent(type="text", text=(
            "Pass total_funds (deposit + mortgage — I return the max affordable "
            "price after SDLT/fees/buffer) and/or price (I cost out that "
            "specific price)."))]

    lines: list[str] = []

    if nonres:
        # Non-residential track (mixed-use / commercial): one rate table, no
        # relief, no surcharges — FTB tracks are residential-only concepts.
        if total_funds:
            lo = max_affordable(int(total_funds), False, renovation, buffer,
                                sdlt_fn=sdlt_nonres)
            lines.append(
                f"Total funds £{int(total_funds):,}, mixed-use/commercial track: "
                f"max affordable price ≈ £{lo:,} "
                f"(price + non-residential SDLT £{sdlt_nonres(lo):,} + fees "
                f"~£{DEFAULT_FEES:,}"
                + (f" + renovation £{renovation:,}" if renovation else "")
                + f" + buffer £{buffer:,}).")
        if price:
            tax = sdlt_nonres(int(price))
            lines.append(
                f"Costing £{int(price):,} as mixed-use/commercial: "
                f"NON-residential SDLT £{tax:,} → all-in ≈ "
                f"£{int(price) + tax + DEFAULT_FEES:,} (price + SDLT + "
                f"~£{DEFAULT_FEES:,} typical fees).")
        if any(v is not None for v in (m_rate, m_band, m_sc, m_gr)) or m_outcode:
            lines.append(
                "(monthly ownership model is residential-only — semi-commercial "
                "lending terms differ too much to assume; not computed here.)")
        lines.append(
            "Notes: non-residential SDLT rates (0% to £150k · 2% £150k–£250k · "
            "5% above). First-time-buyer relief, the additional-dwelling (+5%) "
            "and the non-resident (+2%) surcharges do NOT apply on this track — "
            "they are residential-only. Standard residential mortgages generally "
            "do not apply to mixed-use/commercial property (semi-commercial "
            "lending instead). Fees are typical estimates; not financial or tax "
            "advice — confirm treatment with a conveyancer.")
        return [types.TextContent(type="text", text="\n".join(lines))]

    if total_funds:
        tracks = [(bool(ftb), "")] if ftb is not None else [(True, " (first-time buyer)"), (False, " (standard)")]
        lines.append(
            f"Total funds £{int(total_funds):,} is NOT the same as the price you "
            "can pay — SDLT, fees, renovation and a cash buffer come out of the "
            "same pot. Search with the affordable price below, not the raw budget:")
        for is_ftb, suffix in tracks:
            afford = max_affordable(int(total_funds), is_ftb, renovation, buffer)
            tax = sdlt(afford, is_ftb)
            lines.append(f"  Max affordable price{suffix}: ≈ £{afford:,}")
            lines.append(
                f"    breakdown: price £{afford:,} + SDLT £{tax:,} + fees ~£{DEFAULT_FEES:,}"
                + (f" + renovation £{renovation:,}" if renovation else "")
                + f" + buffer £{buffer:,} ≤ £{int(total_funds):,}")

    if price:
        lines.append(f"Costing £{int(price):,}:")
        if ftb is None:
            lines += _cost_lines(int(price), True, "First-time buyer")
            lines += _cost_lines(int(price), False, "Standard")
            if int(price) > FTB_CEILING:
                lines.append(f"  (above £{FTB_CEILING:,} the first-time-buyer relief is "
                             "void — both tracks identical)")
        else:
            lines += _cost_lines(int(price), bool(ftb),
                                 "First-time buyer" if ftb else "Standard")
        lines += _monthly_block(int(price), m_rate, m_term, m_dep,
                                m_band, m_outcode, m_sc, m_gr)

    lines.append(
        "Notes: England main residence only. Additional-property (+5%) and "
        "non-resident (+2%) surcharges NOT included — ask if they might apply. "
        "Fees are typical estimates (conveyancing, L3 survey, mortgage, moving); "
        "actual quotes vary. Not financial advice.")
    return [types.TextContent(type="text", text="\n".join(lines))]


async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="compute-buying-costs",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
