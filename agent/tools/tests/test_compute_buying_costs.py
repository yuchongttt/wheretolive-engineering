"""compute_buying_costs: total funds ≠ house price (fix for the agent-benchmark round-02 R2-1 loss).

R2-1 comparison: the scenario stated a total budget; both of our runs used it
as the search price cap. The reference agent first deducted stamp duty +
transaction fees + renovation + a cash buffer and arrived at a lower real
viewing price — the single most valuable insight of the round. Tax arithmetic can't rely on the LLM doing it live (the reference agent
getting it right this time was ability plus luck, not dependable), so it is
pinned as a deterministic tool.

Rate anchors (England, in force 2026, gov.uk; the reference agent's R2-1
figures were hand-checked item by item):
  standard: 0-125k 0% / 125-250k 2% / 250k-925k 5% / 925k-1.5M 10% / above 12%
  first-time-buyer relief: 0-300k 0% / 300-500k 5%; **price > £500k voids the relief entirely**
  verified: £594k standard = £19,700; £500k FTB = £10,000; £500k standard = £15,000
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import compute_buying_costs as cbc  # noqa: E402


def _call(**args):
    out = asyncio.run(cbc.handle_call_tool("compute_buying_costs", args))
    return out[0].text


# ---------------------------------------------------------------------------
# SDLT pure functions
# ---------------------------------------------------------------------------

def test_sdlt_standard_rates():
    assert cbc.sdlt(594000, first_time_buyer=False) == 19700
    assert cbc.sdlt(500000, first_time_buyer=False) == 15000
    assert cbc.sdlt(120000, first_time_buyer=False) == 0


def test_sdlt_ftb_relief_and_cliff():
    assert cbc.sdlt(300000, first_time_buyer=True) == 0
    assert cbc.sdlt(500000, first_time_buyer=True) == 10000
    # above £500k the relief is void entirely and standard rates apply — the cliff must show
    assert cbc.sdlt(501000, first_time_buyer=True) == cbc.sdlt(501000, first_time_buyer=False)


# ---------------------------------------------------------------------------
# Solving for the affordable price
# ---------------------------------------------------------------------------

def test_max_affordable_reproduces_r21_worked_example():
    """Worked example (synthetic figures): £700k total funds, renovation £60k,
    buffer £20k, fees £6k → viewing price ≈ £594k."""
    price = cbc.max_affordable(700000, first_time_buyer=False,
                               renovation=60000, buffer=20000, fees=6000)
    assert 590000 <= price <= 596000


def test_tool_output_shows_breakdown_not_just_number():
    text = _call(total_funds=700000, first_time_buyer=False,
                 renovation=60000, buffer=20000)
    assert "594" in text.replace(",", "")   # order of the affordable price
    assert "SDLT" in text or "stamp duty" in text.lower()
    # must say "total funds ≠ price" and tell the model to search with the derived cap
    assert "not the same" in text.lower() or "≠" in text


def test_ftb_unknown_shows_both_tracks():
    """When first-time-buyer status is unknown, show both tracks — never choose for the user."""
    text = _call(price=450000)
    assert "first-time" in text.lower()
    assert "standard" in text.lower()


def test_no_input_is_instructive():
    text = _call()
    assert "total_funds" in text and "price" in text


# ---------------------------------------------------------------------------
# Non-residential / mixed-use SDLT (R2-5 Fix Q): 0% ≤£150k · 2% £150k–£250k · 5% >£250k.
# No FTB relief, no additional-dwelling (+5%) or non-resident (+2%) surcharge —
# all three are residential-only. Anchor: the reference agent's R2-5 example
# £850k → £32,000 (checked band by band by hand).
# ---------------------------------------------------------------------------

def test_sdlt_nonres_rates():
    assert cbc.sdlt_nonres(850000) == 32000     # 0+2k+30k, the R2-5 anchor
    assert cbc.sdlt_nonres(150000) == 0
    assert cbc.sdlt_nonres(250000) == 2000
    assert cbc.sdlt_nonres(975000) == 38250


def test_tool_mixed_use_costs_out_nonres_track():
    out = _call(price=850000, property_class="mixed_use_or_commercial")
    assert "32,000" in out
    # no FTB track on the non-residential track (the relief is residential-only)
    assert "First-time buyer" not in out
    # must spell out the three non-applicable items, so the model doesn't add surcharges itself
    assert "additional-dwelling" in out or "additional-property" in out


def test_tool_residential_default_unchanged():
    out = _call(price=850000)
    assert "First-time buyer" in out and "Standard" in out


# ---------------------------------------------------------------------------
# Monthly ownership model (R2-7 Fix Y): mortgage P&I + council tax £/month (Fix V
# rates) + SC/GR. The reference answer's "total monthly outgoings" table was
# its most decision-friendly deliverable; we held every input and only lacked
# the payment arithmetic and assembly. Anchor: £550k/10%/5.33%/30yr → £2,758
# (the reference agent's example, independently re-checked). Assumptions must
# be labelled — the default rate is "assumed", not a fact.
# ---------------------------------------------------------------------------

def test_monthly_pi_anchor():
    assert abs(cbc.monthly_pi(550000, 10, 5.33, 30) - 2758) <= 5
    assert abs(cbc.monthly_pi(635000, 10, 5.33, 30) - 3184) <= 5
    assert abs(cbc.monthly_pi(550000, 20, 5.33, 30) - 2452) <= 5


def test_price_costing_includes_monthly_block_with_labels(tmp_path, monkeypatch):
    import council_tax as ct
    area = tmp_path / "area_intel.db"
    import sqlite3
    conn = sqlite3.connect(area)
    conn.execute("""CREATE TABLE council_tax_rates (
        year TEXT, borough TEXT, ons_code TEXT,
        band_a REAL, band_b REAL, band_c REAL, band_d REAL,
        band_e REAL, band_f REAL, band_g REAL, band_h REAL, fetched_at TEXT)""")
    conn.execute("INSERT INTO council_tax_rates VALUES ('2026-27','Ealing','E09000009',1426,1663,1901,2139,2614,3088,3565,4278,'t')")
    conn.commit(); conn.close()
    ev = tmp_path / "evaluations.db"
    conn = sqlite3.connect(ev)
    conn.execute("CREATE TABLE outcode_borough (outcode TEXT PRIMARY KEY, borough TEXT, share REAL, source TEXT)")
    conn.execute("INSERT INTO outcode_borough VALUES ('W5','Ealing',0.95,NULL)")
    conn.commit(); conn.close()
    monkeypatch.setattr(ct, "AREA_DB", area)
    monkeypatch.setattr(ct, "EVAL_DB", ev)

    out = _call(price=550000, mortgage_rate=5.33, council_tax_band="D",
                outcode="W5", service_charge_pa=150)
    pi_line = next(ln for ln in out.splitlines() if "P&I" in ln)
    assert "2,758" in pi_line and "30yr term" in pi_line
    assert "2,452" in pi_line                  # the 20% variant (no explicit deposit given)
    # rate-source label (review B#5/B#6): not ASSUMED, but must say caller-supplied
    assert "caller-supplied" in pi_line and "ASSUMED" not in pi_line
    ct_line = next(ln for ln in out.splitlines() if "council tax" in ln)
    assert "178" in ct_line and "Ealing" in ct_line
    # default-rate path: the ASSUMED suffix must be there (deleting the labelling logic turns this red)
    out2 = _call(price=550000, council_tax_band="D", outcode="W5")
    pi2 = next(ln for ln in out2.splitlines() if "P&I" in ln)
    assert "ASSUMED" in pi2 and "confirm a real quote" in pi2
    # lenient shapes: a full-postcode outcode + "Band D" must still resolve council tax
    out3 = _call(price=550000, council_tax_band="Band D", outcode="W5 2AB")
    assert any("178" in ln and "council tax" in ln for ln in out3.splitlines())
    # when unresolvable, say explicitly it's not included, and list the exclusion on the total line
    out4 = _call(price=550000, council_tax_band="D", outcode="ZZ99")
    assert "NOT included" in out4
    assert any("excl. council tax" in ln for ln in out4.splitlines())
    # cash purchase (100% deposit): no payment line, no ASSUMED-quote instruction
    out5 = _call(price=550000, deposit_pct=100)
    assert "cash purchase" in out5 and "P&I" not in out5


def test_monthly_absent_without_price():
    out = _call(total_funds=700000)
    assert "P&I" not in out and "/mo" not in out


# --- Rate vintage (2026-08-18) ---
# SDLT rates are LAW: when a Budget changes them we silently compute wrong
# figures — on a £700k home one band change is thousands of pounds, and users
# pay according to that number. Before this change the vintage lived only in a
# Python comment (invisible to the model) and the user-facing output was a bare
# "SDLT £X" with no "as of". For anything that "freezes an answer into code",
# the minimum bar is: **the output carries its own vintage**.

def test_rates_declare_their_vintage_in_the_user_facing_output():
    lines = "\n".join(cbc._cost_lines(700_000, False, "Standard"))
    assert cbc.RATES_AS_OF in lines, "the output must carry the rates' effective date, not just a comment"


def test_vintage_is_a_real_date():
    import datetime
    datetime.date.fromisoformat(cbc.RATES_AS_OF)


def test_going_stale_escalates_the_wording():
    """A Budget can change the rates at any time. Past the freshness window the wording must escalate to "verify"."""
    fresh = cbc.rates_vintage_note(today="2026-08-18")          # with the real as-of
    stale = cbc.rates_vintage_note(today="2028-08-18")          # two years later
    assert "verify" not in fresh.lower(), fresh
    assert "verify" in stale.lower(), stale
    assert cbc.RATES_AS_OF in fresh and cbc.RATES_AS_OF in stale
