"""get_price_trend geographic granularity: outcodes and sectors must be askable too.

Background (found in a 2026-08-12 user simulation): asked "house-price trend
across all London areas — up or down over the last year?" and "price trend
around Ealing Broadway", the agent queried get_price_trend('W5') / ('N1') /
('EC1') with outcodes, and the tool answered every one with "W5 has no price
analysis yet" — so it told the user "the W5 outcode hasn't had a price-trend
analysis yet" and, to produce numbers anyway, **invented some unit postcodes**
(EC1V 0HB / NW3 1QG / W5 5SA / E14 5AB) and queried those; four out of four
had no data either.

Meanwhile the geo_cagr table was built for exactly this: 323 outcodes, 1,173
sectors; the W5 row is 3y +3.96%/yr (1,214 repeat-sale pairs), 10y +5.0%,
direction=up, 11,122 pairs in total. The data was always there; the tool just
didn't accept questions at that granularity.

Rules:
  'N1 9DT' (unit)  → unit path as before, output unchanged;
  'N1 9' (sector)  → geo_cagr sector level;
  'N1' / 'n1'      → geo_cagr outcode level, and must not collide with N19;
  not found        → honestly say so, as before; never invent.

Also: on the unit path, 5% of rows' median/average were actually computed at
sector level (data_json.area_scope='sector'); the output must say so, or the
agent will tell a buyer "this postcode's median price" as a unit-level fact.

(Extract note: the original file also has a TestAgainstLiveDb class that runs
against the production database; it is not included here.)
"""
import os
import sqlite3
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from get_price_trend import (  # noqa: E402
    _LONDON_ALIASES,
    area_trend_text,
    classify_geo,
)


class TestClassifyGeo:
    def test_unit_postcode(self):
        assert classify_geo("N1 9DT") == ("unit", "N1 9DT")
        assert classify_geo("n19dt") == ("unit", "N1 9DT")
        assert classify_geo("EC1V 0HB") == ("unit", "EC1V 0HB")

    def test_outcode(self):
        assert classify_geo("N1") == ("outcode", "N1")
        assert classify_geo(" w5 ") == ("outcode", "W5")
        assert classify_geo("SW11") == ("outcode", "SW11")
        assert classify_geo("EC1V") == ("outcode", "EC1V")

    def test_outcode_n19_is_not_sector_n1_9(self):
        # 'N19' is Archway's outcode, not sector 9 of N1.
        assert classify_geo("N19") == ("outcode", "N19")
        assert classify_geo("N1 9") == ("sector", "N1 9")

    def test_sector(self):
        assert classify_geo("SW11 3") == ("sector", "SW11 3")
        assert classify_geo("sw113") == ("sector", "SW11 3")


def _mem_db():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute(
        "CREATE TABLE geo_cagr (level TEXT, geo TEXT, cagr_3y REAL, cagr_3y_count INT, "
        "cagr_5y REAL, cagr_5y_count INT, cagr_10y REAL, cagr_10y_count INT, "
        "direction TEXT, sample_count INT, built_at TEXT, PRIMARY KEY(level, geo))"
    )
    c.execute("INSERT INTO geo_cagr VALUES "
              "('outcode','W5',3.96,1214,4.28,2101,5.0,3650,'up',11122,'2026-08-09T03:30:13Z')")
    c.execute("INSERT INTO geo_cagr VALUES "
              "('sector','W5 2',2.1,31,2.4,55,3.0,96,'stable',402,'2026-08-09T03:30:13Z')")
    return c


class TestAreaTrendText:
    def test_outcode_reports_rate_and_counts(self):
        txt = area_trend_text(_mem_db(), "outcode", "W5")
        assert txt is not None
        assert "W5" in txt
        assert "outcode" in txt.lower()
        assert "+3.96" in txt or "+4.0" in txt      # 3y CAGR
        assert "1,214" in txt or "1214" in txt      # repeat-sale pairs in that window
        assert "11,122" in txt or "11122" in txt    # total sample
        assert "up" in txt.lower()

    def test_area_answer_never_claims_a_price_level(self):
        # Price levels have no reliable caliber at area level (each postcode's
        # median uses a different window): give only the rate, and say clearly
        # "for a price level, give a full postcode".
        txt = area_trend_text(_mem_db(), "outcode", "W5")
        assert "median" not in txt.lower() and "£" not in txt
        assert "full postcode" in txt.lower()

    def test_sector_level(self):
        txt = area_trend_text(_mem_db(), "sector", "W5 2")
        assert txt is not None and "sector" in txt.lower() and "W5 2" in txt

    def test_missing_geo_returns_none(self):
        assert area_trend_text(_mem_db(), "outcode", "ZZ9") is None


class TestLondonWide:
    """Whole-city granularity (2026-08-13).

    Background: asked "what's the overall London house-price trend in 2026? is
    now a good time to buy?", the agent ran two WebSearches and **zero local
    tools**, reasoning "our tools query by postcode and no area was named" —
    true: geo_cagr then had only sector/outcode levels. So a question about
    **our own market** was answered entirely from forecasts on the web.

    With the london level added, two rules hold:
      1. State the coverage (19 postal areas incl. the KT/DA/EN/RM commuter
         belt — not the administrative boundary), or "London +4.11%" gets
         repeated as more precise than it is;
      2. Give the **dispersion**. One city-wide average is exactly why macro
         forecasts are useless to a specific buyer; outcodes' 3-year rates run
         from -4.48% to +7.54%, and that spread is something we can say and a
         forecast can't.
    """

    def _london_db(self):
        c = _mem_db()
        c.execute("INSERT INTO geo_cagr VALUES "
                  "('london','London',4.11,219228,4.49,399255,5.26,729137,'up',"
                  "2050687,'2026-08-13T16:00:00Z')")
        return c

    def test_reports_rate_and_pair_counts(self):
        txt = area_trend_text(self._london_db(), "london", "London")
        assert txt is not None
        assert "+4.11" in txt and "219,228" in txt
        assert "2,050,687" in txt

    def test_states_its_coverage_is_not_the_administrative_boundary(self):
        txt = area_trend_text(self._london_db(), "london", "London")
        assert "coverage:" in txt
        assert "not the administrative boundary" in txt

    def test_reports_dispersion_across_outcodes(self):
        txt = area_trend_text(self._london_db(), "london", "London")
        assert "dispersion:" in txt
        # The fixture has a single outcode (W5), so the spread degenerates to it
        # — assert the line exists with both bounds, not specific numbers.
        assert "%/yr to " in txt

    def test_still_refuses_to_quote_a_price_level(self):
        txt = area_trend_text(self._london_db(), "london", "London")
        assert "median" not in txt.lower() and "£" not in txt

    def test_dispersion_line_survives_a_db_without_outcode_rows(self):
        c = sqlite3.connect(":memory:")
        c.row_factory = sqlite3.Row
        c.execute(
            "CREATE TABLE geo_cagr (level TEXT, geo TEXT, cagr_3y REAL, cagr_3y_count INT, "
            "cagr_5y REAL, cagr_5y_count INT, cagr_10y REAL, cagr_10y_count INT, "
            "direction TEXT, sample_count INT, built_at TEXT, PRIMARY KEY(level, geo))"
        )
        c.execute("INSERT INTO geo_cagr VALUES "
                  "('london','London',4.11,219228,4.49,399255,5.26,729137,'up',"
                  "2050687,'2026-08-13T16:00:00Z')")
        txt = area_trend_text(c, "london", "London")
        assert txt is not None and "+4.11" in txt      # must not crash without the dispersion

    def test_aliases_cover_both_languages(self):
        for a in ("london", "greater london", "全伦敦", "伦敦整体"):
            assert a in _LONDON_ALIASES
        # specific place names must not be swallowed into the city-wide caliber
        assert "london bridge" not in _LONDON_ALIASES
        assert "east london" not in _LONDON_ALIASES
