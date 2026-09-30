"""Read layer for data/area_intel.db (offline area intelligence).

Tables (built by scripts/build_geo_lookup.py, ingest_pipr.py,
ingest_imd_census.py, ingest_crime.py):
  postcode_geo      postcode -> LSOA21/OA21/borough (ONSPD, space-insensitive key)
  rent_benchmarks   ONS PIPR borough x month x size rents (stock, not asking)
  imd2025           IoD2025 deciles per domain + London percentile
  census_lsoa       Census 2021 aggregated counts per LSOA
  crime_events / crime_lsoa_month / crime_lsoa_12m
                    police.uk street crime, 36 trailing months (London forces)

All functions fail soft (return None / {}) when the DB or a table is missing,
so callers on /check and chat never hard-fail because of this layer.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_ROOT, "data", "area_intel.db")

PIPR_SIZE_BY_BEDS = {1: "bed1", 2: "bed2", 3: "bed3"}  # 4+ handled below


# Process-wide cached connection. Per-call open/close of a mode=ro handle on a
# WAL database failed intermittently (~35% fallback rate) during the 47k-postcode
# scoring sweep — the readonly open races the -shm/-wal lifecycle. One long-lived
# ordinary connection (reads only by convention) is both reliable and ~free.
_CONN: Optional[sqlite3.Connection] = None


def _conn() -> Optional[sqlite3.Connection]:
    global _CONN
    if _CONN is not None:
        return _CONN
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA query_only=ON")
    conn.row_factory = sqlite3.Row
    _CONN = conn
    return _CONN


def _pc_key(postcode: str) -> str:
    return (postcode or "").upper().replace(" ", "").strip()


def get_geo(postcode: str) -> Optional[dict]:
    """postcode -> {pcds, lat, lon, lad25cd, lsoa21cd, msoa21cd, oa21cd} or None."""
    conn = _conn()
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT pcds, lat, lon, lad25cd, lsoa21cd, msoa21cd, oa21cd, doterm "
            "FROM postcode_geo WHERE pcd_key = ?",
            (_pc_key(postcode),),
        ).fetchone()
        return dict(row) if row else None
    except sqlite3.Error:
        return None


def get_official_rent(postcode: str, beds: Optional[int]) -> Optional[dict]:
    """ONS PIPR benchmark for the listing's borough (latest month).

    Returns {borough, month, size_label, rent, yoy, rent_all_sizes, london_rent}
    or None. `rent` is the borough average for the matching size band; PIPR
    measures the whole private-rental stock (sitting tenants included), so it
    reads below new-let asking rents — label as "official benchmark", not comp.
    """
    conn = _conn()
    if conn is None:
        return None
    try:
        geo = None
        row = conn.execute(
            "SELECT pcds, lad25cd FROM postcode_geo WHERE pcd_key = ?",
            (_pc_key(postcode),),
        ).fetchone()
        if row:
            geo = row["lad25cd"]
        if not geo:
            return None
        latest = conn.execute(
            "SELECT MAX(month) FROM rent_benchmarks"
        ).fetchone()[0]
        if not latest:
            return None
        size_key = "all"
        size_label = "all sizes"
        if beds is not None:
            if beds >= 4:
                size_key, size_label = "bed4plus", "4+ bed"
            elif beds >= 1:
                size_key = PIPR_SIZE_BY_BEDS[beds]
                size_label = f"{beds} bed"

        def _fetch(code: str, sk: str):
            return conn.execute(
                "SELECT area_name, rent, yoy FROM rent_benchmarks "
                "WHERE area_code = ? AND month = ? AND size_key = ?",
                (code, latest, sk),
            ).fetchone()

        r = _fetch(geo, size_key) or _fetch(geo, "all")
        if r is None or r["rent"] is None:
            return None
        r_all = _fetch(geo, "all")
        r_london = _fetch("E12000007", size_key)
        return {
            "borough": r["area_name"],
            "borough_code": geo,
            "month": latest,
            "size_label": size_label,
            "rent": round(r["rent"]),
            "yoy_pct": round(r["yoy"], 1) if r["yoy"] is not None else None,
            "rent_all_sizes": round(r_all["rent"]) if r_all and r_all["rent"] else None,
            "london_rent": round(r_london["rent"]) if r_london and r_london["rent"] else None,
            "source": "ONS Price Index of Private Rents",
        }
    except sqlite3.Error:
        return None


def get_crime_profile(postcode: str) -> Optional[dict]:
    """Neighbourhood (LSOA) crime picture from local police.uk data.

    Returns None when the postcode isn't in London or data is missing.
    trend_pct is 12m vs prior 12m (negative = falling). Percentile is
    London-relative on residential-relevant rate (100 = most crime).
    """
    conn = _conn()
    if conn is None:
        return None
    try:
        row = conn.execute(
            """SELECT g.lsoa21cd, s.asof_month, s.total_12m, s.prior_12m,
                      s.resi_12m, s.pop, s.per_1000, s.resi_per_1000,
                      s.resi_rate_pctile, s.categories_json
               FROM postcode_geo g
               JOIN crime_lsoa_12m s ON s.lsoa21cd = g.lsoa21cd
               WHERE g.pcd_key = ?""",
            (_pc_key(postcode),),
        ).fetchone()
        if row is None:
            return None
        cats = json.loads(row["categories_json"] or "{}")
        top = sorted(cats.items(), key=lambda kv: -kv[1])[:5]
        trend_pct = None
        if row["prior_12m"]:
            trend_pct = round(
                100.0 * (row["total_12m"] - row["prior_12m"]) / row["prior_12m"], 1
            )
        return {
            "lsoa21cd": row["lsoa21cd"],
            "asof_month": row["asof_month"],
            "total_12m": row["total_12m"],
            "prior_12m": row["prior_12m"],
            "trend_pct": trend_pct,
            "pop": row["pop"],
            "per_1000": row["per_1000"],
            "resi_per_1000": row["resi_per_1000"],
            "resi_rate_pctile": row["resi_rate_pctile"],
            "top_categories": [{"category": c, "count": n} for c, n in top],
            "source": "police.uk street-level crime (12 months)",
        }
    except sqlite3.Error:
        return None


def get_demographics(postcode: str) -> Optional[dict]:
    """Census 2021 + IMD 2025 neighbourhood profile (LSOA level)."""
    conn = _conn()
    if conn is None:
        return None
    try:
        row = conn.execute(
            """SELECT g.lsoa21cd, i.lad_name, i.imd_decile, i.london_pctile,
                      i.income_decile, i.employment_decile, i.education_decile,
                      i.health_decile, i.crime_decile, i.barriers_decile,
                      i.livenv_decile, c.*
               FROM postcode_geo g
               JOIN imd2025 i ON i.lsoa21cd = g.lsoa21cd
               LEFT JOIN census_lsoa c ON c.lsoa21cd = g.lsoa21cd
               WHERE g.pcd_key = ?""",
            (_pc_key(postcode),),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)

        def share(part, whole):
            p, w = d.get(part), d.get(whole)
            return round(100.0 * p / w, 1) if p is not None and w else None

        return {
            "lsoa21cd": d["lsoa21cd"],
            "borough": d["lad_name"],
            "imd": {
                "decile": d["imd_decile"],
                "london_pctile": d["london_pctile"],
                "domains": {
                    "income": d["income_decile"],
                    "employment": d["employment_decile"],
                    "education": d["education_decile"],
                    "health": d["health_decile"],
                    "crime": d["crime_decile"],
                    "housing_barriers": d["barriers_decile"],
                    "living_environment": d["livenv_decile"],
                },
                "note": "deciles are England-wide, 1 = most deprived 10%",
            },
            "population": d.get("pop_total"),
            "households": d.get("hh_total"),
            "tenure_pct": {
                "owned": share_sum(d, ["own_outright", "own_mortgage"], "hh_total"),
                "private_rented": share("private_rented", "hh_total"),
                "social_rented": share("social_rented", "hh_total"),
                "shared_ownership": share("shared_ownership", "hh_total"),
            },
            "age_pct": {
                "0-14": share("age_0_14", "age_total"),
                "15-24": share("age_15_24", "age_total"),
                "25-34": share("age_25_34", "age_total"),
                "35-49": share("age_35_49", "age_total"),
                "50-64": share("age_50_64", "age_total"),
                "65+": share("age_65plus", "age_total"),
            },
            "household_pct": {
                "one_person": share("hc_one_person", "hc_total"),
                "couple_no_children": share("hc_couple_no_children", "hc_total"),
                "with_dependent_children": share("hc_dep_children", "hc_total"),
            },
            "occupation_pct": {
                "higher_professional": share("nssec_higher_prof", "nssec_total"),
                "lower_professional": share("nssec_lower_prof", "nssec_total"),
                "intermediate": share("nssec_intermediate", "nssec_total"),
                "routine": share("nssec_routine", "nssec_total"),
                "students": share("nssec_students", "nssec_total"),
            },
            "source": "Census 2021 (LSOA) + English Indices of Deprivation 2025",
        }
    except sqlite3.Error:
        return None


def share_sum(d: dict, parts: list, whole: str):
    vals = [d.get(p) for p in parts]
    w = d.get(whole)
    if any(v is None for v in vals) or not w:
        return None
    return round(100.0 * sum(vals) / w, 1)
