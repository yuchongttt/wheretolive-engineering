"""
Environment Data Evaluators
Queries flood risk, air quality, and parks/green space data

Data sources:
- Flood risk: NaFRA2 (National Flood Risk Assessment 2) WMS - Environment Agency
  - Rivers and Sea: flood risk from rivers and the sea
  - Surface Water: surface water flood risk
- Air quality:
  - LAQN (London Air Quality Network) - free, more precise for London
  - UK-AIR (AURN network) - free, UK-wide coverage
- Parks/green space: OpenStreetMap Overpass API (free, global)
"""

import math
import sqlite3
import json
import requests
from typing import Optional, TypedDict, List
from datetime import datetime

from apis.api_usage import record_api_usage


# =============================================================================
# Type Definitions
# =============================================================================

class FloodRiskResult(TypedDict):
    risk_level: str  # 'very_low', 'low', 'medium', 'high'
    risk_level_zh: str
    rivers_sea_risk: Optional[str]  # rivers/sea risk level
    surface_water_risk: Optional[str]  # surface water risk level
    score: int  # 0-100 score
    data_source: str  # 'nafra2'


class AirQualitySiteInfo(TypedDict):
    name: str
    distance_km: float


class AnnualStats(TypedDict):
    no2: Optional[float]
    pm25: Optional[float]
    pm10: Optional[float]


class AirQualityResult(TypedDict):
    has_data: bool
    score: int  # 0-100
    quality_level: str  # 'good', 'moderate', 'poor', 'very_poor'
    quality_level_zh: str
    nearest_site: Optional[AirQualitySiteInfo]
    annual_stats: Optional[AnnualStats]
    data_source: str  # 'laqn' (London), 'uk_air' (UK-wide), 'none'


class ParkInfo(TypedDict):
    name: str
    distance_m: int


class ParksResult(TypedDict):
    parks_count: int
    parks_within_500m: int
    parks_within_1km: int
    green_score: int  # 0-100
    green_level: str  # 'excellent', 'good', 'moderate', 'poor'
    green_level_zh: str
    nearest_parks: List[ParkInfo]


# =============================================================================
# Database Helpers
# =============================================================================

def _get_db_path() -> str:
    """Get the database path"""
    import os
    return os.path.join(os.path.dirname(__file__), "data", "evaluations.db")


def _ensure_tables():
    """Ensure the cache tables exist"""
    conn = sqlite3.connect(_get_db_path())
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")

    # Flood cache
    conn.execute("""
        CREATE TABLE IF NOT EXISTS flood_cache (
            lat REAL,
            lng REAL,
            data_json TEXT,
            cached_at TEXT,
            PRIMARY KEY (lat, lng)
        )
    """)

    # Air quality cache
    conn.execute("""
        CREATE TABLE IF NOT EXISTS air_quality_cache (
            lat REAL,
            lng REAL,
            data_json TEXT,
            cached_at TEXT,
            PRIMARY KEY (lat, lng)
        )
    """)

    # Parks cache
    conn.execute("""
        CREATE TABLE IF NOT EXISTS parks_cache (
            lat REAL,
            lng REAL,
            data_json TEXT,
            cached_at TEXT,
            PRIMARY KEY (lat, lng)
        )
    """)

    conn.commit()
    conn.close()


def _get_cached(table: str, lat: float, lng: float, ttl_days: int) -> Optional[dict]:
    """Get data from the cache"""
    try:
        conn = sqlite3.connect(_get_db_path())
        conn.execute("PRAGMA busy_timeout = 5000")
        cursor = conn.execute(
            f"""
            SELECT data_json, cached_at
            FROM {table}
            WHERE ABS(lat - ?) < 0.0005 AND ABS(lng - ?) < 0.0005
            ORDER BY ABS(lat - ?) + ABS(lng - ?)
            LIMIT 1
            """,
            (lat, lng, lat, lng)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            data_json, cached_at = row
            # Check the TTL
            cached_time = datetime.fromisoformat(cached_at.replace('Z', '+00:00'))
            if (datetime.now() - cached_time.replace(tzinfo=None)).days < ttl_days:
                return json.loads(data_json)
    except Exception:
        pass
    return None


def _save_cache(table: str, lat: float, lng: float, data: dict):
    """Save data to the cache"""
    try:
        conn = sqlite3.connect(_get_db_path())
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {table}
            (lat, lng, data_json, cached_at)
            VALUES (?, ?, ?, datetime('now'))
            """,
            (round(lat, 4), round(lng, 4), json.dumps(data, ensure_ascii=False))
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# =============================================================================
# Flood Risk Evaluator
# =============================================================================

# NaFRA2 risk level definitions
# High: ≥3.3% annual probability (1 in 30)
# Medium: 1-3.3% annual probability (1 in 100)
# Low: 0.1-1% annual probability (1 in 1000)
# Very Low: <0.1% annual probability (1 in 1000+)
FLOOD_RISK_LEVELS = {
    'high': ('high', '高风险', 20),
    'medium': ('medium', '中等风险', 50),
    'low': ('low', '低风险', 75),
    'very_low': ('very_low', '极低风险', 100),
}

# NaFRA2 WMS endpoints
NAFRA2_WMS = {
    'rivers_sea': 'https://environment.data.gov.uk/spatialdata/nafra2-risk-of-flooding-from-rivers-and-sea/wms',
    'surface_water': 'https://environment.data.gov.uk/spatialdata/nafra2-risk-of-flooding-from-surface-water/wms',
}

NAFRA2_LAYERS = {
    'rivers_sea': 'rofrs_4band',
    'surface_water': 'rofsw',
}


def _haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Compute the great-circle distance between two points (km)"""
    R = 6371  # Earth radius in km

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lng = math.radians(lng2 - lng1)

    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


class NaFRA2Unavailable(Exception):
    """NaFRA2 WMS gave no trustworthy answer (non-200 / timeout / 200 but the body is neither a risk band nor
    "no features were found"). Callers must never treat it as "very low"."""


# Real responses (captured 2026-09-27): not in any risk band → the body is exactly this sentence; in a risk band →
# "Results for FeatureType …\nrisk_band = High\n…".
_NO_FEATURES = "no features were found"


def _query_nafra2_wms(lat: float, lng: float, source: str, timeout: float = 10.0) -> Optional[str]:
    """
    Query NaFRA2 WMS for the flood risk level

    Args:
        lat: latitude (WGS84)
        lng: longitude (WGS84)
        source: 'rivers_sea' or 'surface_water'
        timeout: request timeout

    Returns:
        Risk level ('High', 'Medium', 'Low', 'Very low'); None = EA explicitly answered
        "no features were found" (the point is not in any risk band).

    Raises:
        NaFRA2Unavailable: neither of the two answers above was received in two attempts. Before 2026-09-27
        this case also returned None and was cached as "very low" — from 07-22 EA rejected a large share of our
        requests, and cache rows with "both sources empty" jumped from ~60% to ~93%.
    """
    import re

    wms_url = NAFRA2_WMS.get(source)
    layer = NAFRA2_LAYERS.get(source)

    if not wms_url or not layer:
        return None

    # Use WGS84 coordinates and widen the search area to ensure a hit
    delta = 0.001  # about 100m
    bbox = f"{lat - delta},{lng - delta},{lat + delta},{lng + delta}"

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetFeatureInfo",
        "LAYERS": layer,
        "QUERY_LAYERS": layer,
        "INFO_FORMAT": "text/plain",
        "CRS": "EPSG:4326",
        "BBOX": bbox,
        "WIDTH": "100",
        "HEIGHT": "100",
        "I": "50",
        "J": "50",
    }

    import time as _time
    last = "no attempt"
    for attempt in range(2):
        try:
            resp = requests.get(wms_url, params=params, timeout=timeout)
            if resp.status_code == 200:
                # Parse the response: "risk_band = High" or "risk_band = Very low"
                match = re.search(r'risk_band\s*=\s*([^\n]+)', resp.text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
                if resp.text.strip() == _NO_FEATURES:
                    return None  # EA explicitly says: not in any risk band
                last = f"200 with unexpected body {resp.text[:60]!r}"
            else:
                last = f"HTTP {resp.status_code}"
        except Exception as e:  # noqa: BLE001 — a network error is also "no answer"
            last = f"{type(e).__name__}: {e}"
        if attempt == 0:
            _time.sleep(1)

    raise NaFRA2Unavailable(f"{source}: {last}")


def _get_higher_risk(risk1: Optional[str], risk2: Optional[str]) -> str:
    """
    Compare two risk levels and return the higher one

    Risk order: High > Medium > Low > Very low
    """
    risk_order = {'high': 4, 'medium': 3, 'low': 2, 'very low': 1, 'very_low': 1}

    def normalize(r: Optional[str]) -> str:
        if not r:
            return 'very_low'
        return r.lower().replace(' ', '_')

    def get_order(r: str) -> int:
        return risk_order.get(r.lower().replace('_', ' '), 0)

    r1 = normalize(risk1)
    r2 = normalize(risk2)

    return r1 if get_order(r1) >= get_order(r2) else r2


def evaluate_flood_risk(lat: float, lng: float, timeout: float = 10.0, skip_cache: bool = False) -> Optional[FloodRiskResult]:
    """
    Evaluate the flood risk at a coordinate

    Uses the NaFRA2 (National Flood Risk Assessment 2) WMS API
    Data source: Environment Agency

    Risk level definitions:
    - High: ≥3.3% annual probability (about 1 in 30 years)
    - Medium: 1-3.3% annual probability (about 1 in 100 years)
    - Low: 0.1-1% annual probability (about 1 in 1000 years)
    - Very Low: <0.1% annual probability (>1 in 1000 years)

    Args:
        lat: latitude
        lng: longitude
        timeout: request timeout in seconds
        skip_cache: skip reading from the cache

    Returns:
        FloodRiskResult; if either EA source gives no trustworthy answer → None (not cached). Callers
        (evaluate_single / backfill_env) already treat None as "no data for this item", and the
        environment score redistributes the weights across the remaining three items.
    """
    _ensure_tables()

    # Check the cache (TTL: 90 days)
    if not skip_cache:
        cached = _get_cached("flood_cache", lat, lng, 90)
        if cached:
            return cached

    # Default result
    result: FloodRiskResult = {
        "risk_level": "very_low",
        "risk_level_zh": "极低风险",
        "rivers_sea_risk": None,
        "surface_water_risk": None,
        "score": 100,
        "data_source": "nafra2",
    }

    api_success = False
    try:
        # Query both data sources. If either source gives no answer → overall unknown: with half missing we cannot take "the higher of the two".
        try:
            rivers_sea = _query_nafra2_wms(lat, lng, "rivers_sea", timeout)
            surface_water = _query_nafra2_wms(lat, lng, "surface_water", timeout)
        except NaFRA2Unavailable:
            return None
        record_api_usage("flood_risk_api")
        api_success = True

        result["rivers_sea_risk"] = rivers_sea
        result["surface_water_risk"] = surface_water

        # Take the higher risk level
        combined_risk = _get_higher_risk(rivers_sea, surface_water)

        # Normalise the risk level
        risk_map = {
            'high': ('high', '高风险', 20),
            'medium': ('medium', '中等风险', 50),
            'low': ('low', '低风险', 75),
            'very_low': ('very_low', '极低风险', 100),
        }

        level, level_zh, score = risk_map.get(combined_risk, ('very_low', '极低风险', 100))
        result["risk_level"] = level
        result["risk_level_zh"] = level_zh
        result["score"] = score

    except Exception:
        # On API failure return the default result without caching it
        pass

    # Only save to the cache if the API call succeeded
    if api_success:
        _save_cache("flood_cache", lat, lng, result)

    return result


# =============================================================================
# Air Quality Evaluator
# =============================================================================

# WHO 2021 standards (annual mean, µg/m³)
WHO_STANDARDS = {
    "no2": 10,    # NO2 annual mean
    "pm25": 5,    # PM2.5 annual mean
    "pm10": 15,   # PM10 annual mean
}

AIR_QUALITY_LEVELS = {
    'good': ('good', '良好'),
    'moderate': ('moderate', '中等'),
    'poor': ('poor', '较差'),
    'very_poor': ('very_poor', '很差'),
}

# UK-AIR SOS API (UK-wide AURN network)
UK_AIR_API_BASE = "https://uk-air.defra.gov.uk/sos-ukair/api/v1"


def _query_uk_air(lat: float, lng: float, timeout: float = 10.0) -> Optional[dict]:
    """
    Query the UK-AIR SOS API (UK-wide AURN network)

    Note: the UK-AIR timeseries API returns real-time hourly data, not annual means.
    This is temporarily disabled until the correct annual-mean data API is found.

    Returns: None (temporarily disabled)
    """
    # The UK-AIR timeseries API returns real-time data rather than annual means; temporarily disabled
    # TODO: re-enable once a UK-AIR annual-mean data API is found
    return None


def _calculate_air_quality_score(site_data: dict) -> tuple:
    """
    Compute the air quality score and level from pollutant data

    Returns: (score, quality_level, quality_level_zh)
    """
    ratios = []
    if site_data.get("no2"):
        ratios.append(site_data["no2"] / WHO_STANDARDS["no2"])
    if site_data.get("pm25"):
        ratios.append(site_data["pm25"] / WHO_STANDARDS["pm25"])
    if site_data.get("pm10"):
        ratios.append(site_data["pm10"] / WHO_STANDARDS["pm10"])

    if not ratios:
        return 0, "moderate", "中等"

    avg_ratio = sum(ratios) / len(ratios)
    # Score: smooth sigmoid curve, ratio=1→83, ratio=2→53, ratio=3→22
    score = max(0, min(100, int(5 + 95 / (1 + math.exp(1.5 * (avg_ratio - 2.0))))))

    # Rating (same thresholds as parks/green space)
    if score >= 80:
        return score, "excellent", "优秀"
    elif score >= 60:
        return score, "good", "良好"
    elif score >= 40:
        return score, "moderate", "中等"
    else:
        return score, "poor", "较差"


def _query_laqn(lat: float, lng: float, timeout: float = 10.0) -> Optional[dict]:
    """
    Query the London Air Quality Network (LAQN) API

    Returns PM2.5, PM10 and NO2 annual means from the nearest site (data available for London only)
    """
    try:
        import time as _time
        url = "https://api.erg.ic.ac.uk/AirQuality/Annual/MonitoringObjective/GroupName=London/Year=2023/Json"
        data = None
        for attempt in range(2):
            try:
                resp = requests.get(url, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    record_api_usage("laqn")
                    break
                if attempt == 0:
                    _time.sleep(1)
            except Exception:
                if attempt == 0:
                    _time.sleep(1)
        if data is None:
            return None

        sites = data.get("SiteObjectives", {}).get("Site", [])
        if not sites:
            return None

        nearest_site = None
        min_dist = float('inf')
        site_data = {}

        for site in sites:
            site_lat = float(site.get("@Latitude", 0))
            site_lng = float(site.get("@Longitude", 0))

            if site_lat and site_lng:
                dist = _haversine_distance(lat, lng, site_lat, site_lng)

                # Only consider sites within 5km
                if dist < 5 and dist < min_dist:
                    min_dist = dist
                    nearest_site = {
                        "name": site.get("@SiteName", "Unknown"),
                        "distance_km": round(dist, 2),
                    }

                    objectives = site.get("Objective", [])
                    if not isinstance(objectives, list):
                        objectives = [objectives]

                    for obj in objectives:
                        species = obj.get("@SpeciesCode", "").upper()
                        value = obj.get("@Value")
                        obj_name = obj.get("@ObjectiveName", "").lower()

                        # Only take "annual mean" data; exclude other metrics
                        # The LAQN API returns several Objective types:
                        # - "XX ug/m3 as an annual mean" - annual mean concentration (what we need)
                        # - "XX ug/m3 as a 24 hour mean, not to be exceeded..." - 24-hour mean
                        # - "Capture Rate (%)" - data capture rate (~99%)

                        # Exclude Capture Rate
                        if "capture" in obj_name or "rate (" in obj_name:
                            continue

                        # Only keep records containing "annual mean" (excludes "24 hour mean", etc.)
                        if "annual mean" not in obj_name:
                            continue

                        if value and value != "n/m":
                            try:
                                val = float(value)
                                if species == "NO2":
                                    site_data["no2"] = val
                                elif species == "PM25":
                                    site_data["pm25"] = val
                                elif species == "PM10":
                                    site_data["pm10"] = val
                            except ValueError:
                                pass

        if not nearest_site or not site_data:
            return None

        return {
            "name": nearest_site["name"],
            "distance_km": nearest_site["distance_km"],
            "no2": site_data.get("no2"),
            "pm25": site_data.get("pm25"),
            "pm10": site_data.get("pm10"),
        }

    except Exception:
        return None


def evaluate_air_quality(lat: float, lng: float, timeout: float = 10.0, skip_cache: bool = False) -> AirQualityResult:
    """
    Evaluate the air quality at a coordinate

    Data source priority:
    1. LAQN (London Air Quality Network) - nearest site within 5 km
    2. UK-AIR (AURN network) - UK-wide; currently disabled (_query_uk_air returns None)

    Scoring logic (based on WHO 2021 guideline values), see _calculate_air_quality_score:
    - Compute each pollutant's ratio to the WHO guideline and average them
    - Score = 5 + 95 / (1 + exp(1.5 * (ratio - 2))), i.e. ratio 1 → ~83, 2 → ~53, 3 → ~22
    - ≥80 excellent, ≥60 good, ≥40 moderate, <40 poor

    Args:
        lat: latitude
        lng: longitude
        timeout: request timeout in seconds
        skip_cache: skip reading from the cache

    Returns:
        AirQualityResult
    """
    _ensure_tables()

    # Check the cache (TTL: 365 days - annual data changes slowly)
    if not skip_cache:
        cached = _get_cached("air_quality_cache", lat, lng, 365)
        if cached:
            # Recompute score from annual_stats (avoids cached values from an old formula)
            stats = cached.get("annual_stats") or {}
            if any(stats.get(k) for k in ("no2", "pm25", "pm10")):
                score, level, level_zh = _calculate_air_quality_score(stats)
                cached["score"] = score
                cached["quality_level"] = level
                cached["quality_level_zh"] = level_zh
            return cached

    result: AirQualityResult = {
        "has_data": False,
        "score": 0,
        "quality_level": "moderate",
        "quality_level_zh": "中等",
        "nearest_site": None,
        "annual_stats": None,
        "data_source": "none",
    }

    site_info = None
    data_source = "none"

    # 1. Try LAQN first (London data is more precise)
    site_info = _query_laqn(lat, lng, timeout)
    if site_info:
        data_source = "laqn"
    else:
        # 2. Fallback: UK-AIR (UK-wide AURN network)
        site_info = _query_uk_air(lat, lng, timeout)
        if site_info:
            data_source = "uk_air"

    if site_info:
        result["has_data"] = True
        result["data_source"] = data_source
        result["nearest_site"] = {
            "name": site_info["name"],
            "distance_km": site_info["distance_km"],
        }
        result["annual_stats"] = {
            "no2": site_info.get("no2"),
            "pm25": site_info.get("pm25"),
            "pm10": site_info.get("pm10"),
        }

        # Compute the score
        score, level, level_zh = _calculate_air_quality_score(site_info)
        result["score"] = score
        result["quality_level"] = level
        result["quality_level_zh"] = level_zh

    # Save to the cache
    _save_cache("air_quality_cache", lat, lng, result)

    return result


# =============================================================================
# Parks Evaluator
# =============================================================================

PARKS_LEVELS = {
    'excellent': ('excellent', '优秀'),
    'good': ('good', '良好'),
    'moderate': ('moderate', '中等'),
    'poor': ('poor', '较差'),
}

# Multiple Overpass API endpoints (primary + fallback)
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# Overpass requires a polite User-Agent per API usage policy; requests' default
# "python-requests/X.Y" triggers HTTP 406 from the main endpoint. Naming the
# project + contact email means we get rate-limited instead of banned if we
# ever overstep their fair-use policy.
OVERPASS_HEADERS = {
    "User-Agent": "wheretolive.xyz/1.0 (+https://wheretolive.xyz; contact@wheretolive.xyz)",
}


def evaluate_parks(lat: float, lng: float, timeout: float = 15.0, skip_cache: bool = False) -> ParksResult:
    """
    Evaluate parks and green space near a coordinate

    Uses the OpenStreetMap Overpass API

    Scoring logic:
    - A park within 500m: +30 pts
    - Each additional park within 500m: +10 pts (max +30)
    - Parks within 1km: an extra +3 pts each (max +15)
    - Base score 25 pts

    Args:
        lat: latitude
        lng: longitude
        timeout: request timeout in seconds
        skip_cache: skip reading from the cache

    Returns:
        ParksResult
    """
    _ensure_tables()

    # Accessible-parks polygon path (nearest-edge distance + area-weighted score)
    # for London — authoritative. Replaces the old Overpass "distance to centroid,
    # count polygons" method which dropped/mis-ranked large parks (Royal Parks etc.)
    # and counted a 0.1 ha square the same as 142 ha Hyde Park. Public-access only.
    # Falls back to the cache/Overpass path below outside London or if unavailable.
    try:
        import parks_index
        _idx = parks_index.get_index()
        if _idx.in_bbox(lat, lng):
            _r = _idx.compute(lat, lng)
            result: ParksResult = {
                "parks_count": _r["parks_count"],
                "parks_within_500m": _r["parks_within_500m"],
                "parks_within_1km": _r["parks_within_1km"],
                "green_score": _r["green_score"],
                "green_level": _r["green_level"],
                "green_level_zh": _r["green_level_zh"],
                "nearest_parks": _r["nearest_parks"],
            }
            _save_cache("parks_cache", lat, lng, result)
            return result
    except Exception:
        pass  # shapely/index missing or point outside London → legacy path

    # Check the cache (TTL: 90 days)
    if not skip_cache:
        cached = _get_cached("parks_cache", lat, lng, 90)
        if cached:
            # Recompute green_score from parks_within_500m/1km (avoids cached values from an old formula)
            p500 = cached.get("parks_within_500m", 0)
            p1km_total = cached.get("parks_within_1km", 0)
            p1km_only = p1km_total - p500
            weighted = p500 * 2 + p1km_only
            score = max(5, min(100, int(10 + 17 * math.log(weighted + 1))))
            cached["green_score"] = score
            if score >= 80:
                cached["green_level"], cached["green_level_zh"] = "excellent", "优秀"
            elif score >= 60:
                cached["green_level"], cached["green_level_zh"] = "good", "良好"
            elif score >= 40:
                cached["green_level"], cached["green_level_zh"] = "moderate", "中等"
            else:
                cached["green_level"], cached["green_level_zh"] = "poor", "较差"
            return cached

    result: ParksResult = {
        "parks_count": 0,
        "parks_within_500m": 0,
        "parks_within_1km": 0,
        "green_score": 20,  # base score
        "green_level": "poor",
        "green_level_zh": "较差",
        "nearest_parks": [],
    }

    api_success = False
    try:
        # Public green space query: park/recreation/nature + public playgrounds + named gardens
        query = f"""
        [out:json][timeout:10];
        (
          way["leisure"="park"](around:1000,{lat},{lng});
          way["leisure"="recreation_ground"](around:1000,{lat},{lng});
          way["leisure"="nature_reserve"](around:1000,{lat},{lng});
          way["leisure"="playground"]["access"!="private"](around:1000,{lat},{lng});
          way["leisure"="garden"]["name"](around:1000,{lat},{lng});
          relation["leisure"="park"](around:1000,{lat},{lng});
        );
        out center;
        """

        # Try multiple Overpass endpoints (fall back if the primary fails)
        # The Overpass query has timeout=10s, so give each endpoint a request timeout of at least 12s
        data = None
        per_endpoint_timeout = max(12.0, timeout / len(OVERPASS_ENDPOINTS))
        for endpoint_url in OVERPASS_ENDPOINTS:
            try:
                resp = requests.post(endpoint_url, data={"data": query}, headers=OVERPASS_HEADERS, timeout=per_endpoint_timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    record_api_usage("osm_overpass")
                    api_success = True
                    break
                # 429/504 or similar error: try the next endpoint
            except Exception:
                continue

        if not api_success or data is None:
            raise RuntimeError("All Overpass endpoints failed")

        elements = data.get("elements", [])

        parks_500m = []
        parks_1km = []

        for elem in elements:
            # Get the centre point
            if elem.get("center"):
                park_lat = elem["center"]["lat"]
                park_lng = elem["center"]["lon"]
            elif elem.get("lat") and elem.get("lon"):
                park_lat = elem["lat"]
                park_lng = elem["lon"]
            else:
                continue

            # Compute the distance
            dist_km = _haversine_distance(lat, lng, park_lat, park_lng)
            dist_m = int(dist_km * 1000)

            # Get the name and tags
            tags = elem.get("tags", {})
            leisure_type = tags.get("leisure", "")
            name = tags.get("name", "")  # Only take a real name
            access = tags.get("access", "")

            # Skip private facilities (already filtered in the Overpass query; this is a second line of defence)
            if access == "private":
                continue

            # A garden must have a name (an unnamed garden is most likely a residential garden)
            if leisure_type == "garden" and not name:
                continue

            # Filter out generic names (case-insensitive)
            generic_names = {"park", "garden", "green", "gardens", "playground", "recreation ground", "green space", "open space"}
            name_lower = name.lower().strip() if name else ""
            has_real_name = bool(name and name_lower not in generic_names)

            park_info = {
                "name": name or tags.get("leisure", "Park"),  # for display
                "distance_m": dist_m,
                "_has_real_name": has_real_name,  # internal flag
            }

            if dist_m <= 500:
                parks_500m.append(park_info)
            elif dist_m <= 1000:
                parks_1km.append(park_info)

        # Sort by distance
        parks_500m.sort(key=lambda x: x["distance_m"])
        parks_1km.sort(key=lambda x: x["distance_m"])

        result["parks_within_500m"] = len(parks_500m)
        result["parks_within_1km"] = len(parks_500m) + len(parks_1km)
        result["parks_count"] = result["parks_within_1km"]

        # Only show parks with a real name, and drop the _has_real_name flag
        all_parks = parks_500m + parks_1km
        named_parks = [
            {"name": p["name"], "distance_m": p["distance_m"]}
            for p in all_parks if p.get("_has_real_name")
        ]
        result["nearest_parks"] = named_parks[:5]  # the 5 closest named parks

        # Compute the score: weighted park count → log mapping
        # Weight 2x within 500m, 1x for 500m-1km
        weighted = len(parks_500m) * 2 + len(parks_1km)
        # log(w+1) compresses the long tail: 0→10, 5→40, 10→51, 28→67, 50→77, 100→88, 132→93
        score = max(5, min(100, int(10 + 17 * math.log(weighted + 1))))

        result["green_score"] = min(100, score)

        # Rating
        if score >= 80:
            result["green_level"] = "excellent"
            result["green_level_zh"] = "优秀"
        elif score >= 60:
            result["green_level"] = "good"
            result["green_level_zh"] = "良好"
        elif score >= 40:
            result["green_level"] = "moderate"
            result["green_level_zh"] = "中等"
        else:
            result["green_level"] = "poor"
            result["green_level_zh"] = "较差"

    except Exception:
        # On API failure return the default result without caching it (avoids a failed result polluting the 90-day cache)
        pass

    # Only save to the cache if the API call succeeded
    if api_success:
        _save_cache("parks_cache", lat, lng, result)

    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 environment_evaluators.py <lat> <lng>")
        print("   or: python3 environment_evaluators.py <type> <lat> <lng>")
        print("   where type is: flood, air, parks, all")
        sys.exit(1)

    if len(sys.argv) == 3:
        # Run all by default
        lat, lng = float(sys.argv[1]), float(sys.argv[2])
        eval_type = "all"
    else:
        eval_type = sys.argv[1]
        lat, lng = float(sys.argv[2]), float(sys.argv[3])

    print(f"Evaluating environment at: {lat}, {lng}")
    print()

    if eval_type in ("flood", "all"):
        print("=== Flood Risk ===")
        flood = evaluate_flood_risk(lat, lng)
        print(json.dumps(flood, indent=2, ensure_ascii=False))
        print()

    if eval_type in ("air", "all"):
        print("=== Air Quality ===")
        air = evaluate_air_quality(lat, lng)
        print(json.dumps(air, indent=2, ensure_ascii=False))
        print()

    if eval_type in ("parks", "all"):
        print("=== Parks ===")
        parks = evaluate_parks(lat, lng)
        print(json.dumps(parks, indent=2, ensure_ascii=False))
