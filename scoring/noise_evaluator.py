"""
DEFRA Noise Map Evaluator
Queries road, rail and airport noise data for England

Data source: environment.data.gov.uk WMS GetFeatureInfo
- Round 4 data (2023)
- Lden: 24-hour weighted average (day-evening-night)
"""

import math
import requests
from typing import Optional, TypedDict
from functools import lru_cache

# WMS endpoints
WMS_ENDPOINTS = {
    "road": "https://environment.data.gov.uk/spatialdata/road-noise-all-metrics-england-round-4/wms",
    "rail": "https://environment.data.gov.uk/spatialdata/noise-data/wms",
    "airport": "https://environment.data.gov.uk/spatialdata/airport-noise-all-metrics-england-round-4/wms",
}

# Layer names (Lden = day-evening-night 24-hour weighted average)
LAYERS = {
    "road": "Road_Noise_Lden_England_Round_4_All",
    "rail": "Rail_Noise_Lden_England_Round_4_All",
    "airport": "Airport_Noise_ALL_Lden",
}

# Noise level labels (used only for category labels; the score uses a continuous sigmoid function)
NOISE_LEVEL_LABELS = [
    (50, "quiet", "安静"),
    (55, "fairly_quiet", "较安静"),
    (60, "moderate", "中等"),
    (65, "noisy", "较吵"),
    (70, "very_noisy", "很吵"),
    (float("inf"), "extremely_noisy", "非常吵"),
]


class NoiseResult(TypedDict):
    road_db: Optional[float]
    rail_db: Optional[float]
    airport_db: Optional[float]
    combined_db: Optional[float]
    level: str
    level_zh: str
    score: float
    dominant_source: Optional[str]


def query_wms_noise(lat: float, lng: float, source: str, timeout: float = 10.0) -> Optional[float]:
    """
    Query the decibel level of a single noise source
    
    Args:
        lat: latitude
        lng: longitude
        source: noise source (road/rail/airport)
        timeout: request timeout in seconds
    
    Returns:
        Decibel value, or None if there is no data
    """
    if source not in WMS_ENDPOINTS:
        return None
    
    endpoint = WMS_ENDPOINTS[source]
    layer = LAYERS[source]
    
    # Build a small BBOX (about 20m x 20m)
    delta = 0.0001  # ~10m
    bbox = f"{lat - delta},{lng - delta},{lat + delta},{lng + delta}"
    
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetFeatureInfo",
        "LAYERS": layer,
        "QUERY_LAYERS": layer,
        "INFO_FORMAT": "application/json",
        "CRS": "EPSG:4326",
        "BBOX": bbox,
        "WIDTH": "10",
        "HEIGHT": "10",
        "I": "5",
        "J": "5",
    }
    
    try:
        resp = requests.get(endpoint, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        
        features = data.get("features", [])
        if features:
            gray_index = features[0].get("properties", {}).get("GRAY_INDEX")
            # Filter out NoData sentinel values (typically very large numbers like 3.4e+38)
            # Valid noise readings should be between 0 and 120 dB
            if gray_index is not None and 0 < gray_index < 120:
                return round(float(gray_index), 1)
        return None
    except Exception as e:
        # Fail silently and return None
        return None


def combine_noise_levels(levels: list[float]) -> float:
    """
    Combine decibel levels from multiple noise sources (sound pressure level summation)
    
    Formula: L_total = 10 × log₁₀(Σ 10^(Li/10))
    """
    if not levels:
        return 0
    
    total = sum(10 ** (l / 10) for l in levels)
    return round(10 * math.log10(total), 1)


def _noise_score(db: float) -> float:
    """
    Decibels → score, using a smooth logistic sigmoid curve.
    Low noise scores high, high noise scores low. center=63dB, k=0.18
    ~45dB→92, ~50dB→87, ~55dB→78, ~60dB→63, ~65dB→42, ~70dB→24
    """
    return round(5.0 + 90.0 / (1.0 + math.exp(0.18 * (db - 63.0))), 1)


def get_noise_level(db: float) -> tuple[str, str, float]:
    """
    Return the level and smooth score for a decibel value

    Returns:
        (level_en, level_zh, score)
    """
    score = _noise_score(db)
    for threshold, level_en, level_zh in NOISE_LEVEL_LABELS:
        if db < threshold:
            return level_en, level_zh, score
    return "extremely_noisy", "非常吵", score


def _get_db_path() -> str:
    """Get the database path"""
    import os
    return os.path.join(os.path.dirname(__file__), "data", "evaluations.db")


def _get_noise_cached(lat: float, lng: float) -> Optional[NoiseResult]:
    """Get noise data from the SQLite cache"""
    import sqlite3
    try:
        conn = sqlite3.connect(_get_db_path())
        conn.execute("PRAGMA busy_timeout = 5000")
        cursor = conn.execute(
            """
            SELECT road_db, rail_db, airport_db, combined_db, level, level_zh, score, dominant_source
            FROM noise_cache
            WHERE ABS(lat - ?) < 0.0005 AND ABS(lng - ?) < 0.0005
            ORDER BY ABS(lat - ?) + ABS(lng - ?)
            LIMIT 1
            """,
            (lat, lng, lat, lng)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            # Recompute score/level from combined_db (avoids stale discrete values in the cache)
            combined = row[3]
            if combined is not None:
                level, level_zh, score = get_noise_level(combined)
            else:
                level, level_zh, score = row[4], row[5], row[6]
            return NoiseResult(
                road_db=row[0],
                rail_db=row[1],
                airport_db=row[2],
                combined_db=combined,
                level=level,
                level_zh=level_zh,
                score=score,
                dominant_source=row[7],
            )
    except Exception:
        pass
    return None


def _save_noise_cache(lat: float, lng: float, result: NoiseResult) -> None:
    """Save noise data to the SQLite cache"""
    import sqlite3
    try:
        conn = sqlite3.connect(_get_db_path())
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(
            """
            INSERT OR REPLACE INTO noise_cache 
            (lat, lng, road_db, rail_db, airport_db, combined_db, level, level_zh, score, dominant_source, cached_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (round(lat, 4), round(lng, 4), result["road_db"], result["rail_db"], 
             result["airport_db"], result["combined_db"], result["level"], 
             result["level_zh"], result["score"], result["dominant_source"])
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def evaluate_noise(lat: float, lng: float, timeout: float = 10.0, skip_cache: bool = False) -> NoiseResult:
    """
    Evaluate the noise level at a coordinate

    Args:
        lat: latitude
        lng: longitude
        timeout: timeout in seconds for each request
        skip_cache: skip reading from the cache

    Returns:
        NoiseResult with the decibel level of each noise source and the combined assessment
    """
    # Check the SQLite cache first
    if not skip_cache:
        cached = _get_noise_cached(lat, lng)
        if cached:
            return cached
    
    # Querying the three noise sources in parallel would be faster, but they are queried sequentially for simplicity
    road_db = query_wms_noise(lat, lng, "road", timeout)
    rail_db = query_wms_noise(lat, lng, "rail", timeout)
    airport_db = query_wms_noise(lat, lng, "airport", timeout)
    
    # Collect valid values
    noise_sources = []
    source_names = []
    
    if road_db is not None:
        noise_sources.append(road_db)
        source_names.append(("road", road_db))
    if rail_db is not None:
        noise_sources.append(rail_db)
        source_names.append(("rail", rail_db))
    if airport_db is not None:
        noise_sources.append(airport_db)
        source_names.append(("airport", airport_db))
    
    # Compute combined noise
    if noise_sources:
        combined_db = combine_noise_levels(noise_sources)
        level_en, level_zh, score = get_noise_level(combined_db)
        # Find the dominant noise source
        dominant = max(source_names, key=lambda x: x[1])[0] if source_names else None
    else:
        combined_db = None
        level_en, level_zh, score = "unknown", "未知", 50
        dominant = None
    
    result = NoiseResult(
        road_db=road_db,
        rail_db=rail_db,
        airport_db=airport_db,
        combined_db=combined_db,
        level=level_en,
        level_zh=level_zh,
        score=score,
        dominant_source=dominant,
    )
    
    # Save to the SQLite cache
    if combined_db is not None:
        _save_noise_cache(lat, lng, result)
    
    return result


# Cache postcode coordinate lookups
@lru_cache(maxsize=1000)
def get_postcode_coords(postcode: str) -> Optional[tuple[float, float]]:
    """
    Get the centroid coordinates of a postcode via postcodes.io
    
    Returns:
        (lat, lng) or None
    """
    try:
        clean_pc = postcode.upper().replace(" ", "")
        resp = requests.get(
            f"https://api.postcodes.io/postcodes/{clean_pc}",
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == 200:
                result = data.get("result", {})
                return (result.get("latitude"), result.get("longitude"))
    except Exception:
        pass
    return None


def evaluate_noise_by_postcode(postcode: str, timeout: float = 10.0) -> Optional[NoiseResult]:
    """
    Evaluate noise by postcode
    
    Args:
        postcode: UK postcode
        timeout: request timeout
    
    Returns:
        NoiseResult, or None (if the postcode is invalid)
    """
    coords = get_postcode_coords(postcode)
    if not coords:
        return None
    lat, lng = coords
    return evaluate_noise(lat, lng, timeout)


if __name__ == "__main__":
    import json
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python noise_evaluator.py <lat> <lng>")
        print("   or: python noise_evaluator.py <postcode>")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        # Postcode mode
        postcode = sys.argv[1]
        print(f"Evaluating noise for postcode: {postcode}")
        result = evaluate_noise_by_postcode(postcode)
        if result:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Invalid postcode or no data")
    else:
        # Coordinate mode
        lat, lng = float(sys.argv[1]), float(sys.argv[2])
        print(f"Evaluating noise at: {lat}, {lng}")
        result = evaluate_noise(lat, lng)
        print(json.dumps(result, indent=2, ensure_ascii=False))
