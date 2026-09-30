#!/usr/bin/env python3
"""
Single-property evaluation across all dimensions - CLI version
Accepts the address and price directly
Evaluation dimensions can be controlled via the config file or CLI arguments
Supports caching and ranking
"""

import sys
import os
import json
import argparse
from contextlib import redirect_stdout
from evaluator import HomeEvaluator
from apis.api_usage import get_quota_exceeded_apis, clear_quota_exceeded_apis

# Cache module
try:
    from cache.service import CacheService
    from cache.config import DATA_VERSION, CACHE_VERSION, SCHEMA_VERSION, RANKING_PERIODS, DEFAULT_RANKING_PERIOD
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    DATA_VERSION = "unknown"
    CACHE_VERSION = "unknown"
    SCHEMA_VERSION = 0
    RANKING_PERIODS = {"30d": 30}
    DEFAULT_RANKING_PERIOD = "30d"

# Noise evaluation module
try:
    from noise_evaluator import evaluate_noise
    NOISE_AVAILABLE = True
except ImportError:
    NOISE_AVAILABLE = False

# Environment data evaluation module (flood, air quality, parks)
try:
    from environment_evaluators import evaluate_flood_risk, evaluate_air_quality, evaluate_parks
    ENVIRONMENT_AVAILABLE = True
except ImportError:
    ENVIRONMENT_AVAILABLE = False


def parse_price(price_str: str) -> int:
    """Parse a price string"""
    import re
    cleaned = re.sub(r'[£$,\s]', '', price_str)
    try:
        return int(cleaned)
    except ValueError:
        return 0


def validate_uk_postcode(postcode: str, property_data: dict = None) -> bool:
    """
    Validate a UK postcode: check the format first, then look it up via postcodes.io (the same call also fetches the coordinates, avoiding a second request)

    UK postcode formats:
    - AA9A 9AA, A9A 9AA, A9 9AA, A99 9AA, AA9 9AA, AA99 9AA

    If property_data is passed, the coordinates are written to _api_coords for later use.
    """
    import re
    import requests

    # Normalize the postcode
    cleaned = postcode.upper().replace(" ", "").strip()

    # 1. Format check
    uk_postcode_pattern = r'^[A-Z]{1,2}[0-9][0-9A-Z]?\s?[0-9][A-Z]{2}$'
    if not re.match(uk_postcode_pattern, cleaned):
        return False

    # 2. Fetch the full record directly (validation + coordinates in one request instead of two)
    try:
        resp = requests.get(
            f"https://api.postcodes.io/postcodes/{cleaned}",
            timeout=5
        )
        if resp.status_code == 200:
            pc_data = resp.json().get("result", {})
            if pc_data and property_data is not None:
                lat, lng = pc_data.get("latitude"), pc_data.get("longitude")
                if lat and lng:
                    property_data["_api_coords"] = {"lat": lat, "lng": lng}
            return True
        elif resp.status_code == 404:
            return False
    except Exception:
        pass

    # Let it through if the API fails (to avoid blocking)
    return True


def print_dimension_list():
    """Print all available evaluation dimensions"""
    print("\n" + "="*60)
    print("Available evaluation dimensions")
    print("="*60)

    print("\nMain dimensions:")
    print("-"*60)
    dimensions = {
        "location": ("Location convenience", "commute, public transport, nearby amenities, long-distance travel"),
        "living": ("Living quality", "orientation, noise, property age, layout"),
        "environment": ("Surrounding environment", "safety, parks, air quality"),
        "education": ("Education", "school ratings"),
        "economic": ("Economic factors", "value for money, price analysis, appreciation potential"),
        "other": ("Other factors", "parking, gym, pet-friendly"),
    }

    for key, (name, desc) in dimensions.items():
        print(f"  {key:12s} - {name}: {desc}")

    print("\nLocation convenience sub-dimensions:")
    print("-"*60)
    sub_dimensions = {
        "commute": ("Commute", "💰 Google Routes (TFL for London with --free)"),
        "transit": ("Public transport access", "💰 Google Places (TFL for London with --free)"),
        "amenities": ("Nearby amenities", "💰 Google Places API"),
        "long_distance": ("Long-distance travel access", "💰 Google Routes API"),
    }

    for key, (name, desc) in sub_dimensions.items():
        print(f"  {key:12s} - {name}: {desc}")

    print("\nSurrounding environment sub-dimensions:")
    print("-"*60)
    print(f"  {'safety':12s} - Safety: 🆓 UK Police API")
    print(f"  {'demographics':12s} - Area demographics: 🆓 Postcodes.io + IMD 2025")

    print("\nEconomic factors sub-dimensions:")
    print("-"*60)
    print(f"  {'price_analysis':12s} - Price analysis: 🆓 Land Registry + EPC API")

    print("\n" + "="*60)
    print("API tiers:")
    print("  🆓 Free APIs: TFL, Land Registry, EPC, UK Police")
    print("  💰 Paid APIs: Google Routes, Google Places")
    print("  --free mode: commute/transit use TFL in London, skipped elsewhere")
    print("  --all mode:  Google APIs everywhere")
    print("\nExamples:")
    print("  Free APIs only:  python3 evaluate_single.py <address> --free")
    print("  All APIs:        python3 evaluate_single.py <address> --all")
    print("  Skip education:  python3 evaluate_single.py <address> --skip education")
    print("  Commute only:    python3 evaluate_single.py <address> --only commute")
    print("="*60 + "\n")


def apply_cli_overrides(eval_config: dict, skip: list, only: list) -> dict:
    """
    Override the evaluation config from CLI arguments

    Args:
        eval_config: original evaluation config
        skip: list of dimensions to skip
        only: list of dimensions to evaluate exclusively

    Returns:
        The modified evaluation config
    """
    # Main dimension mapping
    dimension_map = {
        "location": "location_convenience",
        "environment": "surrounding_environment",
        "economic": "economic_factors",
    }

    # Sub-dimension mapping
    sub_dimension_map = {
        "commute": ("location_convenience", "commute"),
        "transit": ("location_convenience", "transit"),
        "amenities": ("location_convenience", "amenities"),
        "long_distance": ("location_convenience", "long_distance"),
        "safety": ("surrounding_environment", "safety"),
        "demographics": ("surrounding_environment", "demographics"),
    }

    # Make sure "dimensions" exists
    if "dimensions" not in eval_config:
        eval_config["dimensions"] = {}

    # With --only, disable everything first, then enable the requested ones
    if only:
        # Disable all main dimensions
        for dim_name in dimension_map.values():
            if dim_name not in eval_config["dimensions"]:
                eval_config["dimensions"][dim_name] = {}
            eval_config["dimensions"][dim_name]["enabled"] = False

        # Enable the requested dimensions
        for item in only:
            if item in dimension_map:
                # Main dimension
                dim_name = dimension_map[item]
                eval_config["dimensions"][dim_name]["enabled"] = True
            elif item in sub_dimension_map:
                # Sub-dimension - enable both the parent and the sub-dimension
                parent, sub = sub_dimension_map[item]
                if parent not in eval_config["dimensions"]:
                    eval_config["dimensions"][parent] = {}
                eval_config["dimensions"][parent]["enabled"] = True

                if "sub_dimensions" not in eval_config["dimensions"][parent]:
                    eval_config["dimensions"][parent]["sub_dimensions"] = {}
                if sub not in eval_config["dimensions"][parent]["sub_dimensions"]:
                    eval_config["dimensions"][parent]["sub_dimensions"][sub] = {}
                eval_config["dimensions"][parent]["sub_dimensions"][sub]["enabled"] = True

    # Handle --skip
    for item in skip:
        if item in dimension_map:
            # Main dimension
            dim_name = dimension_map[item]
            if dim_name not in eval_config["dimensions"]:
                eval_config["dimensions"][dim_name] = {}
            eval_config["dimensions"][dim_name]["enabled"] = False
        elif item in sub_dimension_map:
            # Sub-dimension
            parent, sub = sub_dimension_map[item]
            if parent not in eval_config["dimensions"]:
                eval_config["dimensions"][parent] = {}
            if "sub_dimensions" not in eval_config["dimensions"][parent]:
                eval_config["dimensions"][parent]["sub_dimensions"] = {}
            if sub not in eval_config["dimensions"][parent]["sub_dimensions"]:
                eval_config["dimensions"][parent]["sub_dimensions"][sub] = {}
            eval_config["dimensions"][parent]["sub_dimensions"][sub]["enabled"] = False

    return eval_config


def apply_api_tier_filter(eval_config: dict, tier: str) -> dict:
    """
    Filter the evaluation config by API tier

    Args:
        eval_config: original evaluation config
        tier: API tier ('free' or 'all')

    Returns:
        The modified evaluation config
    """
    if tier == "all":
        return eval_config

    # Make sure "dimensions" exists
    if "dimensions" not in eval_config:
        eval_config["dimensions"] = {}

    dimensions = eval_config["dimensions"]

    # Walk all dimensions and disable those that need a paid API
    for dim_name, dim_config in dimensions.items():
        if not isinstance(dim_config, dict):
            continue

        # Check the main dimension's API tier
        api_tier = dim_config.get("api_tier")
        if api_tier == "paid":
            dim_config["enabled"] = False
            dim_config["_disabled_reason"] = "requires paid API"
            continue

        # Check the sub-dimensions
        sub_dims = dim_config.get("sub_dimensions", {})
        for sub_name, sub_config in sub_dims.items():
            if not isinstance(sub_config, dict):
                continue

            sub_api_tier = sub_config.get("api_tier")

            # "paid" - paid API only; disabled in free mode
            if sub_api_tier == "paid":
                sub_config["enabled"] = False
                sub_config["_disabled_reason"] = "requires paid API"
            # "mixed" - has a free alternative (e.g. TfL); stays enabled
            # "free" - free API; stays enabled
            # "none" - no API needed; stays enabled

    return eval_config


def _census_population(demo: dict) -> int:
    """Extract total population from Census 2021 ethnicity data (most reliable headcount)."""
    census = demo.get("census")
    if not census:
        return 0
    ethnicity = census.get("ethnicity")
    if ethnicity and isinstance(ethnicity, dict):
        return ethnicity.get("total", 0)
    return 0


def _format_census(census_data):
    """Format census data with percentages for JSON output."""
    if not census_data:
        return None
    result = {}
    for dim_key, dim_data in census_data.items():
        if dim_data is None:
            result[dim_key] = None
            continue
        total = dim_data.get("total", 0)
        categories = {}
        for cat_name, count in dim_data.get("categories", {}).items():
            categories[cat_name] = {
                "count": count,
                "percentage": round(count / total * 100, 1) if total > 0 else 0,
            }
        result[dim_key] = {"total": total, "categories": categories}
    return result


def stream_json_output(output: dict, already_streamed: set = None):
    """Emit one JSON line per dimension (stream mode)

    Args:
        output: the full evaluation result
        already_streamed: dimensions already emitted by the parallel tasks, to avoid emitting them twice
    """
    already_streamed = already_streamed or set()

    # Main evaluation dimensions (commute, transit, safety, price_analysis are emitted after the main evaluation finishes)
    # demographics and schools are already emitted separately by the parallel tasks
    dim_keys = ["commute", "transit", "safety", "demographics", "schools", "price_analysis"]
    for dim in dim_keys:
        if dim in already_streamed:
            continue  # skip ones already emitted
        if dim in output:
            line = {"dim": dim, "data": output[dim]}
            # Add score for scored dimensions (use new dimension names)
            if dim == "price_analysis" and output.get("scores"):
                score_data = output["scores"].get("price")
                if score_data:
                    line["score"] = score_data["score"]
            print(json.dumps(line, ensure_ascii=False), flush=True)

    # Environment data & hub commute - already emitted by the parallel tasks, skipped here
    env_keys = ["noise", "flood_risk", "air_quality", "parks", "hub_commute"]
    for key in env_keys:
        if key in already_streamed:
            continue
        if key in output:
            print(json.dumps({"dim": key, "data": output[key]}, ensure_ascii=False), flush=True)

    # Summary line
    summary = {
        "total_score": output.get("total_score"),
        "rating": output.get("rating"),
        "scores": output.get("scores"),
        "weights": output.get("weights"),
        "adjusted_weights": output.get("adjusted_weights"),
        "percentiles": output.get("percentiles"),
        "missing_dims": output.get("missing_dims"),
        "data_completeness": output.get("data_completeness"),
        "address": output.get("address", ""),
        "price": output.get("price", 0),
        "bedrooms": output.get("bedrooms", 0),
    }
    if "_cache" in output:
        summary["_cache"] = output["_cache"]
    if "_ranking" in output:
        summary["_ranking"] = output["_ranking"]
    if "_failed_dims" in output:
        summary["_failed_dims"] = output["_failed_dims"]
    if "_no_data_dims" in output:
        summary["_no_data_dims"] = output["_no_data_dims"]
    if "_paused_dims" in output:
        summary["_paused_dims"] = output["_paused_dims"]
    print(json.dumps({"dim": "_summary", "data": summary}, ensure_ascii=False), flush=True)


def build_json_output(property_data: dict, result: dict, skip_cache: bool = False) -> dict:
    """Build the JSON output structure"""
    from simple_scorer import SimpleScorer

    # Basic info
    output = {
        "address": property_data.get("address", ""),
        "price": property_data.get("price", 0),
        "bedrooms": property_data.get("bedrooms", 0),
        "dimensions": {}
    }

    # Track missing dimensions: API error vs. query returned no results
    failed_dims = []    # API call failed / timed out
    no_data_dims = []   # API worked but returned no data
    paused_dims = []    # API paused (quota)

    # Per-dimension "attempted" flag (set by the parallel runners)
    _dim_to_attempted = {
        "commute": "_main_eval_attempted",
        "transit": "_main_eval_attempted",
        "safety": "_main_eval_attempted",
        "price_analysis": "_main_eval_attempted",
        "demographics": "_demographics_attempted",
        "schools": "_schools_attempted",
    }

    def _mark_missing(dim: str):
        """API called but no data → fetch failed; not called → no data yet"""
        attempted_key = _dim_to_attempted.get(dim)
        if attempted_key and property_data.get(attempted_key):
            failed_dims.append(dim)
        else:
            no_data_dims.append(dim)

    # Commute data
    if "_api_commute_time_minutes" in property_data:
        output["commute"] = {
            "average_minutes": round(property_data["_api_commute_time_minutes"], 1),
            "source": property_data.get("_api_commute_source", "unknown"),
            "destination": property_data.get("_api_commute_destination", "")
        }
        if "_api_commute_details" in property_data:
            details = property_data["_api_commute_details"]
            # Default fare for walking-only commutes
            default_fare = {
                "peak": 0,
                "off_peak": 0,
                "total": 0,
                "low_zone": None,
                "high_zone": None
            }
            if "morning" in details:
                output["commute"]["morning_minutes"] = round(details["morning"]["time_minutes"], 1)
                # Include fare data (use default 0 for walking-only commutes)
                morning_fare = details["morning"].get("fare")
                output["commute"]["morning_fare"] = morning_fare if morning_fare else default_fare
            # The evening peak is no longer computed; write null
            output["commute"]["evening_minutes"] = None
            output["commute"]["evening_fare"] = None
    else:
        _mark_missing("commute")

    # Public transport data
    if "_api_transit_data" in property_data:
        transit = property_data["_api_transit_data"]
        output["transit"] = {
            "lines_count": transit.get("lines_count", 0),
            "lines": transit.get("lines", []),
            "total_weight": transit.get("total_line_weight", 0),
            "score": round(transit.get("tube_score", 0), 1),
            "source": transit.get("api_source", "unknown")
        }
        # Nearest stations
        tube_details = property_data.get("_api_transit_tube_details", [])
        if tube_details:
            output["transit"]["nearest_stations"] = [
                {
                    "name": t.get("name", ""),
                    "walk_minutes": round(t.get("walk_time", 0), 1),
                    "lines": t.get("lines", [])
                }
                for t in tube_details[:5]
            ]
    else:
        # _api_transit_coords present means the API was called fine; it just found no stations
        if "_api_transit_coords" in property_data:
            no_data_dims.append("transit")
        else:
            _mark_missing("transit")

    # Safety data
    if "_api_crime_data" in property_data:
        crime = property_data["_api_crime_data"]
        coords = property_data.get("_api_crime_coords", {})
        output["safety"] = {
            "total_crimes": crime.get("total", 0),
            "coords": coords,
            "breakdown": {
                "anti_social": crime.get("anti-social-behaviour", 0),
                "burglary": crime.get("burglary", 0),
                "violent": crime.get("violent-crime", 0),
                "robbery": crime.get("robbery", 0),
                "theft": crime.get("theft-from-the-person", 0),
                "drugs": crime.get("drugs", 0),
                "weapons": crime.get("possession-of-weapons", 0)
            }
        }
        # If the local police force does not publish data to data.police.uk, mark it unavailable so the frontend/scorer ignore it
        if crime.get("_data_unavailable"):
            output["safety"]["_data_unavailable"] = True
            output["safety"]["force_id"] = crime.get("force_id")
            output["safety"]["force_name"] = crime.get("force_name")
    else:
        if "_api_crime_coords" in property_data:
            no_data_dims.append("safety")
        else:
            _mark_missing("safety")

    # Demographics data
    if "_api_demographics_data" in property_data:
        demo = property_data["_api_demographics_data"]
        output["demographics"] = {
            "postcode": demo.get("postcode", ""),
            "lsoa_code": demo.get("lsoa_code", ""),
            "lsoa_name": demo.get("lsoa_name", ""),
            "local_authority": demo.get("lad_name", ""),
            "imd_decile": demo.get("imd_decile", 0),
            "imd_rank": demo.get("imd_rank", 0),
            "total_lsoas": demo.get("total_lsoas", 33755),
            "income_decile": demo.get("income_decile", 0),
            "employment_decile": demo.get("employment_decile", 0),
            "education_decile": demo.get("education_decile", 0),
            "crime_decile": demo.get("crime_decile", 0),
            "health_decile": demo.get("health_decile", 0),
            "housing_decile": demo.get("housing_decile", 0),
            "environment_decile": demo.get("environment_decile", 0),
            # Population figures — always use the IMD mid-2015 estimates (consistent LSOA 2011 basis)
            # Census 2021 uses the new LSOA boundaries, so its population counts are not comparable with IMD
            "total_population": demo.get("total_population", 0),
            "children_0_15": demo.get("children_0_15", 0),
            "population_16_59": demo.get("population_16_59", 0),
            "population_60_plus": demo.get("population_60_plus", 0),
            "score": round(demo.get("total_score", 0), 1),
            "census": _format_census(demo.get("census")),
        }
    else:
        _mark_missing("demographics")

    # Schools data
    if "_api_schools_data" in property_data:
        schools_details = property_data["_api_schools_data"]
        schools_score = property_data.get("_api_schools_score", 0)
        nearest = []
        from school_evaluator import SchoolEvaluator as _SE
        for s in schools_details.get("nearby_schools", []):
            if s.get("ofsted_rating") is not None:
                label = _SE.OFSTED_LABELS.get(s["ofsted_rating"], "Unknown")
                nearest.append({
                    "name": s["name"],
                    "type": s.get("phase", "Unknown"),
                    "category": s.get("category", "state"),
                    "ofsted_rating": label,
                    "distance_km": s["distance_km"],
                })
                if len(nearest) >= 10:
                    break
        output["schools"] = {
            "score": round(schools_score, 1),
            "total_schools": schools_details.get("total_schools", 0),
            "rated_schools": schools_details.get("rated_schools", 0),
            "outstanding_count": schools_details.get("outstanding_count", 0),
            "good_count": schools_details.get("good_count", 0),
            "ri_count": schools_details.get("ri_count", 0),
            "inadequate_count": schools_details.get("inadequate_count", 0),
            "nearest_schools": nearest,
        }
    else:
        _mark_missing("schools")

    # Price analysis data
    if "_api_price_analysis" in property_data:
        price_data = property_data["_api_price_analysis"]
        summary = price_data.get("summary", {})
        trends = price_data.get("trends", {})

        output["price_analysis"] = {
            "postcode": price_data.get("postcode", ""),
            "area_scope": price_data.get("area_scope", "unit"),
            "area_scope_postcode": price_data.get("area_scope_postcode", ""),
            "years_analyzed": price_data.get("years_analyzed", 0),
            "total_transactions": summary.get("total_transactions", 0),
            "avg_price_transactions": summary.get("avg_price_transactions", 0),
            "average_price": summary.get("average_price", 0),
            "median_price": summary.get("median_price", 0),
            "price_per_sqm": summary.get("price_per_sqm", 0),
            "price_per_sqm_count": summary.get("price_per_sqm_count", 0),
            "trend_direction": trends.get("direction", "unknown"),
            "cagr_10y": trends.get("10_year_cagr"),
            "cagr_10y_count": trends.get("10_year_count", 0),
            "cagr_5y": trends.get("5_year_cagr"),
            "cagr_5y_count": trends.get("5_year_count", 0),
            "cagr_3y": trends.get("3_year_cagr"),
            "cagr_3y_count": trends.get("3_year_count", 0),
            # Full paired-sales stats (used for scoring; the repeat_sales field is only a display sample).
            # Pass None through when missing — the scorer gives None a neutral score; a fabricated 0 would masquerade as "confirmed no sales"
            "repeat_sales_count": price_data.get("repeat_sales_count"),
            "repeat_sales_return_std": price_data.get("repeat_sales_return_std")
        }

        # Repeat-sales examples
        repeat_sales = price_data.get("repeat_sales", [])
        if repeat_sales:
            output["price_analysis"]["repeat_sales"] = [
                {
                    "address": rs.get("address", ""),
                    "previous_sale": {
                        "date": rs.get("previous_sale", {}).get("date", ""),
                        "price": rs.get("previous_sale", {}).get("price", 0)
                    },
                    "current_sale": {
                        "date": rs.get("current_sale", {}).get("date", ""),
                        "price": rs.get("current_sale", {}).get("price", 0)
                    },
                    "total_change_pct": round(rs.get("price_change_pct", 0), 1),
                    "annualized_return": round(rs.get("annualized_return", 0), 2),
                    "years_held": round(rs.get("years_held", 0), 1),
                    "selection_reason": rs.get("selection_reason")
                }
                for rs in repeat_sales
            ]

        # Yearly stats
        yearly = price_data.get("yearly_stats", [])
        if yearly:
            output["price_analysis"]["yearly_stats"] = [
                {
                    "year": y.get("year", 0),
                    "transactions": y.get("count", 0),
                    "average_price": y.get("average_price", 0),
                    "price_per_sqm": y.get("price_per_sqm") or 0
                }
                for y in yearly[-10:]  # last 10 years
            ]
    else:
        _mark_missing("price_analysis")

    quota_apis = get_quota_exceeded_apis()
    if quota_apis:
        paused = set()
        if "google_routes" in quota_apis:
            paused.update(["commute", "long_distance"])
        if "google_places" in quota_apis:
            paused.update(["transit", "amenities", "long_distance"])
        if "google_geocoding" in quota_apis:
            paused.update(["commute", "transit", "amenities", "long_distance"])
        paused_dims = sorted(paused)

    if failed_dims:
        output["_failed_dims"] = failed_dims
    if no_data_dims:
        output["_no_data_dims"] = no_data_dims
    if paused_dims:
        output["_paused_dims"] = paused_dims

    # Get noise data (prefer data prefetched by the parallel threads, to avoid repeat API calls and DB access)
    noise_data = None
    if "_api_noise" in property_data:
        noise_data = property_data["_api_noise"]
        output["noise"] = noise_data
    else:
        coords = property_data.get("_api_coords") or property_data.get("_api_crime_coords")
        if NOISE_AVAILABLE and coords and coords.get("lat") and coords.get("lng"):
            try:
                noise_result = evaluate_noise(coords["lat"], coords["lng"], timeout=8.0, skip_cache=skip_cache)
                if noise_result and noise_result.get("combined_db") is not None:
                    noise_data = noise_result
                    output["noise"] = noise_result
            except Exception:
                pass  # a failed noise fetch does not affect the main flow

    # Get environment data (flood risk, air quality, parks/green space) - use prefetched data or fetch in parallel
    prefetched_env = property_data.get("_prefetched_env", {})
    if prefetched_env:
        # Use the prefetched data (fetched in parallel with the main evaluation)
        if prefetched_env.get("flood"):
            output["flood_risk"] = prefetched_env["flood"]
        if prefetched_env.get("air"):
            output["air_quality"] = prefetched_env["air"]
        if prefetched_env.get("parks"):
            output["parks"] = prefetched_env["parks"]
    elif ENVIRONMENT_AVAILABLE and coords and coords.get("lat") and coords.get("lng"):
        # Fallback: no prefetched data, so fetch in parallel here
        lat, lng = coords["lat"], coords["lng"]
        from concurrent.futures import ThreadPoolExecutor

        def _get_flood():
            try:
                return evaluate_flood_risk(lat, lng, timeout=8.0, skip_cache=skip_cache)
            except Exception:
                return None

        def _get_air():
            try:
                return evaluate_air_quality(lat, lng, timeout=8.0, skip_cache=skip_cache)
            except Exception:
                return None

        def _get_parks():
            try:
                return evaluate_parks(lat, lng, timeout=12.0, skip_cache=skip_cache)
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=3) as env_pool:
            flood_future = env_pool.submit(_get_flood)
            air_future = env_pool.submit(_get_air)
            parks_future = env_pool.submit(_get_parks)

            try:
                flood_result = flood_future.result(timeout=10.0)
                if flood_result:
                    output["flood_risk"] = flood_result
            except Exception:
                pass

            try:
                air_result = air_future.result(timeout=10.0)
                if air_result:
                    output["air_quality"] = air_result
            except Exception:
                pass

            try:
                parks_result = parks_future.result(timeout=14.0)
                if parks_result:
                    output["parks"] = parks_result
            except Exception:
                pass

    # Hub commute data
    if "_api_hub_commute" in property_data:
        output["hub_commute"] = property_data["_api_hub_commute"]

    # Compute the per-dimension scores with SimpleScorer
    scorer = SimpleScorer()
    scorer_data = {
        "address": property_data.get("address", ""),
        "commute": output.get("commute"),
        "transit": output.get("transit"),
        "safety": output.get("safety"),
        "noise": noise_data,
        "flood_risk": output.get("flood_risk"),
        "air_quality": output.get("air_quality"),
        "parks": output.get("parks"),
        "price_analysis": output.get("price_analysis"),
        "schools": output.get("schools"),
        "demographics": output.get("demographics"),
    }
    score_result = scorer.calculate_scores(scorer_data)

    # Add the scoring result to the output
    output["total_score"] = score_result["total_score"]
    output["rating"] = score_result["rating"]
    output["scores"] = score_result["scores"]
    output["weights"] = score_result["weights"]
    output["percentiles"] = score_result.get("percentiles", {})
    output["missing_dims"] = score_result.get("missing_dims", [])
    output["data_completeness"] = score_result.get("data_completeness", "0/5")
    # backward compat
    output["adjusted_weights"] = score_result.get("adjusted_weights", {})

    return output


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a single property across all dimensions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (address only; uses all APIs by default)
  python3 evaluate_single.py "King's Cross, London N1 9AP"

  # Free APIs only (recommended for everyday use)
  python3 evaluate_single.py "EC4M 8AD" --free

  # All APIs (including the paid Google APIs)
  python3 evaluate_single.py "EC4M 8AD" --all

  # Specify price and bedrooms
  python3 evaluate_single.py "King's Cross, London N1 9AP" --price 520000 --bedrooms 2

  # Skip the education evaluation (suits households without children)
  python3 evaluate_single.py "King's Cross, London N1 9AP" --skip education

  # Evaluate only commute and public transport
  python3 evaluate_single.py "King's Cross, London N1 9AP" --only commute transit

  # List all available dimensions
  python3 evaluate_single.py --list-dimensions

API tiers:
  --free   free APIs only: TFL, Land Registry, EPC, UK Police
  --all    all APIs, including the paid Google Routes/Places

Dimension control:
  config.json sets which dimensions are enabled by default
  --skip temporarily skips some dimensions
  --only evaluates only the specified dimensions
        """
    )

    parser.add_argument(
        'address',
        nargs='?',
        help='Property address'
    )

    parser.add_argument(
        '-p', '--price',
        type=str,
        default='0',
        help='Price (e.g. 450000 or "£450,000")'
    )

    parser.add_argument(
        '-b', '--bedrooms',
        type=int,
        default=2,
        help='Number of bedrooms (default: 2)'
    )

    parser.add_argument(
        '-d', '--destination',
        type=str,
        default='',
        help='Commute destination (postcode or address, e.g. "EC2R 8AH")'
    )

    parser.add_argument(
        '--bathrooms',
        type=int,
        default=1,
        help='Number of bathrooms (default: 1)'
    )

    parser.add_argument(
        '--orientation',
        choices=['North', 'South', 'East', 'West', 'Northeast', 'Northwest', 'Southeast', 'Southwest'],
        default='South',
        help='Property orientation (default: South)'
    )

    parser.add_argument(
        '--noise',
        choices=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        default='Medium',
        help='Noise level (default: Medium)'
    )

    parser.add_argument(
        '--crime',
        choices=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        default='Medium',
        help='Crime level (default: Medium)'
    )

    parser.add_argument(
        '--school',
        type=int,
        default=7,
        help='School catchment rating 1-10 (default: 7)'
    )

    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Interactive input mode'
    )

    # Evaluation dimension control arguments
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Config file path (default: config.json)'
    )

    parser.add_argument(
        '--skip',
        nargs='+',
        choices=['commute', 'transit', 'amenities', 'long_distance', 'safety', 'demographics',
                 'location', 'living', 'environment', 'education', 'economic', 'other'],
        default=[],
        help='Skip the specified evaluation dimensions'
    )

    parser.add_argument(
        '--only',
        nargs='+',
        choices=['commute', 'transit', 'amenities', 'long_distance', 'safety', 'demographics',
                 'location', 'living', 'environment', 'education', 'economic', 'other'],
        default=[],
        help='Evaluate only the specified dimensions (ignores the config file)'
    )

    parser.add_argument(
        '--list-dimensions',
        action='store_true',
        help='List all available evaluation dimensions'
    )

    # API tier options
    api_group = parser.add_mutually_exclusive_group()
    api_group.add_argument(
        '--free',
        action='store_true',
        help='Free APIs only (TFL, Land Registry, EPC, UK Police)'
    )
    api_group.add_argument(
        '--all',
        action='store_true',
        help='All APIs, including the paid ones (Google Routes, Google Places)'
    )

    parser.add_argument(
        '--json',
        action='store_true',
        help='Output JSON (for the web API)'
    )

    parser.add_argument(
        '--stream',
        action='store_true',
        help='Streaming output (use with --json; emits one JSON line per dimension)'
    )

    # Cache arguments
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Bypass the cache and force a fresh evaluation'
    )

    parser.add_argument(
        '--ranking-period',
        choices=['1d', '7d', '30d', '1y'],
        default='30d',
        help='Ranking period (default: 30d)'
    )

    parser.add_argument(
        '--cache-stats',
        action='store_true',
        help='Show cache statistics'
    )

    args = parser.parse_args()
    clear_quota_exceeded_apis()

    # Show the dimension list
    if args.list_dimensions:
        print_dimension_list()
        sys.exit(0)

    # Show cache statistics
    if args.cache_stats:
        if not CACHE_AVAILABLE:
            print("Cache module unavailable")
            sys.exit(1)
        cache_service = CacheService()
        stats = cache_service.get_cache_stats()
        print("\nCache statistics:")
        total_records = stats['total_records']
        if isinstance(total_records, dict):
            for dim, cnt in total_records.items():
                print(f"  {dim}: {cnt} records")
        else:
            print(f"  Total records: {total_records}")
        print(f"  Data version: {stats['data_version']}")
        print(f"  Schema version: {stats['schema_version']}")
        for period in ['1d', '7d', '30d', '1y']:
            period_stats = cache_service.get_statistics(period)
            print(f"\n  {period} stats:")
            print(f"    Postcodes evaluated: {period_stats.unique_postcodes}")
            print(f"    Total evaluations: {period_stats.total_evaluations}")
            if period_stats.avg_score:
                print(f"    Average score: {period_stats.avg_score:.1f}")
        sys.exit(0)

    # Interactive mode
    if args.interactive:
        print("\n" + "="*60)
        print("Interactive property evaluation")
        print("="*60)
        address = input("\nProperty address: ").strip()
        price_str = input("Price (e.g. £520,000): ").strip() or "0"
        bedrooms = input("Bedrooms (default 2): ").strip() or "2"
        bathrooms = input("Bathrooms (default 1): ").strip() or "1"

        property_data = {
            "address": address,
            "price": parse_price(price_str),
            "bedrooms": int(bedrooms),
            "bathrooms": int(bathrooms),
            "property_type": "Flat",
            "orientation": "South",
            "noise_level": "Medium",
            "crime_rate": "Medium",
            "school_rating": 7
        }

    # CLI argument mode
    else:
        if not args.address:
            parser.print_help()
            print("\nError: please provide a property address")
            sys.exit(1)

        property_data = {
            "address": args.address,
            "price": parse_price(args.price),
            "bedrooms": args.bedrooms,
            "bathrooms": args.bathrooms,
            "property_type": "Flat",

            # Living quality
            "orientation": args.orientation,
            "noise_level": args.noise,
            "property_age_years": 10,
            "layout_rating": 7,

            # Surroundings
            "crime_rate": args.crime,
            "park_distance_m": 500,
            "air_quality_index": 70,

            # Education
            "school_rating": args.school,

            # Other
            "has_parking": False,
            "gym_nearby": True,
            "pet_friendly": True
        }

    # Validate the postcode format (this also fetches the coordinates, avoiding another postcodes.io call later)
    if not validate_uk_postcode(property_data["address"], property_data):
        if args.json:
            error_output = {
                "error": True,
                "error_type": "invalid_postcode",
                "message": f"Invalid UK postcode: {property_data['address']}",
                "address": property_data["address"]
            }
            if args.stream:
                print(json.dumps(error_output, ensure_ascii=False), flush=True)
            else:
                print(json.dumps(error_output, ensure_ascii=False, indent=2))
            return
        else:
            print(f"\nError: invalid UK postcode format: {property_data['address']}")
            print("Example UK postcodes: SW1A 1AA, EC4M 8AD, N4 3JP")
            sys.exit(1)

    # Load the config file
    config = {}
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Config file {args.config} not found, using default config")
    except json.JSONDecodeError as e:
        print(f"Config file parse error: {e}, using default config")

    # Extract the dimension config for runtime overrides
    eval_config = {"dimensions": config.get("dimensions", {})}

    # Apply CLI overrides
    if args.skip or args.only:
        eval_config = apply_cli_overrides(eval_config, args.skip, args.only)

    # Apply the API tier filter
    api_tier = "all"  # all APIs by default
    if args.free:
        api_tier = "free"
        eval_config = apply_api_tier_filter(eval_config, "free")
    elif args.all:
        api_tier = "all"
        # No filtering needed; use all APIs

    # Save the API tier into the config
    eval_config["_api_tier"] = api_tier

    # Run the evaluation
    if not args.json:
        print(f"\nEvaluating property: {property_data['address']}")
        print(f"Price: £{property_data['price']:,}")
        print(f"Bedrooms: {property_data['bedrooms']}")

        # Show the API tier
        if api_tier == "free":
            print(f"API tier: 🆓 free APIs only (TFL, Land Registry, EPC, UK Police)")
        else:
            print(f"API tier: 💰 all APIs (including Google Routes, Google Places)")

    evaluator = HomeEvaluator(config_path=args.config)
    evaluator.set_eval_config(eval_config)

    # Set the commute destination (if given)
    if args.destination:
        evaluator.set_commute_destination(args.destination)

    # JSON output mode - suppress prints during the evaluation
    if args.json:
        # Redirect stdout to stderr so debug output does not corrupt the JSON
        original_stdout = sys.stdout
        sys.stdout = sys.stderr

        # Set up the cache service
        cache_service = None
        cache_info = None
        ranking_info = None
        skip_cache_read = args.no_cache  # --no-cache only skips cache reads; results are still saved and ranked

        if CACHE_AVAILABLE:
            cache_service = CacheService()
            destination = args.destination or "EC2R 8AH"

            # Check the cache (unless --no-cache)
            if not skip_cache_read:
                cached_result, cache_info = cache_service.get_cached(
                    postcode=property_data["address"],
                    destination=destination
                )

                if cached_result and cache_info.hit:
                    # Fill in the non-dimension fields (these come from the input arguments and are not cached)
                    cached_result["address"] = property_data.get("address", "")
                    cached_result["price"] = property_data.get("price", 0)
                    cached_result["bedrooms"] = property_data.get("bedrooms", 0)
                    # Cache hit: fetch the ranking info
                    ranking_info = cache_service.get_ranking(cached_result, args.ranking_period)
                    # Add the cache and ranking info to the output
                    cached_result["_cache"] = {
                        "hit": True,
                        "cached_at": cache_info.cached_at,
                        "expires_at": cache_info.expires_at,
                        "data_version": cache_info.data_version,
                        "schema_version": cache_info.schema_version,
                    }
                    cached_result["_ranking"] = {
                        "period": ranking_info.period,
                        "period_days": ranking_info.period_days,
                        "total_evaluated": ranking_info.total_evaluated,
                        "percentiles": ranking_info.percentiles,
                        "ranks": ranking_info.ranks,
                    }
                    # Re-derive missing dimensions from the cached data (the cache comes from successful evaluations, so missing = no data yet)
                    # Raw data dimensions
                    _all_dims = ["commute", "transit", "safety", "demographics", "schools", "price_analysis"]
                    # Scored dimensions (computed by simple_scorer, no need to check them)
                    _cached_no_data = [d for d in _all_dims if d not in cached_result]
                    # Old-cache compatibility: drop the stale _failed_dims / _paused_dims
                    cached_result.pop("_failed_dims", None)
                    cached_result.pop("_paused_dims", None)
                    if _cached_no_data:
                        cached_result["_no_data_dims"] = _cached_no_data

                    # Fetch environment data (it has its own cache, independent of the main cache)
                    if ENVIRONMENT_AVAILABLE:
                        try:
                            import requests as _req
                            _pc_resp = _req.get(
                                f"https://api.postcodes.io/postcodes/{property_data['address'].replace(' ', '')}",
                                timeout=5
                            )
                            if _pc_resp.ok:
                                _pc_data = _pc_resp.json().get("result", {})
                                if _pc_data and _pc_data.get("latitude") and _pc_data.get("longitude"):
                                    _lat, _lng = _pc_data["latitude"], _pc_data["longitude"]
                                    # Fetch environment data in parallel (each has its own cache, usually fast)
                                    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _ac
                                    _env_tasks = {}
                                    with _TPE(max_workers=4) as _env_pool:
                                        if NOISE_AVAILABLE and not cached_result.get("noise"):
                                            _env_tasks[_env_pool.submit(evaluate_noise, _lat, _lng, 8.0)] = "noise"
                                        _env_tasks[_env_pool.submit(evaluate_flood_risk, _lat, _lng, 8.0)] = "flood_risk"
                                        _env_tasks[_env_pool.submit(evaluate_air_quality, _lat, _lng, 8.0)] = "air_quality"
                                        _env_tasks[_env_pool.submit(evaluate_parks, _lat, _lng, 12.0)] = "parks"
                                        for _f in _ac(_env_tasks, timeout=15.0):
                                            _dim = _env_tasks[_f]
                                            try:
                                                _r = _f.result()
                                                if _r:
                                                    if _dim == "noise":
                                                        if _r.get("combined_db") is not None:
                                                            cached_result["noise"] = _r
                                                    else:
                                                        cached_result[_dim] = _r
                                            except Exception:
                                                pass
                                    # Recompute the environment score (using the freshly fetched environment data)
                                    from simple_scorer import SimpleScorer
                                    _scorer = SimpleScorer()
                                    _env_score = _scorer._score_environment(
                                        cached_result.get("noise"),
                                        cached_result.get("flood_risk"),
                                        cached_result.get("air_quality"),
                                        cached_result.get("parks")
                                    )
                                    if _env_score and cached_result.get("scores"):
                                        # Calibrate the environment score (same as the fresh path)
                                        _raw_env = _env_score["score"]
                                        _cal_env, _pct_env = _scorer._calibrate_score(_raw_env, "environment")
                                        _env_score["raw_score"] = _raw_env
                                        _env_score["score"] = _cal_env
                                        _env_score["percentile"] = _pct_env
                                        cached_result["scores"]["environment"] = _env_score
                                        # Recompute total_score (weighted average of the calibrated dimensions)
                                        _weights = _scorer.WEIGHTS
                                        _scores = cached_result["scores"]
                                        _tw = sum(_weights[k] for k in _weights if _scores.get(k) is not None)
                                        if _tw > 0:
                                            _total_raw = sum(
                                                _scores[k]["score"] * (_weights[k] / _tw)
                                                for k in _weights if _scores.get(k) is not None
                                            )
                                            # Apply total PCHIP — without this the total score
                                            # would be raw weighted-avg (σ≈7.6) instead of
                                            # calibrated N(65,15). Bug pre-dating 2026-05-07.
                                            _total_cal, _ = _scorer._calibrate_score(_total_raw, "total")
                                            cached_result["total_score"] = round(_total_cal, 1)
                                            cached_result["rating"] = _scorer._get_rating(round(_total_cal, 1))
                        except Exception:
                            pass  # a failed environment fetch does not affect the main flow

                    # Re-score from the cached dimension DATA so scoring-logic
                    # changes (e.g. the multi-hub commute blend) take effect on
                    # cache hits immediately — the scorer is cheap pure math, only
                    # the upstream API data is cached. Env was refreshed above so
                    # every dimension is current here. Falls back to the cached
                    # scores on any failure.
                    try:
                        from simple_scorer import SimpleScorer as _SS
                        _rs = _SS().calculate_scores({
                            "address": property_data.get("address", ""),
                            "commute": cached_result.get("commute"),
                            "transit": cached_result.get("transit"),
                            "safety": cached_result.get("safety"),
                            "noise": cached_result.get("noise"),
                            "flood_risk": cached_result.get("flood_risk"),
                            "air_quality": cached_result.get("air_quality"),
                            "parks": cached_result.get("parks"),
                            "price_analysis": cached_result.get("price_analysis"),
                            "schools": cached_result.get("schools"),
                            "demographics": cached_result.get("demographics"),
                        })
                        cached_result["total_score"] = _rs["total_score"]
                        cached_result["rating"] = _rs["rating"]
                        cached_result["scores"] = _rs["scores"]
                        cached_result["weights"] = _rs["weights"]
                        cached_result["percentiles"] = _rs.get("percentiles", {})
                        cached_result["adjusted_weights"] = _rs.get("adjusted_weights", {})
                        cached_result["data_completeness"] = _rs.get(
                            "data_completeness", cached_result.get("data_completeness"))
                    except Exception:
                        pass  # keep cached scores on any failure

                    # Restore stdout and print the JSON
                    sys.stdout = original_stdout
                    if args.stream:
                        stream_json_output(cached_result)
                    else:
                        print(json.dumps(cached_result, ensure_ascii=False, indent=2))
                    return

        # Cache miss or cache read skipped: run the evaluation
        # Get coordinates for the schools/environment evaluations (skipped if already fetched during validation)
        if "_api_coords" not in property_data:
            try:
                import requests as _req
                _pc_resp = _req.get(f"https://api.postcodes.io/postcodes/{property_data['address'].replace(' ', '')}", timeout=5)
                if _pc_resp.ok:
                    _pc_data = _pc_resp.json().get("result", {})
                    if _pc_data:
                        property_data["_api_coords"] = {"lat": _pc_data["latitude"], "lng": _pc_data["longitude"]}
            except Exception:
                pass

        # =====================================================================
        # True streaming output: emit each dimension as soon as it completes
        # =====================================================================
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        result_holder = [None]
        env_holder = {"flood": None, "air": None, "parks": None}
        stream_lock = threading.Lock()  # keeps stdout writes ordered

        def stream_dim(dim_name: str, data: dict, score: float = None):
            """Emit one dimension's data immediately (written straight to the original stdout)"""
            if not args.stream:
                return
            line = {"dim": dim_name, "data": data}
            if score is not None:
                line["score"] = score
            with stream_lock:
                # Write straight to the original stdout, bypassing a possibly redirected sys.stdout
                _real_stdout.write(json.dumps(line, ensure_ascii=False) + '\n')
                _real_stdout.flush()

        # Suppress the evaluators' print output (original_stdout was saved before the redirect)
        _real_stdout = original_stdout
        _devnull = open(os.devnull, 'w')

        # Redirect stdout to /dev/null once, before any thread starts
        # Threads must not touch sys.stdout; racing threads would leak prints to the real stdout
        sys.stdout = _devnull

        def run_main_eval():
            """Main evaluation (commute, transit, safety, price_analysis)"""
            property_data["_main_eval_attempted"] = True
            try:
                result_holder[0] = evaluator.evaluate(property_data)
            except Exception:
                pass

        def run_demographics():
            """Fetch demographics separately and emit them immediately"""
            property_data["_demographics_attempted"] = True
            try:
                from area_demographics_evaluator import AreaDemographicsEvaluator
                demographics_evaluator = AreaDemographicsEvaluator()
                demographics_evaluator.evaluate(property_data)
                # Emit demographics immediately
                if "_api_demographics_data" in property_data:
                    demo = property_data["_api_demographics_data"]
                    demo_output = {
                        "postcode": demo.get("postcode", ""),
                        "lsoa_code": demo.get("lsoa_code", ""),
                        "lsoa_name": demo.get("lsoa_name", ""),
                        "local_authority": demo.get("lad_name", ""),
                        "imd_decile": demo.get("imd_decile", 0),
                        "imd_rank": demo.get("imd_rank", 0),
                        "total_lsoas": demo.get("total_lsoas", 33755),
                        "income_decile": demo.get("income_decile", 0),
                        "employment_decile": demo.get("employment_decile", 0),
                        "education_decile": demo.get("education_decile", 0),
                        "crime_decile": demo.get("crime_decile", 0),
                        "health_decile": demo.get("health_decile", 0),
                        "housing_decile": demo.get("housing_decile", 0),
                        "environment_decile": demo.get("environment_decile", 0),
                        "total_population": demo.get("total_population", 0),
                        "children_0_15": demo.get("children_0_15", 0),
                        "population_16_59": demo.get("population_16_59", 0),
                        "population_60_plus": demo.get("population_60_plus", 0),
                        "score": round(demo.get("total_score", 0), 1),
                        "census": _format_census(demo.get("census")),
                    }
                    stream_dim("demographics", demo_output)
            except Exception:
                pass

        def run_schools():
            """Fetch schools separately and emit them immediately"""
            property_data["_schools_attempted"] = True
            try:
                from school_evaluator import SchoolEvaluator
                school_eval = SchoolEvaluator()
                coords = property_data.get("_api_coords") or property_data.get("_api_crime_coords") or property_data.get("_api_transit_coords")
                if coords:
                    score, details = school_eval.score(coords["lat"], coords["lng"])
                    property_data["_api_schools_score"] = score
                    property_data["_api_schools_data"] = details
                    # Build nearest_schools (same as build_json_output)
                    nearest = []
                    for s in details.get("nearby_schools", []):
                        if s.get("ofsted_rating") is not None:
                            label = school_eval.OFSTED_LABELS.get(s["ofsted_rating"], "Unknown")
                            nearest.append({
                                "name": s["name"],
                                "type": s.get("phase", "Unknown"),
                                "category": s.get("category", "state"),
                                "ofsted_rating": label,
                                "distance_km": s["distance_km"],
                            })
                            if len(nearest) >= 10:
                                break
                    # Emit schools immediately
                    schools_output = {
                        "score": round(score, 1),
                        "total_schools": details.get("total_schools", 0),
                        "rated_schools": details.get("rated_schools", 0),
                        "outstanding_count": details.get("outstanding_count", 0),
                        "good_count": details.get("good_count", 0),
                        "ri_count": details.get("ri_count", 0),
                        "inadequate_count": details.get("inadequate_count", 0),
                        "nearest_schools": nearest,
                    }
                    stream_dim("schools", schools_output, round(score, 1))
            except Exception:
                pass

        def run_environment():
            """Fetch environment data in parallel and emit each result as it arrives"""
            if not ENVIRONMENT_AVAILABLE:
                return
            coords = property_data.get("_api_coords")
            if not coords or not coords.get("lat") or not coords.get("lng"):
                return
            lat, lng = coords["lat"], coords["lng"]
            _skip = skip_cache_read  # use skip_cache_read from the enclosing scope

            # Define the fetch functions (avoids lambda closure pitfalls)
            def get_flood():
                return evaluate_flood_risk(lat, lng, timeout=8.0, skip_cache=_skip)
            def get_air():
                return evaluate_air_quality(lat, lng, timeout=8.0, skip_cache=_skip)
            def get_parks():
                return evaluate_parks(lat, lng, timeout=12.0, skip_cache=_skip)

            # Fetch in parallel; emit each one as soon as it completes
            with ThreadPoolExecutor(max_workers=3) as env_pool:
                futures = {
                    env_pool.submit(get_flood): "flood_risk",
                    env_pool.submit(get_air): "air_quality",
                    env_pool.submit(get_parks): "parks",
                }
                streamed_dims = set()
                try:
                    for future in as_completed(futures, timeout=20.0):
                        dim_name = futures[future]
                        try:
                            result = future.result(timeout=1.0)
                            if result:
                                if dim_name == "flood_risk":
                                    env_holder["flood"] = result
                                elif dim_name == "air_quality":
                                    env_holder["air"] = result
                                elif dim_name == "parks":
                                    env_holder["parks"] = result
                                stream_dim(dim_name, result)
                                streamed_dims.add(dim_name)
                        except Exception:
                            pass
                except TimeoutError:
                    pass  # after a timeout, shutdown(wait=True) still waits for every future to finish

                # shutdown(wait=True) has run, so all futures are done
                # Pick up futures not handled before the timeout (e.g. parks, where Overpass is slow but eventually succeeds)
                for future, dim_name in futures.items():
                    if dim_name in streamed_dims:
                        continue
                    try:
                        result = future.result(timeout=0)
                        if result:
                            if dim_name == "flood_risk":
                                env_holder["flood"] = result
                            elif dim_name == "air_quality":
                                env_holder["air"] = result
                            elif dim_name == "parks":
                                env_holder["parks"] = result
                            stream_dim(dim_name, result)
                    except Exception:
                        pass

        def run_noise():
            """Fetch noise separately and emit it immediately"""
            if not NOISE_AVAILABLE:
                return
            coords = property_data.get("_api_coords")
            if not coords or not coords.get("lat") or not coords.get("lng"):
                return
            try:
                noise_result = evaluate_noise(coords["lat"], coords["lng"], timeout=8.0, skip_cache=skip_cache_read)
                if noise_result and noise_result.get("combined_db") is not None:
                    property_data["_api_noise"] = noise_result
                    stream_dim("noise", noise_result)
            except Exception:
                pass

        def run_hub_commute():
            """Hub commute: average commute time to the 5 major hubs (runs in parallel; the data is sent with the summary)"""
            try:
                pc = property_data.get("address", "").replace(" ", "")
                if not pc:
                    return
                lc = evaluator.location_evaluator
                if not lc.tfl_api.is_london_postcode(pc):
                    return
                result = lc.evaluate_hub_commute(pc)
                if result:
                    property_data["_api_hub_commute"] = result
            except Exception:
                pass

        # Start all tasks in parallel (including hub_commute)
        with ThreadPoolExecutor(max_workers=6) as pool:
            main_futures = [
                pool.submit(run_main_eval),
                pool.submit(run_demographics),
                pool.submit(run_schools),
                pool.submit(run_environment),
                pool.submit(run_noise),
            ]
            hub_future = pool.submit(run_hub_commute)
            # Wait for the main tasks to finish
            for f in main_futures:
                try:
                    f.result(timeout=120)
                except Exception:
                    pass
            # hub_commute gets a shorter timeout (so it does not slow down the whole run)
            try:
                hub_future.result(timeout=15)
            except Exception:
                pass

        _devnull.close()

        result = result_holder[0]
        property_data["_eval_completed"] = result is not None
        property_data["_prefetched_env"] = env_holder
        output = build_json_output(property_data, result, skip_cache=skip_cache_read)

        # Ranking info (read-only, so unaffected even if the database is locked)
        try:
            if CACHE_AVAILABLE and cache_service:
                ranking_info = cache_service.get_ranking(output, args.ranking_period)
                output["_cache"] = {
                    "hit": False,
                    "cached_at": None,
                    "expires_at": None,
                    "data_version": CACHE_VERSION,
                    "schema_version": SCHEMA_VERSION,
                }
                output["_ranking"] = {
                    "period": ranking_info.period,
                    "period_days": ranking_info.period_days,
                    "total_evaluated": ranking_info.total_evaluated,
                    "percentiles": ranking_info.percentiles,
                    "ranks": ranking_info.ranks,
                }
        except Exception:
            pass

        # Send the output first, then save the cache (best effort)
        sys.stdout = original_stdout
        if args.stream:
            already_streamed = {"demographics", "schools", "noise", "flood_risk", "air_quality", "parks"}
            stream_json_output(output, already_streamed)
            try:
                # Do not cache if any dimension's API failed; the next request will retry
                if CACHE_AVAILABLE and cache_service and not output.get("_failed_dims"):
                    destination = args.destination or "EC2R 8AH"
                    cache_output = {k: v for k, v in output.items() if k not in ("_failed_dims", "_paused_dims")}
                    cache_service.save_result(
                        postcode=property_data["address"],
                        destination=destination,
                        result=cache_output
                    )
            except Exception:
                pass
        else:
            try:
                # Do not cache if any dimension's API failed; the next request will retry
                if CACHE_AVAILABLE and cache_service and not output.get("_failed_dims"):
                    destination = args.destination or "EC2R 8AH"
                    cache_output = {k: v for k, v in output.items() if k not in ("_failed_dims", "_paused_dims")}
                    cache_service.save_result(
                        postcode=property_data["address"],
                        destination=destination,
                        result=cache_output
                    )
            except Exception:
                pass
            print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    result = evaluator.evaluate(property_data)

    # Print the report
    evaluator.print_report(result)

    # Print the API data details
    print("\n" + "="*60)
    print("API data details")
    print("="*60)

    has_api_data = False

    # 1. Commute (if API data is available)
    if "_api_commute_time_minutes" in property_data:
        has_api_data = True
        api_source = property_data.get("_api_commute_source", "google_routes")
        api_name = "TfL Journey Planner" if api_source == "tfl" else "Google Routes API"
        print(f"\n🚇 Commute ({api_name})")
        print(f"   Commute time: {property_data['_api_commute_time_minutes']:.1f} min")

        # Distance (not provided by the TfL API)
        if "_api_commute_distance_km" in property_data:
            print(f"   Commute distance: {property_data['_api_commute_distance_km']:.2f} km")

        # Show the commute details for both directions
        if "_api_commute_details" in property_data:
            details = property_data["_api_commute_details"]
            if "morning" in details:
                morning = details["morning"]
                dist_str = f", {morning['distance_km']:.2f} km" if morning.get('distance_km') else ""
                print(f"   Morning commute: {morning['time_minutes']:.1f} min{dist_str}")
            if "evening" in details:
                evening = details["evening"]
                dist_str = f", {evening['distance_km']:.2f} km" if evening.get('distance_km') else ""
                print(f"   Evening commute: {evening['time_minutes']:.1f} min{dist_str}")

    # 2. Safety (if API data is available)
    if "_api_crime_data" in property_data:
        has_api_data = True
        crime_data = property_data["_api_crime_data"]
        coords = property_data.get("_api_crime_coords", {})

        print("\n🛡️  Safety (UK Police API)")
        print(f"   Coordinates: {coords.get('lat', 0):.6f}, {coords.get('lng', 0):.6f}")
        print(f"   Total: {crime_data.get('total', 0)} crimes")

        # Show the main crime types
        crime_types = {
            "anti-social-behaviour": "Anti-social behaviour",
            "burglary": "Burglary",
            "violent-crime": "Violent crime",
            "robbery": "Robbery",
            "theft-from-the-person": "Theft from the person",
            "drugs": "Drugs",
            "possession-of-weapons": "Weapons possession"
        }

        major_crimes = []
        for crime_type, crime_name in crime_types.items():
            count = crime_data.get(crime_type, 0)
            if count > 0:
                major_crimes.append(f"{crime_name}: {count} crimes")

        if major_crimes:
            print("   Main types:")
            for crime_info in major_crimes[:5]:  # show the top 5
                print(f"     - {crime_info}")

    # 3. Nearby amenities (if API data is available)
    if "_api_amenities_data" in property_data:
        has_api_data = True
        amenities_data = property_data["_api_amenities_data"]
        coords = property_data.get("_api_amenities_coords", property_data.get("_api_crime_coords", {}))

        print("\n🏪 Nearby amenities (Google Places API)")
        if coords:
            print(f"   Coordinates: {coords.get('lat', 0):.6f}, {coords.get('lng', 0):.6f}")
        print(f"   Places: {amenities_data.get('total', 0)} total")
        print(f"   Categories: {amenities_data.get('categories_count', 0)} distinct")
        print(f"   Average rating: {amenities_data.get('average_rating', 0):.2f}⭐")
        print(f"   Highly rated (≥4.0): {amenities_data.get('high_rated', 0)} places")
        print(f"   Popular (≥100 reviews): {amenities_data.get('popular', 0)} places")

        # Show the main amenity categories
        by_category = amenities_data.get("by_category", {})
        if by_category:
            print("   Main amenities:")
            # Sort by count and show the top 5
            sorted_categories = sorted(
                by_category.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )[:5]

            for category, data in sorted_categories:
                count = data["count"]
                avg_rating = data["avg_rating"]
                rating_str = f"(avg {avg_rating:.1f}⭐)" if avg_rating > 0 else ""
                print(f"     - {category}: {count} {rating_str}")

                # Show sample places in this category
                places = data.get("places", [])[:2]  # show the first 2
                for place in places:
                    name = place["name"]
                    rating = place["rating"]
                    reviews = place["review_count"]
                    if rating > 0:
                        print(f"       · {name} ({rating:.1f}⭐, {reviews} reviews)")

    # 4. Public transport access (if API data is available)
    if "_api_transit_data" in property_data:
        has_api_data = True
        transit_data = property_data["_api_transit_data"]
        coords = property_data.get("_api_transit_coords", property_data.get("_api_crime_coords", {}))

        api_source = transit_data.get("api_source", "google_places")
        api_name = "TfL API" if api_source == "tfl" else "Google Places API"
        print(f"\n🚇 Public transport access ({api_name})")

        # Show the reachable lines (the key info)
        lines = transit_data.get("lines", [])
        lines_count = transit_data.get("lines_count", len(lines))
        total_weight = transit_data.get("total_line_weight", 0)

        print(f"   Reachable: {lines_count} lines")
        if lines:
            print(f"   Lines: {', '.join(lines)}")
        print(f"   Line weight: {total_weight} pts")
        print(f"   Overall score: {transit_data.get('tube_score', 0):.1f}/100")

        # Show the nearest stations (short version)
        tube_details = property_data.get("_api_transit_tube_details", [])
        if tube_details:
            print("   Nearest stations:")
            for i, tube in enumerate(tube_details[:3], 1):
                walk_time = tube.get('walk_time', 0)
                print(f"     {i}. {tube.get('name', 'Unknown')} ({walk_time:.0f} min walk)")

    # 5. Long-distance travel access (if API data is available)
    if "_api_long_distance_travel_data" in property_data:
        has_api_data = True
        ld_data = property_data["_api_long_distance_travel_data"]
        coords = property_data.get("_api_long_distance_coords", {})

        print("\n🛫 Long-distance travel access (Google Places API + Routes API)")
        if coords:
            print(f"   Coordinates: {coords.get('lat', 0):.6f}, {coords.get('lng', 0):.6f}")
        print(f"   Airports: {len(ld_data.get('airports', []))} found")
        print(f"   Train stations: {len(ld_data.get('trains', []))} found")
        print(f"   Airport score: {ld_data.get('airport_score', 0):.1f}/100 (weight 60%)")
        print(f"   Train station score: {ld_data.get('train_score', 0):.1f}/100 (weight 40%)")

        # Show the most important airports
        airports = ld_data.get("airports", [])
        if airports:
            print("   Main airports:")
            for i, airport in enumerate(airports[:3], 1):  # show the top 3
                best_mode = airport.get("best_mode", "transit")

                # Get the transit and drive info
                transit_info = airport.get("transit", {})
                drive_info = airport.get("drive", {})

                print(f"     {i}. {airport['name']}")

                # Show public transport info
                transit_time = transit_info.get("duration_minutes", 0)
                transit_dist = transit_info.get("distance_km", 0)
                transit_score = transit_info.get("score", 0)
                if transit_time and transit_time < 9999:
                    best_mark = " ⭐" if best_mode == "transit" else ""
                    print(f"        🚇 Public transport: {transit_time:.0f} min ({transit_dist:.1f} km) - {transit_score:.1f} pts{best_mark}")
                else:
                    print(f"        🚇 Public transport: unavailable")

                # Show driving info
                drive_time = drive_info.get("duration_minutes", 0)
                drive_dist = drive_info.get("distance_km", 0)
                drive_score = drive_info.get("score", 0)
                if drive_time and drive_time < 9999:
                    best_mark = " ⭐" if best_mode == "drive" else ""
                    print(f"        🚗 Drive: {drive_time:.0f} min ({drive_dist:.1f} km) - {drive_score:.1f} pts{best_mark}")
                else:
                    print(f"        🚗 Drive: unavailable")

                print(f"        Importance: {airport['importance_coefficient']:.2f}")

        # Show the main train stations
        trains = ld_data.get("trains", [])
        if trains:
            print("   Main train stations:")
            for i, train in enumerate(trains[:3], 1):  # show the top 3
                best_mode = train.get("best_mode", "transit")

                # Get the transit and drive info
                transit_info = train.get("transit", {})
                drive_info = train.get("drive", {})

                print(f"     {i}. {train['name']}")

                # Show public transport info
                transit_time = transit_info.get("duration_minutes", 0)
                transit_dist = transit_info.get("distance_km", 0)
                transit_score = transit_info.get("score", 0)
                if transit_time and transit_time < 9999:
                    best_mark = " ⭐" if best_mode == "transit" else ""
                    print(f"        🚇 Public transport: {transit_time:.0f} min ({transit_dist:.1f} km) - {transit_score:.1f} pts{best_mark}")
                else:
                    print(f"        🚇 Public transport: unavailable")

                # Show driving info
                drive_time = drive_info.get("duration_minutes", 0)
                drive_dist = drive_info.get("distance_km", 0)
                drive_score = drive_info.get("score", 0)
                if drive_time and drive_time < 9999:
                    best_mark = " ⭐" if best_mode == "drive" else ""
                    print(f"        🚗 Drive: {drive_time:.0f} min ({drive_dist:.1f} km) - {drive_score:.1f} pts{best_mark}")
                else:
                    print(f"        🚗 Drive: unavailable")

                print(f"        Importance: {train['importance_coefficient']:.2f}")

    # 6. Price analysis (if API data is available)
    if "_api_price_analysis" in property_data:
        has_api_data = True
        price_data = property_data["_api_price_analysis"]
        summary = price_data.get("summary", {})
        trends = price_data.get("trends", {})
        target = price_data.get("target_comparison")

        print("\n💰 Price analysis (Land Registry + EPC API)")
        print(f"   Postcode: {price_data.get('postcode', 'N/A')}")
        print(f"   Transactions: {summary.get('total_transactions', 0)} (last {price_data.get('years_analyzed', 5)} years)")
        print(f"   Area average price: £{summary.get('average_price', 0):,}")
        print(f"   Area median price: £{summary.get('median_price', 0):,}")

        if summary.get('price_per_sqm'):
            print(f"   Average price per m²: £{summary.get('price_per_sqm'):,}/m²")

        # Price trend (based on repeat sales)
        direction_map = {"up": "↗️ Rising", "down": "↘️ Falling", "stable": "➡️ Stable"}
        direction = direction_map.get(trends.get("direction", "unknown"), "❓ Unknown")
        print(f"   Trend: {direction}")

        sample_count = trends.get("sample_count", 0)
        if sample_count > 0:
            if trends.get("5_year_cagr") is not None:
                print(f"   5-year average annualized: {trends['5_year_cagr']:+.2f}%/yr ({trends.get('5_year_count', 0)} sales)")
            if trends.get("3_year_cagr") is not None:
                print(f"   3-year average annualized: {trends['3_year_cagr']:+.2f}%/yr ({trends.get('3_year_count', 0)} sales)")

        # Target price comparison
        if target:
            print(f"\n   🎯 Target price analysis: £{target['target_price']:,}")
            vs_avg = target['vs_average']
            vs_med = target['vs_median']
            vs_avg_str = f"+{vs_avg:.1f}%" if vs_avg > 0 else f"{vs_avg:.1f}%"
            vs_med_str = f"+{vs_med:.1f}%" if vs_med > 0 else f"{vs_med:.1f}%"
            print(f"      vs. average: {vs_avg_str}")
            print(f"      vs. median: {vs_med_str}")
            print(f"      Price rank: {target['percentile']:.0f} (percentile)")
            print(f"      Assessment: {target['assessment']}")

    if not has_api_data:
        print("\n⚠️  No API data used; the evaluation is based on default values")
        print("   Configure the Google APIs for more accurate results")

    quota_apis = get_quota_exceeded_apis()
    if quota_apis:
        paused = set()
        if "google_routes" in quota_apis:
            paused.update(["commute", "long_distance"])
        if "google_places" in quota_apis:
            paused.update(["transit", "amenities", "long_distance"])
        if "google_geocoding" in quota_apis:
            paused.update(["commute", "transit", "amenities", "long_distance"])
        if paused:
            paused_list = ", ".join(sorted(paused))
            print(f"\n⏸️  APIs paused: {paused_list}")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
