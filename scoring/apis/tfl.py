#!/usr/bin/env python3
"""Transport for London (TfL) API wrapper"""

import os
import re
import math
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from apis.api_usage import record_api_usage

# Populate TFL_APP_KEY (and friends) from the repo .env for any caller that
# didn't already load it. Best-effort: no dotenv / no .env just means we fall
# back to unauthenticated (still works, slower).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass


# Default transport modes for the TfL journey planner. national-rail must be in here — without it
# the planner is barred from National Rail and can only cobble an answer together from buses: measured
# BR5 3AA → London Bridge degraded from 62 min (Southeastern direct) to 95 min (three buses in a row
# via North Greenwich), and Orpington → London Bridge came out at 101 min. Commute times across the whole
# outer ring (BR/DA/KT/CR/SM/EN/WD) get systematically inflated to the point of looking "unlivable".
#
# This is the single source of truth: scripts/build_sector_hub_commute.py imports it when pre-warming the
# sector×hub table, so the two cannot drift apart again (this bug was exactly that: the default was missing
# it while both deliberately built paths had it). commute_verify.RAILWALK_MODE is a **different definition**
# (walking + rail, no bus), intentionally different — do not merge it in.
DEFAULT_JOURNEY_MODE = "tube,dlr,overground,elizabeth-line,national-rail,bus,walking"


class TfLAPI:
    """Transport for London (TfL) API client wrapper"""

    # The default "python-requests/X.Y" User-Agent is rejected by the API's edge
    # (HTTP 403 / Cloudflare 1010), so every request sends an identifying UA that
    # names the project and a contact address.
    _UA = "wheretolive.xyz/1.0 (+https://wheretolive.xyz; contact@wheretolive.xyz)"

    def __init__(self, app_key: Optional[str] = None):
        """Initialize the TfL API client.

        app_key: the free "500 Requests per min" subscription key. Raises the
        rate limit from ~50 to 500 req/min. Falls back to the TFL_APP_KEY env
        var (.env). None everywhere → unauthenticated (works, just slower).
        """
        self.endpoint = "https://api.tfl.gov.uk"
        self.app_key = app_key or os.environ.get("TFL_APP_KEY") or None

    def _params(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merge the app_key into a request's query params (no-op if unset)."""
        p = dict(params or {})
        if self.app_key:
            p["app_key"] = self.app_key
        return p

    def _headers(self) -> Dict[str, str]:
        return {"User-Agent": self._UA}

    def get_station_lines(
        self,
        station_name: str,
        max_retries: int = 2,
        retry_delay: float = 1.0
    ) -> Optional[List[str]]:
        """
        Get the lines serving a station

        Args:
            station_name: station name (e.g. "King's Cross St. Pancras")
            max_retries: maximum number of retries
            retry_delay: retry delay (seconds)

        Returns:
            List of lines, or None on failure
            e.g.: ["central", "northern", "piccadilly"]
        """
        # Clean up the station name
        # 1. Strip special characters
        clean_name = station_name.replace("'", "").replace(".", "").lower().strip()

        # 2. Strip common suffixes (Station, Underground Station, Tube Station, etc.)
        suffixes_to_remove = [
            r'\s+underground\s+station$',
            r'\s+tube\s+station$',
            r'\s+station$',
            r'\s+dlr\s+station$',
            r'\s+rail\s+station$'
        ]

        for suffix in suffixes_to_remove:
            clean_name = re.sub(suffix, '', clean_name, flags=re.IGNORECASE).strip()

        # Search for the station via the TfL API
        # Covers tube (Underground), dlr (DLR light rail), overground (Overground)
        url = f"{self.endpoint}/StopPoint/Search"
        params = {
            "query": clean_name,
            "modes": "tube,dlr,overground,elizabeth-line"
        }

        last_error = None

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    url,
                    params=self._params(params),
                    headers=self._headers(),
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    matches = data.get("matches", [])

                    if not matches:
                        return None

                    # Find the best-matching station
                    best_match = None
                    for match in matches:
                        match_name = match.get("name", "").lower()
                        # Simple matching logic: contains the main keywords
                        if self._is_station_match(clean_name, match_name):
                            best_match = match
                            break

                    if not best_match and matches:
                        best_match = matches[0]

                    if best_match:
                        station_id = best_match.get("id", "")
                        # Fetch station details
                        record_api_usage("tfl")
                        return self._get_station_details(station_id)

                    return None

                elif response.status_code == 429:
                    last_error = "TfL API rate limited"
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1) * 2)
                        continue

                elif response.status_code >= 500:
                    last_error = f"TfL API server error ({response.status_code})"
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue

                else:
                    return None

            except requests.exceptions.Timeout:
                last_error = "TfL API request timed out"
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue

            except Exception as e:
                last_error = f"TfL API request failed: {str(e)[:100]}"
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue

        return None

    def _is_station_match(self, query: str, station_name: str) -> bool:
        """Check whether a station name matches"""
        # Extract the main keywords
        query_words = set(query.split())
        station_words = set(station_name.split())

        # At least one main keyword matches
        common_words = query_words & station_words
        if len(common_words) >= 1:
            return True

        # Or the query is a substring of the station name
        if query in station_name:
            return True

        return False

    def _get_station_details(self, station_id: str) -> Optional[List[str]]:
        """Fetch station details and extract the list of lines"""
        url = f"{self.endpoint}/StopPoint/{station_id}"

        try:
            response = requests.get(url, params=self._params(), headers=self._headers(), timeout=10)

            if response.status_code == 200:
                data = response.json()
                lines = []

                # Extract lines from lineModeGroups
                # Covers tube, dlr, overground, elizabeth-line
                line_groups = data.get("lineModeGroups", [])
                for group in line_groups:
                    mode_name = group.get("modeName")
                    if mode_name in ["tube", "dlr", "overground", "elizabeth-line"]:
                        line_identifiers = group.get("lineIdentifier", [])
                        lines.extend(line_identifiers)

                # If nothing was found above, try extracting from the lines field
                if not lines:
                    lines_data = data.get("lines", [])
                    for line in lines_data:
                        mode_name = line.get("modeName")
                        if mode_name in ["tube", "dlr", "overground", "elizabeth-line"]:
                            line_id = line.get("id", "")
                            if line_id:
                                lines.append(line_id)

                return lines if lines else None

            return None

        except Exception:
            return None

    # TfL's StopPoint station-name index and the Journey Planner's free-text disambiguator are
    # **two different indexes**. The latter matches "Maidenhead" to "West Ham,
    # MAIDEN ROAD" in East London; the former returns
    # 910GMDNHEAD for the same word, i.e. the real station. It is also a good discriminator:
    # for non-station inputs such as UCL / Soho / a street address / "my office" it returns nothing at all.
    STATION_SEARCH_MODES = "tube,dlr,overground,elizabeth-line,national-rail,tram"

    def search_stop_point(self, query: str, modes: Optional[str] = None) -> List[Dict[str, Any]]:
        """Look up a name in TfL's own stop-point index. Returns matches (possibly empty); raises on
        failure, leaving the fallback strategy to the caller."""
        url = f"{self.endpoint}/StopPoint/Search"
        r = requests.get(
            url,
            params=self._params({"query": query, "modes": modes or self.STATION_SEARCH_MODES}),
            headers=self._headers(),
            timeout=8,
        )
        r.raise_for_status()
        return r.json().get("matches") or []

    def stop_point_children(self, stop_id: str) -> List[Dict[str, Any]]:
        """Children of an aggregate stop (HUB*). A HUB id itself cannot be used as a journey endpoint — measured:
        passing "HUBLST" to the Journey Planner makes it treat the id as free text and match Hillingdon's
        "The Skills Hub"."""
        url = f"{self.endpoint}/StopPoint/{stop_id}"
        r = requests.get(url, params=self._params(), headers=self._headers(), timeout=8)
        r.raise_for_status()
        return r.json().get("children") or []

    def get_nearby_stations(
        self,
        lat: float,
        lng: float,
        radius: int = 800,
        modes: str = "tube,dlr,overground,elizabeth-line"
    ) -> List[Dict[str, Any]]:
        """
        Search for public transport stations near a given location

        Args:
            lat: latitude
            lng: longitude
            radius: search radius (m), default 800 m
            modes: transport modes, comma-separated

        Returns:
            List of stations, each with name, lat, lng, distance, modes, lines
        """
        url = f"{self.endpoint}/StopPoint"
        params = {
            "lat": lat,
            "lon": lng,
            "radius": radius,
            "stopTypes": "NaptanMetroStation,NaptanRailStation",
            "modes": modes
        }

        # Retry strategy: 4 attempts with exponential backoff (3s → 6s → 12s
        # base; doubled for 429). Total span up to ~42s for rate-limit, ~21s
        # for 5xx/network. Previously 2 attempts × 1.5s = 3s total, which
        # couldn't bridge TfL's recurring 5-15min morning rate-limit windows.
        MAX_RETRIES = 4
        BACKOFF_BASE = 3.0

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, params=self._params(params), headers=self._headers(), timeout=15)

                if response.status_code == 200:
                    data = response.json()
                    stop_points = data.get("stopPoints", [])

                    stations = []
                    for sp in stop_points:
                        station_lat = sp.get("lat", 0)
                        station_lng = sp.get("lon", 0)

                        # Compute the distance
                        distance = self._haversine_distance(lat, lng, station_lat, station_lng)

                        # Extract line info
                        lines = []
                        line_groups = sp.get("lineModeGroups", [])
                        for group in line_groups:
                            mode_name = group.get("modeName", "")
                            if mode_name in ["tube", "dlr", "overground", "elizabeth-line", "national-rail"]:
                                line_identifiers = group.get("lineIdentifier", [])
                                if mode_name == "national-rail":
                                    # Treat National Rail as a single line
                                    if "National Rail" not in lines:
                                        lines.append("National Rail")
                                else:
                                    lines.extend(line_identifiers)

                        # If there is no line info, try the lines field
                        if not lines:
                            lines_data = sp.get("lines", [])
                            for line in lines_data:
                                line_id = line.get("id", "")
                                if line_id:
                                    lines.append(line_id)

                        stations.append({
                            "name": sp.get("commonName", sp.get("name", "Unknown")),
                            "lat": station_lat,
                            "lng": station_lng,
                            "distance": distance,
                            "modes": sp.get("modes", []),
                            "lines": lines,
                            "stop_type": sp.get("stopType", ""),
                            "naptan_id": sp.get("naptanId", "")
                        })

                    # Sort by distance
                    stations.sort(key=lambda x: x["distance"])
                    record_api_usage("tfl")
                    return stations

                # Non-200: classify and decide whether to retry.
                is_rate_limit = response.status_code == 429
                is_server_err = response.status_code >= 500
                if not (is_rate_limit or is_server_err):
                    # 4xx other than 429 = bad request / not-found. Don't retry.
                    return []

                if attempt == MAX_RETRIES - 1:
                    print(
                        f"   ⚠️  TfL get_nearby_stations gave up after "
                        f"{MAX_RETRIES} retries (status={response.status_code})"
                    )
                    return []

                wait = BACKOFF_BASE * (2 ** attempt)
                if is_rate_limit:
                    wait *= 2  # back off harder on 429 to let bucket refill
                time.sleep(wait)

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"   ⚠️  TfL get_nearby_stations network error: {str(e)[:50]}")
                    return []
                time.sleep(BACKOFF_BASE * (2 ** attempt))

            except Exception as e:
                # Unexpected error (e.g. JSON parse) — fail fast, don't retry.
                print(f"   ⚠️  TfL get_nearby_stations unexpected error: {str(e)[:50]}")
                return []

        return []

    def _haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Distance between two points (m)"""
        R = 6371000  # Earth radius (m)

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def get_journey(
        self,
        from_location: str,
        to_location: str,
        time_is: str = "Departing",
        mode: str = DEFAULT_JOURNEY_MODE,
        departure_time: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Plan a journey between two points

        Args:
            from_location: origin (a postcode, "lat,lng" coordinates, or an address)
            to_location: destination (a postcode, "lat,lng" coordinates, or an address)
            time_is: "Departing" or "Arriving"
            mode: transport modes
            departure_time: departure time in ISO format, e.g. "2026-01-30T09:00:00"; if not given, 9am on the next working day is used

        Returns:
            Journey info dict with duration_minutes, legs, summary
        """
        def _resolve_disambiguation(
            data: Dict[str, Any], cur_from: str, cur_to: str
        ) -> Optional[tuple]:
            """When TfL returns HTTP 300 (disambiguation), pick the candidate parameterValue with the
            highest matchQuality as the resolved value for that end; an end that is already identified keeps its original value.
            Returns (new_from, new_to) for the retry, or None if unresolvable (empty / no candidates)."""
            def pick(key: str, original: str) -> Optional[str]:
                dis = data.get(key) or {}
                if dis.get("matchStatus") == "identified":
                    return original  # TfL has already resolved this end; use it as-is
                opts = dis.get("disambiguationOptions") or []
                if not opts:
                    return None  # empty / no candidates → unresolvable
                best = max(opts, key=lambda o: o.get("matchQuality", 0))
                return best.get("parameterValue") or None

            new_from = pick("fromLocationDisambiguation", cur_from)
            new_to = pick("toLocationDisambiguation", cur_to)
            if new_from is None or new_to is None:
                return None
            return new_from, new_to

        # If no time is given, use 9am on the next working day
        if departure_time:
            dt = datetime.fromisoformat(departure_time.replace("Z", ""))
        else:
            now = datetime.now()
            # Find the next working day
            days_ahead = 0
            check_date = now
            while check_date.weekday() >= 5:  # 5=Sat, 6=Sun
                check_date += timedelta(days=1)
                days_ahead += 1
            if days_ahead == 0 and now.hour >= 10:  # today is a working day but the morning peak has passed
                check_date += timedelta(days=1)
                while check_date.weekday() >= 5:
                    check_date += timedelta(days=1)
            dt = check_date.replace(hour=9, minute=0, second=0, microsecond=0)

        # TfL API date format: yyyyMMdd, time format: HHmm
        date_str = dt.strftime("%Y%m%d")
        time_str = dt.strftime("%H%M")

        params = {
            "mode": mode,
            "timeIs": time_is,
            "journeyPreference": "LeastTime",
            "date": date_str,
            "time": time_str
        }

        # Origin/destination currently being resolved (may be replaced by a parameterValue after a 300 disambiguation).
        cur_from, cur_to = from_location, to_location
        # range(3): initial request + one disambiguation retry + one retry allowance for network errors (429/5xx).
        for attempt in range(3):
            try:
                url = f"{self.endpoint}/Journey/JourneyResults/{cur_from}/to/{cur_to}"
                response = requests.get(url, params=self._params(params), headers=self._headers(), timeout=20)

                if response.status_code == 200:
                    data = response.json()
                    journeys = data.get("journeys", [])

                    if not journeys:
                        return None

                    # Pick the fastest journey
                    best_journey = min(journeys, key=lambda j: j.get("duration", 9999))
                    duration = best_journey.get("duration", 0)

                    # Parse the journey legs
                    legs = []
                    for leg in best_journey.get("legs", []):
                        leg_info = {
                            "mode": leg.get("mode", {}).get("name", "Unknown"),
                            "duration": leg.get("duration", 0),
                            "from": leg.get("departurePoint", {}).get("commonName", ""),
                            "to": leg.get("arrivalPoint", {}).get("commonName", ""),
                            "instruction": leg.get("instruction", {}).get("summary", "")
                        }

                        # For public transport, add the line info
                        route_options = leg.get("routeOptions", [])
                        if route_options:
                            leg_info["line"] = route_options[0].get("name", "")

                        legs.append(leg_info)

                    # Extract fare information
                    fare_info = None
                    fare_data = best_journey.get("fare")
                    if fare_data:
                        total_cost = fare_data.get("totalCost", 0)
                        fares = fare_data.get("fares", [])

                        total_peak = 0
                        total_off_peak = 0
                        low_zone = None
                        high_zone = None

                        for fare in fares:
                            peak = fare.get("peak", 0)
                            off_peak = fare.get("offPeak", 0)
                            cost = fare.get("cost", 0)

                            if peak == 0 and off_peak == 0:
                                total_peak += cost
                                total_off_peak += cost
                            else:
                                total_peak += peak
                                total_off_peak += off_peak

                            fare_low = fare.get("lowZone", 0)
                            fare_high = fare.get("highZone", 0)
                            if fare_low > 0:
                                if low_zone is None or fare_low < low_zone:
                                    low_zone = fare_low
                                if high_zone is None or fare_high > high_zone:
                                    high_zone = fare_high

                        fare_info = {
                            "total": total_cost / 100,
                            "peak": total_peak / 100,
                            "off_peak": total_off_peak / 100,
                            "low_zone": low_zone,
                            "high_zone": high_zone,
                            "charge_level": "Peak" if total_peak > total_off_peak else "",
                            "is_hopper_fare": any(f.get("isHopperFare", False) for f in fares)
                        }

                    record_api_usage("tfl")
                    return {
                        "duration_minutes": duration,
                        "legs": legs,
                        "summary": f"{duration} min",
                        "fare": fare_info
                    }

                elif response.status_code == 300:
                    # Disambiguation: retry once with the candidate parameterValue that has the highest matchQuality.
                    try:
                        resolved = _resolve_disambiguation(response.json(), cur_from, cur_to)
                    except Exception:  # noqa: BLE001
                        resolved = None
                    if resolved and resolved != (cur_from, cur_to):
                        cur_from, cur_to = resolved
                        continue  # retry immediately with the resolved IDs, no sleep
                    return None  # unresolvable (empty / no candidates / still 300 after resolving)

                # 429/5xx → retry
                if attempt < 2:
                    time.sleep(1.5)

            except Exception as e:
                if attempt == 0:
                    time.sleep(1.5)
                else:
                    print(f"   ⚠️  TfL Journey API error: {str(e)[:50]}")

        return None

    def is_london_postcode(self, postcode: str) -> bool:
        """
        Check whether a postcode is in the London area

        Args:
            postcode: UK postcode

        Returns:
            Whether it is a London postcode
        """
        # Normalize the postcode
        postcode = postcode.upper().replace(" ", "")

        # London postcode prefixes
        london_prefixes = [
            # Central London
            "EC", "WC",
            # East London
            "E",
            # North London
            "N", "NW",
            # South London
            "SE", "SW",
            # West London
            "W",
            # Outer Greater London
            "BR", "CR", "DA", "EN", "HA", "IG", "KT",
            "RM", "SM", "TW", "UB", "WD"
        ]

        for prefix in london_prefixes:
            if postcode.startswith(prefix):
                # Make sure the prefix is followed by a digit
                remaining = postcode[len(prefix):]
                if remaining and remaining[0].isdigit():
                    return True

        return False
