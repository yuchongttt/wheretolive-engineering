#!/usr/bin/env python3
"""UK Police API wrapper"""

import json
import math
import time
import subprocess
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List

from apis.api_usage import record_api_usage


class UKPoliceAPI:
    """UK Police API client wrapper"""

    def __init__(self):
        """
        Initialize the UK Police API client
        This API does not require an API key
        """
        self.endpoint = "https://data.police.uk/api"
        # Cached across calls within one evaluation run.
        self._publishing_forces_cache: Optional[set] = None
        self._force_names_cache: Optional[Dict[str, str]] = None

    def get_publishing_forces(self, lookback_months: int = 6) -> set:
        """
        Forces that appear in the Police UK publication list in the last
        N months. Used to distinguish "force has stopped publishing"
        (e.g. Greater Manchester Police since July 2019) from
        "genuinely zero crimes in this area".

        We read the stop-and-search force list because it closely tracks
        overall publishing status and is one simple list per month.
        """
        if self._publishing_forces_cache is not None:
            return self._publishing_forces_cache
        publishing: set = set()
        try:
            resp = requests.get(
                f"{self.endpoint}/crimes-street-dates",
                timeout=10,
                headers={"User-Agent": "wheretolive.xyz/1.0"},
            )
            if resp.status_code == 200:
                data = resp.json()
                for entry in data[:lookback_months]:
                    for fid in entry.get("stop-and-search", []):
                        publishing.add(fid)
        except Exception as e:
            print(f"  ⚠ get_publishing_forces failed: {e}")
            return set()
        self._publishing_forces_cache = publishing
        return publishing

    def get_force_for_location(self, lat: float, lng: float) -> Optional[Dict[str, str]]:
        """
        Resolve lat/lng → {id, name} of the police force that covers it.
        Returns None on lookup failure.
        """
        try:
            resp = requests.get(
                f"{self.endpoint}/locate-neighbourhood",
                params={"q": f"{lat},{lng}"},
                timeout=8,
                headers={"User-Agent": "wheretolive.xyz/1.0"},
            )
            if resp.status_code != 200:
                return None
            force_id = resp.json().get("force")
            if not force_id:
                return None
            if self._force_names_cache is None:
                forces_resp = requests.get(
                    f"{self.endpoint}/forces",
                    timeout=8,
                    headers={"User-Agent": "wheretolive.xyz/1.0"},
                )
                if forces_resp.status_code == 200:
                    self._force_names_cache = {f["id"]: f["name"] for f in forces_resp.json()}
                else:
                    self._force_names_cache = {}
            return {
                "id": force_id,
                "name": self._force_names_cache.get(force_id, force_id),
            }
        except Exception as e:
            print(f"  ⚠ get_force_for_location failed: {e}")
            return None

    def is_force_publishing(self, lat: float, lng: float) -> Dict[str, Any]:
        """
        Convenience: is the force covering (lat,lng) currently publishing?
        Returns {"publishing": bool | None, "force_id": str, "force_name": str}.
        `publishing=None` means we couldn't determine — callers should
        trust the crime count rather than mark data unavailable.
        """
        force = self.get_force_for_location(lat, lng)
        if not force:
            return {"publishing": None, "force_id": None, "force_name": None}
        publishing_set = self.get_publishing_forces()
        if not publishing_set:
            return {"publishing": None, "force_id": force["id"], "force_name": force["name"]}
        return {
            "publishing": force["id"] in publishing_set,
            "force_id": force["id"],
            "force_name": force["name"],
        }

    def _calculate_square_polygon(
        self,
        lat: float,
        lng: float,
        side_length_km: float = 1.0
    ) -> str:
        """
        Compute a square polygon centred on the given point

        Args:
            lat: latitude of the centre point
            lng: longitude of the centre point
            side_length_km: side length of the square (km), default 1 km

        Returns:
            Polygon string, format: "lat1,lng1:lat2,lng2:lat3,lng3:lat4,lng4"
        """
        # Earth radius (km)
        EARTH_RADIUS_KM = 6371.0

        # Half the side length
        half_side = side_length_km / 2.0

        # Latitude: 1 degree ≈ 111 km (constant)
        lat_offset = half_side / 111.0

        # Longitude: 1 degree ≈ 111 * cos(latitude) km (varies with latitude)
        lng_offset = half_side / (111.0 * math.cos(math.radians(lat)))

        # Compute the square's 4 vertices (clockwise)
        # NW -> NE -> SE -> SW
        vertices = [
            (lat + lat_offset, lng - lng_offset),  # NW
            (lat + lat_offset, lng + lng_offset),  # NE
            (lat - lat_offset, lng + lng_offset),  # SE
            (lat - lat_offset, lng - lng_offset),  # SW
        ]

        # Format as the string the API expects: "lat1,lng1:lat2,lng2:lat3,lng3:lat4,lng4"
        poly_str = ":".join([f"{v[0]:.6f},{v[1]:.6f}" for v in vertices])

        return poly_str

    def get_crimes_at_location(
        self,
        lat: float,
        lng: float,
        date: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        use_polygon: bool = True,
        polygon_size_km: float = 1.0
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get crime data for a given location

        Note: uses curl rather than the requests library, because in some environments requests runs into SSL issues
        (LibreSSL vs OpenSSL compatibility), whereas curl uses the system SSL library and is more stable.

        Args:
            lat: latitude
            lng: longitude
            date: date in "YYYY-MM" format (e.g. "2025-10")
                  If None, data for the month 3 months ago is used
            max_retries: maximum number of retries (currently unused, kept for future extension)
            retry_delay: retry delay (seconds) (currently unused, kept for future extension)
            use_polygon: whether to use a polygon query (default True; otherwise a point query)
            polygon_size_km: side length of the polygon square (km), default 1 km

        Returns:
            List of crime records, or None on failure
        """
        # Compute the polygon or use a point query (computed once, reused by later retries)
        if use_polygon:
            polygon = self._calculate_square_polygon(lat, lng, polygon_size_km)
            print(f"  📐 Query area: {polygon_size_km}km × {polygon_size_km}km square")
        else:
            polygon = None

        # Determine the starting month: if the caller specified one, query only that month; otherwise start 3 months back and
        # step back to earlier months until a non-empty result is found or 3 months have been tried (covers Police API publication lag + sparse months).
        explicit_date = date is not None
        if explicit_date:
            candidates = [date]
        else:
            now = datetime.now()
            candidates = []
            # Start 3 months back, then 4, then 5 — avoids publication lag + sparse false zeros
            for offset in (3, 4, 5):
                target_month = now.month - offset
                target_year = now.year
                while target_month <= 0:
                    target_month += 12
                    target_year -= 1
                candidates.append(f"{target_year}-{target_month:02d}")

        result: Optional[List[Dict[str, Any]]] = None
        used_date = None
        for candidate in candidates:
            print(f"  📅 Query date: {candidate}")
            result = self._get_crimes_with_curl(lat, lng, candidate, polygon)
            if result is None:
                # Network failure — try the next month
                continue
            used_date = candidate
            if len(result) > 0:
                # Got data; use this month's result
                break
            # 0 records returned: a single month can be sparse, so fall back one more month and retry
            print(f"  ⚠ {candidate} returned 0 records, falling back one month")

        if result is None:
            print(f"  ❌ Could not fetch crime data")
        else:
            if used_date and not explicit_date:
                print(f"  ✅ Using month: {used_date} ({len(result)} records)")
            record_api_usage("uk_police")

        return result

    def _get_crimes_with_curl(
        self,
        lat: float,
        lng: float,
        date: str,
        polygon: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch crime data using the curl command"""
        try:
            # Choose the URL depending on whether a polygon is used
            if polygon:
                url = f"{self.endpoint}/crimes-street/all-crime?poly={polygon}&date={date}"
            else:
                url = f"{self.endpoint}/crimes-street/all-crime?lat={lat}&lng={lng}&date={date}"

            result = subprocess.run(
                ["curl", "-s", "--tlsv1.2", "-X", "GET", url],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0 and result.stdout:
                crimes = json.loads(result.stdout)
                if isinstance(crimes, list):
                    print(f"  ✅ Fetched crime data ({len(crimes)} records)")
                    return crimes

            print(f"  ❌ API returned empty data or an invalid format")
            return None

        except subprocess.TimeoutExpired:
            print(f"  ❌ API request timed out (30s)")
            return None
        except json.JSONDecodeError as e:
            print(f"  ❌ Failed to parse data: {str(e)[:100]}")
            return None
        except Exception as e:
            print(f"  ❌ Request failed: {str(e)[:100]}")
            return None

    def count_crimes_by_category(
        self,
        crimes: List[Dict[str, Any]],
        target_categories: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Count crimes by category

        Args:
            crimes: list of crime records
            target_categories: list of crime categories to count
                Counted by default: anti-social-behaviour, burglary, violent-crime,
                        robbery, theft-from-the-person, drugs, possession-of-weapons

        Returns:
            Dict mapping crime category (key) to count (value)
        """
        if target_categories is None:
            target_categories = [
                "anti-social-behaviour",
                "burglary",
                "violent-crime",
                "robbery",
                "theft-from-the-person",
                "drugs",
                "possession-of-weapons"
            ]

        # Initialize the counters
        crime_counts = {category: 0 for category in target_categories}
        crime_counts["other"] = 0
        crime_counts["total"] = 0

        # Count each category
        for crime in crimes:
            category = crime.get("category", "unknown")
            crime_counts["total"] += 1

            if category in target_categories:
                crime_counts[category] += 1
            else:
                crime_counts["other"] += 1

        return crime_counts
