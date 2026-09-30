#!/usr/bin/env python3
"""Google Routes API wrapper"""

import time
import requests
from typing import Dict, Any, Optional

from apis.api_usage import try_acquire_api_quota


class GoogleRoutesAPI:
    """Wrapper for Google Routes API calls"""

    def __init__(self, api_key: str):
        """
        Initialise the Google Routes API client.

        Args:
            api_key: Google API key
        """
        self.api_key = api_key
        self.endpoint = "https://routes.googleapis.com/directions/v2:computeRoutes"

    def compute_route(
        self,
        origin_address: str,
        destination_address: str,
        travel_mode: str = "TRANSIT",
        departure_time: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """
        Compute a route between two places (with retries).

        Args:
            origin_address: origin address
            destination_address: destination address
            travel_mode: travel mode (DRIVE/BICYCLE/WALK/TRANSIT/TWO_WHEELER)
            departure_time: departure time (RFC3339, e.g. "2026-01-23T08:30:00Z")
            max_retries: maximum number of retries
            retry_delay: retry delay (seconds)

        Returns:
            Dict with distance and duration, or None on failure
        """
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "routes.duration,routes.distanceMeters"
        }

        payload = {
            "origin": {
                "address": origin_address
            },
            "destination": {
                "address": destination_address
            },
            "travelMode": travel_mode
        }

        # If a departure time is given, add it to the request
        # Note: DRIVE mode needs routingPreference = TRAFFIC_AWARE to use departureTime
        if departure_time:
            if travel_mode == "DRIVE":
                # DRIVE mode needs traffic awareness enabled to use a departure time
                payload["departureTime"] = departure_time
                payload["routingPreference"] = "TRAFFIC_AWARE"
            elif travel_mode == "TRANSIT":
                # TRANSIT mode can use the departure time directly
                payload["departureTime"] = departure_time
            # WALK and BICYCLE modes do not support departureTime; ignore it

        last_error = None

        for attempt in range(max_retries):
            try:
                if not try_acquire_api_quota("google_routes"):
                    print("  ⚠️  Google Routes API monthly quota exhausted, skipping this call")
                    return None

                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    if "routes" in data and len(data["routes"]) > 0:
                        route = data["routes"][0]
                        return {
                            "distance_meters": route.get("distanceMeters", 0),
                            "duration_seconds": int(route.get("duration", "0s").rstrip("s")),
                            "duration_minutes": int(route.get("duration", "0s").rstrip("s")) / 60,
                            "travel_mode": travel_mode,
                            "success": True
                        }
                    else:
                        last_error = "API response contains no route"
                        if attempt < max_retries - 1:
                            print(f"  ⚠️  {last_error}, retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                            time.sleep(retry_delay)
                            continue
                elif response.status_code == 429:
                    # Rate limited; wait longer
                    last_error = f"API rate limited (429)"
                    wait_time = retry_delay * (attempt + 1) * 2
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  {last_error}, retrying in {wait_time:.1f}s ({attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                elif response.status_code >= 500:
                    # Server error; retry
                    last_error = f"Server error ({response.status_code})"
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  {last_error}, retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue
                else:
                    # Other errors; do not retry
                    last_error = f"API request failed: {response.status_code} - {response.text[:100]}"
                    print(f"  ❌ {last_error}")
                    return None

            except requests.exceptions.Timeout:
                last_error = "API request timed out"
                if attempt < max_retries - 1:
                    print(f"  ⚠️  {last_error}, retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue

            except requests.exceptions.ConnectionError:
                last_error = "Network connection error"
                if attempt < max_retries - 1:
                    print(f"  ⚠️  {last_error}, retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue

            except requests.exceptions.RequestException as e:
                last_error = f"API request error: {str(e)[:100]}"
                if attempt < max_retries - 1:
                    print(f"  ⚠️  {last_error}, retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue

            except Exception as e:
                last_error = f"Error processing response: {str(e)[:100]}"
                if attempt < max_retries - 1:
                    print(f"  ⚠️  {last_error}, retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue

        # All retries failed
        print(f"  ❌ Still failing after {max_retries} retries: {last_error}")
        return None

    def compute_multiple_modes(
        self,
        origin_address: str,
        destination_address: str,
        travel_modes: list = None
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Compute routes for several travel modes.

        Args:
            origin_address: origin address
            destination_address: destination address
            travel_modes: list of travel modes

        Returns:
            Dict keyed by travel mode, value is the route info
        """
        if travel_modes is None:
            travel_modes = ["TRANSIT", "DRIVE", "BICYCLE", "WALK"]

        results = {}
        for mode in travel_modes:
            print(f"Querying route for mode {mode}...")
            results[mode] = self.compute_route(
                origin_address,
                destination_address,
                mode
            )

        return results
