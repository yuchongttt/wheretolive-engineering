#!/usr/bin/env python3
"""Google Geocoding API wrapper"""

import time
import requests
from typing import Dict, Any, Optional

from apis.api_usage import try_acquire_api_quota


class GoogleGeocodingAPI:
    """Wrapper for Google Geocoding API calls"""

    def __init__(self, api_key: str):
        """
        Initialise the Google Geocoding API client.

        Args:
            api_key: Google API key
        """
        self.api_key = api_key
        self.endpoint = "https://maps.googleapis.com/maps/api/geocode/json"

    def get_coordinates(
        self,
        address: str,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """
        Get the latitude/longitude of an address.

        Args:
            address: address string
            max_retries: maximum number of retries
            retry_delay: retry delay (seconds)

        Returns:
            Dict with the coordinates, or None on failure
            Format: {"lat": float, "lng": float, "formatted_address": str}
        """
        params = {
            "address": address,
            "key": self.api_key
        }

        last_error = None

        for attempt in range(max_retries):
            try:
                if not try_acquire_api_quota("google_geocoding"):
                    print("  ⚠️  Google Geocoding API monthly quota exhausted, skipping this call")
                    return None

                response = requests.get(
                    self.endpoint,
                    params=params,
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "OK" and len(data.get("results", [])) > 0:
                        result = data["results"][0]
                        location = result["geometry"]["location"]
                        return {
                            "lat": location["lat"],
                            "lng": location["lng"],
                            "formatted_address": result.get("formatted_address", address),
                            "success": True
                        }
                    else:
                        last_error = f"Geocoding failed: {data.get('status', 'UNKNOWN')}"
                        if attempt < max_retries - 1:
                            print(f"  ⚠️  {last_error}, retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                            time.sleep(retry_delay)
                            continue
                elif response.status_code == 429:
                    # Rate limited
                    last_error = "Geocoding API rate limited (429)"
                    wait_time = retry_delay * (attempt + 1) * 2
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  {last_error}, retrying in {wait_time:.1f}s ({attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                elif response.status_code >= 500:
                    # Server error
                    last_error = f"Geocoding server error ({response.status_code})"
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  {last_error}, retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue
                else:
                    last_error = f"Geocoding request failed: {response.status_code}"
                    print(f"  ❌ {last_error}")
                    return None

            except requests.exceptions.Timeout:
                last_error = "Geocoding request timed out"
                if attempt < max_retries - 1:
                    print(f"  ⚠️  {last_error}, retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue

            except requests.exceptions.RequestException as e:
                last_error = f"Geocoding request error: {str(e)[:100]}"
                if attempt < max_retries - 1:
                    print(f"  ⚠️  {last_error}, retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue

            except Exception as e:
                last_error = f"Error processing Geocoding response: {str(e)[:100]}"
                if attempt < max_retries - 1:
                    print(f"  ⚠️  {last_error}, retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue

        # All retries failed
        print(f"  ❌ Still failing after {max_retries} retries: {last_error}")
        return None
