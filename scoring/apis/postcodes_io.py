#!/usr/bin/env python3
"""Postcodes.io API wrapper"""

import time
import requests
from typing import Dict, Any, Optional

from apis.api_usage import record_api_usage


class PostcodesIOAPI:
    """Postcodes.io API client - free postcode lookup service"""

    def __init__(self):
        """Initialise the Postcodes.io API (free, no authentication required)"""
        self.endpoint = "https://api.postcodes.io"

    def lookup_postcode(
        self,
        postcode: str,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """
        Look up a postcode and get geographic info such as the LSOA code

        Args:
            postcode: UK postcode
            max_retries: maximum number of retries
            retry_delay: retry delay (seconds)

        Returns:
            Postcode info dict containing lsoa, lsoa_code, msoa, admin_district, etc.
        """
        # Normalise the postcode
        cleaned = postcode.replace(" ", "").upper()
        url = f"{self.endpoint}/postcodes/{cleaned}"

        last_error = None

        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == 200 and "result" in data:
                        record_api_usage("postcodes_io")
                        result = data["result"]
                        codes = result.get("codes", {})
                        return {
                            "postcode": result.get("postcode", ""),
                            "lsoa": result.get("lsoa", ""),  # LSOA name
                            "lsoa_code": codes.get("lsoa11", "") or codes.get("lsoa", ""),  # LSOA 2011 code (prefer lsoa11)
                            "lsoa21_code": codes.get("lsoa21", "") or codes.get("lsoa", ""),  # LSOA 2021 code
                            "msoa": result.get("msoa", ""),
                            "msoa_code": codes.get("msoa11", "") or codes.get("msoa", ""),
                            "admin_district": result.get("admin_district", ""),
                            "admin_district_code": codes.get("admin_district", ""),
                            "region": result.get("region", ""),
                            "latitude": result.get("latitude"),
                            "longitude": result.get("longitude"),
                            "success": True
                        }
                    else:
                        last_error = f"Postcode lookup failed: {data.get('error', 'Unknown')}"

                elif response.status_code == 404:
                    last_error = f"Postcode not found: {postcode}"
                    return None

                elif response.status_code == 429:
                    last_error = "Postcodes.io API rate limit"
                    wait_time = retry_delay * (attempt + 1) * 2
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue

                elif response.status_code >= 500:
                    last_error = f"Postcodes.io server error ({response.status_code})"
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue

                else:
                    last_error = f"Request failed: {response.status_code}"
                    return None

            except requests.exceptions.Timeout:
                last_error = "Request timed out"
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue

            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {str(e)[:100]}"
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue

            except Exception as e:
                last_error = f"Error processing response: {str(e)[:100]}"
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue

        print(f"  ❌ Postcodes.io lookup failed: {last_error}")
        return None
