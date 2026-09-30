#!/usr/bin/env python3
"""UK EPC (Energy Performance Certificate) API wrapper"""

import os
import time
import requests
from typing import Dict, Any, Optional, List

from apis.api_usage import record_api_usage


class EPCAPI:
    """UK EPC (Energy Performance Certificate) API client wrapper"""

    def __init__(self, email: str = None, api_key: str = None):
        """
        Initialise the EPC API

        Args:
            email: registered email (can be set via the EPC_EMAIL env var)
            api_key: API key (can be set via the EPC_API_KEY env var)
        """
        self.endpoint = "https://epc.opendatacommunities.org/api/v1/domestic/search"
        # Prefer the passed-in arguments; otherwise read from env vars
        self.email = email or os.getenv("EPC_EMAIL")
        self.api_key = api_key or os.getenv("EPC_API_KEY")

    def _create_auth_header(self) -> Optional[str]:
        """
        Build the HTTP Basic Auth header

        Returns:
            Base64-encoded auth string, or None (if not configured)
        """
        if not self.email or not self.api_key:
            return None

        import base64
        credentials = f"{self.email}:{self.api_key}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"

    def search_by_postcode(
        self,
        postcode: str,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Search EPC records by postcode

        Args:
            postcode: postcode
            max_retries: maximum number of retries
            retry_delay: retry delay (seconds)

        Returns:
            List of EPC records, or None on failure
        """
        auth_header = self._create_auth_header()
        if not auth_header:
            print("  ⚠️  EPC API credentials not configured; skipping EPC data fetch")
            return None

        # Normalise the postcode (remove spaces)
        cleaned_postcode = postcode.replace(" ", "")

        headers = {
            "Authorization": auth_header,
            "Accept": "application/json"
        }

        params = {
            "postcode": cleaned_postcode,
            "size": 100  # return at most 100 records
        }

        last_error = None

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    self.endpoint,
                    headers=headers,
                    params=params,
                    timeout=30
                )

                if response.status_code == 200:
                    record_api_usage("epc")
                    data = response.json()
                    rows = data.get("rows", [])

                    records = []
                    for row in rows:
                        record = {
                            "address": row.get("address", ""),
                            "floor_area": float(row.get("total-floor-area") or 0) if row.get("total-floor-area", "").strip() else 0.0,
                            "current_energy_rating": row.get("current-energy-rating", ""),
                            "property_type": row.get("property-type", ""),
                            "built_form": row.get("built-form", ""),
                            "construction_age_band": row.get("construction-age-band", ""),
                            "lodgement_date": row.get("lodgement-date", "")
                        }
                        records.append(record)

                    if attempt > 0:
                        print(f"  ✅ EPC API: attempt {attempt + 1} succeeded")

                    return records

                elif response.status_code == 401:
                    print("  ❌ EPC API authentication failed; check the email and api_key settings")
                    return None

                elif response.status_code == 403:
                    print("  ❌ EPC API access denied; you may need to re-register")
                    return None

                elif response.status_code == 429:
                    last_error = "EPC API rate limited (429)"
                    wait_time = retry_delay * (attempt + 1) * 2
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  {last_error}; retrying in {wait_time:.1f}s ({attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue

                elif response.status_code >= 500:
                    last_error = f"EPC API server error ({response.status_code})"
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  {last_error}; retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue

                else:
                    last_error = f"EPC API request failed: {response.status_code}"
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  {last_error}; retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"  ❌ {last_error}")
                        return None

            except requests.exceptions.Timeout:
                last_error = "EPC API request timed out"
                if attempt < max_retries - 1:
                    print(f"  ⚠️  {last_error}; retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue

            except requests.exceptions.RequestException as e:
                last_error = f"EPC API request error: {str(e)[:100]}"
                if attempt < max_retries - 1:
                    print(f"  ⚠️  {last_error}; retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue

            except Exception as e:
                last_error = f"Error processing EPC response: {str(e)[:100]}"
                if attempt < max_retries - 1:
                    print(f"  ⚠️  {last_error}; retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue

        print(f"  ❌ EPC API still failing after {max_retries} retries: {last_error}")
        return None


class LocalEPCService:
    """Wrapper for the EPC microservice on the GPU box (:8400); output has the same shape as EPCAPI.search_by_postcode.

    Full local EPC dataset, no quota, LAN latency. It is the only EPC source used by the price
    evaluator (the fallback to the official API was removed after that API was retired).
    Field mapping: floor_area_sqm→floor_area, energy_rating→current_energy_rating.
    If the service is unreachable it silently returns None (callers degrade to "no EPC data"; the rest of the evaluation is unaffected).
    """

    def __init__(self, base_url: str = None, timeout: float = 4.0):
        self.base_url = (
            base_url or os.getenv("EPC_SERVICE_URL") or "http://localhost:8400"
        ).rstrip("/")
        self.timeout = timeout

    def search_by_postcode(self, postcode: str) -> Optional[List[Dict[str, Any]]]:
        try:
            resp = requests.get(
                f"{self.base_url}/epc/postcode",
                # service max is 200; with the default of 50 the target unit in a large building may be missing from the results
                params={"postcode": postcode, "limit": 200},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return None
            rows = resp.json()
        except (requests.RequestException, ValueError):
            return None
        if not isinstance(rows, list):
            return None
        records = []
        for row in rows:
            try:
                floor_area = float(row.get("floor_area_sqm") or 0)
            except (TypeError, ValueError):
                floor_area = 0.0
            records.append({
                "address": row.get("address", "") or "",
                "postcode": (row.get("postcode", "") or "").replace(" ", "").upper(),
                "floor_area": floor_area,
                "current_energy_rating": row.get("energy_rating", "") or "",
                "property_type": row.get("property_type", "") or "",
                "built_form": row.get("built_form", "") or "",
                "construction_age_band": row.get("construction_age_band", "") or "",
                "lodgement_date": row.get("inspection_date", "") or "",
            })
        return records

    def search_by_postcodes(
        self, postcodes: List[str], max_workers: int = 8
    ) -> List[Dict[str, Any]]:
        """Fetch EPC records for several postcodes concurrently; failed postcodes are skipped."""
        from concurrent.futures import ThreadPoolExecutor

        merged: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for recs in pool.map(self.search_by_postcode, postcodes):
                if recs:
                    merged.extend(recs)
        return merged
