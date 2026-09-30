#!/usr/bin/env python3
"""Unified geocoding service"""

import re
from typing import Dict, Any, Optional

from apis.postcodes_io import PostcodesIOAPI
from apis.google_geocoding import GoogleGeocodingAPI


class GeocodingService:
    """
    Unified geocoding service.
    Prefers Postcodes.io (free); falls back to the Google Geocoding API on failure.
    """

    def __init__(self, google_api_key: Optional[str] = None):
        """
        Initialise the geocoding service.

        Args:
            google_api_key: Google API key (optional, used for the fallback)
        """
        self._postcodes_api = None  # lazy initialisation
        self.google_api = GoogleGeocodingAPI(google_api_key) if google_api_key else None

    @property
    def postcodes_api(self):
        """Lazily initialise PostcodesIOAPI"""
        if self._postcodes_api is None:
            self._postcodes_api = PostcodesIOAPI()
        return self._postcodes_api

    def _extract_postcode(self, address: str) -> Optional[str]:
        """
        Extract a UK postcode from an address.

        Args:
            address: address string

        Returns:
            The extracted postcode, or None
        """
        # Normalise the address
        address = address.upper().strip()

        # Full postcode regex: AA9A 9AA, A9A 9AA, A9 9AA, A99 9AA, AA9 9AA, AA99 9AA
        full_pattern = r'[A-Z]{1,2}[0-9][0-9A-Z]?\s*[0-9][A-Z]{2}'
        match = re.search(full_pattern, address)
        if match:
            postcode = match.group(0).replace(" ", "")
            if len(postcode) > 3:
                return f"{postcode[:-3]} {postcode[-3:]}"
            return postcode

        # The whole address is just a postcode
        cleaned = address.replace(" ", "")
        if re.match(r'^[A-Z]{1,2}[0-9][0-9A-Z]?[0-9][A-Z]{2}$', cleaned):
            return f"{cleaned[:-3]} {cleaned[-3:]}"

        return None

    def get_coordinates(
        self,
        address: str,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """
        Get the latitude/longitude of an address.
        Prefers Postcodes.io; falls back to Google Geocoding on failure.

        Args:
            address: address string
            max_retries: maximum number of retries
            retry_delay: retry delay (seconds)

        Returns:
            Dict with the coordinates, or None on failure
            Format: {"lat": float, "lng": float, "formatted_address": str, "source": str}
        """
        # 1. Try to extract a postcode and use Postcodes.io
        postcode = self._extract_postcode(address)
        if postcode:
            result = self.postcodes_api.lookup_postcode(postcode, max_retries, retry_delay)
            if result and result.get("success") and result.get("latitude") and result.get("longitude"):
                return {
                    "lat": result["latitude"],
                    "lng": result["longitude"],
                    "formatted_address": result.get("postcode", address),
                    "source": "postcodes.io",
                    "success": True
                }

        # 2. Fall back to the Google Geocoding API
        if self.google_api:
            result = self.google_api.get_coordinates(address, max_retries, retry_delay)
            if result and result.get("success"):
                result["source"] = "google"
                return result

        return None
