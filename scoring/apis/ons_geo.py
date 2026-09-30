#!/usr/bin/env python3
"""ONS Open Geography API wrapper"""

from typing import Dict, Any, Optional

from apis.postcodes_io import PostcodesIOAPI


class ONSGeoAPI:
    """ONS Open Geography API - postcode to LSOA mapping (free)"""

    def __init__(self):
        """Initialise the ONS Geo API (free, no authentication required)"""
        self.endpoint = "https://api.os.uk/search/names/v1"
        # postcodes.io is actually used as the primary source, as it is simpler and more reliable

    def get_lsoa_from_postcode(self, postcode: str) -> Optional[Dict[str, Any]]:
        """
        Get the LSOA code for a postcode

        Args:
            postcode: UK postcode

        Returns:
            Dict containing lsoa21cd, lsoa21nm, msoa21cd, lad22cd, lad22nm
        """
        # Use postcodes.io as the primary source
        postcodes_api = PostcodesIOAPI()
        result = postcodes_api.lookup_postcode(postcode)

        if not result or not result.get("success"):
            return None

        return {
            "lsoa11cd": result.get("lsoa_code", ""),  # LSOA 2011 code (used by IMD 2019)
            "lsoa21cd": result.get("lsoa21_code", "") or result.get("lsoa_code", ""),  # LSOA 2021 code
            "lsoa_name": result.get("lsoa", ""),
            "msoa_code": result.get("msoa_code", ""),
            "msoa_name": result.get("msoa", ""),
            "lad_code": result.get("admin_district_code", ""),
            "lad_name": result.get("admin_district", ""),
            "region": result.get("region", ""),
            "latitude": result.get("latitude"),
            "longitude": result.get("longitude"),
            "success": True
        }
