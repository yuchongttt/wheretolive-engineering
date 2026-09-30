#!/usr/bin/env python3
"""
Area Demographics Evaluator
Evaluates area socio-economic indicators using ONS Census 2021 and IMD 2025 data

All APIs are free and need no key:
- Postcodes.io: postcode to LSOA mapping
- IMD 2025: deprivation index data (downloaded automatically on first use)
"""

import re
import requests
from typing import Dict, Any, Optional
from api_helper import PostcodesIOAPI, ONSGeoAPI, IMDDataLoader, ONSCensusAPI


class AreaDemographicsEvaluator:
    """Area demographics evaluator"""

    # Scoring weights
    WEIGHTS = {
        "imd": 0.30,         # IMD overall deprivation index
        "income": 0.15,      # Income
        "employment": 0.15,  # Employment
        "education": 0.15,   # Education
        "crime": 0.15,       # Safety (IMD crime index)
        "health": 0.10,      # Health
    }

    def __init__(self):
        """Initialise the evaluator"""
        self.postcodes_api = PostcodesIOAPI()
        self.ons_geo_api = ONSGeoAPI()
        self.imd_loader = IMDDataLoader()
        self.census_api = ONSCensusAPI()

    def _extract_postcode(self, address: str) -> Optional[str]:
        """
        Extract the postcode from an address

        Args:
            address: address string

        Returns:
            The extracted postcode, or None
        """
        # UK postcode regular expressions
        # Full format: AA9A 9AA, A9A 9AA, A9 9AA, A99 9AA, AA9 9AA, AA99 9AA
        full_postcode_pattern = r'[A-Z]{1,2}[0-9][0-9A-Z]?\s*[0-9][A-Z]{2}'

        # Partial postcode format (first half only): E20, N17, SW1A, EC1A, etc.
        partial_postcode_pattern = r'[A-Z]{1,2}[0-9][0-9A-Z]?(?:\s|,|$)'

        # Normalise the address
        address = address.upper().strip()

        # 1. First try to find a full postcode
        match = re.search(full_postcode_pattern, address)
        if match:
            postcode = match.group(0)
            # Normalise the format
            cleaned = postcode.replace(" ", "")
            if len(cleaned) > 3:
                return f"{cleaned[:-3]} {cleaned[-3:]}"
            return cleaned

        # 2. If the whole address is a full postcode
        cleaned = address.replace(" ", "")
        if re.match(r'^[A-Z]{1,2}[0-9][0-9A-Z]?[0-9][A-Z]{2}$', cleaned):
            return f"{cleaned[:-3]} {cleaned[-3:]}"

        # 3. Try to extract a partial postcode (first half only, e.g. E20, N17)
        # Search from the end of the address
        parts = address.replace(",", " ").split()
        for part in reversed(parts):
            part = part.strip()
            if re.match(r'^[A-Z]{1,2}[0-9][0-9A-Z]?$', part):
                # Found a partial postcode; try to look up full info
                # A partial postcode cannot be used directly for IMD lookups, but can be returned for display
                return part

        return None

    def _decile_to_score(self, decile: int, weight: float = 1.0) -> float:
        """
        Convert an IMD Decile (1-10) to a score (0-100)

        IMD Decile:
        - 1 = most deprived / worst 10%
        - 10 = least deprived / best 10%

        Args:
            decile: IMD decile value (1-10)
            weight: weight

        Returns:
            Score (0-100)
        """
        if decile <= 0 or decile > 10:
            return 50.0  # Default mid-range score

        # Adjusted mapping: decile 1 -> 30 pts, decile 10 -> 100 pts
        # Gives lower-decile areas a higher base score
        score = 22 + decile * 7.8
        return score * weight

    def _is_full_postcode(self, postcode: str) -> bool:
        """Check whether this is a full postcode"""
        cleaned = postcode.replace(" ", "").upper()
        return bool(re.match(r'^[A-Z]{1,2}[0-9][0-9A-Z]?[0-9][A-Z]{2}$', cleaned))

    def _lookup_outcode(self, outcode: str) -> Optional[Dict[str, Any]]:
        """
        Look up info for a partial postcode (outcode)

        Args:
            outcode: first half of the postcode, e.g. E20, N17

        Returns:
            Area info
        """
        url = f"https://api.postcodes.io/outcodes/{outcode.upper()}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == 200 and "result" in data:
                    result = data["result"]
                    return {
                        "outcode": result.get("outcode", ""),
                        "admin_district": result.get("admin_district", []),
                        "latitude": result.get("latitude"),
                        "longitude": result.get("longitude"),
                        "success": True
                    }
        except Exception:
            pass
        return None

    def _find_representative_lsoa(self, outcode: str) -> Optional[Dict[str, str]]:
        """
        Find a representative LSOA code for a partial postcode

        Finds a valid full postcode by trying common suffix combinations

        Returns:
            {"lsoa11cd": "...", "lsoa21cd": "..."} or None
        """
        suffixes = [
            "1AA", "0AA", "1AB", "0AB",
            "2AA", "3AA", "4AA", "5AA", "6AA", "7AA", "8AA", "9AA",
            "1AD", "1AE", "1AF", "0AD", "0AE",
            "1BA", "0BA", "1DA", "0DA",
        ]

        for suffix in suffixes:
            full_postcode = f"{outcode} {suffix}"
            geo_data = self.ons_geo_api.get_lsoa_from_postcode(full_postcode)
            if geo_data and geo_data.get("success") and geo_data.get("lsoa11cd"):
                return {
                    "lsoa11cd": geo_data.get("lsoa11cd", ""),
                    "lsoa21cd": geo_data.get("lsoa21cd", "") or geo_data.get("lsoa11cd", ""),
                }
        return None

    def evaluate(self, property_data: Dict[str, Any]) -> float:
        """
        Evaluate area demographics

        Args:
            property_data: property data dict; must contain an 'address' field

        Returns:
            Score (0-100)
        """
        address = property_data.get("address", "")

        if not address:
            print("  ⚠️  Address is empty; cannot evaluate demographics")
            return 50.0

        # 1. Extract the postcode
        postcode = self._extract_postcode(address)
        if not postcode:
            print(f"  ⚠️  Could not extract a postcode from the address: {address}")
            return 50.0

        is_full = self._is_full_postcode(postcode)
        print(f"  📮 Postcode: {postcode} ({'full' if is_full else 'partial'})")

        # 2. Look up the postcode to get the LSOA code
        print(f"  📍 Looking up LSOA info...")

        lsoa11_code = ""
        lsoa21_code = ""
        lsoa_name = ""
        lad_name = ""

        if is_full:
            # Full postcode: look it up directly
            geo_data = self.ons_geo_api.get_lsoa_from_postcode(postcode)
            if geo_data and geo_data.get("success"):
                lsoa11_code = geo_data.get("lsoa11cd", "")
                lsoa21_code = geo_data.get("lsoa21cd", "") or lsoa11_code
                lsoa_name = geo_data.get("lsoa_name", "")
                lad_name = geo_data.get("lad_name", "")
        else:
            # Partial postcode: try to find a representative LSOA
            print(f"  🔍 Partial postcode; trying to find a representative LSOA...")
            lsoa_result = self._find_representative_lsoa(postcode)
            if lsoa_result:
                lsoa11_code = lsoa_result.get("lsoa11cd", "")
                lsoa21_code = lsoa_result.get("lsoa21cd", "") or lsoa11_code
            if lsoa21_code:
                # Get the name from IMD data (IMD 2025 uses LSOA 2021 codes)
                imd_data = self.imd_loader.get_imd_data(lsoa21_code)
                if imd_data:
                    lsoa_name = imd_data.get("lsoa_name", "")
                    lad_name = imd_data.get("lad_name", "")

        if not lsoa21_code:
            print(f"  ❌ Could not get LSOA info")
            return 50.0

        print(f"  ✅ Local Authority: {lad_name}")
        print(f"  ✅ LSOA: {lsoa_name} ({lsoa21_code})")

        # 3. Look up IMD data (IMD 2025 uses LSOA 2021 codes)
        print(f"  📊 Looking up IMD 2025 data...")
        imd_data = self.imd_loader.get_imd_data(lsoa21_code)

        if not imd_data:
            print(f"  ❌ Could not get IMD data (the LSOA code may not be in England)")
            return 50.0

        # 4. Compute component scores
        scores = {}

        # IMD overall index
        imd_decile = imd_data.get("imd_decile", 5)
        scores["imd"] = self._decile_to_score(imd_decile)

        # Income
        income_decile = imd_data.get("income_decile", 5)
        scores["income"] = self._decile_to_score(income_decile)

        # Employment
        employment_decile = imd_data.get("employment_decile", 5)
        scores["employment"] = self._decile_to_score(employment_decile)

        # Education
        education_decile = imd_data.get("education_decile", 5)
        scores["education"] = self._decile_to_score(education_decile)

        # Safety (note: a higher IMD crime decile is better, meaning less crime)
        crime_decile = imd_data.get("crime_decile", 5)
        scores["crime"] = self._decile_to_score(crime_decile)

        # Health
        health_decile = imd_data.get("health_decile", 5)
        scores["health"] = self._decile_to_score(health_decile)

        # 5. Weighted sum
        total_score = 0.0
        for key, weight in self.WEIGHTS.items():
            total_score += scores.get(key, 50.0) * weight

        # 6. Display results
        imd_desc = self._get_imd_description(imd_decile)
        print(f"  📈 IMD Decile: {imd_decile}/10 ({imd_desc})")
        print(f"\n  Component scores:")
        print(f"    - Overall deprivation index: {scores['imd']:.0f}/100")
        print(f"    - Income: {scores['income']:.0f}/100 (Decile {income_decile})")
        print(f"    - Employment: {scores['employment']:.0f}/100 (Decile {employment_decile})")
        print(f"    - Education: {scores['education']:.0f}/100 (Decile {education_decile})")
        print(f"    - Safety: {scores['crime']:.0f}/100 (Decile {crime_decile})")
        print(f"    - Health: {scores['health']:.0f}/100 (Decile {health_decile})")
        print(f"\n  📊 Demographics Score: {total_score:.1f}/100")

        # 7. Cache data in property_data
        property_data["_api_demographics_data"] = {
            "postcode": postcode,
            "lsoa_code": lsoa21_code,  # Uses the LSOA 2021 code (IMD 2025)
            "lsoa_name": lsoa_name,
            "lad_name": lad_name,
            "imd_decile": imd_decile,
            "imd_rank": imd_data.get("imd_rank", 0),
            "total_lsoas": 33755,  # Total number of LSOA 2021 areas in England
            "income_decile": income_decile,
            "employment_decile": employment_decile,
            "education_decile": education_decile,
            "crime_decile": crime_decile,
            "health_decile": health_decile,
            "housing_decile": imd_data.get("housing_decile", 0),
            "environment_decile": imd_data.get("environment_decile", 0),
            # Population data
            "total_population": imd_data.get("total_population", 0),
            "children_0_15": imd_data.get("children_0_15", 0),
            "population_16_59": imd_data.get("population_16_59", 0),
            "population_60_plus": imd_data.get("population_60_plus", 0),
            "scores": scores,
            "total_score": total_score
        }

        # 8. Census 2021 data (for display only; does not affect the score)
        census_lsoa = lsoa21_code or lsoa11_code
        print(f"  📊 Looking up Census 2021 data (LSOA: {census_lsoa})...")
        census_data = self.census_api.get_all_census_data(census_lsoa)
        if census_data:
            property_data["_api_demographics_data"]["census"] = census_data
            dims = [k for k, v in census_data.items() if v is not None]
            print(f"  ✅ Census data: {', '.join(dims)}")
        else:
            print(f"  ⚠️  Failed to fetch Census data")

        return total_score

    def _get_imd_description(self, decile: int) -> str:
        """
        Get a description for an IMD Decile

        Args:
            decile: IMD decile value (1-10)

        Returns:
            Description string
        """
        if decile <= 2:
            return "more deprived"
        elif decile <= 4:
            return "below average"
        elif decile <= 6:
            return "average"
        elif decile <= 8:
            return "above average"
        else:
            return "more affluent"

    def get_detailed_report(self, property_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get a detailed demographics report

        Args:
            property_data: property data dict

        Returns:
            Detailed report dict
        """
        demo_data = property_data.get("_api_demographics_data")
        if not demo_data:
            return None

        return {
            "postcode": demo_data.get("postcode", ""),
            "lsoa": {
                "code": demo_data.get("lsoa_code", ""),
                "name": demo_data.get("lsoa_name", "")
            },
            "local_authority": demo_data.get("lad_name", ""),
            "imd_2025": {
                "overall_decile": demo_data.get("imd_decile", 0),
                "description": self._get_imd_description(demo_data.get("imd_decile", 5)),
                "components": {
                    "income": demo_data.get("income_decile", 0),
                    "employment": demo_data.get("employment_decile", 0),
                    "education": demo_data.get("education_decile", 0),
                    "health": demo_data.get("health_decile", 0),
                    "crime": demo_data.get("crime_decile", 0),
                    "housing": demo_data.get("housing_decile", 0),
                    "environment": demo_data.get("environment_decile", 0)
                }
            },
            "scores": demo_data.get("scores", {}),
            "total_score": demo_data.get("total_score", 0)
        }


# Test code
if __name__ == "__main__":
    import sys

    # Default test postcode
    test_postcode = "EC4M 8AD"  # St Paul's, London

    if len(sys.argv) > 1:
        test_postcode = sys.argv[1]

    print(f"\n{'='*60}")
    print(f"Area Demographics Evaluator test")
    print(f"{'='*60}")
    print(f"Test address: {test_postcode}\n")

    evaluator = AreaDemographicsEvaluator()
    property_data = {"address": test_postcode}

    score = evaluator.evaluate(property_data)

    print(f"\n{'='*60}")
    print(f"Final score: {score:.1f}/100")
    print(f"{'='*60}\n")

    # Show the detailed report
    report = evaluator.get_detailed_report(property_data)
    if report:
        print("\nDetailed report:")
        import json
        print(json.dumps(report, indent=2, ensure_ascii=False))
