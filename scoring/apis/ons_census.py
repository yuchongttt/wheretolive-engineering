#!/usr/bin/env python3
"""ONS Census 2021 API wrapper"""

import time
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional


class ONSCensusAPI:
    """ONS Census 2021 API - census data (free, no authentication required)"""

    # Dataset ID -> (editions version, category mapping)
    DATASET_VERSIONS = {
        "TS044": 1,  # Accommodation type
        "TS054": 1,  # Tenure
        "TS021": 1,  # Ethnic group
        "TS067": 1,  # Highest qualification
    }

    # TS044: Accommodation Type (8 obs, no "does not apply")
    TS044_CATEGORIES = {
        0: "detached",
        1: "semi_detached",
        2: "terraced",
        3: "flat_purpose_built",
        4: "flat_converted",
        5: "flat_other_converted",
        6: "flat_commercial",
        7: "caravan_temporary",
    }

    # TS054: Tenure (9 obs, index 0 = "does not apply")
    TS054_CATEGORIES = {
        1: "owned_outright",
        2: "owned_mortgage",
        3: "shared_ownership",
        4: "social_rent_council",
        5: "social_rent_other",
        6: "private_rent_landlord",
        7: "private_rent_other",
        8: "rent_free",
    }

    # TS021: Ethnicity (20 obs, index 0 = "does not apply")
    # Grouped into 5 high-level categories
    TS021_GROUPS = {
        "asian": range(1, 6),
        "black": range(6, 9),
        "mixed": range(9, 13),
        "white": range(13, 18),
        "other": range(18, 20),
    }

    # TS067: Highest Qualification (8 obs, index 0 = "does not apply")
    TS067_CATEGORIES = {
        1: "no_qualifications",
        2: "level_1",
        3: "level_2",
        4: "apprenticeship",
        5: "level_3",
        6: "level_4_plus",
        7: "other",
    }

    def __init__(self):
        """Initialise the ONS Census API (free, no authentication required)"""
        self.endpoint = "https://api.beta.ons.gov.uk/v1"

    def get_dataset(
        self,
        dataset_id: str,
        lsoa_code: str,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch data for a specific dataset

        Args:
            dataset_id: dataset ID, e.g. "TS054"
            lsoa_code: LSOA code (Census 2021 uses LSOA 2021 codes)
            max_retries: maximum number of retries
            retry_delay: retry delay

        Returns:
            {"total": int, "categories": {"name": count, ...}}
        """
        version = self.DATASET_VERSIONS.get(dataset_id, 1)
        url = (
            f"{self.endpoint}/datasets/{dataset_id}/editions/2021"
            f"/versions/{version}/json?area-type=lsoa,{lsoa_code}"
        )

        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    observations = data.get("observations", [])
                    return self._parse_observations(dataset_id, observations)
                elif response.status_code == 404:
                    return None
            except requests.exceptions.RequestException:
                pass

            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))

        return None

    def _parse_observations(self, dataset_id: str, observations: list) -> Optional[Dict[str, Any]]:
        """Parse the observations array into a categories dict"""
        if not observations:
            return None

        categories = {}

        if dataset_id == "TS044":
            for idx, name in self.TS044_CATEGORIES.items():
                if idx < len(observations):
                    categories[name] = observations[idx]

        elif dataset_id == "TS054":
            for idx, name in self.TS054_CATEGORIES.items():
                if idx < len(observations):
                    categories[name] = observations[idx]

        elif dataset_id == "TS021":
            for group_name, indices in self.TS021_GROUPS.items():
                total = sum(observations[i] for i in indices if i < len(observations))
                categories[group_name] = total

        elif dataset_id == "TS067":
            for idx, name in self.TS067_CATEGORIES.items():
                if idx < len(observations):
                    categories[name] = observations[idx]

        else:
            return None

        total = sum(categories.values())
        return {"total": total, "categories": categories}

    def get_all_census_data(self, lsoa_code: str) -> Optional[Dict[str, Any]]:
        """
        Fetch all Census 2021 data for an LSOA

        Args:
            lsoa_code: LSOA code (Census 2021 uses LSOA 2021 codes)

        Returns:
            Dict containing housing_type, tenure, ethnicity, education_level
        """
        dataset_map = {
            "housing_type": "TS044",
            "tenure": "TS054",
            "ethnicity": "TS021",
            "education_level": "TS067",
        }

        def fetch(item):
            key, dataset_id = item
            return key, self.get_dataset(dataset_id, lsoa_code)

        result = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for key, data in pool.map(fetch, dataset_map.items()):
                result[key] = data

        # Return None if all failed
        if all(v is None for v in result.values()):
            return None

        return result
