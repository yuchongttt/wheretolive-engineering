#!/usr/bin/env python3
"""IMD 2025 (Index of Multiple Deprivation) data loader"""

import os
import csv
import requests
from typing import Dict, Any, Optional


class IMDDataLoader:
    """IMD 2025 deprivation index data loader (uses 2021 LSOA boundaries)"""

    # File 7: All ranks, deciles and scores for the indices of deprivation 2025
    IMD_CSV_URL = "https://assets.publishing.service.gov.uk/media/691ded56d140bbbaa59a2a7d/File_7_IoD2025_All_Ranks_Scores_Deciles_Population_Denominators.csv"
    DATA_DIR = "data"
    CACHE_FILE = "imd_2025.csv"

    def __init__(self):
        """Initialise the IMD data loader"""
        self._data = None
        self._data_by_lsoa = None

    def _ensure_data_dir(self):
        """Ensure the data directory exists"""
        if not os.path.exists(self.DATA_DIR):
            os.makedirs(self.DATA_DIR)

    def _download_imd_data(self) -> bool:
        """
        Download the IMD 2025 data file

        Returns:
            Whether the download succeeded
        """
        self._ensure_data_dir()
        cache_path = os.path.join(self.DATA_DIR, self.CACHE_FILE)

        print(f"  📥 Downloading IMD 2025 data...")

        try:
            response = requests.get(self.IMD_CSV_URL, timeout=60, stream=True)

            if response.status_code == 200:
                with open(cache_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"  ✅ IMD data downloaded: {cache_path}")
                return True
            else:
                print(f"  ❌ Download failed: HTTP {response.status_code}")
                return False

        except Exception as e:
            print(f"  ❌ Download failed: {str(e)[:100]}")
            return False

    def _load_data(self) -> bool:
        """
        Load IMD data into memory

        Returns:
            Whether loading succeeded
        """
        cache_path = os.path.join(self.DATA_DIR, self.CACHE_FILE)

        # Check whether the cache file exists
        if not os.path.exists(cache_path):
            if not self._download_imd_data():
                return False

        try:
            self._data_by_lsoa = {}

            with open(cache_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    lsoa_code = row.get('LSOA code (2021)', '').strip()
                    if not lsoa_code:
                        continue

                    # Parse each indicator
                    # File 7 column name format: "Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)"
                    try:
                        self._data_by_lsoa[lsoa_code] = {
                            "lsoa_code": lsoa_code,
                            "lsoa_name": row.get('LSOA name (2021)', ''),
                            "lad_code": row.get('Local Authority District code (2024)', ''),
                            "lad_name": row.get('Local Authority District name (2024)', ''),

                            # Overall IMD (1 = most deprived, 10 = most affluent)
                            "imd_rank": int(row.get('Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)', 0) or 0),
                            "imd_decile": int(row.get('Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)', 0) or 0),

                            # Domain indices (Decile: 1 = worst, 10 = best)
                            "income_decile": int(row.get('Income Decile (where 1 is most deprived 10% of LSOAs)', 0) or 0),
                            "employment_decile": int(row.get('Employment Decile (where 1 is most deprived 10% of LSOAs)', 0) or 0),
                            "education_decile": int(row.get('Education, Skills and Training Decile (where 1 is most deprived 10% of LSOAs)', 0) or 0),
                            "health_decile": int(row.get('Health Deprivation and Disability Decile (where 1 is most deprived 10% of LSOAs)', 0) or 0),
                            "crime_decile": int(row.get('Crime Decile (where 1 is most deprived 10% of LSOAs)', 0) or 0),
                            "housing_decile": int(row.get('Barriers to Housing and Services Decile (where 1 is most deprived 10% of LSOAs)', 0) or 0),
                            "environment_decile": int(row.get('Living Environment Decile (where 1 is most deprived 10% of LSOAs)', 0) or 0),

                            # IDACI/IDAOPI (child / older-people deprivation indices)
                            "idaci_decile": int(row.get('Income Deprivation Affecting Children Index (IDACI) Decile (where 1 is most deprived 10% of LSOAs)', 0) or 0),
                            "idaopi_decile": int(row.get('Income Deprivation Affecting Older People (IDAOPI) Decile (where 1 is most deprived 10% of LSOAs)', 0) or 0),

                            # Population data (mid 2022)
                            "total_population": int(row.get('Total population: mid 2022', 0) or 0),
                            "children_0_15": int(row.get('Dependent Children aged 0-15: mid 2022', 0) or 0),
                            "population_16_59": int(row.get('Working age population 18-66 (for use with Employment Deprivation Domain): mid 2022', 0) or 0),
                            "population_60_plus": int(row.get('Older population aged 60 and over: mid 2022', 0) or 0),
                        }
                    except (ValueError, TypeError):
                        continue

            print(f"  ✅ Loaded {len(self._data_by_lsoa)} LSOA records")
            return True

        except Exception as e:
            print(f"  ❌ Failed to load IMD data: {str(e)[:100]}")
            return False

    def get_imd_data(self, lsoa_code: str) -> Optional[Dict[str, Any]]:
        """
        Get IMD data for the given LSOA

        Args:
            lsoa_code: LSOA code (Census 2021 uses LSOA 2021 codes)

        Returns:
            IMD data dict including imd_decile, crime_decile, education_decile, etc.
        """
        # Make sure the data is loaded
        if self._data_by_lsoa is None:
            if not self._load_data():
                return None

        # Look up the data
        data = self._data_by_lsoa.get(lsoa_code)

        if data:
            return data

        # Not found: it may be an LSOA 2021 code, so try to find a close match
        # LSOA 2011 and 2021 codes have different formats; simply return None here
        return None

    def get_all_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all IMD data

        Returns:
            Dict keyed by LSOA code
        """
        if self._data_by_lsoa is None:
            self._load_data()
        return self._data_by_lsoa or {}
