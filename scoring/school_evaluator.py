#!/usr/bin/env python3
"""
School Evaluator
Evaluates the quality of schools near a postcode using GIAS school data and Ofsted ratings

Data sources:
- GIAS: https://www.get-information-schools.service.gov.uk/Downloads
  (Establishment fields CSV - contains school location, type and status)
- Ofsted Management Information:
  https://www.gov.uk/government/statistical-data-sets/monthly-management-information-ofsteds-school-inspections-outcomes
  (Most recent inspections sheet - contains Ofsted ratings)
"""

import csv
import math
import os
import re
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

from school_geo import bng_to_latlon, classify_category, INDEPENDENT_TYPES, SPECIAL_TYPES


class SchoolEvaluator:
    """School quality evaluator"""

    # Ofsted rating mapping
    OFSTED_LABELS = {
        1: "Outstanding",
        2: "Good",
        3: "Requires Improvement",
        4: "Inadequate",
    }

    # Establishment type → category mapping (shared with build_schools_table via school_geo)
    _INDEPENDENT_TYPES = INDEPENDENT_TYPES
    _SPECIAL_TYPES = SPECIAL_TYPES

    @staticmethod
    def classify_category(establishment_type: str) -> str:
        """Classify school establishment type into a simple category."""
        return classify_category(establishment_type)

    # Scoring weights
    WEIGHTS = {
        "avg_rating": 0.40,        # Average Ofsted rating of nearby schools
        "outstanding_nearby": 0.30, # Density of Outstanding schools
        "choice_diversity": 0.10,   # School choice diversity (count + primary/secondary coverage)
        "worst_avoidance": 0.20,    # Avoidance of poor schools (few RI/Inadequate)
    }

    # Search radius (km)
    DEFAULT_RADIUS_KM = 1.5

    def __init__(self, data_dir: str = None):
        """
        Initialise the school evaluator

        Args:
            data_dir: path to the data directory
        """
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self.data_dir = data_dir
        self._schools = None  # lazy load
        self._outcode_index = None  # Index grouped by postcode outcode

    def _bng_to_latlon(self, easting: float, northing: float) -> Tuple[float, float]:
        """Convert British National Grid (OSGB36) coordinates to latitude/longitude (delegates to school_geo)."""
        return bng_to_latlon(easting, northing)

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute the Haversine distance between two points (km)"""
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _extract_outcode(postcode: str) -> str:
        """Extract the outcode (first half) from a postcode"""
        cleaned = postcode.replace(" ", "").upper()
        # Full postcode: drop the last 3 characters
        if len(cleaned) >= 5:
            return cleaned[:-3]
        return cleaned

    def _load_data(self):
        """Load and merge GIAS + Ofsted data"""
        if self._schools is not None:
            return

        gias_path = os.path.join(self.data_dir, "edubasealldata.csv")
        ofsted_path = os.path.join(self.data_dir, "ofsted_ratings.csv")

        if not os.path.exists(gias_path):
            raise FileNotFoundError(
                f"GIAS data file not found: {gias_path}\n"
                f"Download it from https://www.get-information-schools.service.gov.uk/Downloads"
            )
        if not os.path.exists(ofsted_path):
            raise FileNotFoundError(
                f"Ofsted ratings file not found: {ofsted_path}\n"
                f"Run the data preparation script first to generate ofsted_ratings.csv"
            )

        # 1. Load Ofsted ratings (URN -> rating)
        print("  📚 Loading Ofsted rating data...")
        ofsted_ratings = {}
        with open(ofsted_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    urn = row["URN"].strip()
                    rating = int(row["OverallEffectiveness"])
                    if 1 <= rating <= 4:
                        ofsted_ratings[urn] = rating
                except (ValueError, KeyError):
                    continue
        print(f"  ✅ Loaded {len(ofsted_ratings)} Ofsted ratings")

        # 2. Load GIAS data, keeping only Open schools that have coordinates
        print("  🏫 Loading GIAS school data...")
        schools = []
        self._outcode_index = defaultdict(list)

        # Valid school phases
        valid_phases = {
            "Primary", "Secondary", "Middle deemed primary",
            "Middle deemed secondary", "All-through", "16 plus",
            "Not applicable",  # some academies
        }

        with open(gias_path, encoding="cp1252") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Open schools only
                if row.get("EstablishmentStatus (name)", "").strip() != "Open":
                    continue

                urn = row.get("URN", "").strip()
                easting = row.get("Easting", "").strip()
                northing = row.get("Northing", "").strip()
                postcode = row.get("Postcode", "").strip()

                if not easting or not northing or not postcode:
                    continue

                try:
                    e = float(easting)
                    n = float(northing)
                except ValueError:
                    continue

                # Convert coordinates
                lat, lon = self._bng_to_latlon(e, n)

                phase = row.get("PhaseOfEducation (name)", "").strip()

                # Classify the stage
                if phase in ("Primary", "Middle deemed primary"):
                    stage = "primary"
                elif phase in ("Secondary", "Middle deemed secondary", "16 plus"):
                    stage = "secondary"
                elif phase == "All-through":
                    stage = "all-through"
                else:
                    stage = "other"

                # Get the Ofsted rating
                rating = ofsted_ratings.get(urn)

                est_type = row.get("TypeOfEstablishment (name)", "").strip()
                school = {
                    "urn": urn,
                    "name": row.get("EstablishmentName", "").strip(),
                    "type": est_type,
                    "category": self.classify_category(est_type),
                    "phase": phase,
                    "stage": stage,
                    "postcode": postcode,
                    "lat": lat,
                    "lon": lon,
                    "ofsted_rating": rating,  # None if no rating
                }

                schools.append(school)

                # Build the outcode index
                outcode = self._extract_outcode(postcode)
                self._outcode_index[outcode].append(school)

        self._schools = schools
        rated = sum(1 for s in schools if s["ofsted_rating"] is not None)
        print(f"  ✅ Loaded {len(schools)} Open schools ({rated} with an Ofsted rating)")

    def _get_nearby_outcodes(self, outcode: str) -> List[str]:
        """
        Get the list of nearby outcodes (to narrow the search)
        Returns the input outcode plus all "neighbours"
        """
        # Simple strategy: return all known outcodes
        # As a performance optimisation, only geographically adjacent ones could be returned
        # but since results are filtered by haversine in the end, a rough candidate set is enough here
        return list(self._outcode_index.keys())

    def find_nearby_schools(
        self,
        lat: float,
        lon: float,
        radius_km: float = None,
        rated_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Find schools near the given coordinates

        Args:
            lat: latitude
            lon: longitude
            radius_km: search radius (km)
            rated_only: only return schools with an Ofsted rating

        Returns:
            List of nearby schools, sorted by distance
        """
        self._load_data()

        if radius_km is None:
            radius_km = self.DEFAULT_RADIUS_KM

        nearby = []
        for school in self._schools:
            dist = self._haversine(lat, lon, school["lat"], school["lon"])
            if dist <= radius_km:
                if rated_only and school["ofsted_rating"] is None:
                    continue
                nearby.append({**school, "distance_km": round(dist, 3)})

        nearby.sort(key=lambda x: x["distance_km"])
        return nearby

    def _distance_weight(self, distance_km: float) -> float:
        """Distance-decay weight: closer means higher weight"""
        # Exponential decay: 0km=1.0, 0.5km=0.78, 1km=0.61, 1.5km=0.47
        return math.exp(-0.5 * distance_km)

    def score(
        self,
        lat: float,
        lon: float,
        radius_km: float = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute the school quality score

        Args:
            lat: latitude
            lon: longitude
            radius_km: search radius

        Returns:
            (score 0-100, details dict)
        """
        if radius_km is None:
            radius_km = self.DEFAULT_RADIUS_KM

        nearby = self.find_nearby_schools(lat, lon, radius_km)
        rated = [s for s in nearby if s["ofsted_rating"] is not None]

        if not rated:
            return 40.0, {
                "total_schools": len(nearby),
                "rated_schools": 0,
                "nearby_schools": nearby[:10],
                "message": "No nearby schools with an Ofsted rating",
            }

        # 1. Weighted average Ofsted rating (Outstanding=4→high score, Inadequate=1→low score)
        # Note: Ofsted 1=Outstanding (best), 4=Inadequate (worst)
        # Convert as 4-rating+1 so that Outstanding=4, Good=3, RI=2, Inadequate=1
        weighted_sum = 0
        weight_sum = 0
        for s in rated:
            w = self._distance_weight(s["distance_km"])
            quality = 5 - s["ofsted_rating"]  # 1→4, 2→3, 3→2, 4→1
            weighted_sum += quality * w
            weight_sum += w

        avg_quality = weighted_sum / weight_sum if weight_sum > 0 else 2.5
        # avg_quality ranges 1-4; map it to 0-100
        avg_rating_score = (avg_quality - 1) / 3 * 100

        # 2. Density of Outstanding schools (within 1km)
        outstanding_1km = [s for s in rated
                          if s["ofsted_rating"] == 1 and s["distance_km"] <= 1.0]
        good_or_better_1km = [s for s in rated
                              if s["ofsted_rating"] <= 2 and s["distance_km"] <= 1.0]

        # Outstanding score: 0 schools=0, 1=40, 2=65, 3=80, 4+=100
        outstanding_count = len(outstanding_1km)
        if outstanding_count == 0:
            outstanding_score = 0
        elif outstanding_count == 1:
            outstanding_score = 40
        elif outstanding_count == 2:
            outstanding_score = 65
        elif outstanding_count == 3:
            outstanding_score = 80
        else:
            outstanding_score = min(100, 80 + outstanding_count * 5)

        # Add a bonus for Good schools
        good_bonus = min(20, len(good_or_better_1km) * 3)
        outstanding_score = min(100, outstanding_score + good_bonus)

        # 3. School choice diversity
        has_primary = any(s["stage"] in ("primary", "all-through") for s in rated)
        has_secondary = any(s["stage"] in ("secondary", "all-through") for s in rated)

        # Base score: points each for having primary and secondary
        diversity_score = 0
        if has_primary:
            diversity_score += 30
        if has_secondary:
            diversity_score += 30

        # Bonus for total count
        total_rated = len(rated)
        if total_rated >= 2:
            diversity_score += 10
        if total_rated >= 5:
            diversity_score += 10
        if total_rated >= 10:
            diversity_score += 10
        if total_rated >= 15:
            diversity_score += 10

        diversity_score = min(100, diversity_score)

        # 4. Poor-school avoidance score
        ri_or_worse = [s for s in rated if s["ofsted_rating"] >= 3]
        inadequate = [s for s in rated if s["ofsted_rating"] == 4]

        if not ri_or_worse:
            avoidance_score = 100
        else:
            ri_ratio = len(ri_or_worse) / len(rated)
            avoidance_score = max(0, 100 - ri_ratio * 120)
            # Extra penalty for Inadequate schools
            if inadequate:
                avoidance_score = max(0, avoidance_score - len(inadequate) * 15)

        # Combined weighting
        final_score = (
            self.WEIGHTS["avg_rating"] * avg_rating_score +
            self.WEIGHTS["outstanding_nearby"] * outstanding_score +
            self.WEIGHTS["choice_diversity"] * diversity_score +
            self.WEIGHTS["worst_avoidance"] * avoidance_score
        )

        # Build details
        rating_dist = defaultdict(int)
        for s in rated:
            rating_dist[s["ofsted_rating"]] += 1

        details = {
            "total_schools": len(nearby),
            "rated_schools": len(rated),
            "outstanding_count": rating_dist.get(1, 0),
            "good_count": rating_dist.get(2, 0),
            "ri_count": rating_dist.get(3, 0),
            "inadequate_count": rating_dist.get(4, 0),
            "outstanding_1km": len(outstanding_1km),
            "avg_quality": round(avg_quality, 2),
            "has_primary": has_primary,
            "has_secondary": has_secondary,
            "component_scores": {
                "avg_rating": round(avg_rating_score, 1),
                "outstanding_nearby": round(outstanding_score, 1),
                "choice_diversity": round(diversity_score, 1),
                "worst_avoidance": round(avoidance_score, 1),
            },
            "nearby_schools": nearby[:20],  # Top 20 closest
        }

        return round(final_score, 2), details

    def evaluate(self, property_data: Dict[str, Any]) -> float:
        """
        Evaluate the quality of schools near a property (compatible with the batch_evaluate interface)

        Args:
            property_data: property data dict; must contain coordinates

        Returns:
            Score (0-100)
        """
        # Get from the unified coordinates
        coords = property_data.get("_api_coords")
        if not coords:
            print("  ⚠️  No coordinates available; cannot evaluate schools")
            return 40.0

        lat = coords["lat"]
        lng = coords["lng"]

        print(f"  🏫 Evaluating nearby schools...")
        score, details = self.score(lat, lng)

        # Store details in property_data for later use
        property_data["_api_schools_data"] = details

        # Print summary
        rated = details["rated_schools"]
        total = details["total_schools"]
        print(f"  ✅ Found {total} schools ({rated} with an Ofsted rating)")

        if rated > 0:
            print(f"     Outstanding: {details['outstanding_count']}, "
                  f"Good: {details['good_count']}, "
                  f"RI: {details['ri_count']}, "
                  f"Inadequate: {details['inadequate_count']}")
            print(f"     Outstanding within 1km: {details['outstanding_1km']}")
            print(f"     Average quality: {details['avg_quality']:.2f}/4.00")

            # Print the 5 closest rated schools
            rated_nearby = [s for s in details["nearby_schools"] if s.get("ofsted_rating")]
            if rated_nearby:
                print(f"     Closest rated schools:")
                for s in rated_nearby[:5]:
                    label = self.OFSTED_LABELS.get(s["ofsted_rating"], "Unknown")
                    print(f"       • {s['name']} ({s['stage']}) - "
                          f"{label} - {s['distance_km']:.2f}km")

        return score


# Standalone test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from api_helper import GeocodingService

    evaluator = SchoolEvaluator()
    geocoding = GeocodingService()

    test_postcodes = ["SW7 2AZ", "E6 1NW", "N1 1AA", "SE1 7PB"]

    for pc in test_postcodes:
        print(f"\n{'='*60}")
        print(f"📍 Test postcode: {pc}")
        print(f"{'='*60}")

        coords = geocoding.get_coordinates(pc)
        if not coords or not coords.get("success"):
            print(f"  ❌ Could not get coordinates")
            continue

        lat, lng = coords["lat"], coords["lng"]
        print(f"  Coordinates: {lat:.6f}, {lng:.6f}")

        score, details = evaluator.score(lat, lng)
        print(f"\n  📊 School score: {score:.2f}/100")
        print(f"  Found {details['total_schools']} schools ({details['rated_schools']} rated)")
        print(f"  Outstanding: {details['outstanding_count']}, "
              f"Good: {details['good_count']}, "
              f"RI: {details['ri_count']}, "
              f"Inadequate: {details['inadequate_count']}")
        print(f"  Outstanding within 1km: {details['outstanding_1km']}")

        print(f"\n  Component scores:")
        for k, v in details["component_scores"].items():
            print(f"    {k}: {v}")

        # Show the closest rated schools
        rated_nearby = [s for s in details["nearby_schools"] if s.get("ofsted_rating")]
        if rated_nearby:
            print(f"\n  Closest rated schools (top 10):")
            for s in rated_nearby[:10]:
                label = SchoolEvaluator.OFSTED_LABELS.get(s["ofsted_rating"], "?")
                print(f"    {s['distance_km']:5.2f}km | {label:<22} | {s['stage']:<12} | {s['name']}")
