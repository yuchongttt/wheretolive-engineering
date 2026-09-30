#!/usr/bin/env python3
"""
Surrounding environment evaluation module
Evaluates safety, greenery, air quality, street conditions, etc.
"""

from typing import Dict, Any, Optional
from api_helper import GeocodingService, UKPoliceAPI


class SurroundingEnvironmentEvaluator:
    """Surrounding environment evaluator"""

    def __init__(self, api_params: Optional[Dict[str, Any]] = None):
        """
        Initialise the surrounding environment evaluator.

        Args:
            api_params: API parameter config
        """
        self.max_score = 100
        self.api_params = api_params or {}

        # Initialise the APIs - uses GeocodingService (prefers Postcodes.io, falls back to Google)
        google_api_key = None
        if api_params and "google_api" in api_params:
            google_api_key = api_params["google_api"]["api_key"]
        self.geocoding_api = GeocodingService(google_api_key)
        self.police_api = UKPoliceAPI()

    def evaluate(self, property_data: Dict[str, Any]) -> float:
        """
        Evaluate the surrounding environment.

        Args:
            property_data: property data

        Returns:
            Score (0-100)
        """
        scores = []

        # Safety
        safety_score = self._evaluate_safety(property_data)
        scores.append(safety_score * 0.30)

        # Greenery / parks
        greenery_score = self._evaluate_greenery(property_data)
        scores.append(greenery_score * 0.20)

        # Air quality
        air_quality_score = self._evaluate_air_quality(property_data)
        scores.append(air_quality_score * 0.25)

        # Street cleanliness
        cleanliness_score = self._evaluate_cleanliness(property_data)
        scores.append(cleanliness_score * 0.15)

        # Neighbourhood
        community_score = self._evaluate_community(property_data)
        scores.append(community_score * 0.10)

        return sum(scores)

    def _evaluate_safety(self, data: Dict[str, Any]) -> float:
        """
        Evaluate safety.
        Uses the UK Police API to get real crime data.

        Scoring logic:
        - Base score: 100 pts
        - Deduct points by total crime count
        - Serious crimes (violence, robbery, weapons) deduct double
        """
        # If the API is not configured, use the placeholder heuristic
        if not self.geocoding_api or not self.police_api:
            return self._evaluate_safety_fallback(data)

        address = data.get("address", "")
        if not address:
            print("  ⚠️  No address provided, cannot evaluate safety")
            return self._evaluate_safety_fallback(data)

        # If crime data is already cached, use it directly
        if "_api_crime_data" in data:
            crime_counts = data["_api_crime_data"]
        else:
            # 1. Get latitude/longitude
            print(f"  📍 Fetching address coordinates...")
            coords = self.geocoding_api.get_coordinates(address)

            if not coords or not coords.get("success"):
                print("  ❌ Could not get address coordinates, using default score")
                return self._evaluate_safety_fallback(data)

            lat = coords["lat"]
            lng = coords["lng"]
            print(f"  ✅ Coordinates: {lat:.6f}, {lng:.6f}")

            # 2. Get crime data (from 3 months ago)
            print(f"  🚓 Querying crime data...")
            crimes = self.police_api.get_crimes_at_location(lat, lng)

            if crimes is None:
                print("  ❌ Could not get crime data, using default score")
                return self._evaluate_safety_fallback(data)

            # 3. Count crimes by category
            crime_counts = self.police_api.count_crimes_by_category(crimes)

            # 4. If total=0, the local police force may have stopped reporting (e.g. GMP has not reported since 2019)
            #    Check via locate-neighbourhood + crimes-street-dates; no force is hardcoded
            if crime_counts.get("total", 0) == 0:
                force_status = self.police_api.is_force_publishing(lat, lng)
                if force_status.get("publishing") is False:
                    crime_counts["_data_unavailable"] = True
                    crime_counts["force_id"] = force_status.get("force_id")
                    crime_counts["force_name"] = force_status.get("force_name")
                    print(
                        f"  ⚠️  {force_status.get('force_name')} does not publish data to data.police.uk; "
                        "skipping the crime dimension for this evaluation"
                    )

            # Cache the result
            data["_api_crime_data"] = crime_counts
            data["_api_crime_coords"] = {"lat": lat, "lng": lng}

        # If the force does not publish data, use the fallback (silently, so a false 0 crimes does not inflate the score)
        if crime_counts.get("_data_unavailable"):
            return self._evaluate_safety_fallback(data)

        # 4. Compute the score from the crime count
        total_crimes = crime_counts["total"]

        # Print crime stats
        print(f"  📊 Crime stats (one month of data):")
        print(f"     Total: {total_crimes}")
        if crime_counts.get("anti-social-behaviour", 0) > 0:
            print(f"     Anti-social behaviour: {crime_counts['anti-social-behaviour']}")
        if crime_counts.get("burglary", 0) > 0:
            print(f"     Burglary: {crime_counts['burglary']}")
        if crime_counts.get("violent-crime", 0) > 0:
            print(f"     Violent crime: {crime_counts['violent-crime']}")
        if crime_counts.get("robbery", 0) > 0:
            print(f"     Robbery: {crime_counts['robbery']}")
        if crime_counts.get("theft-from-the-person", 0) > 0:
            print(f"     Theft from the person: {crime_counts['theft-from-the-person']}")
        if crime_counts.get("drugs", 0) > 0:
            print(f"     Drugs: {crime_counts['drugs']}")
        if crime_counts.get("possession-of-weapons", 0) > 0:
            print(f"     Possession of weapons: {crime_counts['possession-of-weapons']}")

        # Scoring rules (based on a 1 km × 1 km area; baseline: 350 crimes = 60 pts)
        if total_crimes <= 150:
            # Excellent area: 150 crimes or fewer
            final_score = 100
        elif total_crimes <= 350:
            # Good area: 150-350 crimes, linear from 100 down to 60
            # score = 100 - (total_crimes - 150) / (350 - 150) * 40
            final_score = 100 - (total_crimes - 150) / 200 * 40
        else:
            # Fair and poor areas: over 350 crimes, linear from 60 down to 0
            # Reaches 0 pts at 1000 crimes
            final_score = 60 - (total_crimes - 350) / 650 * 60
            # Floor at 0 pts
            final_score = max(0, final_score)

        # Rating (based on total crimes, calibrated for a 1 km² area)
        if total_crimes <= 150:
            rating = "Excellent (≤150 crimes)"
        elif total_crimes <= 250:
            rating = "Good (≤250 crimes)"
        elif total_crimes <= 350:
            rating = "Moderate (≤350 crimes)"
        elif total_crimes <= 500:
            rating = "Fair (≤500 crimes)"
        elif total_crimes <= 700:
            rating = "Poor (≤700 crimes)"
        else:
            rating = "Very poor (>700 crimes)"

        print(f"  ✅ Safety score: {final_score:.2f}/100 ({rating})\n")

        return final_score

    def _evaluate_safety_fallback(self, data: Dict[str, Any]) -> float:
        """Evaluate safety (fallback placeholder heuristic)"""
        crime_rate = data.get("crime_rate", "Medium")
        has_street_lighting = data.get("has_street_lighting", True)
        police_station_distance = data.get("police_station_distance_km", 2)

        crime_scores = {
            "Very Low": 100,
            "Low": 90,
            "Medium": 70,
            "High": 50,
            "Very High": 30
        }

        base_score = crime_scores.get(crime_rate, 70)

        # Street lighting bonus
        lighting_bonus = 10 if has_street_lighting else 0

        # Police station distance score
        police_score = max(0, 100 - (police_station_distance * 10))

        return (base_score * 0.7 + lighting_bonus + police_score * 0.2)

    def _evaluate_greenery(self, data: Dict[str, Any]) -> float:
        """Evaluate greenery / parks (placeholder heuristic)"""
        # TODO: implement the actual evaluation logic
        # Factors: park distance, number of parks, green coverage
        park_distance = data.get("park_distance_m", 1000)
        num_parks_nearby = data.get("num_parks_nearby", 2)
        greenery_coverage = data.get("greenery_coverage_percent", 20)

        # Distance score
        if park_distance <= 200:
            distance_score = 100
        elif park_distance <= 500:
            distance_score = 85
        elif park_distance <= 1000:
            distance_score = 70
        else:
            distance_score = 50

        # Quantity and coverage scores
        quantity_score = min(100, num_parks_nearby * 25)
        coverage_score = min(100, greenery_coverage * 2)

        return (distance_score * 0.5 + quantity_score * 0.2 + coverage_score * 0.3)

    def _evaluate_air_quality(self, data: Dict[str, Any]) -> float:
        """Evaluate air quality (placeholder heuristic)"""
        # TODO: implement the actual evaluation logic
        # Factors: AQI, distance to main roads, distance to industrial areas
        air_quality_index = data.get("air_quality_index", 70)  # 0-100, lower is better
        distance_to_main_road = data.get("distance_to_main_road_m", 100)
        near_industrial_area = data.get("near_industrial_area", False)

        # AQI score (lower AQI is better)
        if air_quality_index <= 50:
            aqi_score = 100
        elif air_quality_index <= 100:
            aqi_score = 80
        elif air_quality_index <= 150:
            aqi_score = 60
        else:
            aqi_score = 40

        # Main-road distance score
        road_score = min(100, distance_to_main_road / 5)

        # Industrial-area penalty
        industrial_penalty = 20 if near_industrial_area else 0

        return max(0, min(100, aqi_score * 0.6 + road_score * 0.4 - industrial_penalty))

    def _evaluate_cleanliness(self, data: Dict[str, Any]) -> float:
        """Evaluate street cleanliness (placeholder heuristic)"""
        # TODO: implement the actual evaluation logic
        # Factors: street cleanliness, waste collection frequency, building maintenance
        street_cleanliness = data.get("street_cleanliness", "Good")
        waste_collection_frequency = data.get("waste_collection_per_week", 2)
        building_maintenance = data.get("building_maintenance", "Good")

        cleanliness_scores = {
            "Excellent": 100,
            "Good": 85,
            "Fair": 70,
            "Poor": 50,
            "Very Poor": 30
        }

        street_score = cleanliness_scores.get(street_cleanliness, 70)
        collection_score = min(100, waste_collection_frequency * 25)
        maintenance_score = cleanliness_scores.get(building_maintenance, 70)

        return (street_score * 0.5 + collection_score * 0.2 + maintenance_score * 0.3)

    def _evaluate_community(self, data: Dict[str, Any]) -> float:
        """Evaluate the neighbourhood (placeholder heuristic)"""
        # TODO: implement the actual evaluation logic
        # Factors: population mix, community atmosphere, community events
        community_diversity = data.get("community_diversity", "High")
        community_atmosphere = data.get("community_atmosphere", "Friendly")
        community_events_per_month = data.get("community_events_per_month", 2)

        diversity_scores = {
            "Very High": 100,
            "High": 90,
            "Medium": 75,
            "Low": 60
        }

        atmosphere_scores = {
            "Very Friendly": 100,
            "Friendly": 85,
            "Neutral": 70,
            "Unfriendly": 50,
            "Very Unfriendly": 30
        }

        diversity_score = diversity_scores.get(community_diversity, 75)
        atmosphere_score = atmosphere_scores.get(community_atmosphere, 70)
        events_score = min(100, community_events_per_month * 20)

        return (diversity_score * 0.3 + atmosphere_score * 0.5 + events_score * 0.2)
