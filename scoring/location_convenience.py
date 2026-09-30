#!/usr/bin/env python3
"""
Location convenience evaluation module
Evaluates commute, transport, shopping, healthcare and other conveniences
"""

import json
import re
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from api_helper import GoogleRoutesAPI, GeocodingService, TfLAPI, load_api_params
from amenities_evaluator import AmenitiesEvaluator
from transit_convenience_evaluator import TransitConvenienceEvaluator
from long_distance_travel_evaluator import LongDistanceTravelEvaluator


class LocationConvenienceEvaluator:
    """Location convenience evaluator"""

    LONDON_HUBS = [
        {"name": "Bank", "postcode": "EC2R 8AH"},
        {"name": "Canary Wharf", "postcode": "E14 5AB"},
        {"name": "King's Cross", "postcode": "N1C 4AX"},
        {"name": "Victoria", "postcode": "SW1V 1JU"},
        {"name": "Liverpool Street", "postcode": "EC2M 7PY"},
    ]

    def __init__(self, api_params: Optional[Dict[str, Any]] = None, commute_config_path: str = "commute_config.json"):
        """
        Initialise the evaluator.

        Args:
            api_params: API parameter config; loaded from file if None
            commute_config_path: path to the commute config file
        """
        self.max_score = 100

        # Load API parameters
        if api_params is None:
            api_params = load_api_params()

        self.api_params = api_params

        # Load the commute config
        self.commute_config = self._load_commute_config(commute_config_path)

        # Initialise the TfL API (free, always available)
        self.tfl_api = TfLAPI()

        # Initialise the APIs
        google_api_key = None
        if api_params and "google_api" in api_params:
            google_api_key = api_params["google_api"]["api_key"]
            self.routes_api = GoogleRoutesAPI(google_api_key)
            self.use_google_api = True
        else:
            self.routes_api = None
            self.use_google_api = False

        # GeocodingService is always available (prefers Postcodes.io, falls back to Google)
        self.geocoding_api = GeocodingService(google_api_key)

        # Initialise the nearby amenities evaluator
        self.amenities_evaluator = AmenitiesEvaluator(api_params)

        # Initialise the public transport convenience evaluator
        self.transit_evaluator = TransitConvenienceEvaluator(api_params)

        # Initialise the long-distance travel evaluator (prefers the TfL API)
        self.long_distance_evaluator = LongDistanceTravelEvaluator(
            places_api=self.amenities_evaluator.places_api if hasattr(self.amenities_evaluator, 'places_api') else None,
            routes_api=self.routes_api,
            geocoding_api=self.geocoding_api,
            tfl_api=self.tfl_api
        )

    def _load_commute_config(self, config_path: str) -> Dict[str, Any]:
        """Load the commute config file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: commute config file {config_path} not found, using defaults")
            return {
                "office_address": "",
                "travel_mode": "TRANSIT",
                "schedule": {
                    "morning": {"enabled": True, "departure_time": None},
                    "evening": {"enabled": True, "departure_time": None}
                },
                "scoring_weights": {"morning": 0.6, "evening": 0.4}
            }
        except json.JSONDecodeError as e:
            print(f"Warning: failed to parse commute config file - {e}")
            return {}

    def _is_sub_dimension_enabled(self, property_data: Dict[str, Any], sub_dim: str) -> bool:
        """
        Check whether a sub-dimension is enabled.

        Args:
            property_data: property data (includes the evaluation config)
            sub_dim: sub-dimension name (commute, transit, amenities, long_distance)

        Returns:
            Whether it is enabled
        """
        eval_config = property_data.get("_eval_config", {})
        dimensions = eval_config.get("dimensions", {})
        location_config = dimensions.get("location_convenience", {})
        sub_dimensions = location_config.get("sub_dimensions", {})
        sub_config = sub_dimensions.get(sub_dim, {})
        # Enabled by default
        return sub_config.get("enabled", True)

    def _evaluate_amenities_sub(self, property_data: Dict[str, Any]) -> float:
        """Evaluate nearby amenities (with API fallback logic)"""
        amenities_score = self.amenities_evaluator.evaluate(property_data, radius=500)
        if amenities_score > 0 and "_api_amenities_data" in property_data:
            return amenities_score
        else:
            # Use the placeholder heuristic data
            shopping_score = self._evaluate_shopping(property_data)
            dining_score = self._evaluate_dining(property_data)
            services_score = self._evaluate_basic_services(property_data)
            return (shopping_score * 0.65 + dining_score * 0.22 + services_score * 0.13)

    def evaluate(self, property_data: Dict[str, Any]) -> float:
        """
        Evaluate location convenience.

        Args:
            property_data: property data

        Returns:
            Score (0-100)
        """
        # Base weight for each sub-dimension
        base_weights = {
            "commute": 0.30,         # commute convenience
            "transit": 0.25,         # public transport access
            "long_distance": 0.10,   # airports / railway stations
            "medical": 0.12,         # healthcare facilities
            "amenities": 0.23        # nearby amenities
        }

        # Build the list of enabled sub-dimension tasks
        # Each item: (key, display_name, callable)
        tasks = []
        skipped = []

        if self._is_sub_dimension_enabled(property_data, "commute"):
            tasks.append(("commute", "Commute", lambda: self._evaluate_commute(property_data)))
        else:
            skipped.append("Commute")

        if self._is_sub_dimension_enabled(property_data, "transit"):
            tasks.append(("transit", "Public transport", lambda: self._evaluate_public_transport(property_data)))
        else:
            skipped.append("Public transport")

        if self._is_sub_dimension_enabled(property_data, "long_distance"):
            tasks.append(("long_distance", "Long-distance travel", lambda: self._evaluate_travel_hubs(property_data)))
        else:
            skipped.append("Long-distance travel")

        # Healthcare facilities are always enabled
        tasks.append(("medical", "Healthcare", lambda: self._evaluate_medical_facilities(property_data)))

        if self._is_sub_dimension_enabled(property_data, "amenities"):
            tasks.append(("amenities", "Nearby amenities", lambda: self._evaluate_amenities_sub(property_data)))
        else:
            skipped.append("Nearby amenities")

        # Print skipped sub-dimensions
        if skipped:
            print(f"  ⏭️  Skipping location convenience sub-dimensions: {', '.join(skipped)}")

        if not tasks:
            print("  ⚠️  All location convenience sub-dimensions are disabled")
            return 0

        # Run all enabled sub-dimensions in parallel
        scores = {}
        enabled_weights = {}
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_to_key = {}
            for key, name, fn in tasks:
                future_to_key[executor.submit(fn)] = (key, name)

            for future in as_completed(future_to_key):
                key, name = future_to_key[future]
                try:
                    scores[key] = future.result()
                    enabled_weights[key] = base_weights[key]
                except Exception as e:
                    print(f"  ⚠️  {name} evaluation raised an exception: {e}")
                    scores[key] = 0
                    enabled_weights[key] = base_weights[key]

        # If no sub-dimension is enabled, return 0
        if not enabled_weights:
            print("  ⚠️  All location convenience sub-dimensions are disabled")
            return 0

        # Renormalise the weights (so they sum to 1)
        total_weight = sum(enabled_weights.values())
        adjusted_weights = {k: v / total_weight for k, v in enabled_weights.items()}

        # Compute the weighted total
        final_score = sum(scores[k] * adjusted_weights[k] for k in scores)

        return final_score

    def _is_free_api_mode(self, property_data: Dict[str, Any]) -> bool:
        """Check whether we are in free-API mode"""
        eval_config = property_data.get("_eval_config", {})
        return eval_config.get("_api_tier") == "free"

    def _extract_postcode(self, address: str) -> Optional[str]:
        """Extract the postcode from an address"""
        postcode_pattern = r'\b([A-Z]{1,2}[0-9][0-9A-Z]?\s*[0-9][A-Z]{2})\b'
        match = re.search(postcode_pattern, address.upper())
        if match:
            return match.group(1).replace(" ", "")
        return None

    def _evaluate_commute(self, data: Dict[str, Any]) -> float:
        """Evaluate commute convenience (prefers the TfL API, falls back to the Google Routes API on failure)"""
        property_address = data.get("address", "")

        if not property_address:
            print("Warning: no property address provided, using default score")
            return self._evaluate_commute_fallback(data)

        # Get the commute config
        office_address = self.commute_config.get("office_address", "")
        if not office_address:
            print("Warning: no workplace configured, using default score")
            return self._evaluate_commute_fallback(data)

        # Check whether we are in London (the TfL API only covers London)
        postcode = self._extract_postcode(property_address)
        is_london = postcode and self.tfl_api.is_london_postcode(postcode)
        office_postcode = self._extract_postcode(office_address)
        office_is_london = office_postcode and self.tfl_api.is_london_postcode(office_postcode)

        # 1. Prefer the TfL API (if both ends are in London)
        if is_london and office_is_london:
            tfl_result = self._evaluate_commute_with_tfl(data, property_address, office_address)
            if tfl_result is not None and tfl_result > 0:
                return tfl_result
            print("  ⚠️  TfL API query failed, trying the Google Routes API...")

        # 2. Fall back to the Google Routes API
        if self.use_google_api and self.routes_api:
            print(f"🗺️  Computing commute with the Google Routes API...")
            travel_mode = self.commute_config.get("travel_mode", "TRANSIT")
            schedule = self.commute_config.get("schedule", {})
            weights = self.commute_config.get("scoring_weights", {"morning": 0.6, "evening": 0.4})

            commute_results = {}
            total_weighted_score = 0
            total_weight = 0

            # Morning commute (home -> office)
            morning_config = schedule.get("morning", {})
            if morning_config.get("enabled", True):
                print(f"🌅 Querying morning commute: {property_address} -> {office_address}")
                morning_result = self.routes_api.compute_route(
                    property_address,
                    office_address,
                    travel_mode,
                    departure_time=morning_config.get("departure_time")
                )

                if morning_result and morning_result.get("success"):
                    morning_time = morning_result["duration_minutes"]
                    morning_distance = morning_result["distance_meters"] / 1000
                    morning_score = self._calculate_commute_score(morning_time, morning_distance, travel_mode)

                    print(f"  ✅ Morning commute: {morning_time:.1f} min, {morning_distance:.2f} km, score: {morning_score:.1f}")

                    commute_results["morning"] = {
                        "time_minutes": morning_time,
                        "distance_km": morning_distance,
                        "score": morning_score,
                        "departure_time": morning_config.get("departure_time", "not set")
                    }

                    weight = weights.get("morning", 0.6)
                    total_weighted_score += morning_score * weight
                    total_weight += weight
                else:
                    print(f"  ❌ Morning commute query failed")

            # Evening commute (office -> home)
            evening_config = schedule.get("evening", {})
            if evening_config.get("enabled", True):
                print(f"🌙 Querying evening commute: {office_address} -> {property_address}")
                evening_result = self.routes_api.compute_route(
                    office_address,  # note: reversed
                    property_address,
                    travel_mode,
                    departure_time=evening_config.get("departure_time")
                )

                if evening_result and evening_result.get("success"):
                    evening_time = evening_result["duration_minutes"]
                    evening_distance = evening_result["distance_meters"] / 1000
                    evening_score = self._calculate_commute_score(evening_time, evening_distance, travel_mode)

                    print(f"  ✅ Evening commute: {evening_time:.1f} min, {evening_distance:.2f} km, score: {evening_score:.1f}")

                    commute_results["evening"] = {
                        "time_minutes": evening_time,
                        "distance_km": evening_distance,
                        "score": evening_score,
                        "departure_time": evening_config.get("departure_time", "not set")
                    }

                    weight = weights.get("evening", 0.4)
                    total_weighted_score += evening_score * weight
                    total_weight += weight
                else:
                    print(f"  ❌ Evening commute query failed")

            # Save the commute results into data
            if commute_results:
                data["_api_commute_details"] = commute_results

                # Compute weighted average time and distance (for display)
                if "morning" in commute_results and "evening" in commute_results:
                    avg_time = (commute_results["morning"]["time_minutes"] * weights.get("morning", 0.6) +
                                commute_results["evening"]["time_minutes"] * weights.get("evening", 0.4))
                    avg_distance = (commute_results["morning"]["distance_km"] * weights.get("morning", 0.6) +
                                    commute_results["evening"]["distance_km"] * weights.get("evening", 0.4))
                    data["_api_commute_time_minutes"] = avg_time
                    data["_api_commute_distance_km"] = avg_distance
                elif "morning" in commute_results:
                    data["_api_commute_time_minutes"] = commute_results["morning"]["time_minutes"]
                    data["_api_commute_distance_km"] = commute_results["morning"]["distance_km"]
                elif "evening" in commute_results:
                    data["_api_commute_time_minutes"] = commute_results["evening"]["time_minutes"]
                    data["_api_commute_distance_km"] = commute_results["evening"]["distance_km"]

                # Return the weighted average score
                if total_weight > 0:
                    final_score = total_weighted_score / total_weight
                    print(f"📊 Overall commute score: {final_score:.1f}/100 ", end="\n\n")
                    return final_score
                else:
                    print("API query failed, using fallback scoring")
                    return self._evaluate_commute_fallback(data)
            else:
                print("All commute queries failed, using fallback scoring")
                return self._evaluate_commute_fallback(data)
        else:
            # No API available (outside London and no Google API); use the fallback
            if not (is_london and office_is_london):
                print("  ⚠️  Outside London, TfL API not available")
            print("  ⚠️  Using fallback scoring")
            return self._evaluate_commute_fallback(data)

    def _evaluate_commute_with_tfl(
        self,
        data: Dict[str, Any],
        property_address: str,
        office_address: str
    ) -> Optional[float]:
        """Evaluate the commute with the TfL Journey Planner (free)

        Returns:
            Score (0-100), or None if the query failed
        """
        print(f"🚇 Computing commute with the TfL Journey Planner...")

        # Commute config (morning peak only)
        weights = {"morning": 1.0}

        # Extract postcodes as the location parameters
        from_postcode = self._extract_postcode(property_address)
        to_postcode = self._extract_postcode(office_address)

        if not from_postcode or not to_postcode:
            print("  ⚠️  Could not extract a postcode from the address")
            return None  # return None so we fall back to the Google API

        commute_results = {}
        total_weighted_score = 0
        total_weight = 0

        # Work out the date of the next working day
        from datetime import datetime, timedelta
        now = datetime.now()
        next_workday = now
        while next_workday.weekday() >= 5:  # skip weekends
            next_workday += timedelta(days=1)
        if next_workday == now and now.hour >= 10:  # today's morning peak has passed
            next_workday += timedelta(days=1)
            while next_workday.weekday() >= 5:
                next_workday += timedelta(days=1)

        # 9am departure time
        morning_time_str = next_workday.replace(hour=9, minute=0, second=0).isoformat()
        # Only query the morning-peak commute (the evening peak is no longer computed)
        print(f"  🌅 Morning commute: {from_postcode} -> {to_postcode}")

        morning_result = None

        try:
            morning_result = self.tfl_api.get_journey(
                from_postcode, to_postcode,
                departure_time=morning_time_str
            )
        except Exception:
            pass

        # Handle the morning result
        if morning_result:
            morning_time = morning_result["duration_minutes"]
            morning_score = self._calculate_commute_score(morning_time, 0, "TRANSIT")
            morning_fare = morning_result.get("fare")

            fare_str = ""
            if morning_fare:
                fare_str = f", £{morning_fare.get('peak', morning_fare.get('total', 0)):.2f}"
            print(f"  ✅ Morning commute: {morning_time:.0f} min{fare_str}, score: {morning_score:.1f}")

            commute_results["morning"] = {
                "time_minutes": morning_time,
                "distance_km": 0,  # the TfL API does not return distance
                "score": morning_score,
                "legs": morning_result.get("legs", []),
                "fare": morning_fare
            }

            weight = weights.get("morning", 0.6)
            total_weighted_score += morning_score * weight
            total_weight += weight
        else:
            print(f"  ⚠️  Morning commute query failed")

        # Save the results
        if commute_results:
            data["_api_commute_details"] = commute_results
            data["_api_commute_source"] = "tfl"
            data["_api_commute_destination"] = office_address

            if "morning" in commute_results:
                data["_api_commute_time_minutes"] = commute_results["morning"]["time_minutes"]

            if total_weight > 0:
                final_score = total_weighted_score / total_weight
                print(f"📊 Overall commute score: {final_score:.1f}/100\n")
                return final_score

        print("  ⚠️  TfL query failed")
        return None  # return None so we fall back to the Google API

    def _calculate_commute_score(self, commute_time: float, distance_km: float, travel_mode: str = "TRANSIT") -> float:
        """
        Compute a score from commute time and distance.

        Args:
            commute_time: commute time (min)
            distance_km: commute distance (km)
            travel_mode: travel mode (TRANSIT, WALK, BICYCLE, DRIVE)

        Returns:
            Score (0-100)
        """
        # Mainly time-based (slightly lowered so scores are more balanced)
        if commute_time <= 15:
            time_score = 95
        elif commute_time <= 25:
            time_score = 85
        elif commute_time <= 35:
            time_score = 75
        elif commute_time <= 45:
            time_score = 65
        elif commute_time <= 60:
            time_score = 50
        elif commute_time <= 75:
            time_score = 35
        else:
            time_score = max(15, 95 - commute_time)

        # Distance score (secondary reference)
        if distance_km <= 5:
            distance_score = 100
        elif distance_km <= 10:
            distance_score = 85
        elif distance_km <= 15:
            distance_score = 70
        elif distance_km <= 20:
            distance_score = 55
        else:
            distance_score = 40

        # If distance is unknown (e.g. TfL mode returns 0), use the time score only
        if distance_km <= 0:
            return time_score

        # Adjust the time/distance weights by travel mode
        if travel_mode == "TRANSIT":
            # Public transport: mostly time (90%), a little distance (10%)
            time_weight = 0.90
            distance_weight = 0.10
        elif travel_mode in ["WALK", "BICYCLE"]:
            # Walking and cycling: time (70%), distance (30%)
            time_weight = 0.70
            distance_weight = 0.30
        elif travel_mode == "DRIVE":
            # Driving: time (80%), distance (20%)
            time_weight = 0.80
            distance_weight = 0.20
        else:
            # Default to the public transport weights
            time_weight = 0.90
            distance_weight = 0.10

        final_score = time_score * time_weight + distance_score * distance_weight

        return final_score

    def _evaluate_commute_fallback(self, data: Dict[str, Any]) -> float:
        """Fallback commute evaluation (uses the provided data)"""
        commute_time = data.get("commute_time_minutes", 60)

        if commute_time <= 20:
            return 100
        elif commute_time <= 30:
            return 85
        elif commute_time <= 45:
            return 70
        elif commute_time <= 60:
            return 50
        else:
            return 30

    def _evaluate_public_transport(self, data: Dict[str, Any]) -> float:
        """
        Evaluate public transport access.
        Uses TransitConvenienceEvaluator.
        Supports the TfL API (free) and the Google Places API.
        """
        if self.transit_evaluator:
            try:
                # TransitConvenienceEvaluator picks TfL or Google internally based on the API mode
                transit_score = self.transit_evaluator.evaluate(data, search_radius=1200)
                if transit_score > 0:
                    return transit_score
            except Exception as e:
                print(f"  ⚠️  Public transport evaluation failed: {e}")

        # Use the fallback
        return self._evaluate_public_transport_fallback(data)

    def _evaluate_public_transport_fallback(self, data: Dict[str, Any]) -> float:
        """Evaluate public transport access (fallback)"""
        # Factors: distance to tube/bus stop, number of lines
        tube_distance = data.get("nearest_tube_distance_m", 1000)
        num_lines = data.get("num_transport_lines", 2)

        distance_score = max(0, 100 - (tube_distance / 10))
        lines_score = min(100, num_lines * 20)

        return (distance_score * 0.7 + lines_score * 0.3)

    def _evaluate_shopping(self, data: Dict[str, Any]) -> float:
        """Evaluate shopping convenience (placeholder heuristic)"""
        # TODO: implement the actual evaluation logic
        # Factors: distance to a large supermarket / shopping centre
        supermarket_distance = data.get("supermarket_distance_m", 1000)

        if supermarket_distance <= 200:
            return 100
        elif supermarket_distance <= 500:
            return 85
        elif supermarket_distance <= 1000:
            return 70
        else:
            return 50

    def _evaluate_travel_hubs(self, data: Dict[str, Any]) -> float:
        """
        Evaluate distance to airports / railway stations.

        Tries real API data; falls back to the placeholder heuristic on failure
        """
        # Try real API data
        if self.use_google_api and self.long_distance_evaluator:
            address = data.get("address", "")
            if address:
                try:
                    score = self.long_distance_evaluator.evaluate(data, address)
                    if score > 0:
                        return score
                except Exception as e:
                    print(f"  ⚠️  Long-distance travel evaluation failed, using default: {e}")

        # Degrade: use the placeholder heuristic
        return self._evaluate_travel_hubs_fallback(data)

    def _evaluate_travel_hubs_fallback(self, data: Dict[str, Any]) -> float:
        """Evaluate distance to airports / railway stations (placeholder heuristic fallback)"""
        airport_distance = data.get("airport_distance_km", 50)
        train_station_distance = data.get("train_station_distance_km", 10)

        airport_score = max(0, 100 - (airport_distance * 2))
        train_score = max(0, 100 - (train_station_distance * 5))

        return (airport_score * 0.4 + train_score * 0.6)

    def _evaluate_medical_facilities(self, data: Dict[str, Any]) -> float:
        """Evaluate healthcare facilities (placeholder heuristic)"""
        # TODO: implement the actual evaluation logic
        # Factors: distance to hospitals/clinics, especially A&E hospitals
        hospital_distance = data.get("hospital_distance_m", 3000)
        has_emergency = data.get("has_emergency_hospital", False)

        base_score = max(0, 100 - (hospital_distance / 30))
        emergency_bonus = 20 if has_emergency else 0

        return min(100, base_score + emergency_bonus)

    def _evaluate_dining(self, data: Dict[str, Any]) -> float:
        """Evaluate dining convenience (placeholder heuristic)"""
        # TODO: implement the actual evaluation logic
        # Factors: density of nearby restaurants and cafes
        num_restaurants = data.get("num_restaurants_nearby", 10)
        num_cafes = data.get("num_cafes_nearby", 5)

        restaurant_score = min(100, num_restaurants * 5)
        cafe_score = min(100, num_cafes * 10)

        return (restaurant_score * 0.6 + cafe_score * 0.4)

    def _evaluate_basic_services(self, data: Dict[str, Any]) -> float:
        """Evaluate basic services (placeholder heuristic)"""
        # TODO: implement the actual evaluation logic
        # Factors: distance to banks, post offices, etc.
        has_bank_nearby = data.get("has_bank_nearby", True)
        has_post_office = data.get("has_post_office", True)

        score = 50
        if has_bank_nearby:
            score += 25
        if has_post_office:
            score += 25

        return score

    def evaluate_hub_commute(self, from_postcode: str) -> Optional[Dict]:
        """Compute the average commute time to London's 5 main employment hubs

        Args:
            from_postcode: origin postcode

        Returns:
            Dict with the average commute time and per-hub details, or None on failure
        """
        import requests
        from datetime import datetime, timedelta

        # Work out 9am on the next working day
        now = datetime.now()
        next_workday = now
        while next_workday.weekday() >= 5:
            next_workday += timedelta(days=1)
        if next_workday == now and now.hour >= 10:
            next_workday += timedelta(days=1)
            while next_workday.weekday() >= 5:
                next_workday += timedelta(days=1)
        dt = next_workday.replace(hour=9, minute=0, second=0)
        date_str = dt.strftime("%Y%m%d")
        time_str = dt.strftime("%H%M")

        def _query_hub(hub):
            hub_pc = hub["postcode"].replace(" ", "")
            try:
                url = f"https://api.tfl.gov.uk/Journey/JourneyResults/{from_postcode}/to/{hub_pc}"
                resp = requests.get(url, params={
                    "mode": "tube,dlr,overground,elizabeth-line,bus,walking",
                    "timeIs": "Departing",
                    "journeyPreference": "LeastTime",
                    "date": date_str,
                    "time": time_str,
                }, timeout=8)
                if resp.status_code == 200:
                    journeys = resp.json().get("journeys", [])
                    if journeys:
                        best = min(journeys, key=lambda j: j.get("duration", 9999))
                        duration = best.get("duration", 0)
                        if duration > 0:
                            return {"name": hub["name"], "minutes": duration}
            except Exception:
                pass
            return None

        # Query all hubs in parallel
        hubs_results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_query_hub, hub) for hub in self.LONDON_HUBS]
            for future in futures:
                result = future.result()
                if result:
                    hubs_results.append(result)

        if not hubs_results:
            return None

        avg_minutes = sum(h["minutes"] for h in hubs_results) / len(hubs_results)
        return {
            "average_minutes": round(avg_minutes, 1),
            "hub_count": len(self.LONDON_HUBS),
            "successful_count": len(hubs_results),
            "hubs": hubs_results
        }