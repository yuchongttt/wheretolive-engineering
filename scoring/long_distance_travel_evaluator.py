#!/usr/bin/env python3
"""
Long Distance Travel Convenience Evaluator

Scores how convenient it is to get from a property to airports and major railway stations
Used to measure accessibility for long-distance travel

Features:
1. Search nearby airports (within 80 km) and major railway stations (within 50 km)
2. Select the 5 most important airports and the 5 most important railway stations
3. Use the Google Routes API to compute public transport and driving times
4. Compute a convenience score from importance and travel time

Score weights:
- Airports: 60%
- Railway stations: 40%

Rationale:
- Reaching an important airport/railway station within 1 hour = 100 pts
- Automatically picks whichever of transit/driving is better
- Accounts for how important the airport/station is (based on annual passenger numbers)
"""

from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


class LongDistanceTravelEvaluator:
    """Long-distance travel convenience evaluator"""

    # Importance data for major UK airports (based on 2023 passenger numbers)
    AIRPORT_IMPORTANCE = {
        "heathrow": {
            "annual_passengers": 79_000_000,
            "importance_score": 1.0,
            "tier": "mega_hub"
        },
        "gatwick": {
            "annual_passengers": 40_000_000,
            "importance_score": 0.9,
            "tier": "major_hub"
        },
        "stansted": {
            "annual_passengers": 28_000_000,
            "importance_score": 0.8,
            "tier": "major_hub"
        },
        "manchester": {
            "annual_passengers": 28_000_000,
            "importance_score": 0.85,
            "tier": "major_hub"
        },
        "luton": {
            "annual_passengers": 16_000_000,
            "importance_score": 0.7,
            "tier": "regional_hub"
        },
        "edinburgh": {
            "annual_passengers": 14_000_000,
            "importance_score": 0.75,
            "tier": "regional_hub"
        },
        "birmingham": {
            "annual_passengers": 11_000_000,
            "importance_score": 0.75,
            "tier": "regional_hub"
        },
        "glasgow": {
            "annual_passengers": 8_000_000,
            "importance_score": 0.7,
            "tier": "regional_hub"
        },
        "london city": {
            "annual_passengers": 5_000_000,
            "importance_score": 0.65,
            "tier": "regional"
        },
        "bristol": {
            "annual_passengers": 6_000_000,
            "importance_score": 0.65,
            "tier": "regional"
        },
    }

    # Importance data for major UK railway stations (based on annual passenger numbers)
    TRAIN_STATION_IMPORTANCE = {
        "waterloo": {
            "annual_entries": 94_000_000,
            "importance_score": 1.0,
            "tier": "mega_hub",
            "features": ["south_coast", "major_hub"]
        },
        "victoria": {
            "annual_entries": 75_000_000,
            "importance_score": 0.95,
            "tier": "mega_hub",
            "features": ["gatwick_express", "south_coast"]
        },
        "liverpool street": {
            "annual_entries": 67_000_000,
            "importance_score": 0.9,
            "tier": "mega_hub",
            "features": ["stansted_express", "east_anglia"]
        },
        "london bridge": {
            "annual_entries": 54_000_000,
            "importance_score": 0.85,
            "tier": "major_hub",
            "features": ["south_coast"]
        },
        "euston": {
            "annual_entries": 44_000_000,
            "importance_score": 0.9,
            "tier": "major_hub",
            "features": ["west_coast_main_line", "intercity"]
        },
        "king's cross": {
            "annual_entries": 34_000_000,
            "importance_score": 0.9,
            "tier": "major_hub",
            "features": ["east_coast_main_line", "intercity"]
        },
        "st pancras": {
            "annual_entries": 27_000_000,
            "importance_score": 0.95,
            "tier": "major_hub",
            "features": ["eurostar", "international"]  # international trains!
        },
        "paddington": {
            "annual_entries": 36_000_000,
            "importance_score": 0.9,
            "tier": "major_hub",
            "features": ["heathrow_express", "west_country"]
        },
        "stratford": {
            "annual_entries": 24_000_000,
            "importance_score": 0.75,
            "tier": "regional_hub",
            "features": ["elizabeth_line", "east_anglia"]
        },
    }

    def __init__(self, places_api=None, routes_api=None, geocoding_api=None, tfl_api=None):
        """
        Initialise the evaluator

        Args:
            places_api: Google Places API instance
            routes_api: Google Routes API instance
            geocoding_api: Google Geocoding API instance
            tfl_api: TfL API instance (preferred for public transport route calculation)
        """
        self.places_api = places_api
        self.routes_api = routes_api
        self.geocoding_api = geocoding_api
        self.tfl_api = tfl_api

        # Load local transport hub data
        self.local_transport_hubs = self._load_local_transport_hubs()

    def _load_local_transport_hubs(self) -> Dict[str, Any]:
        """Load locally stored airport and railway station data"""
        import os
        data_file = os.path.join(os.path.dirname(__file__), "data", "london_transport_hubs.json")
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"  ⚠️  Could not load local transport hub data: {e}")
            return {"airports": [], "train_stations": []}

    def _get_next_saturday_noon(self) -> str:
        """
        Get next Saturday at 12:00 noon (RFC3339 format)

        Long-distance trips usually happen at weekends, so Saturday noon is used as the reference time:
        - avoids interference from weekday rush hours
        - Saturday midday traffic is stable and representative
        - fits how airports and railway stations are actually used

        Returns:
            Time string in RFC3339 format (e.g. "2026-01-31T12:00:00Z")
        """
        now = datetime.now()

        # Compute the number of days until next Saturday
        # weekday(): Mon=0, Tue=1, ..., Sat=5, Sun=6
        days_until_saturday = (5 - now.weekday()) % 7
        if days_until_saturday == 0 and now.hour >= 12:
            # If today is Saturday and it is already past noon, use next Saturday
            days_until_saturday = 7

        next_saturday = now + timedelta(days=days_until_saturday)
        # Set to 12:00 noon
        next_saturday_noon = next_saturday.replace(hour=12, minute=0, second=0, microsecond=0)

        # Return RFC3339 format (UTC; British Summer Time needs to be taken into account)
        return next_saturday_noon.strftime("%Y-%m-%dT%H:%M:%SZ")

    def evaluate(
        self,
        property_data: Dict[str, Any],
        address: str
    ) -> float:
        """
        Evaluate long-distance travel convenience

        Args:
            property_data: property data dict
            address: property address

        Returns:
            Score (0-100)
        """
        # Need the TfL API or the Google Routes API to compute routes
        if not self.tfl_api and not self.routes_api:
            print("  ⚠️  Neither TfL API nor Google Routes API configured; skipping long-distance travel convenience evaluation")
            return 0

        # Check for a data source (local data or places_api)
        has_local_data = bool(self.local_transport_hubs.get("airports") or
                              self.local_transport_hubs.get("train_stations"))
        if not has_local_data and not self.places_api:
            print("  ⚠️  No local data and Google Places API not configured; skipping long-distance travel convenience evaluation")
            return 0

        print("\n🛫 Evaluating long-distance travel convenience...")

        try:
            # Step 1: get coordinates
            lat, lng = self._get_coordinates(property_data, address)
            if not lat or not lng:
                print("  ❌ Could not get coordinates")
                return 0

            # Step 2: search airports and railway stations
            print(f"  🔍 Searching nearby airports and major railway stations...")
            airports_raw = self._search_airports(lat, lng)
            stations_raw = self._search_train_stations(lat, lng)

            # Step 3: select the 5 most important
            top_airports = self._select_top_destinations(airports_raw, "airport", 5)
            top_stations = self._select_top_destinations(stations_raw, "train_station", 5)

            print(f"  ✅ Found {len(top_airports)} important airports, {len(top_stations)} major railway stations")

            # Step 4: evaluate all destinations in parallel (significant speed-up)
            airport_results = []
            station_results = []
            failed_airports = []
            failed_stations = []

            # Prepare all tasks
            all_tasks = []
            for airport in top_airports:
                all_tasks.append(("airport", airport))
            for station in top_stations:
                all_tasks.append(("train_station", station))

            # Run all evaluations in parallel (up to 10 threads)
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_dest = {
                    executor.submit(
                        self._evaluate_single_destination,
                        address,
                        dest,
                        dest_type
                    ): (dest_type, dest)
                    for dest_type, dest in all_tasks
                }

                for future in as_completed(future_to_dest):
                    dest_type, dest = future_to_dest[future]
                    try:
                        result = future.result()
                        if result:
                            if dest_type == "airport":
                                airport_results.append(result)
                            else:
                                station_results.append(result)
                        else:
                            if dest_type == "airport":
                                failed_airports.append(dest["name"])
                            else:
                                failed_stations.append(dest["name"])
                    except Exception as e:
                        if dest_type == "airport":
                            failed_airports.append(dest["name"])
                        else:
                            failed_stations.append(dest["name"])

            # Sort by score and print results
            airport_results.sort(key=lambda x: x["best_score"], reverse=True)
            station_results.sort(key=lambda x: x["best_score"], reverse=True)

            for result in airport_results:
                mode_icon = "🚇" if result["best_mode"] == "transit" else "🚗"
                print(f"    ✈️  {result['name']}: {result['best_score']:.1f} pts "
                      f"({mode_icon} {result[result['best_mode']]['duration_minutes']:.0f} min)")

            for result in station_results:
                mode_icon = "🚇" if result["best_mode"] == "transit" else "🚗"
                print(f"    🚄 {result['name']}: {result['best_score']:.1f} pts "
                      f"({mode_icon} {result[result['best_mode']]['duration_minutes']:.0f} min)")

            # Report failed queries
            if failed_airports:
                print(f"    ⚠️  Route query failed for {len(failed_airports)} airports: {', '.join(failed_airports)}")
            if failed_stations:
                print(f"    ⚠️  Route query failed for {len(failed_stations)} railway stations: {', '.join(failed_stations)}")

            # Step 5: compute the combined score
            airport_score = self._calculate_airport_score(airport_results)
            train_score = self._calculate_train_score(station_results)
            final_score = self._calculate_final_score(airport_score, train_score)

            # Step 6: save details
            property_data["_api_long_distance_travel_data"] = {
                "airports": airport_results,
                "trains": station_results,
                "airport_score": airport_score,
                "train_score": train_score,
                "final_score": final_score
            }

            property_data["_api_long_distance_coords"] = {"lat": lat, "lng": lng}

            print(f"\n  📊 Airport score: {airport_score:.1f}/100 (weight 60%)")
            print(f"  📊 Railway station score: {train_score:.1f}/100 (weight 40%)")
            print(f"  ✅ Long-distance travel convenience: {final_score:.1f}/100")

            return final_score

        except Exception as e:
            print(f"  ❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def _get_coordinates(
        self,
        property_data: Dict[str, Any],
        address: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Get coordinates (prefer existing coordinates)"""
        # Try to get coordinates from earlier API calls
        if "_api_long_distance_coords" in property_data:
            coords = property_data["_api_long_distance_coords"]
            return coords.get("lat"), coords.get("lng")

        if "_api_transit_coords" in property_data:
            coords = property_data["_api_transit_coords"]
            return coords.get("lat"), coords.get("lng")

        if "_api_crime_coords" in property_data:
            coords = property_data["_api_crime_coords"]
            return coords.get("lat"), coords.get("lng")

        # Get coordinates via the Geocoding API
        if self.geocoding_api:
            print("  📍 Fetching address coordinates...")
            result = self.geocoding_api.get_coordinates(address)
            if result:
                lat = result.get("lat")
                lng = result.get("lng")
                print(f"  ✅ Coordinates: {lat}, {lng}")
                return lat, lng

        return None, None

    def _search_airports(self, lat: float, lng: float) -> List[Dict[str, Any]]:
        """Search nearby airports (prefer local data; fall back to the Google Places API when there is none)"""
        # Prefer local data
        local_airports = self.local_transport_hubs.get("airports", [])
        if local_airports:
            # Convert to the same format as the Google Places API, including the postcode for the TfL API
            results = []
            for airport in local_airports:
                results.append({
                    "displayName": {"text": airport["name"]},
                    "location": airport["location"],
                    "postcode": airport.get("postcode"),  # for the TfL API
                    "importance": airport.get("importance", 0.5),
                    "userRatingCount": airport.get("annual_passengers", 0) // 10000,
                    "_source": "local"
                })
            return results

        # No local data: fall back to the Google Places API
        if self.places_api:
            print("  📡 No local airport data; using Google Places API...")
            results = self.places_api.search_nearby(
                lat, lng,
                radius=50000,
                included_types=["airport"],
                max_results=20
            )
            return results or []
        return []

    def _search_train_stations(self, lat: float, lng: float) -> List[Dict[str, Any]]:
        """Search nearby major railway stations (prefer local data; fall back to the Google Places API when there is none)"""
        # Prefer local data
        local_stations = self.local_transport_hubs.get("train_stations", [])
        if local_stations:
            # Convert to the same format as the Google Places API, including the postcode for the TfL API
            results = []
            for station in local_stations:
                results.append({
                    "displayName": {"text": station["name"]},
                    "location": station["location"],
                    "postcode": station.get("postcode"),  # for the TfL API
                    "importance": station.get("importance", 0.5),
                    "userRatingCount": station.get("annual_passengers", 0) // 10000,
                    "_source": "local"
                })
            return results

        # No local data: fall back to the Google Places API
        if self.places_api:
            print("  📡 No local railway station data; using Google Places API...")
            results = self.places_api.search_nearby(
                lat, lng,
                radius=50000,
                included_types=["train_station"],
                excluded_types=["subway_station", "light_rail_station"],
                max_results=20
            )
            return results or []
        return []

    def _select_top_destinations(
        self,
        destinations: List[Dict[str, Any]],
        dest_type: str,
        top_n: int
    ) -> List[Dict[str, Any]]:
        """
        Select the N most important destinations

        Ranking criteria:
        1. Importance score (matched from the database)
        2. Review count (popularity)
        3. Distance (nearer first)
        """
        scored_destinations = []

        for dest in destinations:
            name = dest.get("displayName", {}).get("text", "")
            location = dest.get("location", {})

            # Compute importance score
            if dest_type == "airport":
                importance = self._calculate_airport_importance(name)
            else:
                importance = self._calculate_station_importance(name)

            # Review count (normalised to 0-1)
            review_count = dest.get("userRatingCount", 0)
            popularity = min(1.0, review_count / 1000)

            # Combined score
            score = importance * 0.7 + popularity * 0.3

            scored_destinations.append({
                "name": name,
                "location": location,
                "postcode": dest.get("postcode"),  # for the TfL API
                "importance": importance,
                "review_count": review_count,
                "score": score,
                "raw_data": dest
            })

        # Sort by combined score
        scored_destinations.sort(key=lambda x: x["score"], reverse=True)

        return scored_destinations[:top_n]

    def _calculate_airport_importance(self, airport_name: str) -> float:
        """
        Compute airport importance (0-1.0)

        Based on annual passenger numbers:
        - Mega Hub (≥50M): 1.0
        - Major Hub (20-50M): 0.9
        - Regional Hub (10-20M): 0.75
        - Regional (5-10M): 0.6
        - Small (<5M): 0.4
        """
        name_lower = airport_name.lower()

        for key, data in self.AIRPORT_IMPORTANCE.items():
            if key in name_lower:
                return data["importance_score"]

        # Unknown airport
        return 0.5

    def _calculate_station_importance(self, station_name: str) -> float:
        """
        Compute railway station importance (0-1.0)

        Special bonuses:
        - International trains (Eurostar): +0.1
        - Airport express: +0.05

        Based on annual passenger numbers:
        - Mega Hub (≥50M): 1.0
        - Major Hub (20-50M): 0.85
        - Regional Hub (10-20M): 0.7
        """
        name_lower = station_name.lower()

        for key, data in self.TRAIN_STATION_IMPORTANCE.items():
            if key in name_lower or key.replace("'", "") in name_lower:
                base_score = data["importance_score"]

                # Special bonuses
                bonus = 0
                features = data.get("features", [])
                if "eurostar" in features or "international" in features:
                    bonus += 0.1
                if any("express" in f for f in features):
                    bonus += 0.05

                return min(1.0, base_score + bonus)

        # Unknown station
        return 0.5

    def _evaluate_single_destination(
        self,
        origin_address: str,
        destination: dict,
        dest_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate the convenience of getting to a single destination

        Args:
            origin_address: origin address
            destination: destination info dict
            dest_type: destination type ("airport" or "train_station")

        Returns:
            Result dict covering both transit and driving; the higher-scoring mode is chosen
        """
        try:
            # Use the destination name as the address (the Google Routes API accepts address strings)
            dest_name = destination["name"]

            # For railway stations, if the name lacks "Station" or "London", try appending them to improve recognition
            if dest_type == "train_station":
                if "station" not in dest_name.lower():
                    dest_name = f"{dest_name} Station"
                if "london" not in dest_name.lower() and not dest_name.startswith("London"):
                    dest_name = f"{dest_name}, London"

            # Get next Saturday noon (long-distance trips usually happen at weekends)
            saturday_noon = self._get_next_saturday_noon()

            # Query public transport and driving times
            # Prefer Google Routes (faster), with TfL as the fallback
            transit_result = None
            drive_result = None

            # 1. Prefer Google Routes (query transit and drive in parallel)
            if self.routes_api:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    transit_future = executor.submit(
                        self.routes_api.compute_route,
                        origin_address=origin_address,
                        destination_address=dest_name,
                        travel_mode="TRANSIT",
                        departure_time=saturday_noon
                    )
                    drive_future = executor.submit(
                        self.routes_api.compute_route,
                        origin_address=origin_address,
                        destination_address=dest_name,
                        travel_mode="DRIVE",
                        departure_time=saturday_noon
                    )

                    try:
                        transit_result = transit_future.result(timeout=30)
                        if transit_result:
                            transit_result["_source"] = "google"
                    except Exception:
                        pass

                    try:
                        drive_result = drive_future.result(timeout=30)
                    except Exception:
                        pass

            # 2. If Google fails, fall back to the TfL API (transit only, using postcodes)
            if not transit_result and self.tfl_api:
                dest_postcode = destination.get("postcode")
                if dest_postcode:
                    import re
                    postcode_match = re.search(r'[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}', origin_address.upper())
                    if postcode_match:
                        origin_postcode = postcode_match.group().replace(" ", "")
                        try:
                            tfl_result = self.tfl_api.get_journey(
                                origin_postcode,
                                dest_postcode.replace(" ", ""),
                                departure_time=saturday_noon
                            )
                            if tfl_result:
                                transit_result = {
                                    "duration_minutes": tfl_result["duration_minutes"],
                                    "distance_meters": 0,
                                    "success": True,
                                    "_source": "tfl"
                                }
                        except Exception:
                            pass

            if not transit_result and not drive_result:
                return None

            # 3. Compute the importance coefficient
            importance_coef = destination["importance"]

            # 4. Compute the time coefficient and score
            result = {
                "name": destination["name"],
                "type": dest_type,
                "importance_coefficient": importance_coef,
            }

            if transit_result:
                transit_time = transit_result.get("duration_minutes", 9999)
                transit_distance = transit_result.get("distance_meters", 0) / 1000  # convert to km
                transit_time_coef = self._calculate_travel_time_coefficient(
                    transit_time, "TRANSIT", dest_type
                )
                transit_score = importance_coef * transit_time_coef * 100

                result["transit"] = {
                    "duration_minutes": transit_time,
                    "distance_km": transit_distance,
                    "time_coefficient": transit_time_coef,
                    "score": transit_score
                }
            else:
                result["transit"] = {"score": 0, "duration_minutes": 9999}

            if drive_result:
                drive_time = drive_result.get("duration_minutes", 9999)
                drive_distance = drive_result.get("distance_meters", 0) / 1000  # convert to km
                drive_time_coef = self._calculate_travel_time_coefficient(
                    drive_time, "DRIVE", dest_type
                )
                drive_score = importance_coef * drive_time_coef * 100

                result["drive"] = {
                    "duration_minutes": drive_time,
                    "distance_km": drive_distance,
                    "time_coefficient": drive_time_coef,
                    "score": drive_score
                }
            else:
                result["drive"] = {"score": 0, "duration_minutes": 9999}

            # 5. Pick the better mode
            transit_score = result["transit"]["score"]
            drive_score = result["drive"]["score"]

            if transit_score >= drive_score:
                result["best_mode"] = "transit"
                result["best_score"] = transit_score
            else:
                result["best_mode"] = "drive"
                result["best_score"] = drive_score

            return result

        except Exception as e:
            print(f"      ⚠️  Evaluating {destination['name']} failed: {e}")
            return None

    def _calculate_travel_time_coefficient(
        self,
        travel_time_minutes: float,
        mode: str,
        dest_type: str = "airport"
    ) -> float:
        """
        Compute the time coefficient (0-1.0)

        Airports (public transport):
        - 0-30 min: 1.0 (perfect ⭐⭐⭐)
        - 30-60 min: falls linearly to 0.7 (good ⭐⭐)
        - 60-90 min: falls linearly to 0.35 (fair ⭐)
        - 90-120 min: falls linearly to 0.15 (poor)
        - >120 min: falls slowly

        Airports (driving):
        - 0-20 min: 1.0 (perfect)
        - 20-40 min: falls linearly to 0.7 (good)
        - 40-60 min: falls linearly to 0.4 (fair)
        - 60-90 min: falls linearly to 0.15 (poor)
        - >90 min: falls slowly

        Railway stations (public transport):
        - 0-20 min: 1.0 (perfect)
        - 20-40 min: falls linearly to 0.7 (good)
        - 40-60 min: falls linearly to 0.4 (fair)
        - 60-90 min: falls linearly to 0.15 (poor)
        - >90 min: falls slowly

        Railway stations (driving):
        - 0-15 min: 1.0 (perfect)
        - 15-30 min: falls linearly to 0.7 (good)
        - 30-45 min: falls linearly to 0.4 (fair)
        - 45-75 min: falls linearly to 0.15 (poor)
        - >75 min: falls slowly
        """
        if dest_type == "airport":
            if mode == "TRANSIT":
                # Airport by transit: 30 min = full marks, 60 min = 70 pts
                if travel_time_minutes <= 30:
                    return 1.0
                elif travel_time_minutes <= 60:
                    # 30-60 min: linear from 1.0 down to 0.7
                    return 1.0 - (travel_time_minutes - 30) / 30 * 0.3
                elif travel_time_minutes <= 90:
                    # 60-90 min: linear from 0.7 down to 0.35
                    return 0.7 - (travel_time_minutes - 60) / 30 * 0.35
                elif travel_time_minutes <= 120:
                    # 90-120 min: linear from 0.35 down to 0.15
                    return 0.35 - (travel_time_minutes - 90) / 30 * 0.2
                else:
                    # >120 min: falls slowly
                    return max(0.05, 0.15 - (travel_time_minutes - 120) * 0.01)
            else:  # DRIVE
                # Airport by car: 20 min = full marks, 40 min = 70 pts
                if travel_time_minutes <= 20:
                    return 1.0
                elif travel_time_minutes <= 40:
                    # 20-40 min: linear from 1.0 down to 0.7
                    return 1.0 - (travel_time_minutes - 20) / 20 * 0.3
                elif travel_time_minutes <= 60:
                    # 40-60 min: linear from 0.7 down to 0.4
                    return 0.7 - (travel_time_minutes - 40) / 20 * 0.3
                elif travel_time_minutes <= 90:
                    # 60-90 min: linear from 0.4 down to 0.15
                    return 0.4 - (travel_time_minutes - 60) / 30 * 0.25
                else:
                    # >90 min: falls slowly
                    return max(0.05, 0.15 - (travel_time_minutes - 90) * 0.01)
        else:  # train_station
            if mode == "TRANSIT":
                # Railway station by transit: 20 min = full marks, 40 min = 70 pts
                if travel_time_minutes <= 20:
                    return 1.0
                elif travel_time_minutes <= 40:
                    # 20-40 min: linear from 1.0 down to 0.7
                    return 1.0 - (travel_time_minutes - 20) / 20 * 0.3
                elif travel_time_minutes <= 60:
                    # 40-60 min: linear from 0.7 down to 0.4
                    return 0.7 - (travel_time_minutes - 40) / 20 * 0.3
                elif travel_time_minutes <= 90:
                    # 60-90 min: linear from 0.4 down to 0.15
                    return 0.4 - (travel_time_minutes - 60) / 30 * 0.25
                else:
                    # >90 min: falls slowly
                    return max(0.05, 0.15 - (travel_time_minutes - 90) * 0.01)
            else:  # DRIVE
                # Railway station by car: 15 min = full marks, 30 min = 70 pts
                if travel_time_minutes <= 15:
                    return 1.0
                elif travel_time_minutes <= 30:
                    # 15-30 min: linear from 1.0 down to 0.7
                    return 1.0 - (travel_time_minutes - 15) / 15 * 0.3
                elif travel_time_minutes <= 45:
                    # 30-45 min: linear from 0.7 down to 0.4
                    return 0.7 - (travel_time_minutes - 30) / 15 * 0.3
                elif travel_time_minutes <= 75:
                    # 45-75 min: linear from 0.4 down to 0.15
                    return 0.4 - (travel_time_minutes - 45) / 30 * 0.25
                else:
                    # >75 min: falls slowly
                    return max(0.05, 0.15 - (travel_time_minutes - 75) * 0.01)

    def _calculate_airport_score(self, airports: List[Dict[str, Any]]) -> float:
        """
        Compute the airport convenience score

        Strategy:
        - Take the best airport's score
        - Diversity bonus: 2 or more good airports (≥70 pts) → +5 pts
        """
        if not airports:
            return 0

        best_score = max(a["best_score"] for a in airports)

        # Diversity bonus
        good_airports = [a for a in airports if a["best_score"] >= 70]
        diversity_bonus = 5 if len(good_airports) >= 2 else 0

        return min(100, best_score + diversity_bonus)

    def _calculate_train_score(self, stations: List[Dict[str, Any]]) -> float:
        """
        Compute the railway station convenience score

        Strategy:
        - Take the best railway station's score
        - Diversity bonus: 2 or more good stations (≥70 pts) → +5 pts
        """
        if not stations:
            return 0

        best_score = max(s["best_score"] for s in stations)

        # Diversity bonus
        good_stations = [s for s in stations if s["best_score"] >= 70]
        diversity_bonus = 5 if len(good_stations) >= 2 else 0

        return min(100, best_score + diversity_bonus)

    def _calculate_final_score(
        self,
        airport_score: float,
        train_score: float
    ) -> float:
        """
        Compute the combined score

        Weights:
        - Airports: 60%
        - Railway stations: 40%
        """
        return airport_score * 0.6 + train_score * 0.4
