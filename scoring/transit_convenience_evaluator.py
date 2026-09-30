#!/usr/bin/env python3
"""
Public transport convenience evaluation module
Scores the convenience of nearby tube stations and bus stops, taking line importance and walking time into account
"""

from typing import Dict, Any, List, Optional, Tuple
from api_helper import GeocodingService, GooglePlacesAPI, GoogleRoutesAPI, TfLAPI
import re


class TransitConvenienceEvaluator:
    """Public transport convenience evaluator"""

    # London rail line weights (based on annual ridership and importance; unit: pts)
    # Design principle: 2 major lines ≈ 80 pts, 3 lines = full marks, no diminishing returns for more lines
    # Covers Underground, DLR, Overground, Elizabeth Line
    TUBE_LINE_WEIGHTS = {
        # Top-tier lines (highest ridership, 40 pts/line)
        "central": 40,
        "northern": 40,
        "elizabeth": 40,  # Elizabeth Line
        "elizabeth-line": 40,  # format returned by the TfL API
        "victoria": 40,
        "jubilee": 40,

        # Major lines (35 pts/line)
        "piccadilly": 35,
        "district": 35,
        "metropolitan": 35,
        "circle": 35,

        # Important lines (30 pts/line)
        "bakerloo": 30,
        "hammersmith-city": 30,
        "dlr": 30,

        # Overground family (25 pts/line)
        "london-overground": 25,
        "overground": 25,
        # New Overground line names (after the 2024 split)
        "mildmay": 25,
        "lioness": 25,
        "windrush": 25,
        "weaver": 25,
        "suffragette": 25,
        "liberty": 25,

        # Standard lines (20 pts/line)
        "waterloo-city": 20,

        # National Rail (mainline trains, treated as one line, 20 pts)
        "national rail": 20,
    }

    # Base-score lookup table for major tube stations (fallback data, used when the TfL API fails)
    # All these major hubs have a base score of 100 (max 100 after the distance coefficient is applied)
    MAJOR_STATIONS = {
        # Zone 1 core interchanges (multiple lines, base 100)
        "king's cross": 100,
        "kings cross": 100,
        "st pancras": 100,
        "liverpool street": 100,
        "paddington": 100,
        "waterloo": 100,
        "victoria": 100,
        "oxford circus": 100,
        "bank": 100,
        "monument": 100,
        "london bridge": 100,
        "euston": 100,
        "green park": 100,
        "baker street": 100,
        "bond street": 100,
        "piccadilly circus": 100,
        "leicester square": 100,
        "tottenham court road": 100,
        "holborn": 100,
        "moorgate": 100,
        "farringdon": 100,
        "blackfriars": 100,
        "embankment": 100,
        "charing cross": 100,
        "warren street": 100,
        "south kensington": 100,
        "earl's court": 100,
        "earls court": 100,

        # Zone 2 important stations (base 70-90)
        "stratford": 90,
        "canary wharf": 90,
        "clapham junction": 80,
        "hammersmith": 80,
        "finsbury park": 70,
        "highbury & islington": 70,
        "notting hill gate": 70,
        "westminster": 100,
        "canada water": 70,

        # Default single-line station: base 50
    }

    def __init__(
        self,
        api_params: Optional[Dict[str, Any]] = None,
        tube_weight: float = 0.8,
        bus_weight: float = 0.2
    ):
        """
        Initialise the public transport convenience evaluator

        Args:
            api_params: API parameter configuration
            tube_weight: tube station weight (default 0.8)
            bus_weight: bus stop weight (default 0.2)
        """
        self.api_params = api_params or {}
        self.tube_weight = tube_weight
        self.bus_weight = bus_weight

        # Initialise the TfL API (no auth required, always available)
        self.tfl_api = TfLAPI()

        # Initialise APIs
        google_api_key = None
        if api_params and "google_api" in api_params:
            google_api_key = api_params["google_api"]["api_key"]
            self.places_api = GooglePlacesAPI(google_api_key)
            self.routes_api = GoogleRoutesAPI(google_api_key)
            self.use_google_api = True
        else:
            self.places_api = None
            self.routes_api = None
            self.use_google_api = False

        # GeocodingService is always available (Postcodes.io first, falls back to Google)
        self.geocoding_api = GeocodingService(google_api_key)

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

    def evaluate(
        self,
        property_data: Dict[str, Any],
        search_radius: float = 1200
    ) -> float:
        """
        Evaluate public transport convenience

        Args:
            property_data: property data
            search_radius: search radius (m), default 1200 m

        Returns:
            Score (0-100)
        """
        address = property_data.get("address", "")
        if not address:
            print("  ⚠️  No address provided; cannot evaluate public transport convenience")
            return 0

        # Check API mode
        free_mode = self._is_free_api_mode(property_data)
        postcode = self._extract_postcode(address)
        is_london = postcode and self.tfl_api.is_london_postcode(postcode)

        # Decide which API to use
        if free_mode:
            if is_london:
                # London: use the TfL API (free)
                return self._evaluate_with_tfl(property_data, address, search_radius)
            else:
                # Outside London: skip in free mode (TfL only covers London)
                print("  ⚠️  Free mode: skipping public transport evaluation outside London (TfL API only covers London)")
                return 0
        else:
            # Use the Google API
            if not self.use_google_api:
                print("  ⚠️  Google API not configured; skipping public transport convenience evaluation")
                return 0
            return self._evaluate_with_google(property_data, address, search_radius)

    def _evaluate_with_tfl(
        self,
        property_data: Dict[str, Any],
        address: str,
        search_radius: float
    ) -> float:
        """Evaluate public transport convenience with the TfL API (London only, free)"""
        print(f"  🚇 Searching nearby stations via TfL API (radius {search_radius} m)...")

        # Get coordinates (try to reuse existing ones, otherwise use the postcode extracted from the address as the search parameter)
        lat, lng = self._get_coordinates_for_tfl(property_data, address)
        if lat is None or lng is None:
            print("  ❌ Could not get coordinates for the address")
            return 0

        # Search nearby stations via the TfL API
        stations = self.tfl_api.get_nearby_stations(
            lat, lng,
            radius=int(search_radius),
            modes="tube,dlr,overground,elizabeth-line,national-rail"
        )

        if not stations:
            print("  ⚠️  No nearby public transport stations found")
            return 0

        print(f"  🔍 TfL API found {len(stations)} stations")

        # Convert to the format the evaluator expects
        tube_stations = []
        for station in stations:
            # Compute walking time (assumes 5 km/h walking speed, ×1.3 Manhattan factor to account for straight-line vs actual path)
            distance_m = station.get("distance", 0)
            walk_time = distance_m * 1.3 / 83.33  # 83.33 m/min = 5 km/h, ×1.3 path factor

            tube_stations.append({
                "name": station["name"],
                "walk_time": walk_time,
                "distance": distance_m,
                "lines": station.get("lines", []),
                "lat": station.get("lat"),
                "lng": station.get("lng")
            })

        # First merge the lines of stations with the same name
        unique_stations = self._dedupe_stations_tfl(tube_stations)

        # Filter redundant stations: at most 3, dropping stations whose lines are already covered by a closer station
        unique_stations = self._filter_redundant_stations(unique_stations, max_stations=3)

        # Collect all unique lines
        all_lines = set()
        for station in unique_stations:
            all_lines.update(station.get("lines", []))
        all_lines = sorted(all_lines)

        # Compute the score (using the merged data)
        tube_score = self._evaluate_tube_stations_tfl(property_data, unique_stations)

        # The TfL API does not provide bus stop data separately; give bus stops a default score
        bus_score = 70 if unique_stations else 0

        # Combined score
        final_score = tube_score * self.tube_weight + bus_score * self.bus_weight

        # Compute total line weight
        total_line_weight = sum(
            self.TUBE_LINE_WEIGHTS.get(line.lower(), 20)
            for line in all_lines
        )

        # Save evaluation results
        property_data["_api_transit_data"] = {
            "lines": all_lines,
            "lines_count": len(all_lines),
            "total_line_weight": total_line_weight,
            "tube_stations_count": len(unique_stations),
            "bus_stops_count": 0,
            "tube_score": tube_score,
            "bus_score": bus_score,
            "tube_weight": self.tube_weight,
            "bus_weight": self.bus_weight,
            "final_score": final_score,
            "api_source": "tfl"
        }

        # Save details
        property_data["_api_transit_tube_details"] = unique_stations[:10]
        property_data["_api_transit_coords"] = {"lat": lat, "lng": lng}

        # Print stats
        self._print_transit_stats(property_data["_api_transit_data"])

        return final_score

    def _get_coordinates_for_tfl(
        self,
        property_data: Dict[str, Any],
        address: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Get coordinates (reuse first, then try the Google API or the postcode)"""
        # Try to reuse existing coordinates
        for key in ["_api_transit_coords", "_api_amenities_coords", "_api_crime_coords"]:
            if key in property_data:
                return property_data[key]["lat"], property_data[key]["lng"]

        # If the Google API is available, use it for precise coordinates
        if self.use_google_api and self.geocoding_api:
            print(f"  📍 Fetching address coordinates...")
            coords = self.geocoding_api.get_coordinates(address)
            if coords and coords.get("success"):
                lat = coords["lat"]
                lng = coords["lng"]
                print(f"  ✅ Coordinates: {lat:.6f}, {lng:.6f}")
                property_data["_api_transit_coords"] = {"lat": lat, "lng": lng}
                return lat, lng

        # Try to get approximate coordinates from the postcode (via the free postcodes.io API)
        postcode = self._extract_postcode(address)
        if postcode:
            coords = self._get_coords_from_postcode(postcode)
            if coords:
                print(f"  📍 Coordinates from postcode: {coords[0]:.6f}, {coords[1]:.6f}")
                property_data["_api_transit_coords"] = {"lat": coords[0], "lng": coords[1]}
                return coords

        return None, None

    def _get_coords_from_postcode(self, postcode: str) -> Optional[Tuple[float, float]]:
        """Get coordinates from a postcode (using the free postcodes.io API)"""
        import requests

        # Normalise the postcode
        postcode = postcode.upper().replace(" ", "")
        if len(postcode) > 3:
            postcode = f"{postcode[:-3]} {postcode[-3:]}"

        url = f"https://api.postcodes.io/postcodes/{postcode}"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                result = data.get("result", {})
                lat = result.get("latitude")
                lng = result.get("longitude")
                if lat and lng:
                    return (lat, lng)
        except Exception:
            pass

        return None

    def _evaluate_tube_stations_tfl(
        self,
        property_data: Dict[str, Any],
        stations: List[Dict[str, Any]]
    ) -> float:
        """Evaluate tube station convenience using TfL data"""
        if not stations:
            print("  📊 Tube station score: 0/100 (no tube stations found)")
            return 0

        station_scores = []

        for station in stations:
            lines = station.get("lines", [])
            if not lines:
                continue

            walk_time = station.get("walk_time", 15)

            # Compute total line weight (lines simply add up, no diminishing returns)
            line_weight_sum = sum(
                self.TUBE_LINE_WEIGHTS.get(line.lower(), 20)
                for line in lines
            )

            # Compute distance coefficient
            distance_coefficient = self._calculate_distance_coefficient(walk_time)

            # Compute station score (capped at 100)
            station_score = min(100, line_weight_sum * distance_coefficient)

            station_scores.append({
                "name": station["name"],
                "walk_time": walk_time,
                "lines": lines,
                "line_weight_sum": round(line_weight_sum, 1),
                "distance_coefficient": round(distance_coefficient, 2),
                "score": round(station_score, 1)
            })

        if not station_scores:
            return 0

        # Sort by score
        station_scores.sort(key=lambda x: x["score"], reverse=True)

        # Compute final score (the best-scoring station dominates)
        best_score = station_scores[0]["score"]
        other_scores_sum = sum(s["score"] * 0.1 for s in station_scores[1:4])
        final_score = min(100, best_score + other_scores_sum)

        # Print best-station info
        best = station_scores[0]
        print(f"  📊 Tube station score: {final_score:.1f}/100")
        print(f"     Nearest station: {best['name']}")
        print(f"     Walking time: {best['walk_time']:.1f} min")
        print(f"     Serves {len(best['lines'])} lines")
        print(f"     Line weight: {best['line_weight_sum']} pts")
        print(f"     Distance coefficient: {best['distance_coefficient']:.2f}")
        print(f"     Station score: {best['score']:.1f} pts")

        # Save detailed data
        property_data["_api_transit_tube_details_all"] = station_scores

        return final_score

    def _dedupe_stations_tfl(self, stations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate stations (same name or very close together), merging their line info"""
        # Group by simplified name, merging lines
        station_groups = {}

        for station in stations:
            name = station["name"].lower()
            # Simplified name (strip suffixes such as Station, Underground, DLR, Rail)
            simple_name = name
            for suffix in [" underground station", " rail station", " dlr station",
                          " station", " underground", " dlr", " (london)"]:
                simple_name = simple_name.replace(suffix, "")
            simple_name = simple_name.strip()

            if simple_name not in station_groups:
                # Create a new station record
                station_groups[simple_name] = {
                    "name": station["name"],
                    "walk_time": station.get("walk_time", 15),
                    "distance": station.get("distance", 0),
                    "lines": set(station.get("lines", [])),
                    "lat": station.get("lat"),
                    "lng": station.get("lng")
                }
            else:
                # Merge lines into the existing station
                station_groups[simple_name]["lines"].update(station.get("lines", []))
                # Use the shorter walking time
                if station.get("walk_time", 15) < station_groups[simple_name]["walk_time"]:
                    station_groups[simple_name]["walk_time"] = station.get("walk_time", 15)
                    station_groups[simple_name]["distance"] = station.get("distance", 0)

        # Convert back to a list
        unique = []
        for simple_name, data in station_groups.items():
            unique.append({
                "name": data["name"],
                "walk_time": data["walk_time"],
                "distance": data["distance"],
                "lines": list(data["lines"]),
                "lat": data["lat"],
                "lng": data["lng"]
            })

        # Sort by walking time
        unique.sort(key=lambda x: x["walk_time"])
        return unique

    def _filter_redundant_stations(
        self,
        stations: List[Dict[str, Any]],
        max_stations: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Filter redundant stations:
        1. Prefer the nearest stations
        2. If all of a station's lines are already covered by closer stations, exclude it
        3. Keep at most max_stations stations

        Args:
            stations: station list already sorted by distance (nearest first)
            max_stations: maximum number of stations to keep

        Returns:
            Filtered station list
        """
        if not stations:
            return []

        selected = []
        covered_lines = set()

        for station in stations:
            station_lines = set(line.lower() for line in station.get("lines", []))

            # Check for new lines (not covered by already-selected stations)
            new_lines = station_lines - covered_lines

            if new_lines:
                # This station adds new lines; keep it
                selected.append(station)
                covered_lines.update(station_lines)

                if len(selected) >= max_stations:
                    break

        # Print filtering result
        if len(stations) > len(selected):
            print(f"  📍 Filtered {len(stations)} stations down to {len(selected)} (redundant lines removed)")

        return selected

    def _evaluate_with_google(
        self,
        property_data: Dict[str, Any],
        address: str,
        search_radius: float
    ) -> float:
        """Evaluate public transport convenience with the Google API"""
        # Get or reuse coordinates
        lat, lng = self._get_coordinates(property_data, address)
        if lat is None or lng is None:
            print("  ❌ Could not get coordinates for the address")
            return 0

        # Search nearby tube stations and bus stops
        print(f"  🚇 Searching nearby tube stations and bus stops (radius {search_radius} m)...")

        # Search tube stations (use transit_station, then filter via the TfL API to stations with Underground lines)
        tube_stations = self.places_api.search_nearby(
            lat, lng,
            radius=search_radius,
            included_types=["transit_station"],
            excluded_types=["bus_stop"],
            max_results=20
        )

        # Search bus stops
        bus_stops = self.places_api.search_nearby(
            lat, lng,
            radius=search_radius,
            included_types=["bus_station"],
            max_results=20
        )

        if tube_stations is None and bus_stops is None:
            print("  ❌ Could not get public transport station data")
            return 0

        tube_stations = tube_stations or []
        bus_stops = bus_stops or []

        print(f"  🔍 Found {len(tube_stations)} tube stations, {len(bus_stops)} bus stops")

        # Compute tube station score
        tube_score = self._evaluate_tube_stations(
            property_data, address, tube_stations
        )

        # Compute bus stop score
        bus_score = self._evaluate_bus_stops(
            property_data, address, bus_stops
        )

        # Get the de-duplicated tube station count (from the details)
        actual_tube_count = len(property_data.get("_api_transit_tube_details_all", []))
        if actual_tube_count == 0:
            actual_tube_count = len(tube_stations)  # fall back to the raw count

        # Print the actual count after de-duplication
        if actual_tube_count != len(tube_stations):
            print(f"  ✅ After de-duplication: {actual_tube_count} tube stations (merged {len(tube_stations) - actual_tube_count} duplicate stations)")
        else:
            print(f"  ✅ Confirmed: {actual_tube_count} tube stations")

        # Combined score
        final_score = (
            tube_score * self.tube_weight +
            bus_score * self.bus_weight
        )

        # Collect all lines (from the filtered stations)
        all_lines = set()
        for station in property_data.get("_api_transit_tube_details", []):
            all_lines.update(line.lower() for line in station.get("lines", []))
        all_lines = sorted(all_lines)

        # Compute total line weight
        total_line_weight = sum(
            self.TUBE_LINE_WEIGHTS.get(line, 20)
            for line in all_lines
        )

        # Save evaluation results
        property_data["_api_transit_data"] = {
            "lines": all_lines,
            "lines_count": len(all_lines),
            "total_line_weight": total_line_weight,
            "tube_stations_count": actual_tube_count,
            "bus_stops_count": len(bus_stops),
            "tube_score": tube_score,
            "bus_score": bus_score,
            "tube_weight": self.tube_weight,
            "bus_weight": self.bus_weight,
            "final_score": final_score,
            "api_source": "google"
        }

        # Print stats
        self._print_stats(property_data["_api_transit_data"])

        return final_score

    def _get_coordinates(
        self,
        property_data: Dict[str, Any],
        address: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Get or reuse coordinates"""
        # Try to reuse existing coordinates
        for key in ["_api_transit_coords", "_api_amenities_coords", "_api_crime_coords"]:
            if key in property_data:
                return property_data[key]["lat"], property_data[key]["lng"]

        # Fetch new coordinates
        print(f"  📍 Fetching address coordinates...")
        coords = self.geocoding_api.get_coordinates(address)

        if not coords or not coords.get("success"):
            return None, None

        lat = coords["lat"]
        lng = coords["lng"]
        print(f"  ✅ Coordinates: {lat:.6f}, {lng:.6f}")

        # Cache coordinates
        property_data["_api_transit_coords"] = {"lat": lat, "lng": lng}

        return lat, lng

    def _evaluate_tube_stations(
        self,
        property_data: Dict[str, Any],
        address: str,
        stations: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate tube station convenience

        New algorithm:
        1. Use the TfL API to get each station's actual line list; drop stations with no tube lines
        2. Compute total line weight (sum of all line weights)
        3. Compute distance coefficient (based on walking time)
        4. Station score = total line weight × distance coefficient

        Returns:
            Score (0-100)
        """
        if not stations:
            print("  📊 Tube station score: 0/100 (no tube stations found)")
            return 0

        # Evaluate each tube station
        station_scores = []

        for station in stations:  # check all stations
            station_name = station.get("displayName", {}).get("text", "Unknown")

            # 1. Get line info via the TfL API to verify this is a real tube station
            lines = self.tfl_api.get_station_lines(station_name)

            # Filter: must have lines returned by the TfL API
            # Covers Underground, DLR, Overground, Elizabeth Line
            if not lines:
                # The TfL API returned no lines, so this is not a rail station; skip
                continue

            # 2. Compute walking time
            walk_time = self._estimate_walk_time(station, property_data)

            # 3. Compute total line weight (simple sum, see the method docstring)
            if lines:
                # Use TfL API data
                line_weight_sum = self._calculate_line_weight_with_diminishing_returns(lines)
                lines_display = lines
                data_source = "TfL API"
            else:
                # Use the fallback method (lookup table)
                line_weight_sum = self._get_fallback_line_weight(station_name)
                lines_display = ["lookup-table estimate"]
                data_source = "lookup table"

            # 4. Compute distance coefficient
            distance_coefficient = self._calculate_distance_coefficient(walk_time)

            # 5. Station score = total line weight × distance coefficient (capped at 100)
            station_score = min(100, line_weight_sum * distance_coefficient)

            # Get station coordinates (for distance-based de-duplication)
            location = station.get("location", {})
            lat = location.get("latitude")
            lng = location.get("longitude")

            station_scores.append({
                "name": station_name,
                "walk_time": walk_time,
                "lines": lines_display,
                "line_weight_sum": line_weight_sum,
                "distance_coefficient": distance_coefficient,
                "score": station_score,
                "lat": lat,
                "lng": lng
            })

        if not station_scores:
            return 0

        # De-duplicate stations (different entrances/names of the same station)
        station_scores = self._deduplicate_stations(station_scores)

        # Sort by walking time (for redundancy filtering)
        station_scores.sort(key=lambda x: x["walk_time"])

        # Filter redundant stations: at most 3, dropping stations whose lines are already covered by a closer station
        station_scores = self._filter_redundant_stations(station_scores, max_stations=3)

        # Save all filtered stations (for counting)
        property_data["_api_transit_tube_details_all"] = station_scores

        # Save details (for display)
        property_data["_api_transit_tube_details"] = station_scores

        # Take the best station's score
        best_score = max(s["score"] for s in station_scores)

        # Diversity bonus: based on the actual de-duplicated station count
        # With multiple stations, +2 pts per extra station, up to 10 pts
        diversity_bonus = min(10, (len(station_scores) - 1) * 2)

        final_score = min(100, best_score + diversity_bonus)

        print(f"  📊 Tube station score: {final_score:.1f}/100")
        if station_scores:
            best = station_scores[0]
            print(f"     Nearest station: {best['name']}")
            print(f"     Walking time: {best['walk_time']:.1f} min")
            print(f"     Serves {len(best['lines'])} lines")
            print(f"     Line weight: {best['line_weight_sum']} pts")
            print(f"     Distance coefficient: {best['distance_coefficient']:.2f}")
            print(f"     Station score: {best['score']:.1f} pts")

        return final_score

    def _evaluate_bus_stops(
        self,
        property_data: Dict[str, Any],
        address: str,
        stops: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate bus stop convenience

        Algorithm:
        1. Compute the distance coefficient of the nearest bus stop
        2. Base score = 70 pts (buses are an important mode of transport)
        3. Stop score = base score × distance coefficient
        4. Quantity bonus: 7 or more stops reach full marks

        Scoring:
        - 1 stop: 70 pts (basic convenience)
        - 3 stops: 80 pts (some choice)
        - 5 stops: 90 pts (plenty of choice)
        - 7 or more: 100 pts (bus hub)

        Returns:
            Score (0-100)
        """
        if not stops:
            print("  📊 Bus stop score: 0/100 (no bus stops found)")
            return 0

        # Compute a score for each bus stop (considering all stops)
        stop_scores = []
        total_stops = len(stops)

        for stop in stops[:10]:  # consider the first 10 stops
            stop_name = stop.get("displayName", {}).get("text", "Unknown")
            walk_time = self._estimate_walk_time(stop, property_data)
            distance_coefficient = self._calculate_distance_coefficient(walk_time)

            # Bus stop base score = 70 pts (buses are an important mode of transport)
            stop_score = 70 * distance_coefficient

            stop_scores.append({
                "name": stop_name,
                "walk_time": walk_time,
                "distance_coefficient": distance_coefficient,
                "score": stop_score
            })

        if not stop_scores:
            return 0

        # Keep the first 5 for display
        property_data["_api_transit_bus_details"] = stop_scores[:5]

        # Take the best stop's score
        best_score = max(s["score"] for s in stop_scores)

        # Quantity bonus: +5 pts per extra stop
        # 7 or more stops reach full marks (70 + 6×5 = 100)
        quantity_bonus = (total_stops - 1) * 5

        final_score = min(100, best_score + quantity_bonus)

        print(f"  📊 Bus stop score: {final_score:.1f}/100")
        if stop_scores:
            best = stop_scores[0]
            print(f"     Nearest stop: {best['name']}")
            print(f"     Walking time: {best['walk_time']:.1f} min")
            print(f"     Distance coefficient: {best['distance_coefficient']:.2f}")
            print(f"     Stop score: {best['score']:.1f} pts")

        return final_score

    def _calculate_line_weight_with_diminishing_returns(self, lines: List[str]) -> float:
        """
        Compute total line weight (simple sum, no diminishing returns; the method
        name is historical)

        Design principles:
        - 2 major lines ≈ 80 pts
        - 3 lines = full marks
        - Multiple lines are a genuine advantage and should not diminish

        Weights:
        - Top-tier lines (Central, Northern, Elizabeth, Victoria, Jubilee): 40 pts
        - Major lines (Piccadilly, District, Metropolitan, Circle): 35 pts
        - Important lines (Bakerloo, H&C, DLR): 30 pts
        - Overground family: 25 pts
        - Standard lines: 20 pts

        Args:
            lines: list of lines (e.g. ['central', 'jubilee', 'dlr'])

        Returns:
            Total line weight (uncapped; the station score caps at 100 after the
            distance coefficient is applied)

        Examples:
            1 line, Central (40): 40 pts
            2 lines, Central + Jubilee (40, 40): 80 pts
            3 lines, Central + Jubilee + DLR (40, 40, 30): 110 -> station score 100 (capped)
        """
        if not lines:
            return 0

        # Sum the line weights directly (default 20 pts)
        total_score = sum(
            self.TUBE_LINE_WEIGHTS.get(line.lower(), 20)
            for line in lines
        )

        return total_score

    def _deduplicate_stations(self, stations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        De-duplicate by geographic distance, merging stations that are close together

        Rules:
        - If two stations are < 400 m apart, treat them as the same transport hub (the same interchange)
        - Keep the higher-scoring one
        - Greedy: process in descending score order, so higher-scoring stations are kept first

        Rationale:
        - The 400 m threshold reflects the actual footprint of large transport hubs
        - e.g. Stratford spans several buildings (Station, Bus Station) but is one hub from a convenience standpoint
        - Stations further away (e.g. Stratford International, 500 m away) are treated as separate stations

        Examples:
        - Stratford Station + Bus Station (303 m apart) → merged into 1 station
        - Stratford + Stratford International (500 m apart) → kept as 2 stations

        Args:
            stations: station list (each station must include lat, lng coordinates)

        Returns:
            De-duplicated station list
        """
        if not stations:
            return []

        # Sort by score, descending (higher scores are kept first)
        sorted_stations = sorted(stations, key=lambda x: x["score"], reverse=True)

        # Stations kept so far
        kept_stations = []

        for station in sorted_stations:
            # Check whether it is too close to an already-kept station
            is_duplicate = False

            for kept in kept_stations:
                distance = self._calculate_station_distance(station, kept)

                # If < 400 m apart, treat as a duplicate (same transport hub)
                if distance < 0.4:  # 400 m = 0.4 km
                    is_duplicate = True
                    break

            # Not a duplicate: keep it
            if not is_duplicate:
                kept_stations.append(station)

        return kept_stations

    def _calculate_station_distance(
        self,
        station1: Dict[str, Any],
        station2: Dict[str, Any]
    ) -> float:
        """
        Compute the distance between two stations (km)

        Uses the Haversine formula for great-circle distance

        Args:
            station1: first station (with lat, lng)
            station2: second station (with lat, lng)

        Returns:
            Distance (km)
        """
        import math

        # Get coordinates
        lat1 = station1.get("lat")
        lng1 = station1.get("lng")
        lat2 = station2.get("lat")
        lng2 = station2.get("lng")

        # If either station lacks coordinates, fall back to comparing names
        if lat1 is None or lng1 is None or lat2 is None or lng2 is None:
            core1 = self._extract_core_name(station1["name"].lower())
            core2 = self._extract_core_name(station2["name"].lower())
            if core1 == core2:
                return 0.0  # treat as the same station
            return 1.0  # treat as different stations

        # Compute distance with the Haversine formula
        R = 6371  # Earth radius (km)

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)

        a = (
            math.sin(delta_lat / 2) ** 2 +
            math.cos(lat1_rad) * math.cos(lat2_rad) *
            math.sin(delta_lng / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance_km = R * c

        return distance_km

    def _extract_core_name(self, station_name: str) -> str:
        """
        Extract the core station name

        Strip common suffixes, keeping the main place name

        Args:
            station_name: station name

        Returns:
            Core name
        """
        import re

        # Lowercase
        name = station_name.lower()

        # Strip common suffixes
        name = re.sub(
            r'\s+(station|underground|tube|rail|bus station|dlr|stop [a-z]|stop [0-9]+|\([^)]+\)).*$',
            '',
            name,
            flags=re.IGNORECASE
        ).strip()

        # Special case: if only "bus" is left it may be "Bus Station", which would need context to decide
        # but for now we just return the processed name
        return name

    def _calculate_distance_coefficient(self, walk_time: float) -> float:
        """
        Compute the distance coefficient from walking time

        Rules:
        - 0-5 min: coefficient 1.0 (perfect)
        - 5-10 min: coefficient falls linearly from 1.0 to 0.7 (10 min = 0.7)
        - 10-15 min: coefficient falls linearly from 0.7 to 0.4 (15 min = 0.4)
        - 15-20 min: coefficient falls linearly from 0.4 to 0.2
        - >20 min: coefficient keeps falling, minimum 0.05

        Args:
            walk_time: walking time (min)

        Returns:
            Distance coefficient (0-1.0)
        """
        if walk_time <= 5:
            return 1.0
        elif walk_time <= 10:
            # 5-10 min: linear from 1.0 down to 0.7
            return 1.0 - (walk_time - 5) / 5 * 0.3
        elif walk_time <= 15:
            # 10-15 min: linear from 0.7 down to 0.4
            return 0.7 - (walk_time - 10) / 5 * 0.3
        elif walk_time <= 20:
            # 15-20 min: linear from 0.4 down to 0.2
            return 0.4 - (walk_time - 15) / 5 * 0.2
        else:
            # >20 min: keep falling, minimum 0.05
            return max(0.05, 0.2 - (walk_time - 20) * 0.01)

    def _get_fallback_line_weight(self, station_name: str) -> float:
        """
        Fallback: get the base score from the lookup table by station name
        Used when the TfL API fails

        Major hubs have a base score of 100 (max 100 after the distance coefficient is applied)
        Ordinary stations have a base score of 50

        Args:
            station_name: station name

        Returns:
            Station base score (line weight)
        """
        import re

        station_lower = station_name.lower()

        # Strip common suffixes to get the core station name
        core_name = re.sub(r'\s+(station|underground station|tube station|rail station|bus station|international|city|high street).*$',
                          '', station_lower, flags=re.IGNORECASE).strip()

        # Look up the MAJOR_STATIONS table - requires an exact or very close match
        for station_key, base_score in self.MAJOR_STATIONS.items():
            # Exact match on the core name
            if station_key == core_name or station_key == station_lower:
                return base_score

            # Or the core name starts with the table key (e.g. "king's cross" matches "kings cross")
            if core_name.startswith(station_key) or station_key in core_name.split():
                # But exclude obvious mismatches (e.g. "stratford international" should not match "stratford")
                # Check for extra descriptive words
                extra_words = core_name.replace(station_key, '').strip()
                if not extra_words:  # no extra words, so it is an exact match
                    return base_score

        # Default base score: assume an ordinary single-line station
        return 50

    def _estimate_walk_time(
        self,
        place: Dict[str, Any],
        property_data: Dict[str, Any]
    ) -> float:
        """
        Estimate walking time (min)

        Estimated from straight-line distance, average walking speed 5 km/h

        Args:
            place: place info (with latitude/longitude)
            property_data: property data (with coordinates)

        Returns:
            Walking time (min)
        """
        import math

        # Get property coordinates
        property_lat = None
        property_lng = None

        for key in ["_api_transit_coords", "_api_amenities_coords", "_api_crime_coords"]:
            if key in property_data:
                property_lat = property_data[key]["lat"]
                property_lng = property_data[key]["lng"]
                break

        if property_lat is None or property_lng is None:
            # Cannot compute distance; return default
            return 10.0

        # Get place coordinates
        place_location = place.get("location", {})
        place_lat = place_location.get("latitude")
        place_lng = place_location.get("longitude")

        if place_lat is None or place_lng is None:
            return 10.0

        # Compute straight-line distance (Haversine formula)
        R = 6371  # Earth radius (km)

        lat1_rad = math.radians(property_lat)
        lat2_rad = math.radians(place_lat)
        delta_lat = math.radians(place_lat - property_lat)
        delta_lng = math.radians(place_lng - property_lng)

        a = (
            math.sin(delta_lat / 2) ** 2 +
            math.cos(lat1_rad) * math.cos(lat2_rad) *
            math.sin(delta_lng / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance_km = R * c

        # Walking speed: 5 km/h
        # Actual walking routes are typically 1.3× the straight-line distance (Manhattan distance factor)
        actual_distance = distance_km * 1.3
        walk_time_minutes = (actual_distance / 5) * 60

        return walk_time_minutes


    def _print_stats(self, stats: Dict[str, Any]):
        """Print stats (legacy, kept for compatibility)"""
        print(f"\n  📊 Public transport convenience stats:")
        print(f"     Found {stats['tube_stations_count']} tube stations")
        print(f"     Found {stats['bus_stops_count']} bus stops")
        print(f"     Tube station score: {stats['tube_score']:.1f}/100 (weight {stats['tube_weight']*100:.0f}%)")
        print(f"     Bus stop score: {stats['bus_score']:.1f}/100 (weight {stats['bus_weight']*100:.0f}%)")
        print(f"  ✅ Public transport convenience score: {stats['final_score']:.1f}/100\n")

    def _print_transit_stats(self, stats: Dict[str, Any]):
        """Print public transport stats (focused on lines)"""
        lines = stats.get("lines", [])
        lines_count = stats.get("lines_count", 0)
        total_weight = stats.get("total_line_weight", 0)

        print(f"\n  📊 Public transport convenience:")
        print(f"     Reachable: {lines_count} lines")
        if lines:
            # Display grouped by weight
            top_lines = [l for l in lines if self.TUBE_LINE_WEIGHTS.get(l.lower(), 20) >= 40]
            major_lines = [l for l in lines if 30 <= self.TUBE_LINE_WEIGHTS.get(l.lower(), 20) < 40]
            other_lines = [l for l in lines if self.TUBE_LINE_WEIGHTS.get(l.lower(), 20) < 30]

            if top_lines:
                print(f"     Top-tier lines: {', '.join(top_lines)}")
            if major_lines:
                print(f"     Important lines: {', '.join(major_lines)}")
            if other_lines:
                print(f"     Other lines: {', '.join(other_lines)}")

        print(f"     Line weight: {total_weight} pts (max 100)")
        print(f"     Covers {stats['tube_stations_count']} stations")
        print(f"  ✅ Public transport score: {stats['final_score']:.1f}/100\n")


# Test code
if __name__ == "__main__":
    from api_helper import load_api_params

    # Load API params
    params = load_api_params()

    # Create the evaluator
    evaluator = TransitConvenienceEvaluator(
        api_params=params,
        tube_weight=0.8,
        bus_weight=0.2
    )

    # Test address
    test_property = {
        "address": "King's Cross, London N1 9AP"
    }

    print("=" * 60)
    print("Testing public transport convenience evaluation")
    print("=" * 60)

    score = evaluator.evaluate(test_property, search_radius=1200)

    print(f"\nFinal score: {score:.1f}/100")

    if "_api_transit_data" in test_property:
        print("\nDetailed data:")
        import json
        print(json.dumps(test_property["_api_transit_data"], indent=2, ensure_ascii=False))
