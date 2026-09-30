#!/usr/bin/env python3
"""Google Places API wrapper"""

import time
import requests
from typing import Dict, Any, Optional, List

from apis.api_usage import try_acquire_api_quota


class GooglePlacesAPI:
    """Wrapper for Google Places API calls"""

    def __init__(self, api_key: str):
        """
        Initialise the Google Places API client.

        Args:
            api_key: Google API key
        """
        self.api_key = api_key
        self.endpoint = "https://places.googleapis.com/v1/places:searchNearby"

    def search_nearby(
        self,
        lat: float,
        lng: float,
        radius: float = 300,
        included_types: Optional[List[str]] = None,
        excluded_types: Optional[List[str]] = None,
        max_results: int = 20,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Search for places near a given location.

        Args:
            lat: latitude
            lng: longitude
            radius: search radius (m), default 500 m
            included_types: list of place types to include
                Default: ["supermarket", "shopping_mall", "park", "restaurant", "cafe", "gym"]
            max_results: maximum number of results (1-20)
            max_retries: maximum number of retries
            retry_delay: retry delay (seconds)

        Returns:
            List of places, or None on failure
        """
        if included_types is None:
            included_types = [
                "grocery_store", "shopping_mall", "park",
                "restaurant", "cafe", "gym", "hospital",
                "pharmacy", "bank", "post_office"
            ]

        if excluded_types is None:
            excluded_types = ["hotel"]

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "places.id,places.displayName,places.location,places.types,places.rating,places.userRatingCount"
        }

        payload = {
            "includedTypes": included_types,
            "rankPreference": "POPULARITY",
            "maxResultCount": min(max_results, 20),
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": lat,
                        "longitude": lng
                    },
                    "radius": radius
                }
            }
        }

        last_error = None

        for attempt in range(max_retries):
            try:
                if not try_acquire_api_quota("google_places"):
                    print("  ⚠️  Google Places API monthly quota exhausted, skipping this call")
                    return None

                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=15
                )

                if response.status_code == 200:
                    data = response.json()
                    places = data.get("places", [])
                    places = self._filter_and_dedupe_places(
                        places=places,
                        excluded_types=excluded_types
                    )
                    if attempt > 0:
                        print(f"  ✅ Places API: attempt {attempt + 1} succeeded")
                    return places

                elif response.status_code == 429:
                    # Rate limited
                    last_error = "Places API rate limited (429)"
                    wait_time = retry_delay * (attempt + 1) * 2
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  {last_error}, retrying in {wait_time:.1f}s ({attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue

                elif response.status_code >= 500:
                    # Server error
                    last_error = f"Places API server error ({response.status_code})"
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  {last_error}, retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue

                else:
                    last_error = f"Places API request failed: {response.status_code} - {response.text[:200]}"
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  {last_error}, retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"  ❌ {last_error}")
                        return None

            except requests.exceptions.Timeout:
                last_error = "Places API request timed out"
                if attempt < max_retries - 1:
                    print(f"  ⚠️  {last_error}, retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue

            except requests.exceptions.RequestException as e:
                last_error = f"Places API request error: {str(e)[:100]}"
                if attempt < max_retries - 1:
                    print(f"  ⚠️  {last_error}, retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue

            except Exception as e:
                last_error = f"Error processing Places API response: {str(e)[:100]}"
                if attempt < max_retries - 1:
                    print(f"  ⚠️  {last_error}, retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue

        # All retries failed
        print(f"  ❌ Places API still failing after {max_retries} retries: {last_error}")
        return None

    def _filter_and_dedupe_places(
        self,
        places: List[Dict[str, Any]],
        excluded_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter and de-duplicate a list of places.

        De-duplication strategy (multi-level):
        1. By place_id (most precise)
        2. By name (avoids the same business listed via different entrances)
        3. By name + type combination (different branches of the same brand are also de-duplicated)

        Args:
            places: raw list of places
            excluded_types: list of types to exclude

        Returns:
            Filtered, de-duplicated list of places
        """
        excluded = set(excluded_types or [])
        seen_ids = set()      # de-dupe by place_id
        seen_names = set()    # de-dupe by (normalised) name
        result: List[Dict[str, Any]] = []

        for place in places:
            place_types = set(place.get("types", []) or [])
            if excluded and (place_types & excluded):
                continue

            # Get the place_id
            place_id = place.get("id")

            # Get and normalise the name (for name de-duplication)
            name = place.get("displayName", {}).get("text", "") or ""
            normalized_name = self._normalize_place_name(name)

            # 1. De-dupe by place_id
            if place_id and place_id in seen_ids:
                continue

            # 2. De-dupe by normalised name
            if normalized_name and normalized_name in seen_names:
                continue

            # Add to the seen sets
            if place_id:
                seen_ids.add(place_id)
            if normalized_name:
                seen_names.add(normalized_name)

            result.append(place)

        return result

    def _normalize_place_name(self, name: str) -> str:
        """
        Normalise a place name (for de-duplication).

        Steps:
        - lowercase
        - collapse extra whitespace
        - strip common suffixes (e.g. "Express", "Local")

        Args:
            name: raw name

        Returns:
            Normalised name
        """
        if not name:
            return ""

        # Lowercase
        normalized = name.lower().strip()

        # Collapse extra whitespace
        normalized = " ".join(normalized.split())

        return normalized

    def analyze_places(
        self,
        places: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyse place data and aggregate by type.

        Args:
            places: list of places

        Returns:
            Dict of analysis results
        """
        # Type mapping (maps API place types to the categories we care about)
        # Only everyday amenities: supermarket, restaurant, gym
        # Could add later: park, school, etc.
        type_mapping = {
            "supermarket": "超市",
            "grocery_store": "超市",
            "restaurant": "餐厅",
            "gym": "健身房",
            # Could add later:
            # "park": "公园",
            # "school": "学校",
        }

        # Initialise stats
        stats = {
            "total": len(places),
            "by_category": {},
            "high_rated": 0,  # rating 4.0 or above
            "popular": 0,  # 100+ reviews
            "average_rating": 0,
            "categories_count": 0
        }

        category_places = {}  # places grouped by category
        all_ratings = []

        for place in places:
            types = place.get("types", [])
            rating = place.get("rating", 0)
            user_rating_count = place.get("userRatingCount", 0)
            name = place.get("displayName", {}).get("text", "Unknown")

            # Count high-rated and popular places
            if rating >= 4.0:
                stats["high_rated"] += 1
            if user_rating_count >= 100:
                stats["popular"] += 1
            if rating > 0:
                all_ratings.append(rating)

            # Per-category counts
            for ptype in types:
                if ptype in type_mapping:
                    category = type_mapping[ptype]
                    if category not in category_places:
                        category_places[category] = []

                    category_places[category].append({
                        "name": name,
                        "rating": rating,
                        "review_count": user_rating_count,
                        "types": types
                    })

        # Count and average rating per category
        for category, places_list in category_places.items():
            ratings = [p["rating"] for p in places_list if p["rating"] > 0]
            stats["by_category"][category] = {
                "count": len(places_list),
                "avg_rating": sum(ratings) / len(ratings) if ratings else 0,
                "places": places_list[:3]  # keep the first 3 as examples
            }

        # Overall average rating
        stats["average_rating"] = sum(all_ratings) / len(all_ratings) if all_ratings else 0
        stats["categories_count"] = len(category_places)

        return stats
