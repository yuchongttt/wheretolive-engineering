#!/usr/bin/env python3
"""
Nearby amenities evaluation module
Evaluates nearby shopping, dining, leisure and other everyday amenities
"""

from typing import Dict, Any, List, Optional
from api_helper import GeocodingService, GooglePlacesAPI


class AmenitiesEvaluator:
    """Nearby amenities evaluator"""

    # Weight per amenity category
    # Only the most-used everyday amenities: supermarket, restaurant, gym
    # Could add later: park, school, etc.
    CATEGORY_WEIGHTS = {
        "超市": 1.5,      # supermarket: most important, needed for daily shopping
        "餐厅": 1.2,      # restaurant: important, for eating out
        "健身房": 1.0,    # gym: everyday exercise
        # Could add later:
        # "公园": 1.3,
        # "学校": 1.4,
    }

    def __init__(self, api_params: Optional[Dict[str, Any]] = None):
        """
        Initialise the nearby amenities evaluator.

        Args:
            api_params: API parameter config
        """
        self.api_params = api_params or {}

        # Initialise the APIs
        google_api_key = None
        if api_params and "google_api" in api_params:
            google_api_key = api_params["google_api"]["api_key"]
            self.places_api = GooglePlacesAPI(google_api_key)
            self.use_api = True
        else:
            self.places_api = None
            self.use_api = False

        # GeocodingService is always available (prefers Postcodes.io, falls back to Google)
        self.geocoding_api = GeocodingService(google_api_key)

    def evaluate(
        self,
        property_data: Dict[str, Any],
        radius: float = 500
    ) -> float:
        """
        Evaluate nearby amenity convenience.

        Args:
            property_data: property data
            radius: search radius (m)

        Returns:
            Score (0-100)
        """
        if not self.use_api:
            print("  ⚠️  Google API not configured, skipping amenities evaluation")
            return 0

        address = property_data.get("address", "")
        if not address:
            print("  ⚠️  No address provided, cannot evaluate amenities")
            return 0

        # Use existing coordinates if available
        if "_api_amenities_coords" in property_data:
            lat = property_data["_api_amenities_coords"]["lat"]
            lng = property_data["_api_amenities_coords"]["lng"]
        elif "_api_crime_coords" in property_data:
            # Reuse the coordinates fetched during the crime evaluation
            lat = property_data["_api_crime_coords"]["lat"]
            lng = property_data["_api_crime_coords"]["lng"]
        else:
            # Fetch coordinates
            print(f"  📍 Fetching address coordinates...")
            coords = self.geocoding_api.get_coordinates(address)

            if not coords or not coords.get("success"):
                print("  ❌ Could not get address coordinates")
                return 0

            lat = coords["lat"]
            lng = coords["lng"]
            print(f"  ✅ Coordinates: {lat:.6f}, {lng:.6f}")

            # Cache the coordinates
            property_data["_api_amenities_coords"] = {"lat": lat, "lng": lng}

        # Search nearby places (everyday amenities only: supermarket, restaurant, gym)
        print(f"  🏪 Searching nearby amenities (radius {radius} m)...")
        print(f"     Search types: supermarket, restaurant, gym")
        places = self.places_api.search_nearby(
            lat, lng,
            radius=radius,
            included_types=[
                "grocery_store",  # supermarket / grocery store
                "supermarket",    # supermarket
                "restaurant",     # restaurant
                "gym",            # gym
                # Could add later:
                # "park",         # park
                # "school",       # school
            ],
            excluded_types=["hotel", "lodging"],
            max_results=20
        )

        if places is None:
            print("  ❌ Could not get nearby amenities data")
            return 0

        # Analyse the place data
        stats = self.places_api.analyze_places(places)

        # Cache the analysis result
        property_data["_api_amenities_data"] = stats

        # Compute the score
        score = self._calculate_amenities_score(stats)

        # Print the stats
        self._print_stats(stats, score)

        return score

    def _calculate_amenities_score(self, stats: Dict[str, Any]) -> float:
        """
        Compute the amenities score from the stats.

        Scoring dimensions:
        1. Quantity (40 pts): number of places and type distribution
        2. Quality (40 pts): place ratings and review counts
        3. Diversity (20 pts): variety of place types

        Args:
            stats: place statistics

        Returns:
            Score (0-100)
        """
        # 1. Quantity (40 pts)
        quantity_score = self._calculate_quantity_score(stats)

        # 2. Quality (40 pts)
        quality_score = self._calculate_quality_score(stats)

        # 3. Diversity (20 pts)
        diversity_score = self._calculate_diversity_score(stats)

        # Overall score
        final_score = quantity_score + quality_score + diversity_score

        return min(100, final_score)

    def _calculate_quantity_score(self, stats: Dict[str, Any]) -> float:
        """Compute the quantity score (max 40 pts)"""
        by_category = stats.get("by_category", {})

        # Weighted count
        weighted_count = 0
        for category, data in by_category.items():
            count = data["count"]
            weight = self.CATEGORY_WEIGHTS.get(category, 1.0)
            weighted_count += count * weight

        # Scoring rules (based on the weighted count)
        if weighted_count >= 25:
            score = 40
        elif weighted_count >= 20:
            score = 35
        elif weighted_count >= 15:
            score = 30
        elif weighted_count >= 10:
            score = 25
        elif weighted_count >= 5:
            score = 20
        else:
            score = weighted_count * 3

        return min(40, score)

    def _calculate_quality_score(self, stats: Dict[str, Any]) -> float:
        """Compute the quality score (max 40 pts)"""
        avg_rating = stats.get("average_rating", 0)
        high_rated = stats.get("high_rated", 0)
        popular = stats.get("popular", 0)
        total = stats.get("total", 1)

        # Average rating contribution (20 pts)
        # Ratings usually fall in 3.5-4.5; map to 0-20 pts
        rating_score = 0
        if avg_rating >= 4.3:
            rating_score = 20
        elif avg_rating >= 4.0:
            rating_score = 17
        elif avg_rating >= 3.8:
            rating_score = 14
        elif avg_rating >= 3.5:
            rating_score = 10
        elif avg_rating > 0:
            rating_score = avg_rating * 5

        # Share of high-rated places (10 pts)
        high_rated_ratio = high_rated / total if total > 0 else 0
        high_rated_score = high_rated_ratio * 10

        # Share of popular places (10 pts)
        popular_ratio = popular / total if total > 0 else 0
        popular_score = popular_ratio * 10

        return min(40, rating_score + high_rated_score + popular_score)

    def _calculate_diversity_score(self, stats: Dict[str, Any]) -> float:
        """Compute the diversity score (max 20 pts)"""
        categories_count = stats.get("categories_count", 0)

        # More types is better
        if categories_count >= 10:
            score = 20
        elif categories_count >= 8:
            score = 18
        elif categories_count >= 6:
            score = 15
        elif categories_count >= 4:
            score = 12
        elif categories_count >= 2:
            score = 8
        else:
            score = categories_count * 3

        return min(20, score)

    def _print_stats(self, stats: Dict[str, Any], score: float):
        """Print the stats"""
        print(f"  📊 Nearby amenities stats:")
        print(f"     Total: {stats['total']}")
        print(f"     Types: {stats['categories_count']}")
        print(f"     Average rating: {stats['average_rating']:.2f}")
        print(f"     High-rated places (≥4.0): {stats['high_rated']}")
        print(f"     Popular places (≥100 reviews): {stats['popular']}")

        # Show the main types
        by_category = stats.get("by_category", {})
        if by_category:
            print(f"     Main types:")
            # Sort by count, show the top 5
            sorted_categories = sorted(
                by_category.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )[:5]

            for category, data in sorted_categories:
                count = data["count"]
                avg_rating = data["avg_rating"]
                rating_str = f"(avg {avg_rating:.1f})" if avg_rating > 0 else ""
                print(f"       - {category}: {count} places {rating_str}")

        print(f"  ✅ Amenities score: {score:.2f}/100\n")