#!/usr/bin/env python3
"""
Main module of the property evaluation system
Combines all evaluation dimensions and computes the overall score
"""

import json
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from location_convenience import LocationConvenienceEvaluator
from surrounding_environment import SurroundingEnvironmentEvaluator
from economic_factors import EconomicFactorsEvaluator
from api_helper import load_api_params


class HomeEvaluator:
    """Overall property evaluator"""

    def __init__(
        self,
        config_path: str = "config.json",
        params_path: str = "property_params.json",
        eval_config_path: str = None  # deprecated, kept for backward compatibility
    ):
        """
        Initialize the evaluator

        Args:
            config_path: path to the config file (weights and dimension toggles)
            params_path: path to the params file (API settings, etc.)
            eval_config_path: deprecated; dimension config now lives in config.json
        """
        self.config = self._load_config(config_path)
        self.weights = self.config.get("weights", {})

        # Dimension config is now read from config.json
        self.eval_config = {"dimensions": self.config.get("dimensions", {})}

        # Load API parameters
        self.api_params = load_api_params(params_path)

        # Initialize the per-dimension evaluators
        self.location_evaluator = LocationConvenienceEvaluator(self.api_params)
        self.environment_evaluator = SurroundingEnvironmentEvaluator(self.api_params)
        self.economic_evaluator = EconomicFactorsEvaluator(self.api_params)

    def set_eval_config(self, eval_config: Dict[str, Any]) -> None:
        """
        Set the runtime evaluation config (used for CLI overrides)

        Args:
            eval_config: config dict containing dimensions
        """
        if "dimensions" in eval_config:
            self.eval_config = eval_config
        else:
            self.eval_config = {"dimensions": eval_config}

    def set_commute_destination(self, destination: str) -> None:
        """
        Set the commute destination (overrides the value in the config file)

        Args:
            destination: commute destination address or postcode
        """
        if destination and hasattr(self, 'location_evaluator'):
            self.location_evaluator.commute_config["office_address"] = destination

    def is_dimension_enabled(self, dimension: str) -> bool:
        """Check whether a dimension is enabled"""
        dimensions = self.eval_config.get("dimensions", {})
        dim_config = dimensions.get(dimension, {})
        # Enabled by default
        return dim_config.get("enabled", True)

    def is_sub_dimension_enabled(self, dimension: str, sub_dimension: str) -> bool:
        """Check whether a sub-dimension is enabled"""
        dimensions = self.eval_config.get("dimensions", {})
        dim_config = dimensions.get(dimension, {})

        # If the main dimension is disabled, its sub-dimensions are disabled too
        if not dim_config.get("enabled", True):
            return False

        # Check the sub-dimension config
        sub_dims = dim_config.get("sub_dimensions", {})
        sub_config = sub_dims.get(sub_dimension, {})
        # Enabled by default
        return sub_config.get("enabled", True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load the config file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found, using default config")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return the default config"""
        return {
            "weights": {
                "location_convenience": 0.40,
                "surrounding_environment": 0.25,
                "economic_factors": 0.35,
            }
        }

    def evaluate(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a property across all dimensions

        Args:
            property_data: property data dict

        Returns:
            Dict with the per-dimension scores and the total score
        """
        # Print the enabled dimensions
        self._print_enabled_dimensions()

        # Store the sub-dimension config in property_data for the evaluators to use
        property_data["_eval_config"] = self.eval_config

        # Per-dimension scores (the config decides whether each one is evaluated)
        scores = {}
        skipped = []

        dim_evaluators = [
            ("location_convenience", "Location convenience", self.location_evaluator),
            ("surrounding_environment", "Surrounding environment", self.environment_evaluator),
            ("economic_factors", "Economic factors", self.economic_evaluator),
        ]

        # Split enabled and disabled dimensions
        enabled_dims = []
        for dim_key, dim_name, evaluator in dim_evaluators:
            if self.is_dimension_enabled(dim_key):
                enabled_dims.append((dim_key, dim_name, evaluator))
            else:
                scores[dim_key] = 0
                skipped.append(dim_name)

        # Run all enabled dimension evaluations in parallel
        with ThreadPoolExecutor(max_workers=len(enabled_dims) or 1) as executor:
            future_to_dim = {}
            for dim_key, dim_name, evaluator in enabled_dims:
                future_to_dim[executor.submit(evaluator.evaluate, property_data)] = (dim_key, dim_name)

            for future in as_completed(future_to_dim):
                dim_key, dim_name = future_to_dim[future]
                try:
                    scores[dim_key] = future.result()
                except Exception as e:
                    print(f"\n⚠️  {dim_name} evaluation error: {e}")
                    scores[dim_key] = 0

        # Compute the weighted total (weights auto-adjusted to exclude disabled dimensions)
        adjusted_weights = self._get_adjusted_weights()
        total_score = sum(
            scores[category] * adjusted_weights.get(category, 0)
            for category in scores
        )

        # Print the skipped dimensions
        if skipped:
            print(f"\n⏭️  Skipped dimensions: {', '.join(skipped)}")

        # Build the evaluation result
        result = {
            "property_info": {
                "address": property_data.get("address", "Unknown address"),
                "price": property_data.get("price", 0),
                "bedrooms": property_data.get("bedrooms", 0)
            },
            "scores": scores,
            "weights": adjusted_weights,  # adjusted weights
            "original_weights": self.weights,  # original weights, kept for reference
            "skipped_dimensions": skipped,
            "total_score": round(total_score, 2),
            "rating": self._get_rating(total_score)
        }

        return result

    def _print_enabled_dimensions(self) -> None:
        """Print the enabled evaluation dimensions"""
        dimensions = self.eval_config.get("dimensions", {})
        api_tier = self.eval_config.get("_api_tier", "all")

        if not dimensions:
            print("\n📋 Evaluation config: default (all dimensions)")
            return

        enabled = []
        disabled = []
        disabled_paid = []  # dimensions disabled because they need a paid API

        dimension_names = {
            "location_convenience": "Location convenience",
            "surrounding_environment": "Surrounding environment",
            "economic_factors": "Economic factors",
        }

        for dim_key, dim_name in dimension_names.items():
            if self.is_dimension_enabled(dim_key):
                enabled.append(dim_name)
            else:
                dim_config = dimensions.get(dim_key, {})
                if dim_config.get("_disabled_reason") == "requires paid API":
                    disabled_paid.append(dim_name)
                else:
                    disabled.append(dim_name)

        # Find sub-dimensions disabled because they need a paid API
        sub_disabled_paid = []
        sub_dimension_names = {
            ("location_convenience", "commute"): "Commute",
            ("location_convenience", "transit"): "Public transport",
            ("location_convenience", "amenities"): "Nearby amenities",
            ("location_convenience", "long_distance"): "Long-distance travel",
            ("surrounding_environment", "safety"): "Safety",
            ("economic_factors", "price_analysis"): "Price analysis",
        }

        for (dim_key, sub_key), sub_name in sub_dimension_names.items():
            dim_config = dimensions.get(dim_key, {})
            sub_dims = dim_config.get("sub_dimensions", {})
            sub_config = sub_dims.get(sub_key, {})
            if sub_config.get("_disabled_reason") == "requires paid API":
                sub_disabled_paid.append(sub_name)

        print("\n📋 Evaluation config:")
        if enabled:
            print(f"   ✅ Enabled: {', '.join(enabled)}")
        if disabled:
            print(f"   ⏭️  Skipped: {', '.join(disabled)}")
        if disabled_paid or sub_disabled_paid:
            all_paid = disabled_paid + sub_disabled_paid
            print(f"   💰 Paid API required (skipped): {', '.join(all_paid)}")

    def _get_adjusted_weights(self) -> Dict[str, float]:
        """
        Return the adjusted weights

        Disabled dimensions get weight 0; the remaining weights are rescaled proportionally
        """
        adjusted = {}
        enabled_weight_sum = 0

        # Sum of the original weights of the enabled dimensions
        for category, weight in self.weights.items():
            if self.is_dimension_enabled(category):
                enabled_weight_sum += weight

        # Rescale the weights proportionally
        if enabled_weight_sum > 0:
            for category, weight in self.weights.items():
                if self.is_dimension_enabled(category):
                    # Scale up proportionally so the weights sum to 1.0
                    adjusted[category] = weight / enabled_weight_sum
                else:
                    adjusted[category] = 0
        else:
            # If every dimension is disabled, return the original weights
            adjusted = self.weights.copy()

        return adjusted

    def _get_rating(self, score: float) -> str:
        """Map the total score to a rating"""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Good"
        elif score >= 70:
            return "Fair"
        elif score >= 60:
            return "Average"
        else:
            return "Poor"

    def print_report(self, result: Dict[str, Any]) -> None:
        """Print the evaluation report"""
        print("\n" + "="*60)
        print("Property Evaluation Report")
        print("="*60)

        # Property info
        info = result["property_info"]
        print(f"\nAddress: {info['address']}")
        print(f"Price: £{info['price']:,}")
        print(f"Bedrooms: {info['bedrooms']}")

        # Per-dimension scores
        print("\nCategory Scores:")
        print("-"*60)
        scores = result["scores"]
        weights = result["weights"]

        categories_cn = {
            "location_convenience": "Location convenience",
            "surrounding_environment": "Surrounding environment",
            "economic_factors": "Economic factors",
        }

        for category, score in scores.items():
            weight = weights.get(category, 0)
            weighted_score = score * weight
            cn_name = categories_cn.get(category, category)
            print(f"{cn_name:12s} | Score: {score:5.1f}/100 | Weight: {weight:4.0%} | Weighted: {weighted_score:5.2f}")

        # Total score and rating
        print("-"*60)
        print(f"\nTotal score: {result['total_score']}/100")
        print(f"Rating: {result['rating']}")
        print("="*60 + "\n")


def main():
    """Entry point - example usage"""
    # Sample property data
    sample_property = {
        "address": "123 London Road, Farringdon, London EC1",
        "price": 450000,
        "bedrooms": 2,
        "bathrooms": 1,
        "property_type": "Flat",

        # Location
        "commute_time_minutes": 25,
        "nearest_tube_distance_m": 300,
        "supermarket_distance_m": 200,
        "airport_distance_km": 25,
        "hospital_distance_m": 1500,

        # Living quality
        "orientation": "South",
        "noise_level": "Low",
        "property_age_years": 10,
        "layout_rating": 8,

        # Surroundings
        "crime_rate": "Low",
        "park_distance_m": 400,
        "air_quality_index": 65,

        # Education
        "school_rating": 7,

        # Other
        "has_parking": True,
        "gym_nearby": True,
        "pet_friendly": True
    }

    # Create the evaluator and run the evaluation
    evaluator = HomeEvaluator()
    result = evaluator.evaluate(sample_property)

    # Print the report
    evaluator.print_report(result)


if __name__ == "__main__":
    main()
