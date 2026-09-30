#!/usr/bin/env python3
"""
Economic factors evaluation module
Scores rent/price, service charges, bills, appreciation potential, etc.
"""

from typing import Dict, Any, Optional
from api_helper import load_api_params


class EconomicFactorsEvaluator:
    """Economic factors evaluator"""

    def __init__(self, api_params: Dict[str, Any] = None):
        self.max_score = 100
        self.api_params = api_params or load_api_params()
        self.price_analyzer = None

    def _get_price_analyzer(self):
        """Lazily load the price analyser"""
        if self.price_analyzer is None:
            from price_analysis_evaluator import PriceAnalysisEvaluator
            self.price_analyzer = PriceAnalysisEvaluator(self.api_params)
        return self.price_analyzer

    def _is_price_analysis_enabled(self, property_data: Dict[str, Any]) -> bool:
        """Check whether price analysis is enabled"""
        eval_config = property_data.get("_eval_config", {})
        dimensions = eval_config.get("dimensions", {})
        economic = dimensions.get("economic_factors", {})
        sub_dims = economic.get("sub_dimensions", {})
        price_analysis = sub_dims.get("price_analysis", {})
        return price_analysis.get("enabled", True)

    def evaluate(self, property_data: Dict[str, Any]) -> float:
        """
        Evaluate economic factors

        Args:
            property_data: property data

        Returns:
            Score (0-100)
        """
        scores = []
        weights_used = []

        # Check whether price analysis is enabled
        price_analysis_enabled = self._is_price_analysis_enabled(property_data)

        if price_analysis_enabled:
            # Run price analysis on API data
            price_analysis_result = self._run_price_analysis(property_data)
            if price_analysis_result:
                property_data["_api_price_analysis"] = price_analysis_result

                # Value-for-money score from API data
                price_score = self._evaluate_price_with_api(property_data, price_analysis_result)
                scores.append(price_score * 0.35)
                weights_used.append(0.35)

                # Appreciation-potential score from API data
                appreciation_score = self._evaluate_appreciation_with_api(property_data, price_analysis_result)
                scores.append(appreciation_score * 0.25)
                weights_used.append(0.25)
            else:
                # API failed: use the default evaluation
                price_score = self._evaluate_price_value(property_data)
                scores.append(price_score * 0.35)
                weights_used.append(0.35)

                appreciation_score = self._evaluate_appreciation_potential(property_data)
                scores.append(appreciation_score * 0.25)
                weights_used.append(0.25)
        else:
            # Price analysis not used: use the default evaluation
            price_score = self._evaluate_price_value(property_data)
            scores.append(price_score * 0.35)
            weights_used.append(0.35)

            appreciation_score = self._evaluate_appreciation_potential(property_data)
            scores.append(appreciation_score * 0.25)
            weights_used.append(0.25)

        # Service charge / Council Tax
        fees_score = self._evaluate_fees(property_data)
        scores.append(fees_score * 0.25)
        weights_used.append(0.25)

        # Utility bills (water, electricity, gas, etc.)
        utilities_cost_score = self._evaluate_utilities_cost(property_data)
        scores.append(utilities_cost_score * 0.15)
        weights_used.append(0.15)

        return sum(scores)

    def _run_price_analysis(self, property_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Run price analysis

        Args:
            property_data: property data

        Returns:
            Analysis result dict, or None on failure
        """
        address = property_data.get("address", "")
        if not address:
            return None

        # Extract the postcode from the address
        postcode = self._extract_postcode(address)
        if not postcode:
            print(f"   ⚠️  Could not extract a postcode from the address: {address}")
            return None

        try:
            analyzer = self._get_price_analyzer()
            # Fetch 30 years of data; the average price uses the last 5 years
            result = analyzer.analyze_area_prices(
                postcode=postcode,
                years=30,
                avg_price_years=5,
                target_price=property_data.get("price"),
                filter_outliers=True
            )
            return result if not result.get("error") else None
        except Exception as e:
            print(f"   ⚠️  Price analysis failed: {e}")
            return None

    def _extract_postcode(self, address: str) -> Optional[str]:
        """Extract the postcode from an address"""
        import re
        # UK postcode regex
        postcode_pattern = r'\b([A-Z]{1,2}[0-9][0-9A-Z]?\s*[0-9][A-Z]{2})\b'
        match = re.search(postcode_pattern, address.upper())
        if match:
            postcode = match.group(1).replace(" ", "")
            if len(postcode) > 3:
                return f"{postcode[:-3]} {postcode[-3:]}"
            return postcode
        return None

    def _evaluate_price_with_api(
        self,
        property_data: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> float:
        """Score value for money using API data"""
        target_price = property_data.get("price", 0)
        if target_price <= 0:
            return 70  # default score when there is no price data

        summary = analysis.get("summary", {})
        target_comparison = analysis.get("target_comparison")

        # If there is a target-price comparison
        if target_comparison:
            vs_median = target_comparison.get("vs_median", 0)
            percentile = target_comparison.get("percentile", 50)

            # The further below the median, the higher the score
            if vs_median <= -15:
                price_score = 100
            elif vs_median <= -10:
                price_score = 90
            elif vs_median <= -5:
                price_score = 80
            elif vs_median <= 5:
                price_score = 70
            elif vs_median <= 10:
                price_score = 60
            elif vs_median <= 15:
                price_score = 50
            else:
                price_score = max(30, 50 - (vs_median - 15))

            # Fine-tune by percentile
            if percentile <= 25:
                price_score = min(100, price_score + 10)
            elif percentile >= 75:
                price_score = max(30, price_score - 10)

            return price_score

        # No target comparison: use the area average price
        avg_price = summary.get("average_price", 0)
        if avg_price > 0:
            ratio = target_price / avg_price
            if ratio <= 0.85:
                return 100
            elif ratio <= 0.95:
                return 85
            elif ratio <= 1.05:
                return 70
            elif ratio <= 1.15:
                return 55
            else:
                return max(30, 70 - (ratio - 1.15) * 100)

        return 70

    def _evaluate_appreciation_with_api(
        self,
        property_data: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> float:
        """Score appreciation potential using API data"""
        trends = analysis.get("trends", {})

        # Get the CAGR (defaults to 0 when None)
        cagr_5 = trends.get("5_year_cagr") or 0
        cagr_3 = trends.get("3_year_cagr") or 0
        direction = trends.get("direction", "unknown")

        # 5-year CAGR score
        if cagr_5 >= 8:
            cagr_score = 100
        elif cagr_5 >= 5:
            cagr_score = 85
        elif cagr_5 >= 3:
            cagr_score = 70
        elif cagr_5 >= 1:
            cagr_score = 55
        elif cagr_5 >= 0:
            cagr_score = 45
        else:
            cagr_score = max(20, 45 + cagr_5 * 5)

        # Trend-direction adjustment
        direction_adjustment = {
            "up": 10,
            "stable": 0,
            "down": -15,
            "unknown": 0
        }
        adjustment = direction_adjustment.get(direction, 0)

        # 3-year vs 5-year trend (accelerating/decelerating)
        if cagr_3 is not None and cagr_5 is not None:
            if cagr_3 > cagr_5 + 2:
                adjustment += 5  # accelerating growth
            elif cagr_3 < cagr_5 - 2:
                adjustment -= 5  # decelerating growth

        return min(100, max(20, cagr_score + adjustment))

    def _evaluate_price_value(self, data: Dict[str, Any]) -> float:
        """Score rent/price value for money (placeholder heuristic)"""
        # TODO: implement the actual evaluation logic
        # Factors: price vs market average, price per unit area, value for money
        price = data.get("price", 500000)
        market_average_price = data.get("market_average_price", 550000)
        property_size_sqm = data.get("property_size_sqm", 70)
        bedrooms = data.get("bedrooms", 2)

        # Price comparison score (the further below market, the better)
        price_ratio = price / market_average_price
        if price_ratio <= 0.85:
            price_comparison_score = 100
        elif price_ratio <= 0.95:
            price_comparison_score = 85
        elif price_ratio <= 1.05:
            price_comparison_score = 70
        elif price_ratio <= 1.15:
            price_comparison_score = 55
        else:
            price_comparison_score = 40

        # Unit-price score
        price_per_sqm = price / property_size_sqm if property_size_sqm > 0 else 10000
        # Assume an ideal unit price of £5000-7000/sqm
        if 5000 <= price_per_sqm <= 7000:
            unit_price_score = 100
        elif 4000 <= price_per_sqm < 5000 or 7000 < price_per_sqm <= 8000:
            unit_price_score = 85
        else:
            unit_price_score = max(40, 100 - abs(price_per_sqm - 6000) / 100)

        # Value per bedroom
        price_per_bedroom = price / bedrooms if bedrooms > 0 else 1000000
        if price_per_bedroom <= 200000:
            bedroom_value_score = 100
        elif price_per_bedroom <= 250000:
            bedroom_value_score = 85
        elif price_per_bedroom <= 300000:
            bedroom_value_score = 70
        else:
            bedroom_value_score = 50

        return (
            price_comparison_score * 0.5 +
            unit_price_score * 0.3 +
            bedroom_value_score * 0.2
        )

    def _evaluate_fees(self, data: Dict[str, Any]) -> float:
        """Score service charge / Council Tax (placeholder heuristic)"""
        # TODO: implement the actual evaluation logic
        service_charge_annual = data.get("service_charge_annual", 1200)
        ground_rent_annual = data.get("ground_rent_annual", 300)
        council_tax_annual = data.get("council_tax_annual", 1500)

        total_fees = service_charge_annual + ground_rent_annual + council_tax_annual

        # Total-fees score (lower is better)
        if total_fees <= 2000:
            return 100
        elif total_fees <= 3000:
            return 85
        elif total_fees <= 4000:
            return 70
        elif total_fees <= 5000:
            return 55
        else:
            return max(30, 100 - (total_fees - 5000) / 100)

    def _evaluate_utilities_cost(self, data: Dict[str, Any]) -> float:
        """Score utility bills (water, electricity, gas, etc.) (placeholder heuristic)"""
        # TODO: implement the actual evaluation logic
        estimated_monthly_utilities = data.get("estimated_monthly_utilities", 150)
        has_efficient_heating = data.get("has_efficient_heating", True)
        energy_rating = data.get("energy_rating", "C")  # A-G

        # Monthly bills score
        if estimated_monthly_utilities <= 100:
            bill_score = 100
        elif estimated_monthly_utilities <= 150:
            bill_score = 85
        elif estimated_monthly_utilities <= 200:
            bill_score = 70
        else:
            bill_score = 50

        # Energy-rating score
        energy_scores = {
            "A": 100,
            "B": 90,
            "C": 75,
            "D": 60,
            "E": 45,
            "F": 30,
            "G": 20
        }
        energy_score = energy_scores.get(energy_rating, 60)

        # Efficient-heating bonus
        heating_bonus = 10 if has_efficient_heating else 0

        return min(100, bill_score * 0.5 + energy_score * 0.4 + heating_bonus)

    def _evaluate_appreciation_potential(self, data: Dict[str, Any]) -> float:
        """Score future appreciation potential (placeholder heuristic)"""
        # TODO: implement the actual evaluation logic
        # Factors: area development plans, transport plans, historical appreciation rate
        area_development_plan = data.get("area_development_plan", "Moderate")
        planned_transport_improvements = data.get("planned_transport_improvements", False)
        historical_appreciation_rate = data.get("historical_appreciation_rate_percent", 3)
        regeneration_area = data.get("regeneration_area", False)

        # Area development plan score
        development_scores = {
            "Excellent": 100,
            "Good": 85,
            "Moderate": 70,
            "Limited": 50,
            "None": 40
        }
        development_score = development_scores.get(area_development_plan, 70)

        # Planned transport improvements bonus
        transport_bonus = 20 if planned_transport_improvements else 0

        # Historical appreciation rate score
        if historical_appreciation_rate >= 5:
            appreciation_score = 100
        elif historical_appreciation_rate >= 3:
            appreciation_score = 80
        elif historical_appreciation_rate >= 2:
            appreciation_score = 60
        else:
            appreciation_score = 40

        # Regeneration area bonus
        regeneration_bonus = 15 if regeneration_area else 0

        return min(100,
            development_score * 0.4 +
            appreciation_score * 0.4 +
            transport_bonus * 0.5 +
            regeneration_bonus * 0.5
        )
