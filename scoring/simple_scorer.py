#!/usr/bin/env python3
"""
Postcode scorer — five dimensions, each calibrated onto a common scale.

Every dimension first gets a raw 0-100 score from its own formula. The raw
score is mapped to an empirical London percentile (PCHIP interpolation over
22 breakpoints sampled from 10k-45k scored postcodes, see
CALIBRATION_BREAKPOINTS) and then onto N(65, 15), so "65" always means
"London median" whatever the dimension.
The weighted average of the calibrated dimensions is calibrated once more
("total" breakpoints) so the overall score has the same spread.
- Missing dimension: percentile None, its weight is redistributed.
- Fewer than 3 dimensions: no total score ("Insufficient data").
"""

import json
import math
from typing import Dict, Any, Optional

import numpy as np
from scipy.interpolate import PchipInterpolator


class SimpleScorer:
    """Scorer — percentile-calibrated."""

    # Dimension weights
    WEIGHTS = {
        "transport": 0.20,    # transport (commute + transit) 20%
        "community": 0.25,    # community (crime + IMD) 25%
        "environment": 0.20,  # environment (noise + flood + air + parks) 20%
        "price": 0.25,        # house prices 25%
        "schools": 0.10,      # schools 10%
    }

    # === Score calibration ===
    # Empirical percentile breakpoints from 10,478 London postcodes (2026-04-24).
    # Format: list of (raw_score, percentile) sorted by raw_score.
    # Used to map raw dimension scores → percentile → N(65, 15).
    # Rerun recalibrate_all.py when sample size grows >20% to refresh breakpoints.
    CALIBRATION_BREAKPOINTS = {
        # Recalibrated 2026-06-10 after the multi-hub commute blend (Option B)
        # shifted the transport raw distribution (n=33,616, mean 34.2 → was ~35.7).
        "transport": [
            (8.0, 0.02), (11.1, 0.05), (14.4, 0.10), (17.0, 0.15),
            (19.2, 0.20), (21.3, 0.25), (23.1, 0.30), (25.1, 0.35),
            (27.3, 0.40), (29.5, 0.45), (31.6, 0.50), (33.6, 0.55),
            (35.7, 0.60), (38.2, 0.65), (40.8, 0.70), (44.1, 0.75),
            (47.8, 0.80), (52.6, 0.85), (59.4, 0.90), (69.5, 0.95),
            (73.8, 0.97), (79.2, 0.99),
        ],
        "community": [
            # v9 recalibrated 2026-07-09 (n=44,622): the crime component switched to
            # the London percentile of the offline LSOA per-capita rate
            # (area_intel.crime_lsoa_12m); the raw distribution moved from mean ~68
            # (old API-count basis) to mean 51.3 / std 22.6, so all breakpoints were re-derived.
            (13.6, 0.02), (18.44, 0.05), (22.64, 0.10), (26.28, 0.15),
            (30.12, 0.20), (33.28, 0.25), (36.6, 0.30), (39.72, 0.35),
            (42.92, 0.40), (45.84, 0.45), (49.08, 0.50), (52.28, 0.55),
            (55.2, 0.60), (59.2, 0.65), (63.36, 0.70), (68.16, 0.75),
            (73.24, 0.80), (78.76, 0.85), (84.68, 0.90), (91.32, 0.95),
            (95.6, 0.97), (99.04, 0.99),
        ],
        "environment": [
            (44.2, 0.02), (49.8, 0.05), (54.9, 0.10), (58.2, 0.15),
            (60.4, 0.20), (62.4, 0.25), (64.1, 0.30), (65.8, 0.35),
            (67.5, 0.40), (69.0, 0.45), (70.2, 0.50), (71.6, 0.55),
            (73.0, 0.60), (74.2, 0.65), (75.5, 0.70), (77.0, 0.75),
            (78.2, 0.80), (79.8, 0.85), (81.3, 0.90), (83.8, 0.95),
            (84.8, 0.97), (87.5, 0.99),
        ],
        "price": [
            (35.0, 0.02), (39.5, 0.05), (44.8, 0.10), (48.2, 0.15),
            (50.5, 0.20), (52.7, 0.25), (54.5, 0.30), (56.3, 0.35),
            (58.1, 0.40), (59.9, 0.45), (61.7, 0.50), (63.6, 0.55),
            (65.6, 0.60), (67.5, 0.65), (69.5, 0.70), (72.1, 0.75),
            (74.7, 0.80), (77.3, 0.85), (79.9, 0.90), (83.3, 0.95),
            (85.2, 0.97), (87.7, 0.99),
        ],
        "schools": [
            (55.5, 0.02), (60.3, 0.05), (67.5, 0.10), (72.7, 0.15),
            (75.4, 0.20), (77.2, 0.25), (80.1, 0.30), (82.2, 0.35),
            (83.5, 0.40), (84.7, 0.45), (85.6, 0.50), (86.5, 0.55),
            (87.3, 0.60), (88.4, 0.65), (89.6, 0.70), (90.4, 0.75),
            (91.1, 0.80), (91.8, 0.85), (92.6, 0.90), (93.5, 0.95),
            (93.9, 0.97), (94.6, 0.99),
        ],
        # total: calibrated weighted average → N(65,15). Without this second calibration
        # the CLT squeezes the SD to ~7.5.
        "total": [
            (49.6, 0.02), (52.5, 0.05), (55.2, 0.10), (57.2, 0.15),
            (58.5, 0.20), (59.8, 0.25), (60.9, 0.30), (62.0, 0.35),
            (63.0, 0.40), (64.1, 0.45), (65.1, 0.50), (66.0, 0.55),
            (66.8, 0.60), (67.8, 0.65), (68.9, 0.70), (70.0, 0.75),
            (71.3, 0.80), (72.8, 0.85), (74.5, 0.90), (77.0, 0.95),
            (78.6, 0.97), (81.5, 0.99),
        ],
    }

    CALIBRATION_MEAN = 65.0
    CALIBRATION_STD = 15.0

    @staticmethod
    def _inv_normal_cdf(p: float) -> float:
        """Inverse normal CDF (probit). Rational approximation by Abramowitz & Stegun.

        Clamp at [0.0001, 0.9999] (z=±3.72). Tighter clamp at [0.003, 0.997]
        truncated the tails enough to compress total σ from 15 → 14.6
        (analytically: σ_clipped = 0.974 × σ when z=±2.748). Looser clamp
        keeps σ within ~1% of target without admitting infinity. */"""
        p = max(0.0001, min(0.9999, p))
        if p < 0.5:
            t = math.sqrt(-2.0 * math.log(p))
            return -(t - (2.515517 + 0.802853 * t + 0.010328 * t * t)
                      / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t))
        else:
            t = math.sqrt(-2.0 * math.log(1.0 - p))
            return (t - (2.515517 + 0.802853 * t + 0.010328 * t * t)
                    / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t))

    # Cache for PCHIP interpolators (built once per dimension)
    _pchip_cache: Dict[str, PchipInterpolator] = {}

    def _get_pchip(self, dimension: str) -> Optional[PchipInterpolator]:
        """Get or build a monotone cubic (PCHIP) interpolator for a dimension."""
        if dimension in self._pchip_cache:
            return self._pchip_cache[dimension]
        breakpoints = self.CALIBRATION_BREAKPOINTS.get(dimension)
        if not breakpoints:
            return None
        xs = np.array([v for v, _ in breakpoints])
        ys = np.array([p for _, p in breakpoints])
        interp = PchipInterpolator(xs, ys)
        self._pchip_cache[dimension] = interp
        return interp

    def _calibrate_score(self, raw_score: float, dimension: str) -> tuple:
        """Map raw dimension score → percentile → calibrated N(65, 15) score.

        Uses PCHIP (monotone cubic Hermite) interpolation for smooth mapping.
        Returns (calibrated_score, percentile_0_to_100).
        """
        breakpoints = self.CALIBRATION_BREAKPOINTS.get(dimension)
        if not breakpoints:
            return raw_score, 50.0

        interp = self._get_pchip(dimension)

        # Interpolate to find percentile (0-1)
        if raw_score <= breakpoints[0][0]:
            # Extrapolate below using PCHIP derivative at lower boundary
            slope = float(interp.derivative()(breakpoints[0][0]))
            pct = breakpoints[0][1] + slope * (raw_score - breakpoints[0][0])
            pct = max(0.001, pct)
        elif raw_score >= breakpoints[-1][0]:
            # Extrapolate above using PCHIP derivative at upper boundary
            slope = float(interp.derivative()(breakpoints[-1][0]))
            pct = breakpoints[-1][1] + slope * (raw_score - breakpoints[-1][0])
            pct = min(0.999, pct)
        else:
            pct = float(interp(raw_score))
            pct = max(0.001, min(0.999, pct))

        # Percentile → normal score
        z = self._inv_normal_cdf(pct)
        calibrated = self.CALIBRATION_MEAN + self.CALIBRATION_STD * z
        return round(max(5.0, min(98.0, calibrated)), 1), round(pct * 100, 1)

    # London reference values
    LONDON_AVG_PRICE = 550000       # London average price
    LONDON_AVG_PRICE_PER_SQM = 8000  # London price per m² (legacy; not referenced by the current scoring path)

    # Crime baselines per area type (monthly crimes) — legacy; not referenced by the current scoring path
    AREA_CRIME_BASELINES = {
        "central_commercial": 800,   # central business district (EC, WC)
        "major_hub": 600,            # major transport hubs
        "urban": 400,                # general urban
        "residential": 250,          # residential
    }

    def __init__(self):
        pass

    def calculate_scores(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score every dimension.

        Args:
            data: dict with commute, transit, safety, price_analysis, noise, flood_risk,
                  air_quality, parks, schools, demographics (any may be missing)

        Returns:
            dict with per-dimension scores, percentiles and the total
        """
        scores = {}

        # 1. Transport (commute 60% + transit 40%)
        commute_score = self._score_commute(data["commute"], data.get("address")) if data.get("commute") else None
        transit_score = self._score_transit(data["transit"]) if data.get("transit") else None
        scores["transport"] = self._score_transport(commute_score, transit_score)

        # 2. Community (crime 40% + IMD 60%; noise moved out)
        if data.get("safety"):
            postcode = data.get("address", "")
            scores["community"] = self._score_community(
                data["safety"], postcode, data.get("demographics"))
        else:
            scores["community"] = None

        # 3. Environment (noise 25% + flood 25% + air 25% + parks 25%)
        scores["environment"] = self._score_environment(
            data.get("noise"),
            data.get("flood_risk"),
            data.get("air_quality"),
            data.get("parks")
        )

        # 4. Price
        if data.get("price_analysis"):
            scores["price"] = self._score_price(data["price_analysis"], data.get("demographics"))
        else:
            scores["price"] = None

        # 5. Schools
        if data.get("schools"):
            scores["schools"] = self._score_schools(data["schools"])
        else:
            scores["schools"] = None

        # Calibration: raw score → percentile → N(65, 15)
        # All 5 dimensions share one calibration path (since 2026-04-24 community is
        # calibrated too instead of skipped, so its percentile is real, not a hard-coded 50)
        percentiles = {}
        missing_dims = []

        for dim_key in self.WEIGHTS:
            if scores.get(dim_key) is not None:
                raw = scores[dim_key]["score"]
                calibrated, pct = self._calibrate_score(raw, dim_key)
                scores[dim_key]["raw_score"] = raw
                scores[dim_key]["score"] = calibrated
                scores[dim_key]["percentile"] = pct
                percentiles[dim_key] = pct
            else:
                # Missing dimension → percentile None, not a misleading 50 (p50)
                percentiles[dim_key] = None
                missing_dims.append(dim_key)

        # Completeness threshold: <3 dimensions → "Insufficient data", no total
        # (stops one strong dimension masquerading as a top-25% postcode)
        MIN_DIMS_FOR_TOTAL = 3
        valid_dim_count = len(self.WEIGHTS) - len(missing_dims)

        if valid_dim_count < MIN_DIMS_FOR_TOTAL:
            total_score = None
            rating = "Insufficient data"
        else:
            # Total: weighted average of the calibrated dimensions → one more pass through
            # the "total" PCHIP to restore N(65,15). Without it the CLT squeezes the SD
            # to ~7.5 and every score bunches into 60-70 with no discrimination.
            total_weight = sum(
                self.WEIGHTS[k] for k in self.WEIGHTS if scores.get(k) is not None
            )
            raw_total = sum(
                scores[k]["score"] * (self.WEIGHTS[k] / total_weight)
                for k in self.WEIGHTS if scores.get(k) is not None
            )
            calibrated_total, _ = self._calibrate_score(raw_total, "total")
            total_score = round(calibrated_total, 1)
            rating = self._get_rating(total_score)

        return {
            "scores": scores,
            "weights": self.WEIGHTS,
            "percentiles": percentiles,
            "total_score": total_score,
            "missing_dims": missing_dims,
            "data_completeness": f"{valid_dim_count}/{len(self.WEIGHTS)}",
            "rating": rating,
        }

    # London commute time CDF (cumulative %)
    # Source: Census 2011 + TfL Travel in London Report 16
    # Format: (minutes, cumulative_pct_of_commuters_at_or_below)
    # "Shorter than X% of London commuters" → higher score
    LONDON_COMMUTE_CDF = [
        (5,   3.0),
        (10,  8.0),
        (15, 15.0),
        (20, 23.0),
        (25, 32.0),
        (30, 42.0),
        (35, 52.0),
        (40, 60.0),
        (45, 68.0),
        (50, 74.0),
        (55, 79.0),
        (60, 84.0),
        (70, 90.0),
        (80, 94.0),
        (90, 97.0),
        (120, 99.5),
    ]

    def _commute_percentile(self, minutes: float) -> float:
        """
        Given commute minutes, return the percentile of London commuters
        who commute THIS long or longer. i.e. "beats X% of commuters".
        Shorter commute → higher percentile → better score.
        """
        cdf = self.LONDON_COMMUTE_CDF
        if minutes <= cdf[0][0]:
            return 99.0
        if minutes >= cdf[-1][0]:
            return 0.5

        # Linear interpolation
        for i in range(len(cdf) - 1):
            t0, p0 = cdf[i]
            t1, p1 = cdf[i + 1]
            if t0 <= minutes <= t1:
                frac = (minutes - t0) / (t1 - t0)
                cum_pct = p0 + frac * (p1 - p0)
                # "beats" = 100 - cumulative (shorter is better)
                return round(100.0 - cum_pct, 1)
        return 50.0

    @staticmethod
    def _percentile_to_score(pct: float) -> float:
        """
        Percentile → score curve (legacy; not referenced by the current scoring path).
        Design targets: beats 50% → 60, beats 80% → 82, beats 20% → 35
        Formula as implemented: score = 10 + 90 * (pct/100)^0.7
        """
        if pct <= 0:
            return 5.0
        if pct >= 100:
            return 98.0
        score = 10 + 90 * (pct / 100) ** 0.7
        return round(min(98, max(5, score)), 1)

    # === Multi-hub commute (Option B) ===
    # The commute sub-score blends two signals so the score reflects an area's
    # *general* accessibility, not just the journey to one point:
    #   A) point-to-point commute to the default destination (City of London) —
    #      fine postcode-level geography, but a single hub.
    #   B) weighted-average commute from this postcode's SECTOR to 20 employment
    #      hubs (precomputed sector_hub_commute table) — broad coverage, but
    #      sector-level (coarser) geography.
    # Hub weights tier by employment size (core CBD > secondary > outer/airport).
    HUB_WEIGHTS = {
        'Bank': 3, 'Liverpool Street': 3, 'Oxford Circus': 3, 'Canary Wharf': 3,
        'London Bridge': 3, 'Waterloo': 3, 'Victoria': 3, "King's Cross": 3, 'Farringdon': 3,
        'Paddington': 2, 'Old Street': 2, 'Stratford': 2, 'Hammersmith': 2,
        'Clapham Junction': 1, 'East Croydon': 1, 'Wimbledon': 1, 'Lewisham': 1,
        'Ealing Broadway': 1, 'Highbury & Islington': 1, 'Heathrow': 1,
    }
    COMMUTE_BLEND_A = 0.4   # weight of the point-to-point component (B gets 0.6)

    def _hub_conn(self):
        """Lazy read-only connection to evaluations.db (sector_hub_commute)."""
        import sqlite3
        from pathlib import Path
        if getattr(self, '_hub_db', None) is None:
            db = Path(__file__).parent / 'data' / 'evaluations.db'
            self._hub_db = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        return self._hub_db

    def _hub_weighted_minutes(self, addr_or_postcode: Optional[str]) -> Optional[float]:
        """Weighted-avg commute (min) from this postcode's SECTOR to the 20 hubs.
        Accepts a clean postcode OR a full address (the live eval passes the
        latter) — a regex pulls the UK postcode out either way. None if no
        postcode is found, the sector isn't in the table, or the DB is down."""
        if not addr_or_postcode:
            return None
        import re
        m = re.search(
            r'([A-Z]{1,2}\d[A-Z\d]?)\s*(\d)[A-Z]{2}', addr_or_postcode.upper())
        if not m:
            return None
        sector = f"{m.group(1)} {m.group(2)}"   # 'HA1 1QP' / '…, HA1 1QP' → 'HA1 1'
        try:
            rows = self._hub_conn().execute(
                "SELECT hub, minutes FROM sector_hub_commute WHERE sector = ?", (sector,)
            ).fetchall()
        except Exception:
            return None
        num = den = 0.0
        for hub, minutes in rows:
            if minutes is None:
                continue
            w = self.HUB_WEIGHTS.get(hub, 1)
            num += w * minutes
            den += w
        return (num / den) if den else None

    def _score_commute(self, commute: Dict, postcode: Optional[str] = None) -> Dict[str, Any]:
        """
        Commute score — based on the real distribution of London commute times
        (Census 2011 + TfL). Blends two signals (Option B): point-to-point to the
        City (A, 40%) + weighted average from the sector to 20 employment hubs
        (B, 60%). If the sector is not in the table, falls back to pure A.
        Raw score = percentile; calibration is left to _calibrate_score.
        """
        avg_minutes = commute.get("average_minutes", 60)
        pct_a = self._commute_percentile(avg_minutes)

        hub_min = self._hub_weighted_minutes(postcode)
        if hub_min is not None:
            pct_b = self._commute_percentile(hub_min)
            pct = self.COMMUTE_BLEND_A * pct_a + (1 - self.COMMUTE_BLEND_A) * pct_b
            detail = f"{avg_minutes:.0f} min to the City · {hub_min:.0f} min to 20 hubs (weighted)"
        else:
            pct = pct_a
            detail = f"{avg_minutes:.0f} min"

        return {
            "score": round(pct, 1),
            "percentile_beats": round(pct, 1),
            "minutes": round(avg_minutes, 0),
            "hub_weighted_minutes": round(hub_min, 0) if hub_min is not None else None,
            "detail": detail,
        }

    def _score_transit(self, transit: Dict) -> Dict[str, Any]:
        """
        Transit score.
        Uses the transit evaluator's composite score as-is (already combines
        line count + line weights + walking distance).
        """
        lines_count = transit.get("lines_count", 0)
        transit_score = transit.get("score", 50)

        return {
            "score": round(transit_score, 1),
            "lines_count": lines_count,
            "lines": transit.get("lines", []),
            "detail": f"{lines_count} lines"
        }

    def _score_transport(self, commute_score: Optional[Dict], transit_score: Optional[Dict]) -> Optional[Dict[str, Any]]:
        """
        Transport = commute 60% + transit 40%.
        If only one has data, use it alone; if neither does, return None.
        """
        if commute_score is None and transit_score is None:
            return None

        components = []
        weights = []
        details = []

        # Detail strings are language-agnostic English tokens — frontend
        # can map to ZH via i18n. Previously emitted Chinese labels which
        # leaked Chinese into EN reports on /report.
        if commute_score is not None:
            components.append(commute_score["score"])
            weights.append(0.60)
            details.append(f"Commute {commute_score['score']:.0f}")

        if transit_score is not None:
            components.append(transit_score["score"])
            weights.append(0.40)
            details.append(f"Transit {transit_score['score']:.0f}")

        # Normalize weights if only one component
        total_weight = sum(weights)
        score = sum(c * w for c, w in zip(components, weights)) / total_weight

        return {
            "score": round(score, 1),
            "commute_score": commute_score["score"] if commute_score else None,
            "transit_score": transit_score["score"] if transit_score else None,
            "commute_minutes": commute_score.get("minutes") if commute_score else None,
            "lines_count": transit_score.get("lines_count") if transit_score else None,
            "detail": " + ".join(details)
        }

    # Estimated daytime population within ~1 mile radius by area type
    # Police API returns crimes in ~1 mile radius (π sq mi ≈ 3.14 sq mi)
    # London avg density ~14k/sq mi residential → ~44k in 1mi radius
    # Commercial areas have much higher daytime density
    AREA_DAYTIME_POP_1MI = {
        "central_commercial": 200000,  # EC/WC: massive daytime influx
        "major_hub": 80000,            # Major stations/hubs
        "urban": 45000,                # Inner London residential
        "residential": 30000,          # Outer London residential
    }

    def _score_community(self, safety: Dict, postcode: str,
                         demographics: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Community = crime rate 40% + IMD 60%
        - Crime (v9): London percentile of the offline LSOA per-capita rate
          (area_intel, real Census population denominators); when the offline DB
          has no row / outside London, falls back to the old Police API ~1 mile
          count / daytime-population heuristic
        - IMD: Index of Multiple Deprivation decile (1 = most deprived, 10 = least)
        """
        total_crimes = safety.get("total_crimes", 0)
        area_type = self._get_area_type(postcode)
        crime_unavailable = bool(safety.get("_data_unavailable"))
        force_name = safety.get("force_name")

        # === Crime score (40%) ===
        # v9: the old 1-mile API count had Spearman only 0.544 against the true
        # neighbourhood rate (commercial postcodes over-penalised, small-population
        # LSOAs missed). Fail-soft: any error in the offline layer takes the old path.
        crime_source = "api_1mi"
        lsoa_pctile = None
        _cp = None
        if not crime_unavailable:
            try:
                from apis.area_intel import get_crime_profile
                _cp = get_crime_profile(postcode)
                if _cp and _cp.get("resi_rate_pctile") is not None:
                    lsoa_pctile = _cp["resi_rate_pctile"]
            except Exception:
                lsoa_pctile = None

        daytime_pop = self.AREA_DAYTIME_POP_1MI.get(area_type, 45000)
        if lsoa_pctile is not None:
            # The percentile is already London-relative (100 = most crime) → invert, clamp 2-98
            crime_source = "lsoa_offline"
            crime_rate = _cp.get("resi_per_1000")
            crime_score = max(2.0, min(98.0, 100.0 - lsoa_pctile))
        elif crime_unavailable:
            # The local force does not publish to data.police.uk (e.g. Greater Manchester
            # Police since 2019), so total_crimes=0 does not mean safe. Skip the crime
            # component and give all the weight to IMD.
            crime_rate = None
            crime_score = None
        else:
            crime_rate = (total_crimes / daytime_pop) * 1000

            if crime_rate <= 3:
                crime_score = 95
            elif crime_rate <= 6:
                crime_score = 95 - (crime_rate - 3) * 5       # 95-80
            elif crime_rate <= 10:
                crime_score = 80 - (crime_rate - 6) * 7.5     # 80-50
            elif crime_rate <= 15:
                crime_score = 50 - (crime_rate - 10) * 4      # 50-30
            elif crime_rate <= 25:
                crime_score = 30 - (crime_rate - 15) * 2      # 30-10
            else:
                crime_score = max(2, 10 - (crime_rate - 25) * 0.5)

        # === IMD score (60%) ===
        # imd_decile: 1 (most deprived) to 10 (least deprived)
        # Map linearly: decile 1 → 10, decile 10 → 100
        imd_decile = demographics.get("imd_decile", 0) if demographics else 0
        if imd_decile >= 1:
            imd_score = 10 + (imd_decile - 1) * 10  # 10, 20, 30, ..., 100
        else:
            imd_score = None  # no IMD data

        # === Combined (crime 40%, IMD 60%) ===
        # Handle missing components by redistributing weights
        components = []
        weights = []
        detail_parts = []

        if crime_score is not None:
            components.append(crime_score)
            weights.append(0.40)
            detail_parts.append(f"Crime {crime_score:.0f}")
        elif crime_unavailable:
            detail_parts.append(f"Crime N/A ({force_name or 'force not publishing'})")

        if imd_score is not None:
            components.append(imd_score)
            weights.append(0.60)
            detail_parts.append(f"IMD {imd_score:.0f}")

        # Normalize weights if some components missing
        total_weight = sum(weights)
        if total_weight == 0:
            # Both crime and IMD unavailable — return None so caller can mark dim as _no_data
            return None
        score = sum(c * w for c, w in zip(components, weights)) / total_weight

        return {
            "score": round(score, 1),
            "total_crimes": total_crimes if not crime_unavailable else None,
            "crime_rate_per_1k": round(crime_rate, 1) if crime_rate is not None else None,
            "crime_score": round(crime_score, 1) if crime_score is not None else None,
            "crime_unavailable": crime_unavailable,
            "crime_source": crime_source,
            "force_name": force_name,
            "imd_decile": imd_decile,
            "imd_score": round(imd_score, 1) if imd_score is not None else None,
            "daytime_pop": daytime_pop,
            "area_type": area_type,
            "detail": " + ".join(detail_parts)
        }

    def _score_environment(self, noise: Optional[Dict], flood_risk: Optional[Dict],
                           air_quality: Optional[Dict], parks: Optional[Dict]) -> Optional[Dict[str, Any]]:
        """
        Environment = noise 25% + flood 25% + air 25% + parks 25%
        Missing components: weights are redistributed automatically.
        """
        components = []
        weights = []
        details = []

        # detail uses English tokens like transport ("Commute 64 + Transit 96");
        # pages localize via translateLegacyDetail. Hard-coded Chinese until
        # 2026-09-27. Only a fallback when a page lacks the sub-scores.

        # Noise score (from DEFRA)
        if noise and noise.get("score") is not None:
            components.append(noise["score"])
            weights.append(0.25)
            details.append(f"Noise {noise['score']:.0f}")

        # Flood risk score
        if flood_risk and flood_risk.get("score") is not None:
            components.append(flood_risk["score"])
            weights.append(0.25)
            details.append(f"Flood {flood_risk['score']:.0f}")

        # Air quality score
        if air_quality and air_quality.get("score") is not None:
            components.append(air_quality["score"])
            weights.append(0.25)
            details.append(f"Air {air_quality['score']:.0f}")

        # Parks/green score
        if parks and parks.get("green_score") is not None:
            components.append(parks["green_score"])
            weights.append(0.25)
            details.append(f"Parks {parks['green_score']:.0f}")

        if not components:
            return None

        # Normalize weights if some components missing
        total_weight = sum(weights)
        score = sum(c * w for c, w in zip(components, weights)) / total_weight

        return {
            "score": round(score, 1),
            "noise_score": noise.get("score") if noise else None,
            "flood_score": flood_risk.get("score") if flood_risk else None,
            "air_score": air_quality.get("score") if air_quality else None,
            "parks_score": parks.get("green_score") if parks else None,
            "components_count": len(components),
            "detail": " + ".join(details) if details else "No environment data"
        }

    def _get_area_type(self, postcode: str) -> str:
        """Classify the area type from the postcode."""
        if not postcode:
            return "urban"

        postcode = postcode.upper().replace(" ", "")

        # Central business district
        if postcode.startswith(("EC", "WC")):
            return "central_commercial"

        # Major transport hub districts (King's Cross, Liverpool Street, Paddington, etc.)
        # Use exact district codes to avoid false matches (E1 should not match E15)
        hub_districts = ["N1C", "W2", "SW1", "SE1", "NW1"]
        for hub in hub_districts:
            if postcode.startswith(hub) and (len(postcode) == len(hub) or not postcode[len(hub)].isdigit()):
                return "major_hub"
        # E1 specifically (not E10, E15, etc.)
        if postcode.startswith("E1") and (len(postcode) == 2 or not postcode[2].isdigit()):
            return "major_hub"

        # Outer residential
        outer_prefixes = ["RM", "IG", "DA", "BR", "CR", "SM", "KT", "TW", "UB", "HA", "EN", "WD"]
        for prefix in outer_prefixes:
            if postcode.startswith(prefix):
                return "residential"

        # Default: general urban
        return "urban"

    @staticmethod
    def _growth_to_score(cagr: Optional[float]) -> float:
        """Piecewise map of CAGR → 0-100."""
        if cagr is None:
            return 50.0
        if cagr >= 10:
            return 100.0
        if cagr >= 7:
            return 92 + (cagr - 7) * 2.67   # 92-100
        if cagr >= 5:
            return 82 + (cagr - 5) * 5      # 82-92
        if cagr >= 3:
            return 68 + (cagr - 3) * 7      # 68-82
        if cagr >= 0:
            return 50 + cagr * 6            # 50-68
        return max(0, 50 + cagr * 5)        # negative growth

    def _score_price(self, price_data: Dict, demographics: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Price score — 5 sub-scores
        - Price level 35%: pricier is better, logistic sigmoid
        - Long-term growth 30%: 10-year CAGR (5-year fallback)
        - Recent momentum 20%: 3-year CAGR (5-year fallback)
        - Growth stability 10%: std of repeat-sales annualised returns, lower is better
        - Market activity 5%: repeat-sales volume
        """
        avg_price = price_data.get("average_price", 0)
        cagr_10y = price_data.get("cagr_10y")
        cagr_5y = price_data.get("cagr_5y")
        cagr_3y = price_data.get("cagr_3y")

        # 1. Price level (35%) — pricier is better, logistic sigmoid
        if avg_price > 0:
            price_ratio = avg_price / self.LONDON_AVG_PRICE
            price_level_score = 100.0 / (1.0 + math.exp(-3.0 * (price_ratio - 1.0)))
        else:
            price_level_score = 50.0

        # 2. Long-term growth (30%) — 10-year preferred, 5-year fallback
        lt_cagr = cagr_10y if cagr_10y is not None else cagr_5y
        long_term_score = self._growth_to_score(lt_cagr)

        # 3. Recent momentum (20%) — 3-year preferred, 5-year fallback
        mom_cagr = cagr_3y if cagr_3y is not None else cagr_5y
        momentum_score = self._growth_to_score(mom_cagr)

        # 4. Growth stability (10%) — std of annualised returns over ALL repeat-sale pairs.
        # The repeat_sales field is a curated display sample (≤5 rows, biased) and
        # must not be used for statistics. The curve is centred on the full-population
        # std distribution (2026-07-08 simulation over n=985 postcodes: median std ~10,
        # p25 5.6, p90 25; the old midpoint of 5 was tuned on the biased sample and
        # pushed most postcodes below 10 points). With <5 pairs the std is too noisy,
        # so the score is neutral (min-sample convention); a missing full-population
        # field is neutral too — never fall back to the biased sample.
        std = price_data.get("repeat_sales_return_std")
        txns = price_data.get("repeat_sales_count")
        if std is not None and txns is not None and txns >= 5:
            stability_score = 100.0 / (1.0 + math.exp(0.15 * (std - 10)))
        else:
            stability_score = 50.0

        # 5. Market activity (5%) — number of repeat-sale pairs (0 or missing → neutral)
        if txns:
            activity_score = 100.0 / (1.0 + math.exp(-0.1 * (txns - 30)))
        else:
            activity_score = 50.0

        # Combined score
        score = (price_level_score * 0.35 + long_term_score * 0.3
                 + momentum_score * 0.2 + stability_score * 0.1
                 + activity_score * 0.05)

        # Best available CAGR for display
        best_cagr = lt_cagr if lt_cagr is not None else mom_cagr

        return {
            "score": round(score, 1),
            "price_level_score": round(price_level_score, 1),
            "long_term_score": round(long_term_score, 1),
            "momentum_score": round(momentum_score, 1),
            "stability_score": round(stability_score, 1),
            "activity_score": round(activity_score, 1),
            "cagr": round(best_cagr, 2) if best_cagr is not None else None,
            "avg_price": avg_price,
            "price_per_sqm": price_data.get("price_per_sqm", 0),
            "detail": f"10yr growth {best_cagr:+.1f}%/yr" if best_cagr is not None else "no growth data"
        }

    def _score_schools(self, schools: Dict) -> Dict[str, Any]:
        """
        Schools score.
        Uses the score returned by school_evaluator as-is.
        """
        score = schools.get("score", 0)
        outstanding = schools.get("outstanding_count", 0)
        good = schools.get("good_count", 0)

        parts = []
        if outstanding > 0:
            parts.append(f"{outstanding} Outstanding")
        if good > 0:
            parts.append(f"{good} Good")
        detail = ", ".join(parts) if parts else "No rated schools"

        return {
            "score": round(score, 1),
            "detail": detail,
        }

    def _get_rating(self, score: float) -> str:
        """Percentile rating — N(65,15) calibrated target distribution.

        Empirically verified on a 2000-postcode sample (2026-05-07):
        actual total distribution N(64.39, 15.31), within sampling noise
        of the design target N(65, 15).

        Output format: "Better than X% of postcodes" — single positive
        frame across all score bands. X is clamped to [1, 99] to avoid
        boundary artefacts ("Better than 0%" / "Better than 100%").

        Replaces the older "Top X%" / "Bottom Y%" split which read as
        harshly negative for sub-mean scores even when the score was
        only slightly below the median. Frontend formatRating() in
        web/src/lib/format.ts can normalise either format for display
        compatibility with cached records.
        """
        if score is None:
            return "Insufficient data"
        # Φ(z) = 0.5·(1 + erf(z/√2))
        z = (float(score) - 65.0) / 15.0
        pct_below = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        x = max(1, min(99, round(pct_below * 100.0)))
        return f"Better than {x}% of postcodes"


# Demo
if __name__ == "__main__":
    scorer = SimpleScorer()

    # Synthetic input
    test_data = {
        "address": "EC4M 8AD",
        "commute": {
            "average_minutes": 20,
        },
        "transit": {
            "lines_count": 2,
            "lines": ["central", "northern"],
            "score": 80,
        },
        "safety": {
            "total_crimes": 400,
        },
        "price_analysis": {
            "average_price": 650000,
            "price_per_sqm": 11000,
            "cagr_10y": 3.0,
            "cagr_5y": 1.5,
        }
    }

    result = scorer.calculate_scores(test_data)

    print("=" * 50)
    print("Percentile-calibrated scores")
    print("=" * 50)
    print(f"\nTotal score: {result['total_score']} (London median = 65)")
    print(f"Rating: {result['rating']}")
    print(f"Data completeness: {result['data_completeness']}")
    if result.get('missing_dims'):
        print(f"Missing dimensions (weight redistributed): {result['missing_dims']}")
    print("\nDimensions:")
    for dim, score_data in result['scores'].items():
        if score_data:
            pct = result['percentiles'].get(dim, 50)
            raw = score_data.get('raw_score', score_data['score'])
            print(f"  {dim}: raw={raw:.1f} → calibrated={score_data['score']:.1f} (pct={pct:.1f}%, weight {result['weights'][dim]:.0%}) - {score_data['detail']}")
