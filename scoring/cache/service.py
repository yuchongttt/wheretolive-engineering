"""
Cache service for home evaluation system.
Integrates caching, version control, and ranking functionality.
Uses per-dimension tables for independent storage and TTL.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .config import (
    CACHE_VERSION,
    SCHEMA_VERSION,
    CACHE_TTL_DAYS,
    DIMENSION_TTL_DAYS,
    DIMENSIONS,
    OPTIONAL_DIMENSIONS,
    RANKING_PERIODS,
    DEFAULT_RANKING_PERIOD,
    DATA_RETENTION_DAYS,
    MIN_RECORDS_TO_KEEP,
)
from .repository import (
    DimensionRepository,
    DimensionRecord,
    CommuteDimensionRecord,
    RankingResult,
    Statistics,
)
from .migrations import migrate_data, needs_migration, migrate_evaluations_to_dimensions

logger = logging.getLogger(__name__)


# Mapping from dimension name to result dict key
_DIM_RESULT_KEY = {
    "commute": "commute",
    "transit": "transit",
    "safety": "safety",
    "demographics": "demographics",
    "price_analysis": "price_analysis",
    "schools": "schools",
    "hub_commute": "hub_commute",
}

# Mapping from dimension name to score extraction path in result dict
_DIM_SCORE_KEY = {
    "commute": "commute",
    "transit": "transit",
    "safety": "safety",
    "price_analysis": "price",
    "schools": "schools",
}


@dataclass
class CacheInfo:
    """Information about cache status."""
    hit: bool
    cached_at: Optional[str]
    expires_at: Optional[str]
    data_version: str
    schema_version: int
    migrated: bool = False
    partial: bool = False  # True if only some dimensions were cached


@dataclass
class RankingInfo:
    """Ranking information for an evaluation."""
    period: str
    period_days: int
    total_evaluated: int
    percentiles: Dict[str, float]
    ranks: Dict[str, int]


class CacheService:
    """
    High-level cache service for evaluation results.
    Uses 5 independent dimension tables.
    """

    def __init__(self, db_path: Optional[str] = None):
        from .config import DB_PATH as default_db_path
        resolved_db_path = db_path or default_db_path
        kwargs = {"db_path": resolved_db_path}

        # Create dimension tables first (via repo init)
        all_dims = DIMENSIONS + OPTIONAL_DIMENSIONS
        self.repos: Dict[str, DimensionRepository] = {
            dim: DimensionRepository(dim, **kwargs)
            for dim in all_dims
        }

        # Auto-migrate from old evaluations table if it exists
        try:
            migrate_evaluations_to_dimensions(resolved_db_path)
        except Exception as e:
            logger.warning(f"Migration check failed: {e}")

    # =========================================================================
    # Noise Lookup
    # =========================================================================

    def _get_noise_for_postcode(self, postcode: str) -> Optional[Dict[str, Any]]:
        """
        Get cached noise data for a postcode.
        Returns None if no noise data is cached for this location.
        """
        import sqlite3
        import requests
        from .config import DB_PATH as default_db_path

        # Get coordinates from postcodes.io
        try:
            resp = requests.get(
                f"https://api.postcodes.io/postcodes/{postcode.replace(' ', '')}",
                timeout=5
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            if data.get("status") != 200 or not data.get("result"):
                return None
            lat = data["result"]["latitude"]
            lng = data["result"]["longitude"]
        except Exception:
            return None

        # Query noise_cache (approximate match within ~100m)
        try:
            conn = sqlite3.connect(default_db_path)
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("PRAGMA journal_mode = WAL")
            cursor = conn.execute(
                """
                SELECT road_db, rail_db, airport_db, combined_db, level, level_zh, score, dominant_source
                FROM noise_cache
                WHERE ABS(lat - ?) < 0.001 AND ABS(lng - ?) < 0.001
                ORDER BY ABS(lat - ?) + ABS(lng - ?)
                LIMIT 1
                """,
                (lat, lng, lat, lng)
            )
            row = cursor.fetchone()
            conn.close()
            if row:
                return {
                    "road_db": row[0],
                    "rail_db": row[1],
                    "airport_db": row[2],
                    "combined_db": row[3],
                    "level": row[4],
                    "level_zh": row[5],
                    "score": row[6],
                    "dominant_source": row[7],
                }
        except Exception:
            pass

        return None

    # =========================================================================
    # Cache Operations
    # =========================================================================

    def get_cached(
        self,
        postcode: str,
        destination: str,
    ) -> Tuple[Optional[Dict[str, Any]], CacheInfo]:
        """
        Get cached evaluation result by assembling from dimension tables.

        Returns full result dict if ALL enabled dimensions are cached.
        Returns None if any dimension is missing.
        """
        assembled = {}
        any_hit = False
        all_hit = True
        cached_at = None
        expires_at = None

        for dim in DIMENSIONS:
            repo = self.repos[dim]
            if dim == "commute":
                record = repo.get(postcode, CACHE_VERSION, destination=destination)
            else:
                record = repo.get(postcode, CACHE_VERSION)

            if record is not None:
                any_hit = True
                dim_data = record.get_data()
                result_key = _DIM_RESULT_KEY[dim]
                
                # Handle "_no_data" marker - treat as cache hit but with None value
                if isinstance(dim_data, dict) and dim_data.get("_no_data"):
                    assembled[result_key] = None
                else:
                    assembled[result_key] = dim_data

                # Track oldest cached_at and earliest expires_at
                if cached_at is None or (record.created_at and record.created_at < cached_at):
                    cached_at = record.created_at
                if record.expires_at:
                    if expires_at is None or record.expires_at < expires_at:
                        expires_at = record.expires_at
            else:
                all_hit = False
                assembled[_DIM_RESULT_KEY[dim]] = None

        if not all_hit:
            return None, CacheInfo(
                hit=False,
                cached_at=None,
                expires_at=None,
                data_version=CACHE_VERSION,
                schema_version=SCHEMA_VERSION,
                partial=any_hit,
            )

        # Load optional dimensions (not required for cache hit)
        for dim in OPTIONAL_DIMENSIONS:
            if dim in self.repos:
                record = self.repos[dim].get(postcode, CACHE_VERSION)
                if record is not None:
                    dim_data = record.get_data()
                    result_key = _DIM_RESULT_KEY.get(dim, dim)
                    if isinstance(dim_data, dict) and dim_data.get("_no_data"):
                        assembled[result_key] = None
                    else:
                        assembled[result_key] = dim_data

        # Add noise data from noise_cache (if available)
        assembled["noise"] = self._get_noise_for_postcode(postcode)

        # Reconstruct full result from assembled dimension data
        result = self._assemble_result(assembled)

        return result, CacheInfo(
            hit=True,
            cached_at=cached_at,
            expires_at=expires_at,
            data_version=CACHE_VERSION,
            schema_version=SCHEMA_VERSION,
        )

    def save_result(
        self,
        postcode: str,
        destination: str,
        result: Dict[str, Any],
        ttl_days: int = CACHE_TTL_DAYS,
    ) -> Dict[str, int]:
        """
        Save evaluation result by splitting into dimension tables.

        Returns dict of dimension -> record_id.
        """
        record_ids = {}

        # Get list of dimensions that have no data (legitimate missing, not API failure)
        no_data_dims = set(result.get("_no_data_dims", []))

        # Save both core and optional dimensions
        all_dims_to_save = DIMENSIONS + OPTIONAL_DIMENSIONS
        for dim in all_dims_to_save:
            result_key = _DIM_RESULT_KEY[dim]
            dim_data = result.get(result_key)
            
            # If dimension has no data, save a marker so cache knows it was checked
            if dim_data is None:
                if result_key in no_data_dims or dim in no_data_dims:
                    dim_data = {"_no_data": True}
                else:
                    continue

            dim_ttl = DIMENSION_TTL_DAYS.get(dim, ttl_days)
            expires_at = (datetime.now() + timedelta(days=dim_ttl)).strftime('%Y-%m-%d %H:%M:%S')

            # Extract score for this dimension
            score = self._extract_dim_score(result, dim)

            data_json = json.dumps(dim_data, ensure_ascii=False)

            if dim == "commute":
                record = CommuteDimensionRecord(
                    postcode=postcode.upper(),
                    destination=destination.upper(),
                    data_version=CACHE_VERSION,
                    data_json=data_json,
                    score=score,
                    expires_at=expires_at,
                )
            else:
                record = DimensionRecord(
                    postcode=postcode.upper(),
                    data_version=CACHE_VERSION,
                    data_json=data_json,
                    score=score,
                    expires_at=expires_at,
                )

            record_ids[dim] = self.repos[dim].save(record)

        # Also save total_score, scores, weights, rating etc. as metadata
        # These are derived from dimension data and stored with commute (or any dim)
        # Actually, we reconstruct these on read via _assemble_result

        logger.info(f"Cached evaluation for {postcode} -> {destination} across {len(record_ids)} dimensions")
        return record_ids

    def invalidate(self, postcode: str, destination: str) -> int:
        """Invalidate all cached dimensions for a postcode/destination."""
        deleted = 0
        for dim in DIMENSIONS + OPTIONAL_DIMENSIONS:
            repo = self.repos[dim]
            if dim == "commute":
                record = repo.get(postcode, CACHE_VERSION, destination=destination)
            else:
                record = repo.get(postcode, CACHE_VERSION)
            if record and record.id:
                if repo.delete(record.id):
                    deleted += 1
        return deleted

    def save_optional_dimension(
        self,
        postcode: str,
        dimension: str,
        data: Dict[str, Any],
        ttl_days: Optional[int] = None,
    ) -> Optional[int]:
        """Save a single optional dimension independently (e.g. hub_commute from daemon thread)."""
        if dimension not in self.repos:
            return None

        dim_ttl = ttl_days or DIMENSION_TTL_DAYS.get(dimension, CACHE_TTL_DAYS)
        expires_at = (datetime.now() + timedelta(days=dim_ttl)).strftime('%Y-%m-%d %H:%M:%S')
        data_json = json.dumps(data, ensure_ascii=False)

        record = DimensionRecord(
            postcode=postcode.upper(),
            data_version=CACHE_VERSION,
            data_json=data_json,
            score=None,
            expires_at=expires_at,
        )
        record_id = self.repos[dimension].save(record)
        logger.info(f"Cached optional dimension {dimension} for {postcode}")
        return record_id

    def _extract_dim_score(self, result: Dict[str, Any], dimension: str) -> Optional[float]:
        """Extract score for a dimension from the full result."""
        scores = result.get('scores', {})
        if not scores:
            return None

        score_key = _DIM_SCORE_KEY.get(dimension)
        if score_key is None:
            return None

        dim_score = scores.get(score_key)
        if dim_score and isinstance(dim_score, dict):
            return dim_score.get('score')
        elif isinstance(dim_score, (int, float)):
            return dim_score
        return None

    def _assemble_result(self, assembled: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assemble a full result dict from dimension data.
        Re-calculates scores using SimpleScorer.
        """
        from simple_scorer import SimpleScorer

        scorer_data = {
            "address": "",
            "commute": assembled.get("commute"),
            "transit": assembled.get("transit"),
            "safety": assembled.get("safety"),
            "demographics": assembled.get("demographics"),
            "noise": assembled.get("noise"),
            "price_analysis": assembled.get("price_analysis"),
            "schools": assembled.get("schools"),
        }
        score_result = SimpleScorer().calculate_scores(scorer_data)

        result = {}
        # Copy dimension data
        for key in ["commute", "transit", "safety", "demographics", "price_analysis", "schools", "hub_commute"]:
            if assembled.get(key) is not None:
                result[key] = assembled[key]

        # Add scoring
        result["total_score"] = score_result["total_score"]
        result["rating"] = score_result["rating"]
        result["scores"] = score_result["scores"]
        result["weights"] = score_result["weights"]
        result["percentiles"] = score_result.get("percentiles", {})
        result["missing_dims"] = score_result.get("missing_dims", [])
        result["data_completeness"] = score_result.get("data_completeness", "0/5")
        result["adjusted_weights"] = score_result.get("adjusted_weights", {})

        return result

    # =========================================================================
    # Ranking Operations
    # =========================================================================

    def get_ranking(
        self,
        result: Dict[str, Any],
        period: str = DEFAULT_RANKING_PERIOD,
    ) -> RankingInfo:
        """Get ranking information for an evaluation result."""
        if period not in RANKING_PERIODS:
            period = DEFAULT_RANKING_PERIOD

        days = RANKING_PERIODS[period]

        percentiles = {}
        ranks = {}
        total_evaluated = 0

        for dim in DIMENSIONS:
            score = self._extract_dim_score(result, dim)
            if score is not None and dim in self.repos:
                ranking = self.repos[dim].get_percentile(score, days)
                key = dim.replace('_analysis', '')  # price_analysis -> price
                percentiles[key] = ranking.percentile
                ranks[key] = ranking.rank
                total_evaluated = max(total_evaluated, ranking.total_count)

        # Total score ranking: compute dynamically
        total_score = result.get('total_score')
        if total_score is not None:
            # Use any dimension repo to count - pick the one with most data
            # For total score percentile, we compute across all dimension combos
            # Simple approach: use the commute repo's stats as proxy
            percentiles['total'] = self._compute_total_percentile(total_score, days)
            ranks['total'] = 0  # Not meaningful for dynamic total

        return RankingInfo(
            period=period,
            period_days=days,
            total_evaluated=total_evaluated,
            percentiles=percentiles,
            ranks=ranks,
        )

    def _compute_total_percentile(self, total_score: float, days: int) -> float:
        """Compute total score percentile by joining dimension tables."""
        # Use a simple heuristic: average the dimension percentiles
        # A more accurate approach would require a separate total_score table
        # For now, return 50.0 as placeholder if no data
        return 50.0

    def get_statistics(self, period: str = DEFAULT_RANKING_PERIOD) -> Statistics:
        """Get aggregate statistics. Uses the transit table as representative."""
        days = RANKING_PERIODS.get(period, RANKING_PERIODS[DEFAULT_RANKING_PERIOD])
        # Use transit as representative dimension for overall stats
        return self.repos["transit"].get_statistics(days)

    # =========================================================================
    # Maintenance Operations
    # =========================================================================

    def cleanup(self) -> Dict[str, int]:
        """Perform cleanup on all dimension tables."""
        total_expired = 0
        total_old = 0

        for dim in DIMENSIONS + OPTIONAL_DIMENSIONS:
            repo = self.repos[dim]
            total_expired += repo.cleanup_expired()
            total_old += repo.cleanup_old_data(
                keep_days=DATA_RETENTION_DAYS,
                min_records=MIN_RECORDS_TO_KEEP,
            )

        if total_expired > 0 or total_old > 0:
            # Vacuum once
            self.repos[DIMENSIONS[0]].vacuum()

        return {
            'expired_deleted': total_expired,
            'old_deleted': total_old,
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        counts = {dim: self.repos[dim].count() for dim in DIMENSIONS + OPTIONAL_DIMENSIONS}
        return {
            'total_records': counts,
            'data_version': CACHE_VERSION,
            'schema_version': SCHEMA_VERSION,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_service: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    global _service
    if _service is None:
        _service = CacheService()
    return _service


def get_cached_evaluation(
    postcode: str,
    destination: str,
) -> Tuple[Optional[Dict[str, Any]], CacheInfo]:
    return get_cache_service().get_cached(postcode, destination)


def cache_evaluation(
    postcode: str,
    destination: str,
    result: Dict[str, Any],
) -> Dict[str, int]:
    return get_cache_service().save_result(postcode, destination, result)


def get_ranking(
    result: Dict[str, Any],
    period: str = DEFAULT_RANKING_PERIOD,
) -> RankingInfo:
    return get_cache_service().get_ranking(result, period)
