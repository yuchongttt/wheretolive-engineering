"""
Cache configuration for home evaluation system.

=== Versioning guide ===

When modifying the files below, the corresponding version number must be bumped:

┌─────────────────────┬──────────────────────────────────────────────────────────┐
│ Version constant    │ When to bump                                             │
├─────────────────────┼──────────────────────────────────────────────────────────┤
│ SCORER_VERSION      │ Scoring logic changes (weights, formulas, thresholds)    │
│                     │ Files involved:                                          │
│                     │   - simple_scorer.py                                     │
│                     │   - area_demographics_evaluator.py                       │
│                     │   - location_convenience.py (_calculate_*_score)         │
│                     │   - transit_convenience_evaluator.py                     │
│                     │   - long_distance_travel_evaluator.py                    │
│                     │   - economic_factors.py                                  │
├─────────────────────┼──────────────────────────────────────────────────────────┤
│ DATA_VERSION        │ External data sources refreshed (monthly/quarterly)      │
│                     │ Data involved:                                           │
│                     │   - IMD 2019 data                                        │
│                     │   - ONS Census data                                      │
│                     │   - Crime statistics                                     │
│                     │   - data/london_transport_hubs.json                      │
├─────────────────────┼──────────────────────────────────────────────────────────┤
│ SCHEMA_VERSION      │ Evaluation result schema changes (fields added/removed)  │
│                     │ Also add a migration function in migrations.py           │
└─────────────────────┴──────────────────────────────────────────────────────────┘

CACHE_VERSION is derived automatically from DATA_VERSION + SCORER_VERSION;
a change to either invalidates the cache, so it needs no manual management.
"""

import os
from pathlib import Path

# =============================================================================
# Version Control
# =============================================================================

# Data version: reflects the freshness of external API data
# Format: YYYY.MM (updated monthly) or YYYY.QN (updated quarterly)
# When this changes, cached data will be invalidated
DATA_VERSION = "2026.01"

# Scorer version: reflects the scoring algorithm
# Increment when scoring logic changes (weights, formulas, thresholds)
# v1: Initial scoring
# v2: Adjusted price growth curve, demographics decile mapping,
#     reduced commute scores, transit unchanged
# v3: Refactored dimensions: transport (commute+transit), community (crime+IMD),
#     environment (noise+flood+air+parks), price, schools
# v4: PCHIP smooth calibration, simplified commute scoring, 5-dim total breakpoints
# v5: Total score = weighted average of calibrated dimensions (no double calibration)
# v6: Noise sigmoid smooth curve, transit remove double counting, recalibrated breakpoints
# v7: Price scoring refactored: remove price_level/sqm, use 4 growth sub-dimensions
# v8: Stability/activity sub-scores use full repeat-sales stats
#     (repeat_sales_count + repeat_sales_return_std), not the curated ≤5 display sample;
#     stability sigmoid re-centred for full-population std (mid 5→10, slope 0.5→0.15,
#     <5 pairs → neutral) so the calibrated price distribution stays on N(65,15)
# v9: Community crime component reads the offline LSOA per-capita rate percentile
#     (area_intel.crime_lsoa_12m, police.uk 36m + Census denominators) instead of
#     the 1-mile API count / daytime-pop heuristic (Spearman 0.544 vs true rate);
#     falls back to the old path outside London. Community breakpoints re-derived
#     (n=44,622, raw mean 51.3 / std 22.6). Dim raw data unchanged -> s8 caches
#     migrated verbatim by scripts/migrate_dims_s8_to_s9.py.
SCORER_VERSION = 9

# Combined cache version: used as the actual cache lookup key
# Changes to either DATA_VERSION or SCORER_VERSION invalidate the cache
CACHE_VERSION = f"{DATA_VERSION}.s{SCORER_VERSION}"

# Schema version: reflects the structure of evaluation results
# Increment when data structure changes (add migration in migrations.py)
# v2: Added demographics data (ONS Census + IMD 2019)
# v3: Added IMD rank, population data to demographics
# v4: Added environment data (flood_risk, air_quality, parks) - informational only
# v5: Refactored scores structure: transport, community, environment, price, schools
SCHEMA_VERSION = 5

# =============================================================================
# Cache TTL (Time To Live)
# =============================================================================

# Default cache TTL in days
# Cache expires if: (now - cached_at) > TTL OR data_version changed
CACHE_TTL_DAYS = 30

# Per-dimension TTL in days
DIMENSION_TTL_DAYS = {
    "commute": 7,          # Commute times may vary
    "transit": 30,         # Transit lines rarely change
    "safety": 30,          # Crime data updated monthly
    "demographics": 90,    # Demographics change slowly
    "price_analysis": 90,  # Property prices change slowly
    "hub_commute": 7,      # Commute times may vary
}

# =============================================================================
# Dimension Configuration
# =============================================================================

# All dimension names (raw data dimensions, not scorer dimensions)
# Note: noise, flood_risk, air_quality, parks are fetched directly, not cached via DimensionRepository
# Core dimensions required for cache hit
DIMENSIONS = ["commute", "transit", "safety", "demographics", "price_analysis", "schools"]

# Optional dimensions: cached independently, not required for cache hit
OPTIONAL_DIMENSIONS = ["hub_commute"]

# Dimension table name mapping (raw data tables)
DIMENSION_TABLES = {
    "commute": "dim_commute",
    "transit": "dim_transit",
    "safety": "dim_safety",
    "demographics": "dim_demographics",
    "price_analysis": "dim_price_analysis",
    "schools": "dim_schools",
    "hub_commute": "dim_hub_commute",
    # Scorer dimension tables (computed)
    "transport": "dim_transport",
    "community": "dim_community",
    "environment": "dim_environment",
}

# Dimensions available for ranking (scorer dimensions)
RANKING_DIMENSIONS = {
    "transport": "dim_transport",
    "community": "dim_community",
    "environment": "dim_environment",
    "price": "dim_price_analysis",
    "schools": "dim_schools",
}

# =============================================================================
# Database Configuration
# =============================================================================

# Default database path (relative to project root)
_DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "evaluations.db"

# Allow override via environment variable
DB_PATH = os.getenv("EVALUATION_DB_PATH", str(_DEFAULT_DB_PATH))

# =============================================================================
# Ranking Configuration
# =============================================================================

# Available ranking periods
RANKING_PERIODS = {
    "1d": 1,
    "7d": 7,
    "30d": 30,
    "1y": 365,
}

# Default ranking period
DEFAULT_RANKING_PERIOD = "30d"

# =============================================================================
# Cleanup Configuration
# =============================================================================

# Keep data for this many days (older data will be cleaned up)
DATA_RETENTION_DAYS = 365

# Minimum records to keep (don't delete if below this threshold)
MIN_RECORDS_TO_KEEP = 100
