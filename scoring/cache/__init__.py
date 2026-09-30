"""
Cache module for home evaluation system.
Provides per-dimension caching and ranking functionality.
"""

from .config import (
    DATA_VERSION, SCORER_VERSION, CACHE_VERSION, SCHEMA_VERSION,
    CACHE_TTL_DAYS, DB_PATH, DIMENSIONS, OPTIONAL_DIMENSIONS, DIMENSION_TABLES,
)
from .service import CacheService

__all__ = [
    'DATA_VERSION',
    'SCORER_VERSION',
    'CACHE_VERSION',
    'SCHEMA_VERSION',
    'CACHE_TTL_DAYS',
    'DB_PATH',
    'DIMENSIONS',
    'OPTIONAL_DIMENSIONS',
    'DIMENSION_TABLES',
    'CacheService',
]
