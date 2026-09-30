"""
Data structure migrations for evaluation cache.

When the evaluation result structure changes:
1. Increment SCHEMA_VERSION in config.py
2. Add a migration function here: migrate_v{old}_to_v{new}
3. Register it in MIGRATIONS dict

Migrations are applied lazily when reading cached data.

Also contains the one-time migration from the old `evaluations` table
to the new per-dimension tables (dim_commute, dim_transit, etc.).
"""

from typing import Dict, Any, Callable, Optional
import logging
import sqlite3
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Type alias for migration functions
MigrationFunc = Callable[[Dict[str, Any]], Dict[str, Any]]

# =============================================================================
# Migration Functions
# =============================================================================

def migrate_v1_to_v2(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Example migration from schema v1 to v2.

    Uncomment and modify when you need to migrate.

    Example changes:
    - Rename field: 'old_name' -> 'new_name'
    - Add new field with default value
    - Restructure nested objects
    """
    # Example: rename a field
    # if 'old_field_name' in data:
    #     data['new_field_name'] = data.pop('old_field_name')

    # Example: add new field with default
    # if 'new_required_field' not in data:
    #     data['new_required_field'] = 'default_value'

    # Example: restructure scores
    # if 'commute_score' in data and 'scores' not in data:
    #     data['scores'] = {
    #         'commute': data.pop('commute_score'),
    #         'transit': data.pop('transit_score'),
    #         ...
    #     }

    return data


def migrate_v2_to_v3(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Example migration from schema v2 to v3.
    Add your migration logic here when needed.
    """
    return data


def migrate_v3_to_v4(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migration from schema v3 to v4.
    Adds environment data fields: flood_risk, air_quality, parks.
    These are informational only (no scoring impact).
    """
    # No data migration needed - new fields will be populated on re-evaluation
    return data


def create_environment_cache_tables(db_path: str) -> None:
    """
    Create cache tables for environment data (flood, air quality, parks).
    Called during schema migration to v4.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")

    # Flood risk cache (TTL: 90 days)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS flood_cache (
            lat REAL,
            lng REAL,
            data_json TEXT,
            cached_at TEXT,
            PRIMARY KEY (lat, lng)
        )
    """)

    # Air quality cache (TTL: 365 days)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS air_quality_cache (
            lat REAL,
            lng REAL,
            data_json TEXT,
            cached_at TEXT,
            PRIMARY KEY (lat, lng)
        )
    """)

    # Parks cache (TTL: 90 days)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS parks_cache (
            lat REAL,
            lng REAL,
            data_json TEXT,
            cached_at TEXT,
            PRIMARY KEY (lat, lng)
        )
    """)

    conn.commit()
    conn.close()
    logger.info("Created environment cache tables (flood, air_quality, parks)")


# =============================================================================
# Migration Registry
# =============================================================================

# Register migrations: (from_version, to_version) -> migration_function
MIGRATIONS: Dict[tuple, MigrationFunc] = {
    # (1, 2): migrate_v1_to_v2,
    # (2, 3): migrate_v2_to_v3,
}


# =============================================================================
# Migration Engine
# =============================================================================

def migrate_data(
    data: Dict[str, Any],
    from_version: int,
    to_version: int
) -> Dict[str, Any]:
    """
    Migrate data from one schema version to another.

    Applies migrations sequentially: v1 -> v2 -> v3 -> ...

    Args:
        data: The evaluation result data to migrate
        from_version: Current schema version of the data
        to_version: Target schema version

    Returns:
        Migrated data compatible with to_version

    Raises:
        ValueError: If migration path doesn't exist
    """
    if from_version == to_version:
        return data

    if from_version > to_version:
        # Downgrade not supported
        logger.warning(
            f"Cannot downgrade from v{from_version} to v{to_version}. "
            "Returning data as-is."
        )
        return data

    current_version = from_version
    migrated_data = data.copy()

    while current_version < to_version:
        next_version = current_version + 1
        migration_key = (current_version, next_version)

        if migration_key in MIGRATIONS:
            logger.info(f"Migrating data from v{current_version} to v{next_version}")
            migrated_data = MIGRATIONS[migration_key](migrated_data)
        else:
            # No migration needed for this step (schema compatible)
            logger.debug(
                f"No migration registered for v{current_version} to v{next_version}, "
                "assuming backward compatible"
            )

        current_version = next_version

    return migrated_data


def needs_migration(from_version: int, to_version: int) -> bool:
    """Check if any migrations exist between versions."""
    for v in range(from_version, to_version):
        if (v, v + 1) in MIGRATIONS:
            return True
    return False


# =============================================================================
# Validation
# =============================================================================

def validate_schema_version(data: Dict[str, Any], expected_version: int) -> bool:
    """
    Validate that data conforms to expected schema version.

    This is a basic check. Add more specific validations as needed.
    """
    # Add version-specific validation rules here
    required_fields_by_version = {
        1: ['address', 'total_score'],
        # 2: ['address', 'total_score', 'new_required_field'],
    }

    required_fields = required_fields_by_version.get(expected_version, [])

    for field in required_fields:
        if field not in data:
            logger.warning(f"Missing required field '{field}' for schema v{expected_version}")
            return False

    return True


# =============================================================================
# One-time Migration: evaluations -> dim_* tables
# =============================================================================

def migrate_evaluations_to_dimensions(db_path: str) -> Dict[str, int]:
    """
    Migrate data from the old `evaluations` table into the 5 new dimension tables.
    After migration, renames old table to `evaluations_legacy`.

    Returns dict with migration counts per dimension.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row

    # Check if old table exists
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    if 'evaluations' not in tables:
        conn.close()
        return {}

    # Check if already migrated
    if 'evaluations_legacy' in tables:
        conn.close()
        return {}

    logger.info("Starting migration from evaluations to dimension tables...")

    counts = {
        "commute": 0,
        "transit": 0,
        "safety": 0,
        "demographics": 0,
        "price_analysis": 0,
    }

    rows = conn.execute('SELECT * FROM evaluations').fetchall()

    for row in rows:
        try:
            result = json.loads(row['result_json'])
        except (json.JSONDecodeError, TypeError):
            continue

        postcode = row['postcode']
        destination = row['destination']
        data_version = row['data_version']
        created_at = row['created_at']
        expires_at = row['expires_at']

        # Extract dimension data from result_json and insert into dim tables
        dim_map = {
            "commute": result.get("commute"),
            "transit": result.get("transit"),
            "safety": result.get("safety"),
            "demographics": result.get("demographics"),
            "price_analysis": result.get("price_analysis"),
        }

        # Extract scores
        scores_dict = result.get("scores", {})
        score_map = {
            "commute": row['commute_score'],
            "transit": row['transit_score'],
            "safety": row['safety_score'],
            "demographics": None,
            "price_analysis": row['price_score'],
        }

        for dim_name, dim_data in dim_map.items():
            if dim_data is None:
                continue

            data_json = json.dumps(dim_data, ensure_ascii=False)
            score = score_map.get(dim_name)

            try:
                if dim_name == "commute":
                    conn.execute('''
                        INSERT OR IGNORE INTO dim_commute
                        (postcode, destination, data_version, data_json, score, created_at, expires_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (postcode, destination, data_version, data_json, score, created_at, expires_at))
                else:
                    table = f"dim_{dim_name}"
                    conn.execute(f'''
                        INSERT OR IGNORE INTO {table}
                        (postcode, data_version, data_json, score, created_at, expires_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (postcode, data_version, data_json, score, created_at, expires_at))

                counts[dim_name] += 1
            except Exception as e:
                logger.warning(f"Failed to migrate {dim_name} for {postcode}: {e}")

    # Rename old table
    conn.execute('ALTER TABLE evaluations RENAME TO evaluations_legacy')
    conn.commit()
    conn.close()

    logger.info(f"Migration complete: {counts}")
    return counts
