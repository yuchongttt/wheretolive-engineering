"""
Data access layer for dimension-based evaluation cache.
Each dimension (commute, transit, safety, demographics, price_analysis) has its own table.
"""

import re
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager

from .config import DB_PATH, DIMENSION_TABLES, DIMENSIONS


_FULL_PC_RE = re.compile(r"^[A-Z]{1,2}[0-9][A-Z0-9]?[0-9][A-Z]{2}$")


def _norm_pc(pc: Optional[str]) -> str:
    """Canonical, space-insensitive cache key for a UK postcode.

    "N1 7GZ" and "N17GZ" are the same postcode; the listing source stores it unspaced
    while report URLs / user input pass it spaced. Keying on a single canonical
    form collapses the two so a postcode is cached (and read) exactly once.

    We normalise to the SPACED form ("N1 7GZ") — the incode is always the last 3
    chars — because that is how ~99.7% of the existing cache is already stored;
    keying unspaced would orphan every one of those rows and trigger a full
    recompute. Outcode-only or non-standard values are just upper/space-stripped.
    """
    s = (pc or "").upper().replace(" ", "")
    if _FULL_PC_RE.match(s):
        return f"{s[:-3]} {s[-3:]}"
    return s


@dataclass
class DimensionRecord:
    """Represents a cached dimension result (postcode-only dimensions)."""
    postcode: str
    data_version: str
    data_json: str
    score: Optional[float] = None
    created_at: Optional[str] = None
    expires_at: Optional[str] = None
    id: Optional[int] = None

    def get_data(self) -> dict:
        return json.loads(self.data_json)


@dataclass
class CommuteDimensionRecord:
    """Represents a cached commute dimension result (postcode + destination)."""
    postcode: str
    destination: str
    data_version: str
    data_json: str
    score: Optional[float] = None
    created_at: Optional[str] = None
    expires_at: Optional[str] = None
    id: Optional[int] = None

    def get_data(self) -> dict:
        return json.loads(self.data_json)


@dataclass
class RankingResult:
    """Represents ranking statistics for a dimension."""
    dimension: str
    score: float
    percentile: float  # 0-100, higher is better
    rank: int
    total_count: int
    period_days: int


@dataclass
class Statistics:
    """Overall statistics for a time period."""
    unique_postcodes: int
    total_evaluations: int
    avg_score: Optional[float]
    min_score: Optional[float]
    max_score: Optional[float]
    period_days: int


class DimensionRepository:
    """
    Repository for a single dimension table.
    Commute dimension uses (postcode, destination, data_version) as key.
    Other dimensions use (postcode, data_version) as key.
    """

    def __init__(self, dimension: str, db_path: str = DB_PATH):
        self.dimension = dimension
        self.table_name = DIMENSION_TABLES[dimension]
        self.is_commute = (dimension == "commute")
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA mmap_size = 30000000000")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            if self.is_commute:
                conn.execute(f'''
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        postcode TEXT NOT NULL,
                        destination TEXT NOT NULL,
                        data_version TEXT NOT NULL,
                        data_json TEXT NOT NULL,
                        score REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        expires_at DATETIME,
                        UNIQUE(postcode, destination, data_version)
                    )
                ''')
                conn.execute(f'''
                    CREATE INDEX IF NOT EXISTS idx_{self.dimension}_lookup
                    ON {self.table_name}(postcode, destination)
                ''')
            else:
                conn.execute(f'''
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        postcode TEXT NOT NULL,
                        data_version TEXT NOT NULL,
                        data_json TEXT NOT NULL,
                        score REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        expires_at DATETIME,
                        UNIQUE(postcode, data_version)
                    )
                ''')
                conn.execute(f'''
                    CREATE INDEX IF NOT EXISTS idx_{self.dimension}_postcode
                    ON {self.table_name}(postcode)
                ''')

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def get(
        self,
        postcode: str,
        data_version: str,
        destination: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Get cached dimension data.
        For commute: requires destination.
        For others: destination is ignored.
        """
        with self._get_connection() as conn:
            if self.is_commute:
                cursor = conn.execute(f'''
                    SELECT * FROM {self.table_name}
                    WHERE postcode = ?
                      AND destination = ?
                      AND data_version = ?
                      AND (expires_at IS NULL OR expires_at > datetime('now'))
                ''', (_norm_pc(postcode), _norm_pc(destination), data_version))
                row = cursor.fetchone()
                if row:
                    return CommuteDimensionRecord(
                        id=row['id'],
                        postcode=row['postcode'],
                        destination=row['destination'],
                        data_version=row['data_version'],
                        data_json=row['data_json'],
                        score=row['score'],
                        created_at=row['created_at'],
                        expires_at=row['expires_at'],
                    )
            else:
                cursor = conn.execute(f'''
                    SELECT * FROM {self.table_name}
                    WHERE postcode = ?
                      AND data_version = ?
                      AND (expires_at IS NULL OR expires_at > datetime('now'))
                ''', (_norm_pc(postcode), data_version))
                row = cursor.fetchone()
                if row:
                    return DimensionRecord(
                        id=row['id'],
                        postcode=row['postcode'],
                        data_version=row['data_version'],
                        data_json=row['data_json'],
                        score=row['score'],
                        created_at=row['created_at'],
                        expires_at=row['expires_at'],
                    )
        return None

    def save(self, record) -> int:
        """Save or update a dimension record (upsert). Returns record ID."""
        with self._get_connection() as conn:
            if self.is_commute:
                rec = record  # CommuteDimensionRecord
                conn.execute(f'''
                    INSERT INTO {self.table_name} (
                        postcode, destination, data_version, data_json, score,
                        created_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, datetime('now'), ?)
                    ON CONFLICT(postcode, destination, data_version)
                    DO UPDATE SET
                        data_json = excluded.data_json,
                        score = excluded.score,
                        created_at = datetime('now'),
                        expires_at = excluded.expires_at
                ''', (
                    _norm_pc(rec.postcode),
                    _norm_pc(rec.destination),
                    rec.data_version,
                    rec.data_json,
                    rec.score,
                    rec.expires_at,
                ))
                cursor = conn.execute(f'''
                    SELECT id FROM {self.table_name}
                    WHERE postcode = ? AND destination = ? AND data_version = ?
                ''', (_norm_pc(rec.postcode), _norm_pc(rec.destination), rec.data_version))
            else:
                rec = record  # DimensionRecord
                conn.execute(f'''
                    INSERT INTO {self.table_name} (
                        postcode, data_version, data_json, score,
                        created_at, expires_at
                    ) VALUES (?, ?, ?, ?, datetime('now'), ?)
                    ON CONFLICT(postcode, data_version)
                    DO UPDATE SET
                        data_json = excluded.data_json,
                        score = excluded.score,
                        created_at = datetime('now'),
                        expires_at = excluded.expires_at
                ''', (
                    _norm_pc(rec.postcode),
                    rec.data_version,
                    rec.data_json,
                    rec.score,
                    rec.expires_at,
                ))
                cursor = conn.execute(f'''
                    SELECT id FROM {self.table_name}
                    WHERE postcode = ? AND data_version = ?
                ''', (_norm_pc(rec.postcode), rec.data_version))

            return cursor.fetchone()['id']

    def delete(self, record_id: int) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute(
                f'DELETE FROM {self.table_name} WHERE id = ?',
                (record_id,)
            )
            return cursor.rowcount > 0

    # =========================================================================
    # Ranking Operations
    # =========================================================================

    def get_percentile(self, score: float, days: int) -> RankingResult:
        """Calculate percentile ranking for a score in this dimension."""
        with self._get_connection() as conn:
            cursor = conn.execute(f'''
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN score > ? THEN 1 ELSE 0 END) as above
                FROM {self.table_name}
                WHERE created_at > datetime('now', ? || ' days')
                  AND score IS NOT NULL
            ''', (score, f'-{days}'))

            row = cursor.fetchone()
            total = row['total'] or 0
            above = row['above'] or 0
            rank = above + 1

            if total <= 1:
                percentile = 100.0 if total == 1 else 50.0
            else:
                percentile = max(0.0, (total - rank) / (total - 1) * 100)

            return RankingResult(
                dimension=self.dimension,
                score=score,
                percentile=round(percentile, 1),
                rank=rank,
                total_count=total,
                period_days=days,
            )

    def get_statistics(self, days: int) -> Statistics:
        with self._get_connection() as conn:
            cursor = conn.execute(f'''
                SELECT
                    COUNT(DISTINCT postcode) as unique_postcodes,
                    COUNT(*) as total_evaluations,
                    AVG(score) as avg_score,
                    MIN(score) as min_score,
                    MAX(score) as max_score
                FROM {self.table_name}
                WHERE created_at > datetime('now', ? || ' days')
            ''', (f'-{days}',))

            row = cursor.fetchone()
            return Statistics(
                unique_postcodes=row['unique_postcodes'] or 0,
                total_evaluations=row['total_evaluations'] or 0,
                avg_score=round(row['avg_score'], 1) if row['avg_score'] else None,
                min_score=round(row['min_score'], 1) if row['min_score'] else None,
                max_score=round(row['max_score'], 1) if row['max_score'] else None,
                period_days=days,
            )

    # =========================================================================
    # Cleanup Operations
    # =========================================================================

    def cleanup_old_data(self, keep_days: int, min_records: int = 100) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(f'SELECT COUNT(*) as cnt FROM {self.table_name}')
            total = cursor.fetchone()['cnt']

            if total <= min_records:
                return 0

            cursor = conn.execute(f'''
                DELETE FROM {self.table_name}
                WHERE created_at < datetime('now', ? || ' days')
                  AND id NOT IN (
                      SELECT id FROM {self.table_name}
                      ORDER BY created_at DESC
                      LIMIT ?
                  )
            ''', (f'-{keep_days}', min_records))

            return cursor.rowcount

    def cleanup_expired(self) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(f'''
                DELETE FROM {self.table_name}
                WHERE expires_at IS NOT NULL
                  AND expires_at < datetime('now')
            ''')
            return cursor.rowcount

    def vacuum(self):
        with self._get_connection() as conn:
            conn.execute('VACUUM')

    def count(self) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(f'SELECT COUNT(*) as cnt FROM {self.table_name}')
            return cursor.fetchone()['cnt']
