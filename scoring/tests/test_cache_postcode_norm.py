"""Cache postcode keys must be space-insensitive.

The listing source stores postcodes unspaced ("N17GZ"); user-typed / report-URL
paths pass them spaced ("N1 7GZ"). Because the dimension cache keyed on the raw
(upper-cased but un-stripped) postcode, the same postcode was cached TWICE and a
read with the other spacing missed.

Run: pytest tests/test_cache_postcode_norm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cache.repository import DimensionRepository, DimensionRecord, CommuteDimensionRecord

VER = "test-v1"
FUTURE = "2099-01-01 00:00:00"


def _rec(pc):
    return DimensionRecord(postcode=pc, data_version=VER, data_json="{}", score=1.0, expires_at=FUTURE)


def test_spaced_write_unspaced_read_hits(tmp_path):
    repo = DimensionRepository("safety", db_path=str(tmp_path / "c.db"))
    repo.save(_rec("N1 7GZ"))
    assert repo.get("N17GZ", VER) is not None      # unspaced read must find the spaced write
    assert repo.get("n1 7gz", VER) is not None      # and be case-insensitive


def test_both_spellings_collapse_to_one_row(tmp_path):
    repo = DimensionRepository("safety", db_path=str(tmp_path / "c.db"))
    repo.save(_rec("N1 7GZ"))
    repo.save(_rec("N17GZ"))
    assert repo.count() == 1                          # not two duplicate rows


def test_commute_postcode_and_destination_normalized(tmp_path):
    repo = DimensionRepository("commute", db_path=str(tmp_path / "c.db"))
    repo.save(CommuteDimensionRecord(
        postcode="N1 7GZ", destination="EC2R 8AH",
        data_version=VER, data_json="{}", score=1.0, expires_at=FUTURE))
    assert repo.get("N17GZ", VER, destination="EC2R8AH") is not None
    # writing the other spelling must not create a second row
    repo.save(CommuteDimensionRecord(
        postcode="N17GZ", destination="EC2R8AH",
        data_version=VER, data_json="{}", score=1.0, expires_at=FUTURE))
    assert repo.count() == 1
