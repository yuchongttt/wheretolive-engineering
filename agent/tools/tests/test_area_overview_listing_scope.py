"""get_area_overview: section headers must name the caliber of their numbers.

The on-market count is computed per OUTCODE (a single unit postcode is too
sparse), but its header once printed the unit postcode the user typed: an
outcode-wide count of several hundred appeared under a unit postcode that had a
handful, and a user was told that many "active listings in this postcode" — the
model then built a further inference on that wrong caliber. The other two sections of the same renderer already said
`(outcode {outcode})`; only this one had slipped.

(Extract note: the original file also runs the tool against a fixture whose
DDL is copied from the production database's sqlite_master; those cases need
the production DB and are not included here. What remains is the source-level
guard that the other sections keep their outcode labels.)
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_other_sections_keep_their_outcode_labels():
    """The `(outcode X)` labels on the "last 12 months" and "rent & yield"
    sections are an existing guarantee and must not be knocked off.

    Checked section by section, not by counting — a count would be tripped by
    later headers of the same kind (e.g. the rent section's "no data" branch),
    which are themselves correct outcode labels.
    """
    src = (REPO / "get_area_overview.py").read_text().splitlines()
    for section in ("Last 12 months", "Rent & yield by size"):
        hits = [ln for ln in src if f"## {section}" in ln]
        assert hits, section
        for ln in hits:
            assert "outcode {outcode}" in ln, ln
