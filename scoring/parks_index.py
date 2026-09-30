#!/usr/bin/env python3
"""Accessible-park green-space scoring from OS Open Greenspace polygons.

Replaces the old per-point Overpass "distance to park centroid, count polygons"
approach (which dropped/mis-ranked large parks like the Royal Parks and counted
a 0.1 ha square the same as 142 ha Hyde Park). Here we:
  - load public-access park POLYGONS (data/london_parks.json, built from OS Open
    Greenspace by scripts/convert_os_greenspace.py — the official Ordnance Survey
    accessible-greenspace dataset, so private residents-only squares are excluded),
  - measure distance to each park's nearest EDGE (0 if you're inside it),
  - weight the green score by park AREA (big parks dominate, with diminishing
    returns), and
  - name only real parks (≥ NAME_MIN_AREA), so no "Living Wall" / hotel gardens.

Local equirectangular projection to metres — accurate to a fraction of a percent
for the <1 km distances / areas we need, anywhere in London. No pyproj dependency.
"""
import json
import math
import os
from functools import lru_cache

from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.strtree import STRtree

_LAT0, _LNG0 = 51.50, -0.12
_M_LAT = 111320.0
_M_LNG = 111320.0 * math.cos(math.radians(_LAT0))

MIN_COUNT_AREA = 1000.0  # m² — ≥0.1 ha to count as a "park" (drops point-sized
                         # memorial gardens / slivers that padded the raw count)
NAME_MIN_AREA = 2000.0   # m² — ≥0.2 ha to be named a "park"
SCORE_K = 6.0            # saturation constant for the area-weighted score
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "data", "london_parks.json")

# Greater London bbox the polygon dataset covers — outside it we can't score.
BBOX = (51.26, -0.55, 51.72, 0.36)


def _proj_ring(ring):
    return [((lon - _LNG0) * _M_LNG, (lat - _LAT0) * _M_LAT) for lon, lat in ring]


class ParkIndex:
    def __init__(self, geoms, metas):
        self.geoms = geoms
        self.metas = metas
        self.tree = STRtree(geoms)

    def in_bbox(self, lat, lng):
        return BBOX[0] <= lat <= BBOX[2] and BBOX[1] <= lng <= BBOX[3]

    def compute(self, lat, lng):
        px, py = (lng - _LNG0) * _M_LNG, (lat - _LAT0) * _M_LAT
        pt = Point(px, py)
        near = []
        for i in self.tree.query(pt.buffer(1000)):
            i = int(i)
            if self.metas[i]["area_m2"] < MIN_COUNT_AREA:
                continue  # too small to count as a park — honest count only
            d = self.geoms[i].distance(pt)
            if d <= 1000:
                near.append((d, i))
        near.sort(key=lambda x: x[0])

        c500 = sum(1 for d, _ in near if d <= 500)
        c1km = len(near)

        eff = 0.0
        for d, i in near:
            area_ha = self.metas[i]["area_m2"] / 10000.0
            eff += math.sqrt(area_ha) * max(0.0, 1.0 - d / 1000.0)
        score = int(round(100 * (1 - math.exp(-eff / SCORE_K))))
        level = ("excellent" if score >= 80 else "good" if score >= 55
                 else "fair" if score >= 30 else "limited")

        named, seen = [], set()
        for d, i in near:
            m = self.metas[i]
            nm = m["name"]
            if nm and m["area_m2"] >= NAME_MIN_AREA and nm not in seen:
                seen.add(nm)
                named.append({"name": nm, "distance_m": int(d)})
            if len(named) >= 5:
                break

        return {
            "parks_within_500m": c500,
            "parks_within_1km": c1km,
            "parks_count": c1km,
            "green_score": score,
            "green_level": level,
            "green_level_zh": {"excellent": "优秀", "good": "良好", "fair": "一般", "limited": "有限"}[level],
            "nearest_parks": named,
        }


def load_index(path=DEFAULT_PATH):
    parks = json.load(open(path))
    geoms, metas = [], []
    for p in parks:
        polys = []
        for ring in p.get("rings", []):
            if len(ring) < 4:
                continue
            try:
                poly = Polygon(_proj_ring(ring))
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_valid and poly.area > 0:
                    polys.append(poly)
            except Exception:
                continue
        if not polys:
            continue
        geom = polys[0] if len(polys) == 1 else MultiPolygon(
            [g for g in polys if g.geom_type == "Polygon"]).buffer(0)
        if geom.is_empty or geom.area <= 0:
            continue
        geoms.append(geom)
        metas.append({"name": p.get("name", ""), "area_m2": geom.area})
    return ParkIndex(geoms, metas)


@lru_cache(maxsize=1)
def get_index(path=DEFAULT_PATH):
    """Process-wide singleton so the STRtree is built once."""
    return load_index(path)
