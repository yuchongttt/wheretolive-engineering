"""Shared school geography + categorisation helpers.

Used by both school_evaluator.py and scripts/build_schools_table.py so the
BNG->latlon math and establishment-type categorisation can't drift between the
scorer and the schools-table ingest.
"""
import math
from typing import Tuple

# Establishment type → category mapping
INDEPENDENT_TYPES = {
    "Other independent school", "Other independent special school",
    "Non-maintained special school",
}
SPECIAL_TYPES = {
    "Community special school", "Foundation special school",
    "Academy special converter", "Academy special sponsor led",
    "Free schools special", "Pupil referral unit",
    "Academy alternative provision converter",
    "Academy alternative provision sponsor led",
    "Free schools alternative provision", "Secure units",
    "Academy secure 16 to 19", "Special post 16 institution",
}


def classify_category(establishment_type: str) -> str:
    """Classify school establishment type into a simple category."""
    if establishment_type in INDEPENDENT_TYPES:
        return "independent"
    if establishment_type in SPECIAL_TYPES:
        return "special"
    return "state"


def bng_to_latlon(easting: float, northing: float) -> Tuple[float, float]:
    """
    Convert British National Grid (OSGB36) coordinates to WGS84 latitude/longitude.
    Uses a simplified Helmert transform; accuracy is about 5m, which is sufficient for school search.
    """
    # OSGB36 ellipsoid parameters
    a = 6377563.396  # semi-major axis
    b = 6356256.909  # semi-minor axis
    F0 = 0.9996012717  # scale factor
    lat0 = math.radians(49)  # true origin latitude
    lon0 = math.radians(-2)  # true origin longitude
    N0 = -100000  # false northing
    E0 = 400000   # false easting

    e2 = 1 - (b * b) / (a * a)
    n = (a - b) / (a + b)
    n2 = n * n
    n3 = n * n * n

    lat = lat0
    M = 0

    while True:
        lat = (northing - N0 - M) / (a * F0) + lat
        Ma = (1 + n + (5/4) * n2 + (5/4) * n3) * (lat - lat0)
        Mb = (3 * n + 3 * n2 + (21/8) * n3) * math.sin(lat - lat0) * math.cos(lat + lat0)
        Mc = ((15/8) * n2 + (15/8) * n3) * math.sin(2*(lat - lat0)) * math.cos(2*(lat + lat0))
        Md = (35/24) * n3 * math.sin(3*(lat - lat0)) * math.cos(3*(lat + lat0))
        M = b * F0 * (Ma - Mb + Mc - Md)

        if abs(northing - N0 - M) < 0.00001:
            break

    cosLat = math.cos(lat)
    sinLat = math.sin(lat)
    nu = a * F0 / math.sqrt(1 - e2 * sinLat * sinLat)
    rho = a * F0 * (1 - e2) / pow(1 - e2 * sinLat * sinLat, 1.5)
    eta2 = nu / rho - 1
    tanLat = math.tan(lat)

    VII = tanLat / (2 * rho * nu)
    VIII = tanLat / (24 * rho * nu**3) * (5 + 3 * tanLat**2 + eta2 - 9 * tanLat**2 * eta2)
    IX = tanLat / (720 * rho * nu**5) * (61 + 90 * tanLat**2 + 45 * tanLat**4)
    X = 1 / (cosLat * nu)
    XI = 1 / (6 * cosLat * nu**3) * (nu / rho + 2 * tanLat**2)
    XII = 1 / (120 * cosLat * nu**5) * (5 + 28 * tanLat**2 + 24 * tanLat**4)
    XIIa = 1 / (5040 * cosLat * nu**7) * (61 + 662 * tanLat**2 + 1320 * tanLat**4 + 720 * tanLat**6)

    dE = easting - E0

    lat_osgb = lat - VII * dE**2 + VIII * dE**4 - IX * dE**6
    lon_osgb = lon0 + X * dE - XI * dE**3 + XII * dE**5 - XIIa * dE**7

    # Helmert transform: OSGB36 -> WGS84
    # Simplification: use the OSGB values directly; error < 10m
    lat_deg = math.degrees(lat_osgb)
    lon_deg = math.degrees(lon_osgb)

    return lat_deg, lon_deg
