# agents/geospatial/restrictions.py

from typing import Dict, Any, Tuple
from shapely.geometry import Point, Polygon

# Domain bounds for Indian Ocean Operational Region
MIN_LAT, MAX_LAT = 5.0, 25.0
MIN_LON, MAX_LON = 65.0, 95.0

# Known EEZ / Restricted Polygons / Bounding Boxes (e.g. Naval Exercise / Protected Sanctuary)
RESTRICTED_BOUNDARIES = [
    # Polygon 1: Offshore Security Restriction Zone
    {
        "name": "Naval Defense Restricted Zone", 
        "polygon": Polygon([
            (84.80, 9.90),
            (85.20, 9.90),
            (85.20, 10.10),
            (84.80, 10.10),
            (84.80, 9.90)
        ])
    },
    # Polygon 2: Marine Biosphere Core Sanctuary
    {
        "name": "Gulf of Mannar Marine Sanctuary", 
        "polygon": Polygon([
            (78.80, 8.80),
            (79.30, 8.80),
            (79.30, 9.20),
            (78.80, 9.20),
            (78.80, 8.80)
        ])
    }
]

def is_in_domain(lat: float, lon: float) -> bool:
    """
    Checks if coordinates fall within the supported Indian Ocean domain (5°N–25°N, 65°E–95°E).
    """
    return MIN_LAT <= lat <= MAX_LAT and MIN_LON <= lon <= MAX_LON


def is_eez_restricted(lat: float, lon: float) -> Tuple[bool, str]:
    """
    Checks if coordinates fall inside any restricted EEZ or security zone.
    Returns (is_restricted, zone_name).
    """
    point = Point(lon, lat)
    for zone in RESTRICTED_BOUNDARIES:
        if zone["polygon"].contains(point):
            return True, zone["name"]
    return False, "AUTHORIZED_FISHING_ZONE"
