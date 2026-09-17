# agents/geospatial/coastline.py

import os
import json
import logging
from typing import Tuple, Optional
from shapely.geometry import Point, shape, MultiLineString, LineString
from shapely.ops import nearest_points
from shapely.strtree import STRtree

from .distance import haversine_distance, bearing

logger = logging.getLogger(__name__)

_COASTLINE_TREE: Optional[STRtree] = None
_COASTLINE_GEOMS = []

def _load_coastline():
    global _COASTLINE_TREE, _COASTLINE_GEOMS
    if _COASTLINE_TREE is not None:
        return

    geojson_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "coastline_clipped.geojson")
    if not os.path.exists(geojson_path):
        logger.warning(f"Coastline GeoJSON not found at {geojson_path}")
        return

    try:
        with open(geojson_path, "r") as f:
            data = json.load(f)
        
        for feature in data.get("features", []):
            geom = shape(feature["geometry"])
            if isinstance(geom, (LineString, MultiLineString)):
                _COASTLINE_GEOMS.append(geom)
            
        if _COASTLINE_GEOMS:
            _COASTLINE_TREE = STRtree(_COASTLINE_GEOMS)
            
    except Exception as e:
        logger.error(f"Error loading coastline geometry: {e}")


def get_nearest_coastline_point(lat: float, lon: float) -> Optional[Tuple[float, float]]:
    """
    Returns the nearest (lat, lon) on the actual coastline geometry.
    """
    _load_coastline()
    if not _COASTLINE_TREE:
        return None
        
    origin = Point(lon, lat)
    
    nearest_idx = _COASTLINE_TREE.nearest(origin)
    if nearest_idx is None:
        return None
        
    nearest_geom = _COASTLINE_GEOMS[nearest_idx]
    
    # Get the exact nearest point on that geometry
    pt_origin, pt_nearest = nearest_points(origin, nearest_geom)
    return pt_nearest.y, pt_nearest.x


def distance_to_coast(lat: float, lon: float) -> Optional[float]:
    """
    Calculates geodesic distance (in km) to the nearest coastline point.
    """
    nearest_pt = get_nearest_coastline_point(lat, lon)
    if nearest_pt is None:
        return None
        
    return haversine_distance(lat, lon, nearest_pt[0], nearest_pt[1])


def bearing_to_coast(lat: float, lon: float) -> Optional[float]:
    """
    Calculates bearing (in degrees) to the nearest coastline point.
    """
    nearest_pt = get_nearest_coastline_point(lat, lon)
    if nearest_pt is None:
        return None
        
    return bearing(lat, lon, nearest_pt[0], nearest_pt[1])
