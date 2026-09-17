# agents/geospatial/graph.py

import math
from typing import List, Tuple
from .restrictions import is_in_domain, is_eez_restricted
try:
    from global_land_mask import globe
    HAS_LAND_MASK = True
except ImportError:
    HAS_LAND_MASK = False

class LocalNavGraph:
    """
    Dynamically generates a 2D geographic navigation graph between a start and end location.
    Nodes are geographic coordinates snapped to a grid of specified resolution.
    """
    def __init__(self, start_lat: float, start_lon: float, goal_lat: float, goal_lon: float, resolution_km: float = 5.0, padding_km: float = 50.0):
        self.resolution_km = resolution_km
        
        # Calculate bounding box with padding
        delta_lat_pad = padding_km / 111.0
        # rough approx for max cos(lat)
        max_lat = max(abs(start_lat), abs(goal_lat))
        delta_lon_pad = padding_km / (111.0 * math.cos(math.radians(max_lat))) if max_lat < 89 else padding_km / 111.0
        
        self.min_lat = min(start_lat, goal_lat) - delta_lat_pad
        self.max_lat = max(start_lat, goal_lat) + delta_lat_pad
        self.min_lon = min(start_lon, goal_lon) - delta_lon_pad
        self.max_lon = max(start_lon, goal_lon) + delta_lon_pad
        
        self.delta_lat = resolution_km / 111.0
        
    def _get_delta_lon(self, lat: float) -> float:
        return self.resolution_km / (111.0 * math.cos(math.radians(lat)))

    def get_neighbors(self, lat: float, lon: float) -> List[Tuple[float, float]]:
        """
        Returns valid neighboring (lat, lon) nodes in an 8-way grid.
        Filters out land, out-of-domain, and restricted zones.
        """
        neighbors = []
        delta_lon = self._get_delta_lon(lat)
        
        offsets = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]
        
        for dlat, dlon in offsets:
            n_lat = round(lat + dlat * self.delta_lat, 4)
            n_lon = round(lon + dlon * delta_lon, 4)
            
            # Check bounding box
            if not (self.min_lat <= n_lat <= self.max_lat and self.min_lon <= n_lon <= self.max_lon):
                continue
                
            # Check domain
            if not is_in_domain(n_lat, n_lon):
                continue
                
            # Check land
            if HAS_LAND_MASK and globe.is_land(n_lat, n_lon):
                continue
                
            # Check hard restrictions (geofences)
            is_restr, _ = is_eez_restricted(n_lat, n_lon)
            if is_restr:
                continue
                
            neighbors.append((n_lat, n_lon))
            
        return neighbors
