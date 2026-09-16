# Proto/location/marine_identity.py

from typing import Optional
from dataclasses import dataclass
from .gazetteer import GAZETTEER
from .location_metadata import LOCATION_METADATA
from agents.geospatial.distance import haversine_distance, bearing, compass_direction

@dataclass
class MarineIdentity:
    display_name: str
    reference_landmark: Optional[str]
    distance_from_landmark_km: Optional[float]
    bearing_from_landmark_deg: Optional[float]
    compass_direction: Optional[str]
    region: Optional[str]
    latitude: float
    longitude: float


class MarineIdentityResolver:
    @staticmethod
    def resolve(
        lat: float, 
        lon: float, 
        official_landmark: Optional[str] = None, 
        official_bearing: Optional[str] = None, 
        official_distance: Optional[float] = None,
        region: Optional[str] = None
    ) -> MarineIdentity:
        """
        Resolves marine coordinates into a human-readable display identity.
        Never invents new place names.
        """
        # 1. If we have an official INCOIS PFZ landmark, use it directly.
        if official_landmark and official_bearing and official_distance is not None:
            display = f"{official_distance:.1f} km {official_bearing} of {official_landmark}"
            return MarineIdentity(
                display_name=display,
                reference_landmark=official_landmark,
                distance_from_landmark_km=official_distance,
                bearing_from_landmark_deg=None,
                compass_direction=official_bearing,
                region=region,
                latitude=lat,
                longitude=lon
            )

        # 2. Find the nearest verified coastal landmark from GAZETTEER
        nearest_coast_name = None
        min_dist = float("inf")
        nearest_lat, nearest_lon = None, None
        nearest_region = region

        for g_key, (g_lat, g_lon, g_name) in GAZETTEER.items():
            g_meta = LOCATION_METADATA.get(g_key, {})
            # Only use coastal points or fishing ports for reference
            if g_meta.get("coastal_access", True) and g_key not in ["kolkata", "calcutta", "bengaluru", "pune", "ranchi"]:
                dist = haversine_distance(lat, lon, g_lat, g_lon)
                if dist < min_dist:
                    min_dist = dist
                    nearest_coast_name = g_name
                    nearest_lat = g_lat
                    nearest_lon = g_lon
                    nearest_region = g_meta.get("region", region)

        if nearest_coast_name and min_dist < 200.0:
            brng = bearing(nearest_lat, nearest_lon, lat, lon)
            comp_dir = compass_direction(brng)
            
            # If the candidate is virtually AT the landmark
            if min_dist < 2.0:
                display = f"Near {nearest_coast_name} Coast"
            else:
                display = f"{min_dist:.1f} km {comp_dir} of {nearest_coast_name}"
                
            return MarineIdentity(
                display_name=display,
                reference_landmark=nearest_coast_name,
                distance_from_landmark_km=round(min_dist, 1),
                bearing_from_landmark_deg=round(brng, 1),
                compass_direction=comp_dir,
                region=nearest_region,
                latitude=lat,
                longitude=lon
            )

        # 3. Region fallback if no coastal landmark within 200km
        fallback_region = region or "Marine Candidate"
        display = f"{fallback_region} — {lat:.2f}°N, {lon:.2f}°E"
        
        return MarineIdentity(
            display_name=display,
            reference_landmark=None,
            distance_from_landmark_km=None,
            bearing_from_landmark_deg=None,
            compass_direction=None,
            region=fallback_region,
            latitude=lat,
            longitude=lon
        )
