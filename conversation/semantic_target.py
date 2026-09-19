import re
from enum import Enum
from typing import Optional, Dict, Any, Tuple, List
from pydantic import BaseModel
from schemas.contracts import GeoLocation, SpatialConstraint

class DistanceMode(str, Enum):
    EXACT = "EXACT"
    APPROXIMATE = "APPROXIMATE"
    MAXIMUM = "MAXIMUM"
    MINIMUM = "MINIMUM"

class TargetType(str, Enum):
    USER_SPECIFIED_LOCATION = "USER_SPECIFIED_LOCATION"
    NEAREST_FISHING_ZONE = "NEAREST_FISHING_ZONE"
    FISHING_AREA = "FISHING_AREA"
    PFZ = "PFZ"
    OFFSHORE_DISTANCE = "OFFSHORE_DISTANCE"
    NEARBY_FISHING_AREA = "NEARBY_FISHING_AREA"

class SemanticTarget(BaseModel):
    target_type: TargetType
    reference_location: Optional[str] = None
    distance_km: Optional[float] = None
    distance_mode: Optional[DistanceMode] = None
    provenance: Optional[str] = None
    resolved: bool = False
    original_text: str

def parse_semantic_target(text: str, spatial_constraint: Optional[SpatialConstraint] = None) -> SemanticTarget:
    t_lower = text.lower()
    
    # 1. Detect target type
    is_fishing = any(kw in t_lower for kw in ["fishing", "fish", "pfz", "machli", "machhli"])
    is_offshore = "offshore" in t_lower
    is_nearest = "nearest" in t_lower or "closest" in t_lower
    is_nearby = "near" in t_lower or "nearby" in t_lower or "paas" in t_lower or "kache" in t_lower
    
    target_type = TargetType.FISHING_AREA
    if is_fishing:
        if is_nearest:
            target_type = TargetType.NEAREST_FISHING_ZONE
        elif is_nearby:
            target_type = TargetType.NEARBY_FISHING_AREA
        elif "pfz" in t_lower:
            target_type = TargetType.PFZ
        else:
            target_type = TargetType.FISHING_AREA
    elif is_offshore:
        target_type = TargetType.OFFSHORE_DISTANCE
    else:
        target_type = TargetType.FISHING_AREA
        
    # 2. Detect distance and mode
    distance_km = None
    distance_mode = DistanceMode.APPROXIMATE
    
    if spatial_constraint and spatial_constraint.distance_km is not None:
        distance_km = spatial_constraint.distance_km
        if "within" in t_lower or "under" in t_lower or "less than" in t_lower:
            distance_mode = DistanceMode.MAXIMUM
    else:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:km|kilometer|kilometre)s?", t_lower)
        if match:
            distance_km = float(match.group(1))
            if "within" in t_lower or "under" in t_lower or "less than" in t_lower:
                distance_mode = DistanceMode.MAXIMUM
            elif "about" in t_lower or "approximately" in t_lower or "around" in t_lower:
                distance_mode = DistanceMode.APPROXIMATE
            
    return SemanticTarget(
        target_type=target_type,
        distance_km=distance_km,
        distance_mode=distance_mode,
        original_text=text
    )

def resolve_semantic_target(target: SemanticTarget, ref_loc: GeoLocation) -> Tuple[Optional[GeoLocation], str, Optional[float], Optional[str]]:
    """
    Resolves a semantic target to a physical location.
    Returns: (resolved_location, provenance, actual_distance_km, error_message)
    """
    from agents.geospatial.grid import generate_candidate_grid_points
    try:
        from data_sources.incois import get_pfz_advisory
        from agents.geospatial.distance import haversine_distance
    except ImportError:
        get_pfz_advisory = None
        haversine_distance = None

    dist_req = target.distance_km
    mode = target.distance_mode
    is_fishing = target.target_type in (
        TargetType.FISHING_AREA, TargetType.PFZ, TargetType.NEAREST_FISHING_ZONE, TargetType.NEARBY_FISHING_AREA
    )

    if is_fishing and get_pfz_advisory:
        candidates = get_pfz_advisory(ref_loc.latitude, ref_loc.longitude, count=20)
        if candidates:
            best_cand = None
            best_diff = float("inf")
            min_dist = float("inf")
            
            for c in candidates:
                c_dist = haversine_distance(ref_loc.latitude, ref_loc.longitude, c.latitude, c.longitude)
                if c_dist < min_dist:
                    min_dist = c_dist
                
                if dist_req is not None:
                    if mode == DistanceMode.MAXIMUM:
                        if c_dist <= dist_req:
                            if c_dist < best_diff:
                                best_diff = c_dist
                                best_cand = c
                    else: # APPROXIMATE
                        diff = abs(c_dist - dist_req)
                        if diff < best_diff:
                            best_diff = diff
                            best_cand = c
                else: # nearest or nearby
                    if c_dist < best_diff:
                        best_diff = c_dist
                        best_cand = c
                        
            if not best_cand and dist_req is not None and mode == DistanceMode.MAXIMUM:
                err_msg = (f"No fishing zone was found within {dist_req} km of {ref_loc.name or 'the origin'}. "
                           f"The nearest available INCOIS PFZ is approximately {min_dist:.1f} km away.")
                return None, "", None, err_msg
                
            if best_cand:
                actual_dist = haversine_distance(ref_loc.latitude, ref_loc.longitude, best_cand.latitude, best_cand.longitude)
                loc = GeoLocation(latitude=best_cand.latitude, longitude=best_cand.longitude, name=target.original_text)
                return loc, "OFFICIAL_PFZ", actual_dist, None
                
    # Fallback or Non-Fishing Offshore target
    if target.target_type == TargetType.OFFSHORE_DISTANCE or not is_fishing:
        fallback_dist = dist_req if dist_req is not None else 50.0
        cands = generate_candidate_grid_points(ref_loc.latitude, ref_loc.longitude, radii_km=[fallback_dist])
        if len(cands) > 1:
            loc = GeoLocation(latitude=cands[1]["latitude"], longitude=cands[1]["longitude"], name=target.original_text)
            return loc, "GENERATED_OFFSHORE_TARGET", fallback_dist, None
            
    # Generic fishing fallback if PFZ failed
    fallback_dist = dist_req if dist_req is not None else 10.0
    cands = generate_candidate_grid_points(ref_loc.latitude, ref_loc.longitude, radii_km=[fallback_dist])
    if len(cands) > 1:
        loc = GeoLocation(latitude=cands[1]["latitude"], longitude=cands[1]["longitude"], name=target.original_text)
        return loc, "EXISTING_FISHING_CANDIDATE", fallback_dist, None

    return None, "", None, "Could not resolve semantic destination to a valid marine location."
