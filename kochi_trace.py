import os
os.environ["ORCA_PFZ_SOURCE"] = "live"

from data_sources.incois import fetch_live_incois_pfz
from agents.geospatial.distance import haversine_distance, bearing, compass_direction

print("=== RAW INCOIS FETCH ===")
lat, lon = 9.9312, 76.2673
sector = "SEC005" # Kerala
live_candidates = fetch_live_incois_pfz(sector)

print("Exact normalized PFZ object returned immediately after fetch_live_incois_pfz():")
if live_candidates:
    cand = live_candidates[0]
    cand_dict = cand.model_dump()
    # Serialize safely
    import json
    # Convert datetime objects to string
    for k, v in cand_dict.items():
        if hasattr(v, "isoformat"):
            cand_dict[k] = v.isoformat()
            
    print(json.dumps(cand_dict, indent=2))
    
    print("\n=== ORCA-DERIVED VALUES ===")
    dist_km = haversine_distance(lat, lon, cand.latitude, cand.longitude)
    bearing_deg = bearing(lat, lon, cand.latitude, cand.longitude)
    comp_dir = compass_direction(bearing_deg)
    
    print(f"query coordinate: {lat}, {lon}")
    print(f"PFZ coordinate: {cand.latitude}, {cand.longitude}")
    print(f"query->PFZ distance: {round(dist_km, 1)} km")
    print(f"query->PFZ bearing: {round(bearing_deg, 1)}°")
    print(f"query->PFZ compass direction: {comp_dir}")
else:
    print("No candidates found.")

