import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
import os
os.environ["ORCA_PFZ_SOURCE"] = "live"
from data_sources.incois import get_pfz_advisory

# Test locations
locations = [
    ("Kochi, Kerala", 9.9312, 76.2673),
    ("Digha, Odisha/WB border", 21.6266, 87.5085),
    ("Haldia, West Bengal", 22.0257, 88.0583)
]

print("=== Phase 5 Validation ===")
for name, lat, lon in locations:
    print(f"\nQuerying near: {name} (Lat: {lat}, Lon: {lon})")
    candidates = get_pfz_advisory(lat, lon, count=3)
    if not candidates:
        print("  -> No PFZ candidates found.")
    else:
        for i, c in enumerate(candidates):
            print(f"  Candidate {i+1}:")
            print(f"    Raw PFZ coordinate (from INCOIS text): {c.incois_distance_km_range}km {c.incois_direction} of somewhere")
            print(f"    Normalized coordinate: {c.latitude}, {c.longitude}")
            print(f"    Distance: {c.distance_from_landmark} km (ORCA calc) vs {c.incois_distance_km_range} km (INCOIS raw)")
            print(f"    Direction: {c.bearing_from_landmark} (ORCA calc) vs {c.incois_direction} (INCOIS raw)")
            print(f"    Depth: {c.depth} m (Normalized) vs {c.incois_depth_m_range} m (INCOIS raw)")
            print(f"    Source: {c.source}")
