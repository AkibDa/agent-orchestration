import os
import json
from data_sources.incois import get_pfz_advisory
from agents.geospatial.grid import generate_candidate_grid_points

tests = {
    "Kochi": (9.93, 76.26),
    "Digha": (21.62, 87.50),
    "Haldia": (22.06, 88.06),
    "Inland (Delhi)": (28.61, 77.20),
    "Offshore (Indian Ocean)": (5.0, 70.0)
}

print("=== PFZ ADVISORY TESTS ===")
for name, (lat, lon) in tests.items():
    print(f"\nLocation: {name} ({lat}, {lon})")
    cands = get_pfz_advisory(lat, lon, count=3)
    if not cands:
        print("  Result: No PFZ nearby or all expired.")
    else:
        for c in cands:
            print(f"  PFZ centroid: ({c.latitude}, {c.longitude})")
            print(f"  Distance: {c.distance_from_landmark} km")
            print(f"  Bearing: {c.bearing_from_landmark}")
            print(f"  Probability: {c.confidence}")
            print(f"  Source: {c.source}")
            print(f"  Validity: {c.validity_window}")

print("\n=== GRID GENERATION (LAND MASK) TESTS ===")
for name, (lat, lon) in tests.items():
    print(f"\nLocation: {name} ({lat}, {lon})")
    grid = generate_candidate_grid_points(lat, lon, radii_km=[10.0], bearings_deg=[0.0, 90.0, 180.0, 270.0])
    print(f"  Generated {len(grid)} candidate points (including origin).")
    for g in grid:
        print(f"  - Point {g['id']}: ({g['latitude']}, {g['longitude']}) - {g['compass_direction']}")
