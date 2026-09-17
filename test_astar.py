import time
from agents.geospatial.astar import plan_safe_route_astar

def run_test(name, start, goal, env_data={}):
    print(f"\n--- Test: {name} ---")
    print(f"Start: {start}, Goal: {goal}")
    res = plan_safe_route_astar(start[0], start[1], goal[0], goal[1], env_data, resolution_km=10.0)
    print(f"Status: {res['route_status']}")
    print(f"Distance: {res['total_distance_km']} km")
    print(f"ETA: {res['estimated_travel_time_h']} h")
    print(f"Hazards: {res['hazards_encountered']}")
    print(f"Nodes eval: {res['nodes_evaluated']}")
    print(f"Time: {res['execution_time_ms']} ms")
    if res['route_coordinates']:
        print(f"Path length (nodes): {len(res['route_coordinates'])}")
        
print("Starting benchmarks...\n")

# 1. coastal -> offshore (Kochi -> Arabian Sea)
run_test("Coastal to Offshore", (9.93, 76.26), (9.0, 75.0))

# 2. offshore -> offshore
run_test("Offshore to Offshore", (8.0, 75.0), (7.0, 74.0))

# 3. route crossing land (Arabian Sea to Bay of Bengal across South India)
run_test("Route crossing land", (10.0, 75.0), (10.0, 80.0))

# 4. route crossing restricted polygon (Naval Defense Zone is [84.80, 9.90] to [85.20, 10.10])
run_test("Route crossing Restricted Polygon", (10.0, 84.0), (10.0, 86.0))

# 5. start already in restricted area
run_test("Start in Restricted", (10.0, 85.0), (10.0, 86.0))

