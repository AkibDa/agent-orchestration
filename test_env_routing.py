import time
from agents.geospatial import astar
from agents.geospatial.restrictions import RESTRICTED_BOUNDARIES
from shapely.geometry import Polygon
import agents.weather.model

def synthetic_weather_predict(lat, lon, timestamp=None, weather_context=None, cyclone_context=None):
    mode = weather_context.get("mode", "NONE") if weather_context else "NONE"
    
    res = "NORMAL"
    if mode == "HIGH_WAVE":
        if 10.0 <= lat <= 11.0 and 70.3 <= lon <= 70.7:
            res = "DANGEROUS"
            
    if mode == "HIGH_WIND":
        if 12.0 <= lat <= 13.0 and 73.2 <= lon <= 73.8:
            res = "CAUTION"

    if mode == "COMBINED":
        if 15.0 <= lat <= 16.0 and 71.2 <= lon <= 71.8:
            res = "CAUTION"
        if 16.0 <= lat <= 17.0 and 71.2 <= lon <= 71.8:
            res = "DANGEROUS"
            
    return {"risk_level": res}

agents.weather.model.predict = synthetic_weather_predict

def run_test(name, start, goal, mode, resolution_km=10.0):
    print(f"\n--- Test: {name} ---")
    
    env_data = {"weather_data": {"mode": mode}}
    res = astar.plan_safe_route_astar(start[0], start[1], goal[0], goal[1], env_data, resolution_km=resolution_km)
    
    print(f"Status: {res['route_status']}")
    print(f"Distance: {res['total_distance_km']} km")
    print(f"Hazards Encountered: {res['hazards_encountered']}")
    print(f"Env Cost Penalty: {res['environmental_cost']}")
    print(f"Nodes eval: {res['nodes_evaluated']}")
    print(f"Time: {res['execution_time_ms']} ms")
    return res

print("Starting Environmental Routing Tests...\n")

run_test("Test 1: High Wave Corridor (Penalties Enabled)", (9.5, 70.5), (11.5, 70.5), "HIGH_WAVE")
run_test("Test 2: High Wind Corridor (Penalties Enabled)", (11.5, 73.5), (13.5, 73.5), "HIGH_WIND")

original_boundaries = list(RESTRICTED_BOUNDARIES)
RESTRICTED_BOUNDARIES.append({
    "name": "Test Restricted Zone",
    "polygon": Polygon([
        (70.1, 9.8), (70.4, 9.8), (70.4, 11.2), (70.1, 11.2), (70.1, 9.8)
    ])
})
run_test("Test 3: Restricted Polygon Overlapping Safe Corridor", (9.5, 70.5), (11.5, 70.5), "HIGH_WAVE")
RESTRICTED_BOUNDARIES.pop()

RESTRICTED_BOUNDARIES.append({
    "name": "Test Combined Restricted Zone",
    "polygon": Polygon([
        (70.8, 15.5), (71.3, 15.5), (71.3, 16.5), (70.8, 16.5), (70.8, 15.5)
    ])
})
run_test("Test 4: Combined Hazards (Penalties Enabled)", (14.5, 71.5), (17.5, 71.5), "COMBINED")
run_test("Test 5: Baseline No Penalties (Combined Route)", (14.5, 71.5), (17.5, 71.5), "NONE")

RESTRICTED_BOUNDARIES.pop()

