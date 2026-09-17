import time
from agents.geospatial import astar
from agents.geospatial.restrictions import RESTRICTED_BOUNDARIES
from shapely.geometry import Polygon

def synthetic_weather_predict(lat, lon, weather_context=None, cyclone_context=None):
    mode = weather_context.get("mode", "NONE") if weather_context else "NONE"
    
    res = "NORMAL"
    if mode == "HIGH_WAVE":
        if 10.0 <= lat <= 11.0 and 70.0 <= lon <= 71.0:
            res = "DANGEROUS"
            
    if mode == "HIGH_WIND":
        if 12.0 <= lat <= 13.0 and 72.0 <= lon <= 75.0:
            res = "CAUTION"

    if mode == "COMBINED":
        if 15.0 <= lat <= 16.0 and 70.0 <= lon <= 74.0:
            res = "CAUTION"
        if 16.0 <= lat <= 17.0 and 71.0 <= lon <= 72.0:
            res = "DANGEROUS"
            
    print(f"[PREDICT] lat={lat} lon={lon} mode={mode} result={res}")
    return {"risk_level": res}

import agents.geospatial.astar
agents.geospatial.astar.weather_predict = synthetic_weather_predict

def run_test(name, start, goal, mode, resolution_km=10.0):
    print(f"\n--- Test: {name} ---")
    
    env_data = {"weather_data": {"mode": mode}}
    res = agents.geospatial.astar.plan_safe_route_astar(start[0], start[1], goal[0], goal[1], env_data, resolution_km=resolution_km)
    
    print(f"Status: {res['route_status']}")
    print(f"Distance: {res['total_distance_km']} km")
    print(f"Hazards Encountered: {res['hazards_encountered']}")
    print(f"Nodes eval: {res['nodes_evaluated']}")
    return res

run_test("Test 4: Combined Hazards (Penalties Enabled)", (14.5, 71.5), (17.5, 71.5), "COMBINED")

