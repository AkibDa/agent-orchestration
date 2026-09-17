from agents.geospatial.coastline import distance_to_coast
from agents.geospatial.agent import GeospatialAgent
from schemas.contracts import QueryPlan, GeoLocation

lat, lon = 9.93, 76.26 # Kochi
dist = distance_to_coast(lat, lon)
print(f"Distance to coast from Kochi: {dist} km")

lat, lon = 28.61, 77.20 # Delhi
dist = distance_to_coast(lat, lon)
print(f"Distance to coast from Delhi: {dist} km")

agent = GeospatialAgent()
plan = QueryPlan(
    query="nearest coast from delhi",
    intent="marine_geography",
    operation="DISTANCE_TO_COAST",
    language="en",
    location=GeoLocation(latitude=28.61, longitude=77.20, name="Delhi")
)
res = agent.run(plan, {})
print("Agent Result:")
print(res.data.get('distance_to_coast_summary'))

