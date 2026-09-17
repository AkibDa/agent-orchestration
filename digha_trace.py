import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

os.environ["ORCA_PFZ_SOURCE"] = "live"

from schemas.contracts import QueryPlan, GeoLocation

# Create a mock plan
plan = QueryPlan(
    query="Where is the nearest PFZ today in Digha?",
    intent="FIND_PFZ",
    location=GeoLocation(name="Digha", latitude=21.6266, longitude=87.5085),
    language="en"
)

# Run PFZ agent directly
from agents.pfz.agent import PFZAgent
pfz_agent = PFZAgent()

context = {}
result = pfz_agent.run(plan, context)

print("--- PFZ Agent Result ---")
if result.data and 'candidates' in result.data:
    for i, c in enumerate(result.data['candidates']):
        print(f"Candidate {i}: {c.model_dump_json(indent=2)}")
else:
    print(result.data)

