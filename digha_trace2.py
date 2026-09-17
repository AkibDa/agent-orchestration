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
    language="en",
    result_type="PFZ_RESULT"
)

# Run PFZ agent directly
from agents.pfz.agent import PFZAgent
pfz_agent = PFZAgent()

context = {}
result = pfz_agent.run(plan, context)

print("--- Response Generation ---")
from conversation.response_generator import generate_multilingual_response

# Simulate what the orchestrator does
recommendation_result = result.data.copy()
recommendation_result["result_type"] = plan.result_type
if result.location:
    recommendation_result["location"] = result.location.model_dump()

text, segments = generate_multilingual_response(recommendation_result, language=plan.language, context={"pfz": result})
print(text)
