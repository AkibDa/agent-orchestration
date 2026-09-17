import sys
import os
import asyncio

os.environ["ORCA_PFZ_SOURCE"] = "live"

from orchestrator.orchestrator import ORCAOrchestrator
from schemas.contracts import QueryPlan, GeoLocation
from conversation.response import generate_response

async def main():
    plan = QueryPlan(
        query="Where is the nearest PFZ today in Digha?",
        intent="FIND_PFZ",
        operation="NEAREST_PFZ_SEARCH",
        location=GeoLocation(name="Digha", latitude=21.6266, longitude=87.5085),
        target_location=GeoLocation(name="Digha", latitude=21.6266, longitude=87.5085),
        language="en",
        result_type="PFZ_RESULT"
    )
    
    orch = ORCAOrchestrator()
    res = orch.execute(plan, {})
    
    text, seg, timing = generate_response(plan, res)
    print("--- RESPONSE ---")
    print(text)

asyncio.run(main())
