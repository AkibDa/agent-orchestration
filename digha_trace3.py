import sys
import os
import asyncio

os.environ["ORCA_PFZ_SOURCE"] = "live"

from conversation.parser import ORCAParser
from orchestrator.orchestrator import ORCAOrchestrator
from conversation.response import generate_response

async def main():
    parser = ORCAParser(use_mlx=False)
    # mock parser
    plan = parser.parse("Where is the nearest PFZ today in Digha?", context={})
    
    # We will override the parsed plan manually to ensure speed and bypass mlx load
    plan.intent = "FIND_PFZ"
    plan.operation = "NEAREST_PFZ_SEARCH"
    plan.location.name = "Digha"
    plan.location.latitude = 21.6266
    plan.location.longitude = 87.5085
    plan.language = "en"
    
    orch = ORCAOrchestrator()
    res = orch.execute(plan, {})
    
    text, seg, timing = generate_response(plan, res)
    print("--- RESPONSE ---")
    print(text)

asyncio.run(main())
