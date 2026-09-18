import asyncio
from conversation.model import get_conversation_model
from conversation.router import llm_route_stateful
from location.resolver import extract_location
from conversation.state import ConversationState

async def main():
    conv_model = get_conversation_model()
    state = ConversationState()
    state.location_role = "REFERENCE"
    state.location_text = "Kochi"
    state.reference_location_text = "Kochi"
    state.target_location_text = None
    state.intent = "safe_route"
    
    query = "to the fishing area"
    action_str, plan, result, timings = llm_route_stateful(query, conv_model, extract_location, state)
    
    print(f"Action: {action_str}")
    if plan:
        print(f"Operation: {plan.operation}")
        print(f"Intent: {plan.intent}")

if __name__ == "__main__":
    asyncio.run(main())
