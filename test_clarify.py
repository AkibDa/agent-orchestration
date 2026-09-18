import asyncio
from conversation.model import get_conversation_model
from conversation.router import llm_route_stateful
from location.resolver import extract_location
from conversation.state import ConversationState

async def main():
    conv_model = get_conversation_model()
    # Mocking previous context where system asked for destination
    state = ConversationState()
    state.location_role = "REFERENCE"
    state.location_text = "Kochi"
    state.reference_location_text = "Kochi"
    state.target_location_text = None
    state.intent = "safe_route"
    
    # User's follow-up reply
    query = "fishing area"
    action_str, plan, result, timings = llm_route_stateful(query, conv_model, extract_location, state)
    
    print(f"Action: {action_str}")
    if plan:
        print(f"Operation: {plan.operation}")
        print(f"Intent: {plan.intent}")
        print(f"Target: {plan.target_location.name if plan.target_location else None}")
        print(f"Ref: {plan.reference_location.name if plan.reference_location else None}")
        print(f"Clarification: {plan.clarification_reason}")

if __name__ == "__main__":
    asyncio.run(main())
