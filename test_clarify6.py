import asyncio
from conversation.model import get_conversation_model
from conversation.router import llm_route_stateful
from location.resolver import extract_location
from conversation.state import ConversationState

async def main():
    class MockModel:
        def __init__(self, orig):
            self.orig = orig
            self.last_route_timings = {}
        def reset_turn_stats(self):
            pass
        def extract(self, sys_prompt, query):
            from schemas.extraction import ExtractionResult, Intent, Action, ActionType, LocationItem, LocationRole, Language
            return ExtractionResult(
                language=Language.en,
                intent=Intent.safe_route,
                action=Action.CLARIFY,
                action_type=ActionType.ROUTE,
                locations=[
                    LocationItem(text="Kochi", role=LocationRole.REFERENCE),
                    LocationItem(text="fishing area", role=LocationRole.TARGET)
                ],
                activity="fishing",
                time_relative="today"
            )
    
    conv_model = MockModel(get_conversation_model())
    state = ConversationState()
    state.location_role = "REFERENCE"
    state.location_text = "Kochi"
    state.reference_location_text = "Kochi"
    state.target_location_text = None
    state.intent = "safe_route"
    
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
