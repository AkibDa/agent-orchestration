from conversation.router import llm_route_stateful
from location.resolver import extract_location
from conversation.state import ConversationState
from schemas.extraction import ExtractionResult, Language, Intent, Action, ActionType, LocationItem, LocationRole

state = ConversationState()
state.intent = "safe_route"
state.reference_location_text = "Kochi"
state.clarify_rounds = 1
state.language = "en"

class MockModel:
    def reset_turn_stats(self): pass
    def extract(self, prompt, query):
        return ExtractionResult(
            language=Language.en,
            intent=Intent.safe_route,
            action=Action.ORCA_QUERY,
            action_type=ActionType.ASSESS,
            locations=[LocationItem(text="fishing area", role=LocationRole.TARGET)]
        )

model = MockModel()
query = "to the fishing area"
action, plan, result, timings = llm_route_stateful(query, model, extract_location, state)
print(f"Final: Action={action}, intent={plan.intent}, op={plan.operation}, tgt={plan.target_location}")
