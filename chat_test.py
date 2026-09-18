from agents.registry import AgentRegistry
from orchestrator.engine import OrcaOrchestrator
from conversation.state import ConversationState
from conversation.model import get_conversation_model

registry = AgentRegistry()
registry.discover_and_register("agents")

engine = OrcaOrchestrator(registry)

state = ConversationState()
query = "I want to go fishing near Kochi tomorrow morning. Which area should I choose considering weather, sea conditions, fishing productivity and safety?"

result = engine.process_query(query, state)
print("=== RESULT ===")
if "plan" in result:
    print("Plan Operation:", result["plan"].operation)
    print("Plan Location:", result["plan"].location.name if result["plan"].location else None)
print("Response:", result.get("response", ""))
