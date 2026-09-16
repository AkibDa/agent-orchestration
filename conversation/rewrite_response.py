import re
import uuid

def wrap(text, agents):
    return text, [{"id": f"seg_{uuid.uuid4().hex[:8]}", "text": text, "agents": agents}]

def rewrite():
    with open("Proto/conversation/response.py", "r") as f:
        content = f.read()

    # 1. Update respond signature
    content = content.replace(
        "def respond(\n  query: str,\n  plan: QueryPlan,\n  exec_result: dict,\n  conv_model,\n  deadline: float = None,\n  request_id: str = None\n) -> tuple[str, dict]:",
        "def respond(\n  query: str,\n  plan: QueryPlan,\n  exec_result: dict,\n  conv_model,\n  deadline: float = None,\n  request_id: str = None\n) -> tuple[str, list, dict]:"
    )

    # 2. Update get_geography_factual_response calls in respond
    content = content.replace(
        "return get_geography_factual_response(query, plan), timings",
        "text = get_geography_factual_response(query, plan)\n    import uuid\n    return text, [{\"id\": f\"seg_{uuid.uuid4().hex[:8]}\", \"text\": text, \"agents\": []}], timings"
    )
    
    # 3. Update get_deterministic_clarification calls
    content = content.replace(
        "return get_deterministic_clarification(plan), timings",
        "text = get_deterministic_clarification(plan)\n    import uuid\n    return text, [{\"id\": f\"seg_{uuid.uuid4().hex[:8]}\", \"text\": text, \"agents\": []}], timings"
    )

    # 4. Update COMPARE_FISHING_REGIONS
    content = content.replace(
        "return evidence.get(\"reason\", \"Comparison completed.\"), timings",
        "text = evidence.get(\"reason\", \"Comparison completed.\")\n    import uuid\n    return text, [{\"id\": f\"seg_{uuid.uuid4().hex[:8]}\", \"text\": text, \"agents\": [\"rules\"]}], timings"
    )
    
    # 5. Update generate_multilingual_response
    content = content.replace(
        "fast_text = generate_multilingual_response(rec_payload, language=plan.language, context=ctx)",
        "fast_text, segments = generate_multilingual_response(rec_payload, language=plan.language, context=ctx)"
    )
    content = content.replace(
        "return fast_text, timings",
        "return fast_text, segments, timings"
    )
    
    # 6. Update _get_deterministic_response
    content = content.replace(
        "deterministic_res = _get_deterministic_response(plan, evidence)",
        "deterministic_res = _get_deterministic_response(plan, evidence)\n  segments = []"
    )
    content = content.replace(
        "return validate_response_script(deterministic_res, plan.language), timings",
        "text = validate_response_script(deterministic_res, plan.language)\n    import uuid\n    segments = [{\"id\": f\"seg_{uuid.uuid4().hex[:8]}\", \"text\": text, \"agents\": [\"rules\"]}]\n    return text, segments, timings"
    )
    
    # 7. Update _fallback_text
    content = content.replace(
        "return _fallback_text(plan, evidence), timings",
        "text = _fallback_text(plan, evidence)\n    import uuid\n    return text, [{\"id\": f\"seg_{uuid.uuid4().hex[:8]}\", \"text\": text, \"agents\": []}], timings"
    )
    
    # 8. Update LLM synthesis
    # text = validate_response_script(text, plan.language), timings
    content = content.replace(
        "return validate_response_script(text, plan.language), timings",
        "text = validate_response_script(text, plan.language)\n  # Extract context agents\n  ctx_agents = [k for k in exec_result.get(\"context\", {}).keys()]\n  import uuid\n  segments = [{\"id\": f\"seg_{uuid.uuid4().hex[:8]}\", \"text\": text, \"agents\": ctx_agents}]\n  return text, segments, timings"
    )

    with open("Proto/conversation/response.py", "w") as f:
        f.write(content)

rewrite()
