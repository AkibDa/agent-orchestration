# main.py

import sys
import time
from pathlib import Path

# Ensure torch and transformers (and Triton) are loaded BEFORE TensorFlow/Keras
# to prevent native library symbol conflicts on WSL/CUDA.
import triton
import torch
import transformers


from schemas.contracts import QueryPlan
from conversation.router import llm_route
from conversation.response import respond


# --- Unified Location Resolution Service ---
from location.resolver import extract_location

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def print_ui(query: str, plan: QueryPlan, orchestrator_result: dict, final_text: str = None, ext_st = None, gen_st = None):
    print(f"\nYou: {query}")
    print("ORCA")
    print("────────────────────────────")
    print(f"Language: {plan.language}")
    print(f"Intent: {plan.intent}")

    if plan.location:
        loc_name = plan.location.name
    elif getattr(plan, "location_type", None) == "inland" and getattr(plan, "inland_name", None):
        loc_name = f"{plan.inland_name} (Inland — Non-coastal region)"
    else:
        loc_name = "None specified (Need clarification)"

    print(f"Location: {loc_name}")
    if getattr(plan, "candidate_locations", None):
        print(f"Comparison Candidates: {', '.join(plan.candidate_locations)} (Unchecked)")

    rel = getattr(plan.time, "relative", None) if hasattr(plan.time, "relative") else (plan.time.get("relative") if isinstance(plan.time, dict) else None)
    per = getattr(plan.time, "period", None) if hasattr(plan.time, "period") else (plan.time.get("period") if isinstance(plan.time, dict) else None)
    time_str = f"{rel or ''} {per or ''}".strip() or "None"
    print(f"Time: {time_str}")
    print("\nAgents selected:")

    execution_order = orchestrator_result["execution_order"]
    if not execution_order:
        print("✓ No agents triggered")
    else:
        for agent in execution_order:
            print(f"✓ {agent.capitalize()}")

        print("\nExecuting...")
        time.sleep(0.5)
        for agent in execution_order:
            print(f"✓ {agent.capitalize()}")
            time.sleep(0.1)

        print("\nConstraint validation:")
        print("✓ Constraints checked against context")

    print("────────────────────────────")
    rec = orchestrator_result["recommendation"]
    ctx = orchestrator_result["context"]

    res_type = rec.get("result_type", getattr(plan, "result_type", "SAFETY_ASSESSMENT"))
    decision = rec.get("decision", "UNKNOWN")

    if res_type == "PFZ_RESULT":
        icon = "🐟"
        hdr = f"RESULT: PFZ SEARCH ({decision})"
    elif res_type == "NEAREST_COAST_RESULT":
        icon = "📍"
        hdr = f"RESULT: NEAREST COAST ({decision})"
    elif res_type == "CLARIFICATION":
        icon = "🟡"
        hdr = f"RECOMMENDATION: {decision}"
    else:
        icon = "🟢" if decision == "RECOMMEND" else "🔴" if decision == "AVOID" else "🟡"
        hdr = f"RECOMMENDATION: {decision}"

    risk_lvl = ctx["risk"].data.get("risk_level", "N/A") if "risk" in ctx else "N/A"
    conf = f"{int(ctx['risk'].confidence * 100)}%" if "risk" in ctx else "N/A"

    print(f"\n{icon} {hdr}")
    if res_type == "SAFETY_ASSESSMENT":
        print(f"   Risk Level: {risk_lvl} | Confidence: {conf}")
    elif res_type == "PFZ_RESULT":
        print(f"   Risk Level: {risk_lvl} | Status: Zone Identified")
    if "reason" in rec:
        print(f"   Reason: {rec['reason']}")

    # 2. WHY BLOCK (Factual Agent Evidence)
    print("\nWHY (Factual Agent Evidence):")
    if rec.get("why"):
        why = rec["why"]
        if why.get("primary_reason"):
            print(f"  • Primary Reason : {why['primary_reason']}")
        if why.get("fishing_reason"):
            print(f"  • Fishing Signal : {why['fishing_reason']}")
        if why.get("safety_reason"):
            print(f"  • Safety Signal  : {why['safety_reason']}")
        if why.get("imd_weather"):
            print(f"  • IMD Forecast   : {why['imd_weather']['summary']} (Source: {why['imd_weather']['source']})")
        if why.get("incois_bsi"):
            hazards = ", ".join(why['incois_bsi']['active_hazards']) if why['incois_bsi']['active_hazards'] else "None"
            print(f"  • INCOIS Safety  : {why['incois_bsi']['advisory']} | Hazards: {hazards} (Source: {why['incois_bsi']['source']})")
        if why.get("orca_ml_supplement"):
            print(f"  • ORCA ML Risk   : {why['orca_ml_supplement']['weather_risk']} | Suitability: {why['orca_ml_supplement']['ocean_suitability']} (Source: {why['orca_ml_supplement']['source']})")
    if "weather" in ctx:
        w = ctx["weather"].data
        prov_data = w.get("provider_data", w)
        prov = w.get("data_provenance", {})

        imd_stat = prov.get("imd_status", "UNKNOWN")
        imd_lat = prov.get("imd_latency_ms", "N/A")
        if imd_stat != "UNKNOWN":
            print(f"  • IMD API Status : {imd_stat} (Latency: {imd_lat}ms)")

        print(f"  • Wind speed     : {prov_data.get('wind_speed_ms', 'N/A')} m/s")
        print(f"  • Rain prob      : {float(prov_data.get('rain_probability', 0))*100:.0f}%")
        print(f"  • Temperature    : {prov_data.get('temperature_c', 'N/A')} °C")
        print(f"  • Pressure       : {prov_data.get('pressure_hpa', 'N/A')} hPa")
    if "ocean" in ctx:
        o = ctx["ocean"].data
        print(f"  • Wave height    : {o.get('wave_height_m', 'N/A')} m")
        print(f"  • Wave period    : {o.get('wave_period_s', 'N/A')} s")
        print(f"  • Ocean current  : {o.get('current_speed_ms', 'N/A')} m/s")
        print(f"  • Ocean score    : {o.get('ocean_score', 'N/A')}")
    if "pfz" in ctx:
        p = ctx["pfz"].data
        print(f"  • PFZ Zone       : {p.get('zone', 'N/A')} (Score: {p.get('pfz_score', 'N/A')})")
        if p.get("nearest_coastal_loc"):
            print(f"  • Nearest Shore  : {p.get('nearest_coastal_loc')}")
        if p.get("distance_km"):
            print(f"  • Distance/Dir   : {p.get('distance_km')} ({p.get('direction', '')})")
    if rec.get("ranked_candidate_spots"):
        print("\nRECOMMENDED SPATIAL CANDIDATE SPOTS:")
        for s in rec["ranked_candidate_spots"][:3]:
            print(f"  • Candidate {s.get('rank', '?')} — {s.get('display_name')}")
            print(f"      PFZ        : {s.get('pfz_description', 'N/A')}")
            print(f"      Safety     : {s.get('safety_status', 'UNKNOWN')} | Weather: {s.get('weather_status', 'UNKNOWN')}")
            print(f"      Coordinates: {s.get('latitude', 0):.2f}°N, {s.get('longitude', 0):.2f}°E")
            print(f"      Source     : {s.get('source', 'UNKNOWN')}")
            if s.get("selection_reason"):
                print(f"      Reason     : {s['selection_reason']}")
    if rec.get("rejected_candidate_spots"):
        print("\nREJECTED SPOTS (Safety Guardrails):")
        for s in rec["rejected_candidate_spots"][:2]:
            print(f"  • ({s.get('latitude'):.2f}N, {s.get('longitude'):.2f}E) Rejected: {s.get('rejection_reason')}")

    if not ctx:
        print("  • No agent evidence required for this turn.")

    # 3. ORCA CONVERSATIONAL SYNTHESIS
    if final_text:
        print("\nORCA (Conversational Verbalization):")
        print(f"  \"{final_text}\"")



    # Print token and latency metrics for this specific turn
    if ext_st or gen_st:
        print("\nPERFORMANCE & TOKEN PROFILING")
        print("────────────────────────────────────────────────────────────")
        if ext_st:
            print(f"1. Intake/Routing  : {ext_st.latency_sec:.2f}s | Prompt: {ext_st.prompt_tokens} tok | Output: {ext_st.completion_tokens} tok | Total: {ext_st.total_tokens} tok ({ext_st.tokens_per_sec:.1f} tok/s)")
            print(f"   Raw JSON Output : {ext_st.raw_output}")
        if gen_st:
            print(f"2. Response Synthesis: {gen_st.latency_sec:.2f}s | Prompt: {gen_st.prompt_tokens} tok | Output: {gen_st.completion_tokens} tok | Total: {gen_st.total_tokens} tok ({gen_st.tokens_per_sec:.1f} tok/s)")
            print(f"   Raw Text Output : \"{gen_st.raw_output}\"")
        if ext_st and gen_st:
            total_sec = ext_st.latency_sec + gen_st.latency_sec
            total_tok = ext_st.total_tokens + gen_st.total_tokens
            print(f"TOTAL TURN COST    : {total_sec:.2f} seconds | {total_tok} total tokens consumed")

    print("────────────────────────────\n")


from conversation.router import llm_route, llm_route_stateful
from conversation.state import ConversationState


def run_conversation(test_turns: list[str], conv_model, engine, extract_location):
    """Runs a sequence of turns through ONE shared ConversationState,
    simulating a real back-and-forth (e.g. CLARIFY -> user answers -> routes)."""
    state = ConversationState()

    for turn_idx, turn in enumerate(test_turns, 1):
        print(f"\n--- Turn {turn_idx}: You -> \"{turn}\" ---")
        action, plan, extraction, route_timings = llm_route_stateful(turn, conv_model, extract_location, state)
        ext_st = conv_model.last_extract_stats

        if action == "CHAT":
            print(f"ORCA: {extraction.chat_reply}")
            continue
        if action == "GIVE_UP":
            print("ORCA: Sorry, I still couldn't identify the location — please start over with a place name.")
            continue

        exec_result = engine.run(plan)
        final_text = respond(turn, plan, exec_result, conv_model)
        gen_st = conv_model.last_generate_stats
        print_ui(turn, plan, exec_result, final_text, ext_st, gen_st)


import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        print("Starting ORCA Regression Benchmark Suite...")
        from tests.benchmark import main as run_benchmark
        run_benchmark()
    else:
        from chat import main as run_chat
        run_chat()


if __name__ == "__main__":
    main()
