# Proto/chat.py
"""Interactive ORCA chat — the actual product experience with Stage Latency Profiling.
Run with: python Proto/chat.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import sys
from pathlib import Path

PROTO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROTO_DIR.parent
if str(PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(PROTO_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure torch and transformers (and Triton) are loaded BEFORE TensorFlow/Keras
# to prevent native library symbol conflicts on WSL/CUDA.
# Skip entirely on macOS — importing torch loads native OpenMP libs that
# cannot be unloaded and cause a segfault when XGBoost loads its own copy.
import platform
if platform.system() != "Darwin":
    try:
        import torch
        if torch.cuda.is_available():
            try:
                import triton  # type: ignore
            except ImportError:
                pass
            import transformers
    except ImportError:
        pass

from orchestrator.engine import OrcaOrchestrator
from agents.weather.agent import WeatherAgent
from agents.ocean.agent import OceanAgent
from agents.ocean_state.agent import OceanStateAgent
from agents.tide.agent import TideAgent
from agents.geospatial.agent import GeospatialAgent
from agents.risk.agent import RiskAgent
from agents.pfz.agent import PFZAgent
from agents.productivity.agent import FishProductivityAgent
from agents.rules.safety_agent import SafetyRuleAgent
from agents.rules.recommendation_agent import RecommendationAgent
from agents.marine_safety.agent import MarineSafetyAgent

from agents.ocean.model import get_ocean_suitability_model
from data_sources.incois import load_ocean_datasets
from agents.pfz.model import get_pfz_model
from agents.weather.model import get_weather_model
from agents.productivity.model import get_productivity_model
from agents.risk.model import get_marine_risk_model

from conversation.model import get_conversation_model
from conversation.router import llm_route_stateful
from conversation.response import respond
from conversation.response_generator import generate_multilingual_response
from conversation.state import ConversationState
from location.resolver import extract_location


def build_engine() -> OrcaOrchestrator:
    registry = {
        "weather": WeatherAgent(),
        "ocean": OceanAgent(),
        "ocean_state": OceanStateAgent(),
        "tide": TideAgent(),
        "pfz": PFZAgent(),
        "geospatial": GeospatialAgent(),
        "risk": RiskAgent(),
        "productivity": FishProductivityAgent(),
        "marine_safety": MarineSafetyAgent(),
        "safety_rules": SafetyRuleAgent(),
        "recommendation": RecommendationAgent(),
    }
    return OrcaOrchestrator(registry=registry)


def print_banner():
    print("=" * 65)
    print("  🌊 ORCA — Autonomous Marine Safety & Fishing Assistant")
    print("  Supports questions in Bengalish, Bengali, English, & Hinglish")
    print("  Commands: /reset (clear session) | /debug (toggle logs) | /quit")
    print("=" * 65)


def print_latency_breakdown(
    lang_detect_s: float,
    qwen_extract_s: float,
    location_res_s: float,
    planning_s: float,
    domain_agents_s: float,
    decision_engine_s: float,
    synthesis_s: float,
    total_s: float,
):
    print("\nLATENCY BREAKDOWN")
    print("────────────────────────────────────────")
    print(f"Language detection     {lang_detect_s:6.2f}s  ({lang_detect_s*1000.0:7.1f} ms)")
    print(f"Qwen extraction        {qwen_extract_s:6.2f}s  ({qwen_extract_s*1000.0:7.1f} ms)")
    print(f"Location resolution    {location_res_s:6.2f}s  ({location_res_s*1000.0:7.1f} ms)")
    print(f"Planning & State       {planning_s:6.2f}s  ({planning_s*1000.0:7.1f} ms)")
    print(f"Domain Agents          {domain_agents_s:6.2f}s  ({domain_agents_s*1000.0:7.1f} ms)")
    print(f"Risk / Decision engine {decision_engine_s:6.2f}s  ({decision_engine_s*1000.0:7.1f} ms)")
    print(f"Response synthesis     {synthesis_s:6.2f}s  ({synthesis_s*1000.0:7.1f} ms)")
    print("────────────────────────────────────────")
    print(f"TOTAL                  {total_s:6.2f}s  ({total_s*1000.0:7.1f} ms)")


def handle_turn(query: str, conv_model, engine, state: ConversationState, debug: bool):
    t0_start = time.perf_counter()

    # Stage 1: Router & Intake
    action, plan, extraction, route_timings = llm_route_stateful(query, conv_model, extract_location, state)

    ext_st = getattr(conv_model, 'last_extract_stats', None)
    if debug and ext_st:
        print(f"  [DEBUG] completion_tokens={ext_st.completion_tokens} prompt_tokens={ext_st.prompt_tokens} tok/s={ext_st.tokens_per_sec:.1f}")
        print(f"  [DEBUG] raw_output={ext_st.raw_output}")

    t_lang_s = route_timings.get("lang_detect_ms", 0.0) / 1000.0
    t_extract_s = route_timings.get("qwen_extract_ms", 0.0) / 1000.0
    t_loc_s = route_timings.get("location_res_ms", 0.0) / 1000.0
    t_plan_s = route_timings.get("planning_ms", 0.0) / 1000.0

    if debug:
        loc_str = plan.location.name if (plan and plan.location) else "None"
        ref_str = plan.reference_location.name if (plan and plan.reference_location) else "None"
        tgt_str = plan.target_location.name if (plan and plan.target_location) else "None"
        print(f"\n  [DEBUG] Action={action} | Intent={plan.intent if plan else extraction.intent.value} | Location={loc_str} | Lang={plan.language if plan else 'N/A'}")
        print(f"  [DEBUG] RefLoc={ref_str} | TargetLoc={tgt_str} | Operation={plan.operation if plan else 'N/A'} | LocRole={plan.location_role.value if plan else 'N/A'}")
        qt = plan.time if plan else None
        qt_str = f"relative={qt.relative} offset={qt.offset_days} hour={qt.hour} min={qt.minute} period={qt.period}" if qt else "None"
        print(f"  [DEBUG] QueryTime={qt_str}")
        print(f"  [DEBUG] State Known={state.known_summary()}")

    if action == "CHAT":
        t0_synth = time.perf_counter()
        reply_text = extraction.chat_reply
        t_synth_s = time.perf_counter() - t0_synth
        t_total_s = time.perf_counter() - t0_start
        if debug:
            print_latency_breakdown(
                lang_detect_s=t_lang_s,
                qwen_extract_s=t_extract_s,
                location_res_s=t_loc_s,
                planning_s=t_plan_s,
                domain_agents_s=0.0,
                decision_engine_s=0.0,
                synthesis_s=t_synth_s,
                total_s=t_total_s,
            )
        if not reply_text:
            reply_text = "I encountered an internal error and could not process your query. Could you please rephrase or try again?"
        print(f"\nORCA ({t_total_s*1000.0:.1f}ms): {reply_text}")
        return

    if action == "NON_COASTAL_ERROR":
        t0_synth = time.perf_counter()
        NON_COASTAL_MESSAGES = {
            "en": "{location} is not a coastal location. Please provide a coastal location or fishing port.",
            "bn-Latn": "{location} ekta inland region jekhane kono sea access nei. Fishing-er jonno ekta coastal location ba port specify korun.",
            "bn": "{location} একটি উপকূলীয় স্থান নয়। অনুগ্রহ করে একটি উপকূলীয় স্থান বা মাছ ধরার বন্দর দিন।",
            "hi-Latn": "{location} ek inland area hai jahan direct ocean access nahi hai. Kripya koi coastal town ya port batayein.",
        }
        loc_name = plan.inland_name if plan and plan.inland_name else "This location"
        lang = plan.language if plan and plan.language in NON_COASTAL_MESSAGES else "en"
        reply_text = NON_COASTAL_MESSAGES[lang].format(location=loc_name)
        t_synth_s = time.perf_counter() - t0_synth
        t_total_s = time.perf_counter() - t0_start
        if debug:
            print_latency_breakdown(
                lang_detect_s=t_lang_s,
                qwen_extract_s=t_extract_s,
                location_res_s=t_loc_s,
                planning_s=t_plan_s,
                domain_agents_s=0.0,
                decision_engine_s=0.0,
                synthesis_s=t_synth_s,
                total_s=t_total_s,
            )
        print(f"\nORCA ({t_total_s*1000.0:.1f}ms): {reply_text}")
        return

    if action == "GIVE_UP":
        t0_synth = time.perf_counter()
        give_up_msg = "I couldn't identify the location from your query. Where are you planning to go fishing or navigate?"
        t_synth_s = time.perf_counter() - t0_synth
        t_total_s = time.perf_counter() - t0_start
        if debug:
            print_latency_breakdown(
                lang_detect_s=t_lang_s,
                qwen_extract_s=t_extract_s,
                location_res_s=t_loc_s,
                planning_s=t_plan_s,
                domain_agents_s=0.0,
                decision_engine_s=0.0,
                synthesis_s=t_synth_s,
                total_s=t_total_s,
            )
        print(f"\nORCA ({t_total_s*1000.0:.1f}ms): {give_up_msg}")
        state.clear()
        return

    if action == "CLARIFY":
        t0_synth = time.perf_counter()
        from conversation.router import build_clarification_context
        from conversation.response import generate_clarification_response

        ctx = build_clarification_context(plan, state)
        state.clarification_context = ctx
        clarification_msg = generate_clarification_response(ctx, conv_model)
        state.pending_clarification = clarification_msg
        state.previous_clarifications.append(clarification_msg)
        t_synth_s = time.perf_counter() - t0_synth
        t_total_s = time.perf_counter() - t0_start
        if debug:
            print_latency_breakdown(
                lang_detect_s=t_lang_s,
                qwen_extract_s=t_extract_s,
                location_res_s=t_loc_s,
                planning_s=t_plan_s,
                domain_agents_s=0.0,
                decision_engine_s=0.0,
                synthesis_s=t_synth_s,
                total_s=t_total_s,
            )
        print(f"\nORCA ({t_total_s*1000.0:.1f}ms): {clarification_msg}")
        return

    # Stage 2: ORCA Engine Execution
    TOTAL_BUDGET_S = 5.0
    deadline = t0_start + TOTAL_BUDGET_S
    exec_result = engine.run(plan, deadline=deadline)
    stage_timings = exec_result.get("stage_timings", {})
    agent_timings = exec_result.get("agent_timings", {})
    t_domain_s = stage_timings.get("domain_agents_ms", 0.0) / 1000.0
    t_decision_s = stage_timings.get("decision_engine_ms", 0.0) / 1000.0

    rec_output = exec_result.get("recommendation", {})
    action_code = rec_output.get("decision", "CLEAR_WEATHER_LOW_YIELD")
    ctx = exec_result.get("context", {})

    # Stage 3: Multilingual Response Generation
    t0_synth = time.perf_counter()
    lang_key = "en"
    if plan and plan.language:
        if plan.language in ["bn-Latn", "bn_en"]:
            lang_key = "bn_en"
        elif plan.language in ["bn"]:
            lang_key = "bn"
        elif plan.language in ["hi-Latn", "hi"]:
            lang_key = "hi-Latn"
        elif plan.language in ["hn"]:
            lang_key = "hn"

    rec_payload = rec_output.copy()
    if plan and plan.location:
        rec_payload["latitude"] = plan.location.latitude
        rec_payload["longitude"] = plan.location.longitude

    final_text = generate_multilingual_response(rec_payload, language=lang_key, context=ctx)

    # Translations are now handled entirely via localized templates and the localized why_dict.
    # We removed the LLM translation step here to strictly enforce < 5 seconds latency.

    t_synth_s = time.perf_counter() - t0_synth
    t_total_s = time.perf_counter() - t0_start

    # ═══════════════════════════════════════════════════════════════
    # DISPLAY: ORCA RECOMMENDATION (always shown)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  🌊 ORCA RECOMMENDATION")
    print(f"{'═' * 80}")
    if not final_text:
        final_text = "I couldn't process that query due to an internal error — could you rephrase, or try again?"
    print(f"\n  {final_text}\n")

    # ═══════════════════════════════════════════════════════════════
    # DISPLAY: PER-AGENT OUTPUT & TIMING (Debug Only)
    # ═══════════════════════════════════════════════════════════════
    execution_order = exec_result.get("execution_order", [])
    if debug:
        print(f"{'─' * 80}")
        print(f"  📊 AGENT EXECUTION REPORT  (Execution Order: {' → '.join(execution_order)})")
        print(f"{'─' * 80}")

        for agent_name in execution_order:
            agent_res = ctx.get(agent_name)
            if not agent_res: continue
            
            d = agent_res.data
            decision = d.get("decision", "N/A")
            title = d.get("action_title", "N/A")
            ranked = d.get("ranked_candidate_spots", [])
            rejected = d.get("rejected_candidate_spots", [])
            print(f"       Decision: {decision} │ Title: {title}")
            print(f"       Ranked Spots: {len(ranked)} │ Rejected Spots: {len(rejected)}")

            # Show warnings if any
            if agent_res.warnings:
                for w in agent_res.warnings:
                    print(f"       ⚠️  {w}")

    if debug:
        # ═══════════════════════════════════════════════════════════════
        # DISPLAY: WHY THIS RECOMMENDATION (Debug Only)
        # ═══════════════════════════════════════════════════════════════
        why = rec_output.get("why", {})
        if why:
            print(f"\n{'─' * 80}")
            print(f"  🧠 WHY THIS RECOMMENDATION")
            print(f"{'─' * 80}")
            print(f"  Primary Reason   : {why.get('primary_reason', 'N/A')}")
            print(f"  Fishing Reason   : {why.get('fishing_reason', 'N/A')}")
            print(f"  Safety Reason    : {why.get('safety_reason', 'N/A')}")
            dist = why.get("distance_km")
            coast = why.get("nearest_coast_name")
            if dist is not None and dist > 0:
                print(f"  Nearest Coast    : {coast} ({dist:.1f} km)")

        # Show candidate spots if any
        ranked_sp = rec_output.get("ranked_candidate_spots", [])
        if ranked_sp:
            print(f"\n{'─' * 80}")
            print(f"  🎯 TOP CANDIDATE FISHING SPOTS")
            print(f"{'─' * 80}")
            for s in ranked_sp[:5]:
                rank = s.get("rank", "?")
                eligible = s.get("eligible", True)
                e_icon = "✅" if eligible else "❌"
                print(f"  {e_icon} Candidate {rank} — {s.get('display_name')}")
                print(f"       PFZ        : {s.get('pfz_description', 'N/A')}")
                print(f"       Safety     : {s.get('safety_status', 'UNKNOWN')} | Weather: {s.get('weather_status', 'UNKNOWN')}")
                print(f"       Coordinates: {s.get('latitude', 0):.2f}°N, {s.get('longitude', 0):.2f}°E")
                print(f"       Source     : {s.get('source', 'UNKNOWN')}")
                if s.get("selection_reason"):
                    print(f"       Reason     : {s['selection_reason']}")

        # Show active warnings
        active_warnings = rec_output.get("warnings", [])
        if active_warnings:
            print(f"\n{'─' * 80}")
            print(f"  ⚠️  ACTIVE WARNINGS")
            print(f"{'─' * 80}")
            for w in active_warnings:
                print(f"  • {w}")

    if debug:
        # ═══════════════════════════════════════════════════════════════
        # DISPLAY: LATENCY BREAKDOWN (Debug Only)
        # ═══════════════════════════════════════════════════════════════
        print_latency_breakdown(
            lang_detect_s=t_lang_s,
            qwen_extract_s=t_extract_s,
            location_res_s=t_loc_s,
            planning_s=t_plan_s,
            domain_agents_s=t_domain_s,
            decision_engine_s=t_decision_s,
            synthesis_s=t_synth_s,
            total_s=t_total_s,
        )

        cand_timings = stage_timings.get("candidate_timings", [])
        if cand_timings:
            print("\nCANDIDATE LATENCY")
            print("────────────────────────────────────────")
            for c in cand_timings:
                print(f"  {c['location']:20s}  {c['total_ms']:7.1f} ms")
            print("\nTIER LATENCY (Summed across candidates)")
            print("────────────────────────────────────────")
            tier_sums = {}
            for c in cand_timings:
                for t in c['tiers']:
                    tier_sums[t['tier_name']] = tier_sums.get(t['tier_name'], 0.0) + t['total_ms']
            for t_name, t_ms in sorted(tier_sums.items()):
                print(f"  {t_name:20s}  {t_ms:7.1f} ms")

        # Per-agent timing summary
        if agent_timings:
            print("\nPER-AGENT LATENCY")
            print("────────────────────────────────────────")
            for a_name in execution_order:
                a_ms = agent_timings.get(a_name, 0.0)
                print(f"  {a_name:20s}  {a_ms:7.1f} ms")
            print("────────────────────────────────────────")


def print_startup_breakdown(
    qwen_load_s: float,
    netcdf_load_s: float,
    pfz_model_s: float,
    weather_model_s: float,
    prod_model_s: float,
    risk_model_s: float,
    ocean_model_s: float,
    qwen_warmup_s: float,
    engine_init_s: float,
    total_s: float,
):
    print("\nSTARTUP BREAKDOWN")
    print("────────────────────────────────────────")
    print(f"Qwen 4B LLM load       : {qwen_load_s:6.2f}s  ({qwen_load_s*1000.0:8.1f} ms)")
    print(f"Qwen warmup (extract)  : {qwen_warmup_s:6.2f}s  ({qwen_warmup_s*1000.0:8.1f} ms)")
    print(f"NetCDF datasets load   : {netcdf_load_s:6.2f}s  ({netcdf_load_s*1000.0:8.1f} ms)")
    print(f"Ocean model load       : {ocean_model_s:6.2f}s  ({ocean_model_s*1000.0:8.1f} ms)")
    print(f"PFZ XGBoost load       : {pfz_model_s:6.2f}s  ({pfz_model_s*1000.0:8.1f} ms)")
    print(f"Weather XGBoost load   : {weather_model_s:6.2f}s  ({weather_model_s*1000.0:8.1f} ms)")
    print(f"Risk model load        : {risk_model_s:6.2f}s  ({risk_model_s*1000.0:8.1f} ms)")
    print(f"Productivity LSTM load : {prod_model_s:6.2f}s  ({prod_model_s*1000.0:8.1f} ms)")
    print(f"Orchestrator engine    : {engine_init_s:6.2f}s  ({engine_init_s*1000.0:8.1f} ms)")
    print("────────────────────────────────────────")
    print(f"TOTAL STARTUP TIME     : {total_s:6.2f}s  ({total_s*1000.0:8.1f} ms)\n")


def main():
    print("Initializing ORCA System & Pre-warming NetCDF datasets & ML models...")
    t0_start = time.perf_counter()

    t0 = time.perf_counter()
    conv_model = get_conversation_model()
    t_qwen_s = time.perf_counter() - t0

    print("Warming up Qwen model caches...")
    t0 = time.perf_counter()
    from conversation.prompts import EXTRACTION_SYSTEM_PROMPT, RESPONSE_SYSTEM_PROMPT_TEMPLATE
    conv_model.extract(EXTRACTION_SYSTEM_PROMPT, "warmup query test digha kal safe")
    conv_model.generate_text(RESPONSE_SYSTEM_PROMPT_TEMPLATE.format(language_desc="English."), "warmup")
    t_qwen_warmup_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    load_ocean_datasets()
    t_netcdf_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    get_ocean_suitability_model()
    t_ocean_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    get_pfz_model()
    t_pfz_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    get_weather_model()
    t_weather_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    get_marine_risk_model()
    t_risk_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    get_productivity_model()
    t_prod_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    engine = build_engine()
    t_engine_s = time.perf_counter() - t0

    t_total_s = time.perf_counter() - t0_start

    print_startup_breakdown(
        qwen_load_s=t_qwen_s,
        netcdf_load_s=t_netcdf_s,
        pfz_model_s=t_pfz_s,
        weather_model_s=t_weather_s,
        prod_model_s=t_prod_s,
        risk_model_s=t_risk_s,
        ocean_model_s=t_ocean_s,
        qwen_warmup_s=t_qwen_warmup_s,
        engine_init_s=t_engine_s,
        total_s=t_total_s,
    )

    state = ConversationState()
    debug = False

    print_banner()

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ("/quit", "/exit"):
            print("Goodbye.")
            break
        if query.lower() == "/reset":
            state.clear()
            print("ORCA: Session state cleared.")
            continue
        if query.lower() == "/debug":
            debug = not debug
            print(f"ORCA: Debug logging {'ENABLED' if debug else 'DISABLED'}.")
            continue

        handle_turn(query, conv_model, engine, state, debug)


if __name__ == "__main__":
    main()
