# Proto/orchestrator/engine.py

import time
import logging
import concurrent.futures
from datetime import datetime, timezone
from typing import Dict, List, Any
import contextvars

from schemas.contracts import QueryPlan, AgentResult, BaseAgent, GeoLocation

logger = logging.getLogger(__name__)


# Agent dependency graph.
#
# Domain agents can run independently where possible.
# Decision/safety agents run only after the information they depend on
# is available. Recommendation is deliberately placed after the safety
# and risk layers so the final recommendation can be arbitrated by
# apply_constraints().
AGENT_DEPENDENCIES = {
    "weather": [],
    "ocean": [],
    "ocean_state": [],
    "tide": [],
    "pfz": ["ocean"],
    "productivity": ["ocean"],
    "geospatial": [],
    "marine_safety": ["weather", "ocean"],
    "risk": ["weather", "ocean", "geospatial"],
    "safety_rules": ["weather", "ocean", "geospatial", "marine_safety"],
    "recommendation": [
        "weather",
        "ocean",
        "pfz",
        "productivity",
        "geospatial",
        "risk",
        "safety_rules",
    ],
}

def validate_operation_requirements(plan: QueryPlan) -> tuple[bool, str]:
    op = getattr(plan, "operation", None)
    loc = plan.location or plan.target_location or plan.reference_location
    loc_type = getattr(plan, "location_type", "unknown")
    inland_name = getattr(plan, "inland_name", None)

    if op in ("TEMPORAL_PFZ_GUIDANCE", "COMPARE_FISHING_REGIONS", "COMPARE_LOCATIONS", "FISHING_SAFETY_TRADEOFF", "GENERAL_MARINE_QUERY", "SELECT_BEST_FISHING_OPTION", "NEAREST_PFZ_SEARCH"):
        return True, ""

    if plan.spatial_constraint and loc:
        return True, ""

    if op == "DISTANCE_TO_COAST":
        if loc:
            return True, ""
        return False, "Please specify a location or starting point to calculate distance to coast."

    if loc is None:
        if loc_type == "inland":
            return (
                False,
                f"{inland_name or 'This area'} is an inland region with no direct access "
                "to an ocean or sea. Please specify a coastal location.",
            )
        return False, "Which coastal region or starting point should I search around?"

    if loc_type == "inland" and op not in (
        "DISTANCE_TO_COAST",
        "TEMPORAL_PFZ_GUIDANCE",
        "COMPARE_FISHING_REGIONS",
        "COMPARE_LOCATIONS",
        "SELECT_BEST_FISHING_OPTION",
        "NEAREST_PFZ_SEARCH",
        "FIND_FISHING_SPOTS"
    ):
        return (
            False,
            f"{inland_name or 'This area'} is an inland region with no direct access "
            "to an ocean or sea. Please specify a coastal location or harbor.",
        )

    return True, ""


class OrcaOrchestrator:
    def __init__(self, registry: Dict[str, BaseAgent]):
        self.registry = registry

    def resolve_dependencies(self, requested_agents: List[str]) -> List[str]:
        visited = set()
        execution_order = []

        def visit(agent_name: str):
            if agent_name in visited:
                return
            for dep in AGENT_DEPENDENCIES.get(agent_name, []):
                visit(dep)
            visited.add(agent_name)
            execution_order.append(agent_name)

        for agent in requested_agents:
            if agent in self.registry or agent in AGENT_DEPENDENCIES:
                visit(agent)

        return execution_order

    def _group_into_tiers(self, execution_order: List[str]) -> List[List[str]]:
        tiers = []
        completed = set()
        remaining = list(execution_order)

        while remaining:
            current_tier = []
            for name in remaining:
                deps = AGENT_DEPENDENCIES.get(name, [])
                if all(dep not in execution_order or dep in completed for dep in deps):
                    current_tier.append(name)

            if not current_tier:
                current_tier = [remaining[0]]

            tiers.append(current_tier)
            for name in current_tier:
                completed.add(name)
                remaining.remove(name)

        return tiers

    def apply_constraints(self, plan: QueryPlan, context: Dict[str, AgentResult]) -> Dict[str, Any]:
        from schemas.data_catalog import DATA_CATALOG
        from schemas.contracts import UNKNOWN

        # Check hard invariant: if agent failed, output is UNKNOWN
        for agent_name, agent_res in context.items():
            if not hasattr(agent_res, "status"):
                continue
            if agent_res.status not in ("SUCCESS", "DEGRADED", "MOCKED", "success"):
                agent_res.data = {k: "UNKNOWN" for k in agent_res.data}

        safety_res = context.get("safety_rules")
        risk_res = context.get("risk")
        safety_data = safety_res.data if safety_res else {}
        risk_data = risk_res.data if risk_res else {}

        is_comparison_op = getattr(plan, "operation", None) in (
            "COMPARE_FISHING_REGIONS", "COMPARE_LOCATIONS", "SELECT_BEST_FISHING_OPTION"
        )
        
        if is_comparison_op and "recommendation" in context:
            # For comparisons, top-level risk shouldn't be UNKNOWN just because the global context lacks it.
            # We can extract the safest risk score or just rely on the recommendation agent's text.
            # To avoid "UNKNOWN" in the final API response, we'll mark it as COMPARISON.
            safety_clearance = "COMPARISON"
            risk_level = "COMPARISON"
            rule_violations = []
        else:
            safety_clearance = safety_data.get("safety_clearance", "UNKNOWN") if safety_res and safety_res.status in ("SUCCESS", "DEGRADED", "MOCKED", "success") else "UNKNOWN"
            risk_level = risk_data.get("risk_level", "UNKNOWN") if risk_res and risk_res.status in ("SUCCESS", "DEGRADED", "MOCKED", "success") else "UNKNOWN"
            rule_violations = safety_data.get("rule_violations", [])

        matrix = {}
        completeness_score = 1.0

        if "recommendation" in context:
            rec_data = context["recommendation"].data
            decision = rec_data.get("action_code", "COMPLETED")
            action_title = rec_data.get("action_title", "ADVISORY")
            ranked_spots = rec_data.get("ranked_candidate_spots", [])
            rec_text = rec_data.get("recommendation_text", "Operational analysis complete.")

            # Data quality checks
            weather_res = context.get("weather")
            ocean_res = context.get("ocean")
            weather_failed = weather_res and weather_res.status not in ("SUCCESS", "DEGRADED", "MOCKED", "success", "DEGRADED_SYNTHETIC")
            ocean_failed = ocean_res and ocean_res.status not in ("SUCCESS", "DEGRADED", "MOCKED", "success", "DEGRADED_SYNTHETIC")
            weather_degraded = weather_res and weather_res.status in ("DEGRADED", "DEGRADED_SYNTHETIC")
            ocean_degraded = ocean_res and ocean_res.status in ("DEGRADED", "DEGRADED_SYNTHETIC")
            pfz_res = context.get("pfz")
            pfz_degraded = pfz_res and pfz_res.status == "DEGRADED_SYNTHETIC"

            marine_safety_res = context.get("marine_safety")
            if is_comparison_op:
                pass
            elif decision in ("UNSUPPORTED_LOCATION", "COASTAL_STATE_LOCATION", "DISTANCE_TO_COAST_RESULT"):
                pass # Do not override geographic or domain constraint responses
            elif (safety_clearance == "RESTRICTED" or risk_level in ["DANGEROUS", "DANGER"] or "CYCLONE_HAZARD_ALERT" in rule_violations or "CATASTROPHIC HAZARD" in rec_text.upper() or (marine_safety_res and marine_safety_res.data.get("catastrophic_active"))):
                decision = "CANCEL_VOYAGE"
                action_title = "DO NOT EMBARK / RETURN TO HARBOR"
                ranked_spots = []
                # Only overwrite the text if it's a generic completion message or the agent didn't handle it
                if "No live catastrophic hazard" not in rec_text and "CATASTROPHIC HAZARD" not in rec_text.upper():
                    rec_text = "Severe marine hazard alert active. Operational fishing voyages are strictly CANCELLED."
            elif safety_clearance == "UNKNOWN" or risk_level == "UNKNOWN" or weather_failed or ocean_failed:
                decision = "LIMITED / INSUFFICIENT_DATA"
                action_title = "CAUTION: INSUFFICIENT DATA"
                rec_text = "Unable to fully verify marine safety conditions due to missing data. Please exercise extreme caution."
                ranked_spots = []
            elif weather_degraded or ocean_degraded or pfz_degraded:
                decision = "LIMITED / DEGRADED_SYNTHETIC"
                action_title = "CAUTION: SYNTHETIC DATA"
                rec_text = "Analysis is based on synthetic or mock data because live feeds were unavailable. Do not rely on this recommendation for critical safety decisions."

            # Apply count limit
            count_limit = getattr(plan, "count", 1)
            op = getattr(plan, "operation", "")
            if count_limit <= 1 and op in ("FIND_FISHING_SPOTS", "NEAREST_PFZ_SEARCH", "ROUTE_TO_FISHING_AREA"):
                count_limit = 3
            if ranked_spots and len(ranked_spots) > count_limit:
                ranked_spots = ranked_spots[:count_limit]

            why_dict = rec_data.get("why", {})
            if decision == "CANCEL_VOYAGE":
                if "No catastrophic hazard" not in why_dict.get("primary_reason", "") and "Catastrophic hazard" not in why_dict.get("primary_reason", ""):
                    why_dict["primary_reason"] = "Severe marine hazard alert active. Operations cancelled."
                    why_dict["safety_reason"] = "Active hazards detected in safety rules engine."
            elif decision == "LIMITED / INSUFFICIENT_DATA":
                why_dict["primary_reason"] = "Data from critical marine/weather agents is unavailable."
                why_dict["safety_reason"] = "Missing sensor data."

            return {
                "result_type": getattr(plan, "result_type", "SAFETY_ASSESSMENT"),
                "decision": decision,
                "action_code": decision,
                "action_title": action_title,
                "risk_level": risk_level,
                "reason": rec_text,
                "recommendation_text": rec_text,
                "distance_km": rec_data.get("distance_km"),
                "nearest_coast_name": rec_data.get("nearest_coast_name"),
                "why": why_dict,
                "pfz_summary": rec_data.get("pfz_summary", {}),
                "ranked_candidate_spots": ranked_spots,
                "rejected_candidate_spots": rec_data.get("rejected_candidate_spots", []),
                "warnings": rec_data.get("active_warnings", []),
                "data_matrix": matrix,
                "completeness_score": completeness_score
            }

        return {
          "result_type": "SAFETY_ASSESSMENT",
          "decision": "CANCEL_VOYAGE" if safety_clearance == "RESTRICTED" else "RECOMMEND",
          "risk_level": risk_level,
          "reason": "Safety arbitration complete.",
          "data_matrix": matrix,
          "completeness_score": completeness_score
        }

    def _run_agent_safely(self, agent: BaseAgent, plan: QueryPlan, context: Dict[str, AgentResult], deadline: float = None) -> tuple[AgentResult, float]:
        t0 = time.perf_counter()
        
        time_left = None
        if deadline is not None:
            time_left = max(0.1, deadline - t0)
            context["_time_left"] = time_left
            
        try:
            result = agent.run(plan, context)
        except Exception as e:
            logger.warning(f"Agent {agent.name} failed: {e}")
            loc = plan.target_location or plan.location or plan.reference_location
            result = AgentResult(
                agent=agent.name,
                status="error",
                location=loc,
                timestamp=datetime.now(timezone.utc),
                data={"error": f"Agent execution failed: {str(e)}"},
                confidence=0.0,
                sources=[],
                warnings=[f"{agent.name.upper()}_FAILURE: Internal execution error."]
            )
        return result, (time.perf_counter() - t0) * 1000.0

    def generate_candidates(self, plan: QueryPlan) -> List[GeoLocation]:
        if plan.compare_locations:
            return plan.compare_locations

        loc = plan.location or plan.reference_location or plan.target_location
        if not loc:
            return []

        from agents.geospatial.grid import offset_coordinate
        from location.location_metadata import LOCATION_METADATA

        candidates = [loc]
        loc_name = (loc.name or "").lower()
        coastal = LOCATION_METADATA.get(loc_name, {}).get("coastal_access", True) if loc_name else True

        # If the origin is likely inland, project 8 candidates offshore
        if not coastal or plan.operation in ("FIND_FISHING_SPOTS", "NEAREST_PFZ_SEARCH"):
            candidates = []
            for b in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]:
                lat, lon = offset_coordinate(loc.latitude, loc.longitude, 20.0, b)
                candidates.append(GeoLocation(latitude=lat, longitude=lon, name=f"{loc.name} {b}° Offshore"))

        return candidates

    def run(self, plan: QueryPlan, deadline: float = None, request_id: str = None, bypass_db_lookup: bool = False) -> Dict[str, Any]:
        from orchestrator.handlers import SPECIALIZED_HANDLERS
        handler = SPECIALIZED_HANDLERS.get(plan.intent)
        print(f"DEBUG engine.py: plan.intent is {plan.intent}, type {type(plan.intent)}, handler found: {handler}")
        if handler:
            return handler.run(plan, self)

        t0_orc = time.perf_counter()

        is_valid_op, req_reason = validate_operation_requirements(plan)
        if not is_valid_op:
            return {
                "plan_id": plan.intent,
                "execution_order": [],
                "context": {},
                "recommendation": {
                    "result_type": "CLARIFICATION",
                    "decision": "NEEDS_CLARIFICATION",
                    "action_code": "NEEDS_CLARIFICATION",
                    "risk_level": "N/A",
                    "reason": req_reason
                },
                "stage_timings": {"domain_agents_ms": 0.0, "decision_engine_ms": 0.0, "total_engine_ms": (time.perf_counter() - t0_orc) * 1000.0}
            }

        execution_order = self.resolve_dependencies(plan.agents)
        tiers = self._group_into_tiers(execution_order)

        # Candidate Generation
        candidates = self.generate_candidates(plan)
        if not candidates:
            # Fallback to current location if candidate generation yields nothing
            loc = plan.location or plan.reference_location or plan.target_location
            if loc: 
                candidates = [loc]
            else:
                return {
                    "plan_id": plan.intent,
                    "execution_order": [],
                    "context": {},
                    "recommendation": {
                        "result_type": "CLARIFICATION",
                        "decision": "NEEDS_CLARIFICATION",
                        "action_code": "NEEDS_CLARIFICATION",
                        "risk_level": "N/A",
                        "reason": "Could not determine any coastal locations for this query."
                    },
                    "stage_timings": {"domain_agents_ms": 0.0, "decision_engine_ms": 0.0, "total_engine_ms": (time.perf_counter() - t0_orc) * 1000.0}
                }
            if loc: candidates = [loc]

        is_comparison = plan.operation in ("COMPARE_FISHING_REGIONS", "COMPARE_LOCATIONS", "SELECT_BEST_FISHING_OPTION")

        all_contexts = [{"_bypass_db_lookup": bypass_db_lookup} for _ in candidates]
        cand_plans = [plan.model_copy(update={"location": cand, "target_location": cand}) for cand in candidates]
        agent_timings = {}
        t_domain_ms = 0.0
        t_decision_ms = 0.0
        
        candidate_timings = [
            {
                "location": cand.name or "Unknown",
                "total_ms": 0.0,
                "tiers": [],
                "_t0": time.perf_counter()
            } for cand in candidates
        ]

        for i, tier in enumerate(tiers):
            loc_tier = [a for a in tier if a != "recommendation"]
            if not loc_tier:
                continue
            
            t0_tier = time.perf_counter()
            
            for c_idx in range(len(candidates)):
                candidate_timings[c_idx]["tiers"].append({
                    "tier_name": f"Tier {i+1}",
                    "total_ms": 0.0,
                    "agents": {}
                })
            
            tasks = []
            for c_idx in range(len(candidates)):
                for a in loc_tier:
                    if a in self.registry:
                        tasks.append((c_idx, a))
                        
            # CONCURRENT DOMAIN AGENT EXECUTION
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, max(1, len(tasks)))) as executor:
                futures = {
                    executor.submit(contextvars.copy_context().run, self._run_agent_safely, self.registry[a], cand_plans[c_idx], all_contexts[c_idx]): (c_idx, a)
                    for c_idx, a in tasks
                }
                for future in concurrent.futures.as_completed(futures):
                    c_idx, a = futures[future]
                    res, elapsed = future.result()
                    all_contexts[c_idx][a] = res
                    
                    # Store timings safely
                    agent_timings[a] = agent_timings.get(a, 0.0) + elapsed
                    candidate_timings[c_idx]["tiers"][-1]["agents"][a] = elapsed
                    
            elapsed_tier = (time.perf_counter() - t0_tier) * 1000.0
            
            for c_idx in range(len(candidates)):
                candidate_timings[c_idx]["tiers"][-1]["total_ms"] = elapsed_tier

            if any(a in {"weather", "ocean", "pfz", "geospatial", "productivity"} for a in loc_tier):
                t_domain_ms += elapsed_tier
            else:
                t_decision_ms += elapsed_tier
                
        for c_idx in range(len(candidates)):
            candidate_timings[c_idx]["total_ms"] = (time.perf_counter() - candidate_timings[c_idx]["_t0"]) * 1000.0
            del candidate_timings[c_idx]["_t0"]

        # Build final context intelligently to prevent global context pollution from multiple candidates
        if is_comparison and len(candidates) > 1:
            final_context = {"comparison_meta": AgentResult(agent="engine", status="success", timestamp=datetime.now(timezone.utc), data={"contexts": all_contexts}, confidence=1.0, sources=[])}
        else:
            final_context = all_contexts[0] if all_contexts else {}

        if "recommendation" in execution_order and "recommendation" in self.registry:
            t0_rec = time.perf_counter()
            rec_res, elapsed = self._run_agent_safely(self.registry["recommendation"], plan, final_context, deadline)
            final_context["recommendation"] = rec_res
            agent_timings["recommendation"] = elapsed
            t_decision_ms += elapsed

        recommendation = self.apply_constraints(plan, final_context)

        # Build agent_execution audit
        agents_called = []
        reverse_deps = {a: [] for a in execution_order}
        for a in execution_order:
            for dep in AGENT_DEPENDENCIES.get(a, []):
                if dep in reverse_deps:
                    reverse_deps[dep].append(a)
                    
        for a in execution_order:
            if a in plan.agents:
                reason_called = f"Explicitly requested by intent '{plan.intent}'"
            else:
                reason_called = f"Required as a dependency for downstream agents"
                
            used_by = [dep for dep in reverse_deps.get(a, []) if dep in final_context and getattr(final_context[dep], "status", "") in ("SUCCESS", "DEGRADED", "MOCKED", "success")]
            
            if a in final_context:
                agent_res = final_context[a]
                
                inputs_used = {}
                outputs = {}
                output_reason = "No output explanation provided."
                sources = getattr(agent_res, "sources", [])
                score_source = None
                score_reason = None
                
                if hasattr(agent_res, "audit") and agent_res.audit:
                    inputs_used = agent_res.audit.inputs_used
                    outputs = agent_res.audit.outputs or getattr(agent_res, "data", {})
                    output_reason = agent_res.audit.output_reason
                    if agent_res.audit.sources:
                        sources = agent_res.audit.sources
                    score_source = agent_res.audit.score_source
                    score_reason = agent_res.audit.score_reason
                else:
                    outputs = getattr(agent_res, "data", {})
                
                status_val = getattr(agent_res, "status", "ERROR")
                # Map old status values to new explicit values if they are 'success' or 'error'
                if status_val == "success": status_val = "SUCCESS"
                elif status_val == "error": status_val = "ERROR"
                
                agent_info = {
                    "agent": a,
                    "status": status_val,
                    "reason_called": reason_called,
                    "inputs_used": inputs_used,
                    "outputs": outputs,
                    "output_reason": output_reason,
                    "sources": sources,
                    "confidence": getattr(agent_res, "confidence", 0.0),
                    "used_by": used_by
                }
                if score_source is not None:
                    agent_info["score_source"] = score_source
                if score_reason is not None:
                    agent_info["score_reason"] = score_reason
                
                agents_called.append(agent_info)
            else:
                agents_called.append({
                    "agent": a,
                    "status": "NOT_AVAILABLE" if a in self.registry else "SKIPPED",
                    "reason_called": reason_called,
                    "inputs_used": {},
                    "outputs": {},
                    "output_reason": "Agent was planned but not executed or its output was unavailable.",
                    "sources": [],
                    "confidence": 0.0,
                    "used_by": used_by
                })
        
        agent_execution = {
            "planned_agents": plan.agents,
            "execution_order": execution_order,
            "agents_called": agents_called
        }

        return {
            "plan_id": plan.intent,
            "execution_order": execution_order,
            "context": final_context,
            "agent_execution": agent_execution,
            "recommendation": recommendation,
            "agent_timings": agent_timings,
            "stage_timings": {
                "domain_agents_ms": t_domain_ms,
                "decision_engine_ms": t_decision_ms,
                "total_engine_ms": (time.perf_counter() - t0_orc) * 1000.0,
                "candidate_timings": candidate_timings
            }
        }
