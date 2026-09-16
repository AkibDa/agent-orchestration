# Proto/agents/rules/fishing_search.py

from typing import List, Dict, Any
from agents.geospatial.distance import haversine_distance, bearing, compass_direction
from agents.geospatial.restrictions import is_eez_restricted, is_in_domain

def rank_fishing_candidates(
    origin_lat: float,
    origin_lon: float,
    candidate_evaluations: List[Dict[str, Any]],
    top_k: int = 3
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Ranks valid fishing candidates deterministically using multi-objective scoring.
    
    SAFETY GUARDRAIL: Hard-filters out any restricted, out-of-domain, or hazardous points FIRST,
    recording explicit rejection reasons.
    
    score = 0.50 * pfz_probability - 0.25 * (distance_km / 30.0) - 0.25 * risk_penalty
    """
    risk_penalties = {"NORMAL": 0.0, "CAUTION": 0.5, "DANGEROUS": 1.0, "UNSUPPORTED_LOCATION": 1.0}

    scored_candidates = []
    rejected_candidates = []

    for cand in candidate_evaluations:
        c_lat = cand["latitude"]
        c_lon = cand["longitude"]

        is_restr, restr_name = is_eez_restricted(c_lat, c_lon)
        in_dom = is_in_domain(c_lat, c_lon)
        s_clear = cand.get("safety_clearance", "CLEARED")
        w_risk = cand.get("weather_risk", "NORMAL")

        # HARD SAFETY FILTER: Discard restricted, out-of-domain, or hazardous candidates
        if is_restr:
            rejected_candidates.append({
                "id": cand.get("id", "spot"),
                "latitude": c_lat,
                "longitude": c_lon,
                "eligible": False,
                "rejection_reason": f"Restricted zone boundary ({restr_name})"
            })
            continue
        if not in_dom:
            rejected_candidates.append({
                "id": cand.get("id", "spot"),
                "latitude": c_lat,
                "longitude": c_lon,
                "eligible": False,
                "rejection_reason": "Out of operational domain (Indian Ocean 5°N–25°N, 65°E–95°E)"
            })
            continue
        if s_clear == "RESTRICTED" or w_risk == "DANGEROUS":
            rejected_candidates.append({
                "id": cand.get("id", "spot"),
                "latitude": c_lat,
                "longitude": c_lon,
                "eligible": False,
                "rejection_reason": f"Hazardous safety clearance ({s_clear}, {w_risk} risk)"
            })
            continue

        dist = cand.get("distance_km", haversine_distance(origin_lat, origin_lon, c_lat, c_lon))
        brng = bearing(origin_lat, origin_lon, c_lat, c_lon) if dist > 0 else 0.0
        comp_dir = compass_direction(brng) if dist > 0 else "CENTER"

        pfz_prob = cand.get("pfz_probability", 0.5)
        risk_pen = risk_penalties.get(w_risk, 0.5)
        norm_dist_penalty = min(dist / 30.0, 1.0)
        
        pfz_component = 0.50 * pfz_prob
        dist_component = 0.25 * norm_dist_penalty
        risk_component = 0.25 * risk_pen
        score = pfz_component - dist_component - risk_component
        
        sel_reason = f"PFZ probability {int(pfz_prob*100)}% with {w_risk} weather risk at {round(dist,1)}km {comp_dir}"
        
        score_source = "RULE_BASED_WEIGHTED"
        score_reason = (
            f"Composite score combines PFZ probability (+{pfz_component:.3f}), "
            f"distance penalty (-{dist_component:.3f}), and weather-risk penalty (-{risk_component:.3f})."
        )

        scored_cand = {
            "id": cand.get("id", "spot"),
            "latitude": c_lat,
            "longitude": c_lon,
            "distance_km": round(dist, 1),
            "bearing_deg": round(brng, 1),
            "compass_direction": comp_dir,
            "pfz_probability": round(pfz_prob, 4),
            "weather_risk": w_risk,
            "safety_clearance": s_clear,
            "eligible": True,
            "rejection_reason": None,
            "is_restricted": False,
            "composite_score": round(score, 4),
            "score_source": score_source,
            "score_reason": score_reason,
            "selection_reason": sel_reason,
            "recommended": (score > 0.10)
        }
        scored_candidates.append(scored_cand)

    # Sort deterministically by composite_score descending
    scored_candidates.sort(key=lambda x: x["composite_score"], reverse=True)

    # Assign rank index
    for idx, spot in enumerate(scored_candidates, 1):
        spot["rank"] = idx

    return scored_candidates[:top_k], rejected_candidates
