# conversation/response.py

from schemas.contracts import QueryPlan
from conversation.model import ConversationModel
from conversation.prompts import RESPONSE_SYSTEM_PROMPT_TEMPLATE, CLARIFICATION_SYSTEM_PROMPT_TEMPLATE
from conversation.prompts import RESPONSE_SYSTEM_PROMPT_TEMPLATE
import time
import logging

logger = logging.getLogger(__name__)

# Keyword sets used only as a last-resort hard guardrail — this is a
# safety net, not the primary control (the primary control is the
# grounding instruction + evidence in the prompt).
UNSAFE_CLAIM_MARKERS = [
  "safe hai", "safe rahega", "safe hobe", "nirapod", "is safe",
  "will be safe", "conditions are good", "ok hai jaana",
]
SAFE_CLAIM_MARKERS = [
  "safe nahi", "not safe", "unsafe", "avoid", "dangerous",
  "nirapod na", "risky",
]


def build_evidence(exec_result: dict) -> dict:
  rec = exec_result["recommendation"]
  ctx = exec_result.get("context", {})
  evidence = {
    "decision": rec.get("decision") or rec.get("action_code"),
    "risk_level": rec.get("risk_level", rec.get("weather_summary", {}).get("risk_level", "UNKNOWN")),
    "reason": rec.get("recommendation_text") or rec.get("reason"),
    "why": rec.get("why", {}),
  }
  if "weather" in ctx:
    w = ctx["weather"].data
    w_prov = w.get("provider_data", {}).get("weather", {})
    evidence["wind_speed_ms"] = w_prov.get("wind_speed_ms")
    evidence["rain_probability"] = w_prov.get("rain_probability")
    evidence["temperature_c"] = w_prov.get("temperature_c")
    evidence["pressure_hpa"] = w_prov.get("pressure_hpa")
  if "ocean" in ctx:
    o = ctx["ocean"].data
    o_state = o.get("incois_ocean_state", {})
    evidence["wave_height_m"] = o_state.get("wave_height")
    evidence["wave_period_s"] = o_state.get("wave_period")
    evidence["current_speed_ms"] = o_state.get("current_speed")
    evidence["ocean_score"] = o.get("ocean_score")
    evidence["sst_c"] = o_state.get("sst")
  elif "ocean_state" in ctx:
    o = ctx["ocean_state"].data
    o_state = o.get("incois_ocean_state", {})
    evidence["wave_height_m"] = o_state.get("wave_height")
    evidence["wave_period_s"] = o_state.get("wave_period")
    evidence["current_speed_ms"] = o_state.get("current_speed")
    evidence["ocean_score"] = o.get("ocean_score")
    evidence["sst_c"] = o_state.get("sst")
  if "pfz" in ctx:
    p = ctx["pfz"].data
    evidence["pfz_zone"] = p.get("zone")
    evidence["pfz_score"] = p.get("pfz_score", p.get("pfz_probability"))
    evidence["nearest_coastal_loc"] = p.get("nearest_coastal_loc")
    evidence["direction"] = p.get("direction")
    evidence["distance_km"] = p.get("distance_km")
  if rec.get("ranked_candidate_spots"):
    evidence["ranked_candidate_spots"] = rec.get("ranked_candidate_spots")
  return evidence



def _violates_decision(text: str, decision: str) -> bool:
  lower = text.lower()
  unsafe_decisions = ["AVOID", "CANCEL_VOYAGE"]
  safe_decisions = ["RECOMMEND", "OPTIMAL_FISHING_VOYAGE", "CLEAR_WEATHER_LOW_YIELD", "CANDIDATE_SEARCH_COMPLETE"]
  
  if decision in unsafe_decisions:
    return any(m in lower for m in UNSAFE_CLAIM_MARKERS) and not any(
      m in lower for m in SAFE_CLAIM_MARKERS
    )
  if decision in safe_decisions:
    return any(m in lower for m in SAFE_CLAIM_MARKERS) and not any(
      m in lower for m in UNSAFE_CLAIM_MARKERS
    )
  return False


def _fallback_text(plan: QueryPlan, evidence: dict) -> str:
  loc = plan.location.name if plan.location else "the requested area"
  metrics = []
  if "wave_height_m" in evidence and evidence["wave_height_m"] is not None:
    metrics.append(f"wave height {evidence['wave_height_m']}m")
  if "wind_speed_ms" in evidence and evidence["wind_speed_ms"] is not None:
    metrics.append(f"wind speed {evidence['wind_speed_ms']} m/s")
  metrics_str = f" ({', '.join(metrics)})" if metrics else ""

  if evidence["decision"] in ("AVOID", "CANCEL_VOYAGE"):
    return f"{loc} mein conditions safe nahi hain abhi{metrics_str}. {evidence.get('reason', '')}".strip()
  if evidence["decision"] in ("RECOMMEND", "OPTIMAL_FISHING_VOYAGE", "CLEAR_WEATHER_LOW_YIELD", "CANDIDATE_SEARCH_COMPLETE"):
    return f"{loc} mein conditions theek hain{metrics_str}. {evidence.get('reason', '')}".strip()
  if evidence["decision"] in ("EXERCISE_CAUTION", "LIMITED_COASTAL_FISHING", "UNVERIFIED_SAFETY", "LIMITED / INSUFFICIENT_DATA"):
    return f"{loc} ke marine conditions puri tarah verify nahi ho paaye{metrics_str}. {evidence.get('reason', '')}".strip()
  if getattr(plan, "location_type", None) == "inland" and getattr(plan, "inland_name", None):
    return f"{plan.inland_name} is an inland region with no direct access to an ocean or sea. Please specify a coastal location."
  return "Location clarify kijiye — kripya coastal town ya port ka naam batayein."


def _sanitize_response(text: str) -> str:
  # Post-processing filter for quality artifacts
  text = text.replace(" IS safe", ".").replace("is safe.", ".").replace("koi dard nahi, ", "").replace("koi dard nahi", "")
  text = text.replace("nigher", "nearest")
  text = " ".join(text.split()).strip()
  return text

CLARIFICATION_TEMPLATES = {
  "inland": {
    "hi-Latn": "{inland_name} inland area hai jahan direct ocean access nahi hai. Fishing ke liye kripya coastal location (jaise Kochi ya Digha) specify karein.",
    "bn-Latn": "{inland_name} ekta inland region jekhane kono sea access nei. Fishing-er jonno ekta coastal location (jemon Digha ba Kochi) bolo.",
    "bn": "{inland_name} একটি অভ্যন্তরীণ এলাকা যেখানে সমুদ্রের সরাসরি প্রবেশাধিকার নেই। মাছ ধরার জন্য অনুগ্রহ করে একটি উপকূলীয় স্থান (যেমন দিঘা বা কোচি) উল্লেখ করুন।",
    "en": "{inland_name} is an inland region with no direct access to an ocean or sea. Please specify a coastal location (e.g. Kochi, Digha, Chennai, Mumbai)."
  },
  "missing_location": {
    "hi-Latn": "Kripya coastal location ya port ka naam batayein (jaise Kochi, Digha, Chennai, Mumbai).",
    "bn-Latn": "Please ekta coastal location ba port-er naam bolo (jemon Digha, Kochi, Chennai, Mumbai).",
    "bn": "অনুগ্রহ করে একটি উপকূলীয় স্থান বা বন্দরের নাম বলুন (যেমন দিঘা, কোচি, চেন্নাই, মুম্বাই)।",
    "en": "Please specify a coastal location or port (e.g. Kochi, Digha, Chennai, Mumbai)."
  },
  "missing_location_with_region": {
    "hi-Latn": "{region} mein kis specific location ke paas? Kripya kisi coastal town ya port ka naam batayein.",
    "bn-Latn": "{region}-er kon specific location-er kache? Please ekta coastal town ba port-er naam bolo.",
    "bn": "{region} এর কোন নির্দিষ্ট স্থানের কাছে? অনুগ্রহ করে একটি উপকূলীয় শহর বা বন্দরের নাম বলুন।",
    "en": "Where specifically in {region}? Please provide a coastal town or port name."
  },
  "missing_comparison_candidates": {
    "hi-Latn": "Aapko {reference} ko {region} mein kis jagah ke saath compare karna hai? Kripya un 2 jagahon ke naam batayein.",
    "bn-Latn": "Apni {reference}-ke {region}-er kon jaygar sathe compare korte chan? Please oi 2to jaygar naam bolun.",
    "bn": "আপনি {reference} কে {region} এর কোন জায়গার সাথে তুলনা করতে চান? অনুগ্রহ করে ওই ২টো জায়গার নাম বলুন।",
    "en": "Which locations in {region} do you want to compare with {reference}? Please provide the names of the 2 candidate spots."
  }
}


def get_deterministic_clarification(plan: QueryPlan) -> str:
  lang = plan.language if plan.language in ["hi-Latn", "bn-Latn", "en"] else "hi-Latn"
  if getattr(plan, "location_type", None) == "inland" and getattr(plan, "inland_name", None):
    tmpl = CLARIFICATION_TEMPLATES["inland"].get(lang, CLARIFICATION_TEMPLATES["inland"]["en"])
    return tmpl.format(inland_name=plan.inland_name)
  return CLARIFICATION_TEMPLATES["missing_location"].get(lang, CLARIFICATION_TEMPLATES["missing_location"]["en"])


LANGUAGE_DESCRIPTIONS = {
    "bn-Latn": "Bengali/Banglish (Romanized Bengali, e.g. using Bengali words like 'safe ache', 'somundro', 'hawa', 'na'). Do NOT respond in Hindi.",
    "hi-Latn": "Hindi/Hinglish (Romanized Hindi, e.g. using Hindi words like 'safe hai', 'samundar', 'hawa', 'nahi'). Do NOT respond in Bengali.",
    "en": "English."
}


def get_geography_factual_response(query: str, plan: QueryPlan) -> str:
  q_lower = query.lower()
  loc_name = plan.location.name if plan.location else (plan.inland_name or "this location")
  lang = plan.language if plan.language in ["hi-Latn", "bn-Latn", "en"] else "en"

  # Common marine geography fact patterns
  if "arabian sea" in q_lower or "arabian" in q_lower:
    if "digha" in q_lower or "bengal" in q_lower or "kolkata" in q_lower or "puri" in q_lower:
      if lang == "bn-Latn":
        return f"Na, {loc_name} Arabian Sea-te noy. Digha/West Bengal Bharat-er East Coast-e Bay of Bengal (Bongo-posagor)-er part."
      elif lang == "hi-Latn":
        return f"Nahi, {loc_name} Arabian Sea ka part nahi hai. Digha/West Bengal East Coast par Bay of Bengal mein hai."
      else:
        return f"No, {loc_name} is not part of the Arabian Sea. Digha and the West Bengal coast are located on the Bay of Bengal on India's East Coast."
    elif "kerala" in q_lower or "kochi" in q_lower or "goa" in q_lower or "mumbai" in q_lower:
      if lang == "bn-Latn":
        return f"Haan, {loc_name} India-r West Coast-e Arabian Sea-r part."
      elif lang == "hi-Latn":
        return f"Haan, {loc_name} West Coast par Arabian Sea mein hai."
      else:
        return f"Yes, {loc_name} is located along the Arabian Sea on India's West Coast."

  if "satellite" in q_lower or "pfz" in q_lower:
    if lang == "bn-Latn":
      return "Satellite PFZ zone macher thikana dekhay, kintu sekhane jawar aage weather ar wave height safe kina seta obosshoi check korte hobe."
    elif lang == "hi-Latn":
      return "Satellite PFZ zone fish location dikhata hai, lekin wahan jaane se pehle weather aur wave safety check karna zaroori hai."
    else:
      return "Satellite PFZ indicators show potential fish aggregation based on ocean temperature and chlorophyll, but you must always verify weather and wave safety conditions before going out."

  if lang == "bn-Latn":
    return f"{loc_name} somundro ar upokul bhogolik obosthaner jonno safe routing provide kora hoy."
  elif lang == "hi-Latn":
    return f"{loc_name} ke marine geography aur coastal safety details verified hain."
  return f"{loc_name} is located along the Indian coastal waters."

def validate_response_script(text: str, language: str) -> str:
    import re
    if not text: return text
    
    has_bengali = bool(re.search(r'[\u0980-\u09FF]', text))
    
    if language == "bn-Latn" and has_bengali:
        # Fallback to Latin script and strip Bengali unicode
        text = re.sub(r'[\u0980-\u09FF]+', '', text)
        text = "Banglish-e likhun. " + text.strip()
        return text
    elif language == "bn" and not has_bengali:
        # Expected Bengali script, but didn't get it
        pass
    
    return text

def generate_clarification_response(
    clarification_context: dict,
    conv_model: ConversationModel
) -> str:
    lang = clarification_context.get("language", "en")
    if lang not in ["hi-Latn", "bn-Latn", "bn", "en"]:
        lang = "en"
        
    reason = clarification_context.get("clarification_reason")
    region = clarification_context.get("region", "")
    known_locs = clarification_context.get("known_locations", {})
    ref_loc = known_locs.get("reference", "")
    
    if reason == "MISSING_COMPARISON_CANDIDATES":
        tmpl = CLARIFICATION_TEMPLATES["missing_comparison_candidates"][lang]
        return validate_response_script(tmpl.format(reference=ref_loc or "apnar location", region=region or "ai onchol"), lang)
        
    if reason == "MISSING_LOCATION":
        if region:
            tmpl = CLARIFICATION_TEMPLATES["missing_location_with_region"][lang]
            return validate_response_script(tmpl.format(region=region), lang)
        else:
            tmpl = CLARIFICATION_TEMPLATES["missing_location"][lang]
            return validate_response_script(tmpl, lang)
            
    # Fallback
    tmpl = CLARIFICATION_TEMPLATES["missing_location"][lang]
    return validate_response_script(tmpl, lang)


def _get_deterministic_response(plan: QueryPlan, evidence: dict) -> str | None:
    decision = evidence.get("decision")
    if decision not in ["RECOMMEND", "EXERCISE_CAUTION", "AVOID", "CANCEL_VOYAGE", "OPTIMAL_FISHING_VOYAGE", "CLEAR_WEATHER_LOW_YIELD", "LIMITED_COASTAL_FISHING", "UNVERIFIED_SAFETY", "VERIFICATION_REQUIRED", "LIMITED / INSUFFICIENT_DATA", "LIMITED / DEGRADED_SYNTHETIC"]:
        return None
        
    lang = plan.language if plan.language in ["hi-Latn", "bn-Latn", "en"] else "hi-Latn"
    loc = plan.location.name if plan.location else "the requested area"
    
    wind = evidence.get("wind_speed_ms")
    wave = evidence.get("wave_height_m")
    
    if wind is not None and wave is not None:
        metrics = f"wind ({wind} m/s) aur wave height ({wave} m)" if lang == "hi-Latn" else f"wind ({wind} m/s) ar wave height ({wave} m)" if lang == "bn-Latn" else f"wind ({wind} m/s) and wave height ({wave} m)"
    else:
        metrics = "marine conditions"
    
    pfz_prob = int(evidence.get("pfz_score", 0) * 100) if evidence.get("pfz_score") is not None else 0
    pfz_msg = ""
    if pfz_prob > 0:
        pfz_msg = f"(PFZ Likelihood: {pfz_prob}%)"
            
    risk = evidence.get("risk_level", "MODERATE")
    reason = evidence.get("reason", f"Risk level is {risk}.")
    
    # 1. Determine base conclusion based on decision
    if decision in ["RECOMMEND", "OPTIMAL_FISHING_VOYAGE", "CLEAR_WEATHER_LOW_YIELD"]:
        conclusion = {"en": "Conditions are safe.", "hi-Latn": "Conditions safe hain.", "bn-Latn": "Conditions safe ache."}
    elif decision in ["EXERCISE_CAUTION", "LIMITED_COASTAL_FISHING", "UNVERIFIED_SAFETY", "VERIFICATION_REQUIRED", "LIMITED / INSUFFICIENT_DATA", "LIMITED / DEGRADED_SYNTHETIC"]:
        conclusion = {"en": "Exercise caution.", "hi-Latn": "Caution rakhein.", "bn-Latn": "Sabdhanota obolombon korun."}
    elif decision in ["AVOID", "CANCEL_VOYAGE"]:
        conclusion = {"en": "It is not safe.", "hi-Latn": "Jana safe nahi hai.", "bn-Latn": "Jaoa safe noy."}
    else:
        conclusion = {"en": "", "hi-Latn": "", "bn-Latn": ""}
        
    # 2. Extract Intent
    intent = getattr(plan, "intent", "marine_conditions")
    if hasattr(intent, "value"):
        intent = intent.value
        
    # Handle Catastrophic Hazard Override
    if "CATASTROPHIC" in reason.upper() or "EARTHQUAKE" in reason.upper() or "TSUNAMI" in reason.upper() or "CYCLONE" in reason.upper():
        if lang == "hi-Latn":
            return f"Nahi — kripya abhi fishing ke liye na jaaiye. {reason} Official marine clearance aane tak samundar se door rahein."
        elif lang == "bn-Latn":
            return f"Na — doya kore ekhon fishing-e jaben na. {reason} Official marine clearance na asha porjonto somundro theke dure thakun."
        else:
            return f"No — do not go fishing right now. {reason} Wait for an official marine/coastal clearance before going out."

    # 3. Apply Intent-Specific Templates
    if intent in ["nearest_pfz", "pfz_search"]:
        if lang == "hi-Latn":
            return f"{loc} ke paas fishing spot check kiya gaya {pfz_msg}. {conclusion['hi-Latn']} {reason}"
        elif lang == "bn-Latn":
            return f"{loc}-er kache fishing spot check kora hoyeche {pfz_msg}. {conclusion['bn-Latn']} {reason}"
        else:
            return f"PFZ search near {loc} completed {pfz_msg}. {conclusion['en']} {reason}"
            
    elif intent == "hazard_alert":
        if lang == "hi-Latn":
            return f"{loc} ke liye hazard alert: {conclusion['hi-Latn']} {reason}"
        elif lang == "bn-Latn":
            return f"{loc}-er hazard alert: {conclusion['bn-Latn']} {reason}"
        else:
            return f"Hazard alert for {loc}: {conclusion['en']} {reason}"
            
    elif intent in ["marine_safety_forecast", "marine_safety"]:
        if lang == "hi-Latn":
            return f"{loc} mein aaj safety status: {conclusion['hi-Latn']} {metrics} observed hain. {reason}"
        elif lang == "bn-Latn":
            return f"{loc}-te aaj safety status: {conclusion['bn-Latn']} {metrics} dekha jachhe. {reason}"
        else:
            return f"Safety forecast for {loc}: {conclusion['en']} {metrics} are observed. {reason}"
            
    elif intent == "marine_conditions":
        if lang == "hi-Latn":
            return f"{loc} ki marine conditions: {metrics}. {conclusion['hi-Latn']} {reason}"
        elif lang == "bn-Latn":
            return f"{loc}-er marine conditions: {metrics}. {conclusion['bn-Latn']} {reason}"
        else:
            return f"Marine conditions for {loc} are {metrics}. {conclusion['en']} {reason}"

    elif intent == "fishing_zone_analysis":
        if lang == "hi-Latn":
            return f"Fishing zone analysis for {loc}: {conclusion['hi-Latn']} {pfz_msg} {reason}"
        elif lang == "bn-Latn":
            return f"{loc}-er fishing zone analysis: {conclusion['bn-Latn']} {pfz_msg} {reason}"
        else:
            return f"Fishing zone analysis for {loc}: {conclusion['en']} {pfz_msg} {reason}"
            
    elif intent == "safe_route":
        if lang == "hi-Latn":
            return f"{loc} ka safe route assessment: {conclusion['hi-Latn']} {reason}"
        elif lang == "bn-Latn":
            return f"{loc}-er safe route assessment: {conclusion['bn-Latn']} {reason}"
        else:
            return f"Safe route assessment for {loc}: {conclusion['en']} {reason}"

    elif intent == "productivity_analysis":
        if lang == "hi-Latn":
            return f"{loc} ki productivity analysis {pfz_msg}. {conclusion['hi-Latn']} {reason}"
        elif lang == "bn-Latn":
            return f"{loc}-er productivity analysis {pfz_msg}. {conclusion['bn-Latn']} {reason}"
        else:
            return f"Productivity analysis for {loc} {pfz_msg}. {conclusion['en']} {reason}"
            
    elif intent == "hazardous_zone_filter":
        if lang == "hi-Latn":
            return f"Hazardous zone filter for {loc}: {conclusion['hi-Latn']} {reason}"
        elif lang == "bn-Latn":
            return f"{loc}-er hazardous zone filter: {conclusion['bn-Latn']} {reason}"
        else:
            return f"Hazardous zone filter for {loc}: {conclusion['en']} {reason}"

    # Default fallback (similar to older logic)
    if lang == "hi-Latn":
        return f"{loc} mein {conclusion['hi-Latn']} {metrics} hain. {reason}"
    elif lang == "bn-Latn":
        return f"{loc}-te {conclusion['bn-Latn']} {metrics} ache. {reason}"
    else:
        return f"{conclusion['en']} in {loc}. {metrics}. {reason}"

def respond(
  query: str,
  plan: QueryPlan,
  exec_result: dict,
  conv_model,
  deadline: float = None,
  request_id: str = None
) -> tuple[str, list, dict]:
  timings = {"prompt_build_ms": 0.0, "inference_ms": 0.0, "parse_ms": 0.0, "deterministic_ms": 0.0}
  t_start = time.perf_counter()
  
  evidence = build_evidence(exec_result)

  # GEOGRAPHY & FACTUAL QUESTIONS: Direct factual geography response
  is_factual = getattr(plan, "result_type", None) == "GEOGRAPHY_RESULT" or getattr(plan, "intent", None) == "marine_geography"
  is_generic_explain = getattr(plan, "action_type", None) == "EXPLAIN" and getattr(plan, "explanation_target", None) != "fishing_availability"
  if is_factual or is_generic_explain:
    timings["deterministic_ms"] = (time.perf_counter() - t_start) * 1000.0
    text = get_geography_factual_response(query, plan)
    import uuid
    return text, [{"id": f"seg_{uuid.uuid4().hex[:8]}", "text": text, "agents": []}], timings

  # DETERMINISTIC SHORT-CIRCUIT: For clarification / inland turns without location, bypass 2nd LLM call!
  if evidence.get("decision") == "NEEDS_CLARIFICATION" or (plan.location is None and evidence.get("decision") not in ["COMPARE_FISHING_REGIONS", "GEOGRAPHY_RESULT"]):
    timings["deterministic_ms"] = (time.perf_counter() - t_start) * 1000.0
    text = get_deterministic_clarification(plan)
    import uuid
    return text, [{"id": f"seg_{uuid.uuid4().hex[:8]}", "text": text, "agents": []}], timings

  # STRUCTURED COMPARISON SHORT-CIRCUIT: Bypass LLM to preserve formatted comparison text and safety warnings
  if evidence.get("decision") == "COMPARE_FISHING_REGIONS":
    timings["deterministic_ms"] = (time.perf_counter() - t_start) * 1000.0
    text = evidence.get("reason", "Comparison completed.")
    import uuid
    return text, [{"id": f"seg_{uuid.uuid4().hex[:8]}", "text": text, "agents": ["rules"]}], timings

  # FAST-PATH DETERMINISTIC RENDERING (From Handlers)
  from conversation.response_generator import generate_multilingual_response
  rec = exec_result.get("recommendation", {})
  if rec.get("result_type") in ["HAZARD_RESULT", "PFZ_RESULT", "SAFETY_FORECAST_RESULT", "CONDITIONS_RESULT", "PRODUCTIVITY_RESULT", "FISHING_IMPACT_RESULT"]:
    rec_payload = rec.copy()
    if plan and plan.location:
        rec_payload["latitude"] = plan.location.latitude
        rec_payload["longitude"] = plan.location.longitude
        if not rec_payload.get("location"):
            rec_payload["location"] = {"name": plan.location.name}
    ctx = exec_result.get("context", {})
    fast_text, segments = generate_multilingual_response(rec_payload, language=plan.language, context=ctx)
    if fast_text:
        timings["deterministic_ms"] = (time.perf_counter() - t_start) * 1000.0
        return fast_text, segments, timings

  # FALLBACK DETERMINISTIC RENDERING
  # Only runs for standard fishing queries that have a clear decision
  deterministic_res = _get_deterministic_response(plan, evidence)
  segments = []
  if deterministic_res:
    timings["deterministic_ms"] = (time.perf_counter() - t_start) * 1000.0
    text = validate_response_script(deterministic_res, plan.language)
    import uuid
    segments = [{"id": f"seg_{uuid.uuid4().hex[:8]}", "text": text, "agents": ["rules"]}]
    return text, segments, timings

  t_prompt = time.perf_counter()
  lang_desc = LANGUAGE_DESCRIPTIONS.get(plan.language, LANGUAGE_DESCRIPTIONS["hi-Latn"])
  system_prompt = RESPONSE_SYSTEM_PROMPT_TEMPLATE.format(language_desc=lang_desc)
  compact_summary = {
    "evaluated_location": plan.location.name if plan.location else "None",
    "decision": evidence.get("decision"),
    "risk": evidence.get("risk_level"),
    "reason": evidence.get("reason"),
    "primary_why": evidence.get("why", {}).get("primary_reason", evidence.get("reason")),
  }
  if evidence.get("why", {}).get("fishing_reason"):
    compact_summary["fishing_why"] = evidence["why"]["fishing_reason"]
  if evidence.get("why", {}).get("safety_reason"):
    compact_summary["safety_why"] = evidence["why"]["safety_reason"]
  if evidence.get("why", {}).get("imd_weather"):
    compact_summary["imd_weather"] = evidence["why"]["imd_weather"]["summary"]
  if evidence.get("why", {}).get("incois_bsi"):
    compact_summary["bsi_advisory"] = evidence["why"]["incois_bsi"]["advisory"]
    compact_summary["bsi_hazards"] = evidence["why"]["incois_bsi"]["active_hazards"]
  if plan.user_constraint:
    compact_summary["user_constraint"] = plan.user_constraint
  if "wind_speed_ms" in evidence and evidence["wind_speed_ms"] is not None:
    compact_summary["wind_speed_ms"] = evidence["wind_speed_ms"]
  if "wave_height_m" in evidence and evidence["wave_height_m"] is not None:
    compact_summary["wave_height_m"] = evidence["wave_height_m"]
  if "rain_probability" in evidence and evidence["rain_probability"] is not None:
    compact_summary["rain_probability"] = f"{int(evidence['rain_probability']*100)}%"
  if "sst_c" in evidence and evidence["sst_c"] is not None:
    compact_summary["sst_c"] = evidence["sst_c"]
  if getattr(plan, "candidate_locations", None):
    compact_summary["unchecked_candidate_locations"] = plan.candidate_locations
  if "pfz_zone" in evidence:
    compact_summary["pfz_zone"] = evidence.get("pfz_zone")
    if evidence.get("distance_km"):
      compact_summary["pfz_distance"] = f"{evidence.get('distance_km')} {evidence.get('direction', '')}".strip()
  if evidence.get("ranked_candidate_spots"):
    spots_summary = []
    for s in evidence["ranked_candidate_spots"][:3]:
      spots_summary.append(f"Spot #{s.get('rank')}: {s.get('distance_km')}km {s.get('compass_direction')} (PFZ: {int(s.get('pfz_probability',0)*100)}%, Risk: {s.get('weather_risk')})")
    compact_summary["recommended_spots"] = spots_summary

  user_message = f"Query: {query}\nDecision Summary: {compact_summary}"

  timings["prompt_build_ms"] = (time.perf_counter() - t_prompt) * 1000.0

  if deadline and time.perf_counter() > deadline:
    logger.warning("Deadline exceeded before LLM synthesis. Falling back to deterministic response.")
    text = _fallback_text(plan, evidence)
    import uuid
    return text, [{"id": f"seg_{uuid.uuid4().hex[:8]}", "text": text, "agents": []}], timings

  t_inf = time.perf_counter()
  try:
    text = conv_model.generate_text(
        system_prompt, user_message, max_new_tokens=35, temperature=0.3
    )
    timings["inference_ms"] = (time.perf_counter() - t_inf) * 1000.0
    
    t_parse = time.perf_counter()
    text = _sanitize_response(text)
    timings["parse_ms"] = (time.perf_counter() - t_parse) * 1000.0
  except Exception as e:
    logger.error(f"LLM synthesis failed: {e}", exc_info=True)
    text = _fallback_text(plan, evidence)

  # Hard guardrail: code-level check, never trust the prompt alone for safety claims
  if _violates_decision(text, evidence["decision"]):
    logger.warning(f"Safety guardrail override: LLM response violated decision={evidence['decision']}. Falling back to deterministic text. request_id={request_id}")
    text = _fallback_text(plan, evidence)

  text = validate_response_script(text, plan.language)
  # Extract context agents
  ctx_agents = [k for k in exec_result.get("context", {}).keys()]
  import uuid
  segments = [{"id": f"seg_{uuid.uuid4().hex[:8]}", "text": text, "agents": ctx_agents}]
  return text, segments, timings