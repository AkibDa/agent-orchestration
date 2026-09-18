# conversation/response_generator.py

from typing import Dict, Any, List, Tuple
import uuid

RESPONSE_TEMPLATES = {
    "OPTIMAL_FISHING_VOYAGE": {
        "en": "Recommended fishing voyage! High potential fishing zone identified (probability {pfz_prob_pct}%) with safe ocean and weather conditions.",
        "bn": "মাছ ধরার জন্য উপযুক্ত সময়! উচ্চ সম্ভাবনাময় মাছের এলাকা (সম্ভাবনা {pfz_prob_pct}%) চিহ্নিত করা হয়েছে, আবহাওয়া ও সমুদ্র পরিস্থিতি সম্পূর্ণ নিরাপদ।",
        "bn_en": "Mach dhorar jonno khub bhalo shomoy! High probability fishing zone (probability {pfz_prob_pct}%) pawa geche, abohawa aar shomudro shompurno safe.",
        "hi-Latn": "Recommended fishing voyage! High potential fishing zone mil gaya (probability {pfz_prob_pct}%), aur ocean aur weather conditions safe hain."
    },
    "LIMITED_COASTAL_FISHING": {
        "en": "Proceed with caution. Potential fishing zone detected (probability {pfz_prob_pct}%), but safety conditions require caution. Stay near shore (<10 km).",
        "bn": "সতর্কতার সাথে যান। মাছের এলাকা (সম্ভাবনা {pfz_prob_pct}%) পাওয়া গেছে, তবে নিরাপত্তা সতর্কতা রয়েছে। উপকূলের ১০ কিমির মধ্যে থাকুন।",
        "bn_en": "Shotorkotar shathe jaan. Macher area (probability {pfz_prob_pct}%) pawa geche, kintu safety warning ache. Upokuler 10 km er moddhe thakun.",
        "hi-Latn": "Dhyan se aage badhein. Potential fishing zone detect hua hai (probability {pfz_prob_pct}%), par safety conditions mein caution zaroori hai. Kinare ke paas rahein (<10 km)."
    },
    "CLEAR_WEATHER_LOW_YIELD": {
        "en": "Favorable weather for navigation, but current ocean features do not indicate dense fish aggregation at this location.",
        "bn": "সমুদ্র ও আবহাওয়া নিরাপদ, তবে এই স্থানে প্রচুর মাছের উপস্থিতি পাওয়া যায়নি।",
        "bn_en": "Shomudro ebong abohawa safe, kintu ei jaigay bhalo macher khobor pawa jayni.",
        "hi-Latn": "Mausam aur samundar navigation ke liye theek hai, lekin yahan machliyon ki zyada ummeed nahi hai."
    },
    "EXERCISE_CAUTION": {
        "en": "Exercise caution at sea. {recommendation_text}",
        "bn": "সমুদ্রে সতর্ক থাকুন। {recommendation_text}",
        "bn_en": "Shomudre shotorko thakun. {recommendation_text}",
        "hi-Latn": "Samundar mein dhyan rakhein. {recommendation_text}"
    },
    "CANCEL_VOYAGE": {
        "en": "DANGER / DO NOT EMBARK: {recommendation_text}",
        "bn": "বিপদ / যাত্রা বাতিল করুন: {recommendation_text}",
        "bn_en": "BIPOD / Jatra batil korun: {recommendation_text}",
        "hi-Latn": "KHATRA / TRIP CANCEL KAREIN: {recommendation_text}"
    },
    "LIMITED / INSUFFICIENT_DATA": {
        "en": "WARNING: Insufficient official marine data available at this time to guarantee safety. {recommendation_text}",
        "bn": "সতর্কতা: সমুদ্রের নিরাপত্তা নিশ্চিত করার জন্য পর্যাপ্ত তথ্য নেই। {recommendation_text}",
        "bn_en": "WARNING: Sea safety confirm korar jonno porjapto data nei. {recommendation_text}",
        "hi-Latn": "WARNING: Samundar ki safety confirm karne ke liye data poora nahi hai. {recommendation_text}"
    },
    "DISTANCE_TO_COAST_RESULT": {
        "en": "The nearest coastline/sea is approximately {dist_km:.1f} km away at {nearest_coast}.",
        "bn": "নিকটবর্তী সমুদ্র/উপকূলের দূরত্ব প্রায় {dist_km:.1f} কিমি ({nearest_coast})।",
        "bn_en": "Nearest sea/coastline distance ta lagbhag {dist_km:.1f} km ({nearest_coast}).",
        "hi-Latn": "Sabse paas ki coastline ya samundar lagbhag {dist_km:.1f} km door hai ({nearest_coast})."
    },
    "COASTAL_STATE_LOCATION": {
        "en": "This is a coastal region directly bordering the sea (0 km distance to coast).",
        "bn": "এটি একটি উপকূলীয় অঞ্চল যা সরাসরি সমুদ্রের সাথে সংযুক্ত (দূরত্ব ০ কিমি)।",
        "bn_en": "Eta holo ekta coastal region ta shorasori sea er sathe attached (distance 0 km).",
        "hi-Latn": "Yeh ek coastal region hai jo seedhe samundar se juda hai (0 km distance to coast)."
    },
    "UNSUPPORTED_LOCATION": {
        "en": "UNSUPPORTED LOCATION: Coordinate ({lat:.2f}°N, {lon:.2f}°E) is outside the supported Indian Ocean domain (5°N–25°N, 65°E–95°E). Model predictions are not available for this area.",
        "bn": "অসমর্থিত এলাকা: নির্দিষ্ট অবস্থানটি ({lat:.2f}°N, {lon:.2f}°E) আমাদের সমর্থিত ভারত মহাসাগর অঞ্চলের (5°N–25°N, 65°E–95°E) বাইরে। এই এলাকার পূর্বাভাস পাওয়া যাবে না।",
        "bn_en": "UNSUPPORTED LOCATION: Ei location ({lat:.2f}°N, {lon:.2f}°E) amader supported Indian Ocean domain (5°N–25°N, 65°E–95°E) er baire. Ei jaigar jonno forecast pawa jabe na.",
        "hi-Latn": "UNSUPPORTED LOCATION: Yeh location ({lat:.2f}°N, {lon:.2f}°E) hamare supported Indian Ocean domain (5°N–25°N, 65°E–95°E) ke bahar hai. Is area ke liye forecast available nahi hai."
    },
    "NEEDS_CLARIFICATION": {
        "en": "Location clarification needed: Please specify a valid coastal location or port (e.g. Digha, Kochi, Chennai, Mumbai). Non-coastal or inland regions do not have direct ocean fishing access.",
        "bn": "অবস্থান নির্দিষ্ট করুন: অনুগ্রহ করে একটি সঠিক উপকূলীয় অঞ্চল বা বন্দরের নাম বলুন (যেমন দীঘা, কোচি, চেন্নাই)। স্থলভাগের এলাকা থেকে সরাসরি সমুদ্রে মাছ ধরার সুযোগ নেই।",
        "bn_en": "Location clarify kijiye: Samundar/ocean theke bhitorer region-e direct fishing access nei. Kripya ekta coastal location ba port-er naam bolo (jemon Digha ba Kochi).",
        "hi-Latn": "Kripya coastal location ya port ka naam batayein (jaise Kochi, Digha, Chennai, Mumbai). Inland ya non-coastal areas mein direct ocean fishing possible nahi hai."
    },
    "TEMPORAL_PFZ_GUIDANCE": {
        "en": "For tomorrow's fishing trip, use the PFZ information valid for tomorrow, not today's PFZ. Check tomorrow's weather and marine safety forecast before embarking.",
        "bn": "আগামীকালের মাছ ধরার যাত্রার জন্য আজকের নয়, আগামীকালের PFZ পূর্বাভাস অনুসরণ করুন। রওনা হওয়ার আগে আগামীকালের আবহাওয়া ও সমুদ্র নিরাপত্তা দেখে নিন।",
        "bn_en": "Kaler fishing trip er jonno ajker noy, kaler PFZ forecast follow korun. Jatra shuru korar age kaler weather ebong marine safety check kore nin.",
        "hi-Latn": "Kal ke fishing trip ke liye aaj ka nahi, kal ka PFZ forecast follow karein. Safar shuru karne se pehle kal ka weather aur marine safety zaroor check kar lein."
    },
    "COMPARE_FISHING_REGIONS": {
        "en": "{recommendation_text}",
        "bn": "{recommendation_text}",
        "bn_en": "{recommendation_text}",
        "hi-Latn": "{recommendation_text}"
    },
    "FISHING_SAFETY_TRADEOFF": {
        "en": "Safety always comes first. Even if the PFZ report shows fish aggregation, going out in unsafe sea conditions is not advisable. Wait for safe weather and sea conditions, then follow the PFZ advisory.",
        "bn": "নিরাপত্তাই প্রথম। মাছের উপস্থিতি থাকলেও সমুদ্র অশান্ত হলে যাওয়া উচিত নয়। আবহাওয়া ও সমুদ্র নিরাপদ হওয়ার পর মাছ ধরতে যান।",
        "bn_en": "Safety always comes first. PFZ report e mach thakleo jokhon sea unsafe, takhon jaoa uchit noy. Safe weather aar safe sea condition-e gele best result paben.",
        "hi-Latn": "Safety sabse pehle. Agar PFZ report mein machli ho, phir bhi unsafe samundar mein jana theek nahi. Safe weather aur safe sea condition mein hi fishing ke liye jaayein."
    },
    "GENERAL_MARINE_QUERY": {
        "en": "For general marine and fishing queries, please provide a specific coastal location or region so I can give you detailed safety and productivity information.",
        "bn": "সাধারণ সমুদ্র ও মাছ ধরা সংক্রান্ত প্রশ্নের জন্য, অনুগ্রহ করে নির্দিষ্ট কোনো উপকূল বা অঞ্চলের নাম বলুন।",
        "bn_en": "Samundro ba fishing niye general query-r jonno kripya ekta specific coastal location ba region-er naam bolun jate detailed safety information dite pari.",
        "hi-Latn": "Samundar ya fishing se jude general sawaalon ke liye, kripya kisi coastal location ya region ka naam batayein taaki main detailed safety aur productivity information de sakun."
    }
}

def _make_segments(parts: List[Tuple[str, List[str]]]) -> Tuple[str, List[Dict[str, Any]]]:
    segs = []
    text = ""
    for txt, agents in parts:
        text += txt
        if txt.strip():
            segs.append({
                "id": f"seg_{uuid.uuid4().hex[:8]}",
                "text": txt,
                "agents": agents
            })
    return text, segs

def generate_multilingual_response(recommendation_result: Dict[str, Any], language: str = "en", context: Dict[str, Any] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Translates structured decision payloads into natural, fisherman-tailored responses
    in English ('en'), Bengali ('bn'), or Bengalish ('bn_en'). Includes top-ranked candidate spots if available.
    Returns (text, segments)
    """
    lang = language.lower()
    if lang in ["bn-latn", "bn_en"]:
        lang = "bn-Latn"
    elif lang in ["hi-latn", "hi_en"]:
        lang = "hi-Latn"
    elif lang not in ["en", "bn"]:
        lang = "en"

    result_type = recommendation_result.get("result_type")
    
    # ---------------------------------------------------------
    # Specialized Fast Path Formatters (No LLM, completely deterministic)
    # ---------------------------------------------------------
    if result_type == "HAZARD_RESULT" and context:
        w_speed = recommendation_result.get("wind_speed_ms", "unknown")
        w_height = recommendation_result.get("wave_height_m", "unknown")
        
        has_cyclone = recommendation_result.get("hazards", {}).get("cyclone", False)
        has_lightning = recommendation_result.get("hazards", {}).get("lightning", False)
        warnings = recommendation_result.get("hazards", {}).get("warnings", [])
        
        active_hazards = []
        if has_cyclone: active_hazards.append("Cyclone")
        if has_lightning: active_hazards.append("Lightning")
        for w in warnings:
            if isinstance(w, str) and w not in ["CYCLONE", "LIGHTNING"]:
                active_hazards.append(w.replace("_", " ").title())
                
        is_safe = not active_hazards
        hazard_str = ", ".join(active_hazards) if active_hazards else "None"
        severity = "SEVERE" if has_cyclone or has_lightning else "MODERATE" if warnings else "NONE"
        
        loc_name = recommendation_result.get("location", {}).get("name", "Target location") if isinstance(recommendation_result.get("location"), dict) else "Target location"
        
        if is_safe:
            if lang == "bn":
                return _make_segments([
                    (f"{loc_name}: কোনো সক্রিয় সামুদ্রিক বিপদের সংকেত পাওয়া যায়নি। ", ["marine_safety", "risk"]),
                    (f"বর্তমানে বাতাসের গতি {w_speed} m/s এবং ঢেউয়ের উচ্চতা {w_height} m। ", ["weather", "ocean_state"]),
                    ("এই পরিস্থিতি নিরাপদ।", ["marine_safety", "risk"])
                ])
            elif lang == "bn-Latn":
                return _make_segments([
                    (f"{loc_name}: Kono active marine hazard detect hoyni. ", ["marine_safety", "risk"]),
                    (f"Ekhon wind speed {w_speed} m/s ar wave height {w_height} m. ", ["weather", "ocean_state"]),
                    ("Conditions safe.", ["marine_safety", "risk"])
                ])
            elif lang == "hi-Latn":
                return _make_segments([
                    (f"{loc_name}: Koi active marine hazard detect nahi hua. ", ["marine_safety", "risk"]),
                    (f"Abhi wind speed {w_speed} m/s aur wave height {w_height} m hai. ", ["weather", "ocean_state"]),
                    ("Conditions safe hain.", ["marine_safety", "risk"])
                ])
            else:
                return _make_segments([
                    (f"{loc_name}: No active marine hazards detected. ", ["marine_safety", "risk"]),
                    (f"Current wind is {w_speed} m/s and wave height is {w_height} m. ", ["weather", "ocean_state"]),
                    ("Conditions are safe.", ["marine_safety", "risk"])
                ])
        else:
            if lang == "bn":
                return _make_segments([
                    (f"{loc_name}: সতর্কতা! সক্রিয় বিপদের সংকেত: {hazard_str} (Severity: {severity})। ", ["marine_safety", "risk"]),
                    (f"বাতাসের গতি {w_speed} m/s এবং ঢেউয়ের উচ্চতা {w_height} m। ", ["weather", "ocean_state"]),
                    ("সমুদ্রে যাওয়া অনিরাপদ।", ["marine_safety", "risk"])
                ])
            elif lang == "bn-Latn":
                return _make_segments([
                    (f"{loc_name}: Alert! Active hazard: {hazard_str} (Severity: {severity}). ", ["marine_safety", "risk"]),
                    (f"Wind speed {w_speed} m/s ar wave height {w_height} m. ", ["weather", "ocean_state"]),
                    ("Sea conditions unsafe.", ["marine_safety", "risk"])
                ])
            elif lang == "hi-Latn":
                return _make_segments([
                    (f"{loc_name}: Alert! Active hazard: {hazard_str} (Severity: {severity}). ", ["marine_safety", "risk"]),
                    (f"Wind speed {w_speed} m/s aur wave height {w_height} m hai. ", ["weather", "ocean_state"]),
                    ("Samundar mein jana unsafe hai.", ["marine_safety", "risk"])
                ])
            else:
                return _make_segments([
                    (f"{loc_name}: Alert! Active hazards: {hazard_str} (Severity: {severity}). ", ["marine_safety", "risk"]),
                    (f"Current wind is {w_speed} m/s and wave height is {w_height} m. ", ["weather", "ocean_state"]),
                    ("The sea is considered unsafe.", ["marine_safety", "risk"])
                ])

    if result_type == "PFZ_RESULT" and context:
        pfz_summary = recommendation_result.get("pfz_summary", {})
        
        loc_dict = recommendation_result.get("location")
        if isinstance(loc_dict, dict) and loc_dict.get("name"):
            loc_name = loc_dict["name"]
        else:
            loc_name = "Target location"
        
        if recommendation_result.get("found"):
            cand = recommendation_result.get("candidate", {})
            dist = cand.get("distance_km", "Unknown")
            bearing = cand.get("bearing", "Unknown")
            lat = cand.get("latitude", "Unknown")
            lon = cand.get("longitude", "Unknown")
            prob = cand.get("probability")
            val = recommendation_result.get("validity_window")
            
            inc_dist = cand.get("incois_distance_km_range")
            inc_depth = cand.get("incois_depth_m_range")
            inc_dir = cand.get("incois_direction")
            landing_center = cand.get("landing_center")
            ref_name = landing_center if landing_center else loc_name
            source = recommendation_result.get("source", "Unknown")
            
            qualified = recommendation_result.get("qualified", False)
            threshold = recommendation_result.get("decision_threshold", 0.85) * 100
            
            prob_str = f"Estimated PFZ probability {int(prob * 100)}%" if prob is not None else ""
            val_str = f"forecast validity {val}" if val else ""
            prob_str_en = f"Estimated PFZ probability is {int(prob * 100)}%" if prob is not None else ""
            val_str_en = f"with a forecast validity of {val}" if val else ""
            
            # Combine them gracefully if either is present
            meta_parts = [p for p in [prob_str, val_str] if p]
            meta_str = ", ".join(meta_parts)
            meta_parts_en = [p for p in [prob_str_en, val_str_en] if p]
            meta_str_en = ", ".join(meta_parts_en)
            
            inc_str = ""
            if inc_dist:
                inc_str = f" [INCOIS: {inc_dist} km {inc_dir}, depth {inc_depth} m]"
            
            if qualified:
                if lang == "bn":
                    segs = [(f"{ref_name}-এর কাছাকাছি nearest PFZ: {loc_name} থেকে প্রায় {dist} km {bearing}-এ, location {lat}°N, {lon}°E।{inc_str} ", ["pfz", "geospatial"])]
                    if meta_str: segs.append((f"{meta_str}।", ["pfz"]))
                    return _make_segments(segs)
                elif lang == "bn-Latn" or lang == "bn_en":
                    segs = [(f"{ref_name}-r kachakachi nearest PFZ: {loc_name} theke praye {dist} km {bearing}-e, location {lat}°N, {lon}°E.{inc_str} ", ["pfz", "geospatial"])]
                    if meta_str: segs.append((f"{meta_str}.", ["pfz"]))
                    return _make_segments(segs)
                elif lang == "hi-Latn":
                    segs = [(f"{ref_name} ke paas nearest PFZ: {loc_name} se lagbhag {dist} km {bearing}, location {lat}°N, {lon}°E.{inc_str} ", ["pfz", "geospatial"])]
                    if meta_str: segs.append((f"{meta_str}.", ["pfz"]))
                    return _make_segments(segs)
                else:
                    segs = [(f"Nearest PFZ from {ref_name}: approximately {dist} km {bearing} of {loc_name}, at {lat}°N, {lon}°E.{inc_str} ", ["pfz", "geospatial"])]
                    if meta_str_en: segs.append((f"{meta_str_en}.", ["pfz"]))
                    return _make_segments(segs)
            else:
                prob_val = int(prob * 100) if prob is not None else 0
                if lang == "bn":
                    return _make_segments([
                        (f"{ref_name}-এর কাছাকাছি একটি PFZ সিগন্যাল পাওয়া গেছে: {loc_name} থেকে প্রায় {dist} km {bearing}-এ, location {lat}°N, {lon}°E{inc_str}, ", ["pfz", "geospatial"]),
                        (f"কিন্তু এটি {threshold:.0f}% qualification threshold অতিক্রম করেনি (সম্ভাবনা {prob_val}%)।", ["pfz"])
                    ])
                elif lang == "bn-Latn" or lang == "bn_en":
                    return _make_segments([
                        (f"{ref_name}-r kachakachi ekta PFZ signal pawa geche: {loc_name} theke praye {dist} km {bearing}-e, location {lat}°N, {lon}°E{inc_str}, ", ["pfz", "geospatial"]),
                        (f"kintu eta {threshold:.0f}% qualification threshold reach koreni (probability {prob_val}%).", ["pfz"])
                    ])
                elif lang == "hi-Latn":
                    return _make_segments([
                        (f"{ref_name} ke paas ek PFZ signal mila hai: {loc_name} se lagbhag {dist} km {bearing}, location {lat}°N, {lon}°E{inc_str}, ", ["pfz", "geospatial"]),
                        (f"par yeh {threshold:.0f}% qualification threshold tak nahi pohocha (probability {prob_val}%).", ["pfz"])
                    ])
                else:
                    return _make_segments([
                        (f"A PFZ signal from {ref_name} was detected approximately {dist} km {bearing} of {loc_name}, at {lat}°N, {lon}°E{inc_str}, ", ["pfz", "geospatial"]),
                        (f"but it did not reach the {threshold:.0f}% qualification threshold (probability {prob_val}%).", ["pfz"])
                    ])
        else:
            if lang == "bn" or lang == "bn-Latn":
                return _make_segments([(f"{loc_name}: Kachakachi kono Potential Fishing Zone (PFZ) pawa jayni.", ["pfz"])])
            elif lang == "hi-Latn":
                return _make_segments([(f"{loc_name}: Aas-paas koi Potential Fishing Zone (PFZ) nahi mila.", ["pfz"])])
            else:
                return _make_segments([(f"{loc_name}: No Potential Fishing Zone (PFZ) signal detected nearby.", ["pfz"])])
                
    if result_type == "SAFETY_FORECAST_RESULT" and context:
        clearance = recommendation_result.get("clearance", "UNKNOWN")
        reason = recommendation_result.get("reason", "")
        time_period = recommendation_result.get("time_period", "today")
        w_speed = recommendation_result.get("wind_speed_ms", "unknown")
        w_height = recommendation_result.get("wave_height_m", "unknown")
        warnings = recommendation_result.get("warnings", [])
        
        warn_str = ", ".join(warnings) if warnings else "None"
        
        loc_name = recommendation_result.get("location", {}).get("name", "Target location") if isinstance(recommendation_result.get("location"), dict) else "Target location"
        
        if clearance in ["CLEARED", "CAUTION"]:
            if lang == "bn":
                return _make_segments([
                    (f"{loc_name}: {time_period} সমুদ্রযাত্রা {clearance}। কারণ: {reason}। ", ["safety_rules", "risk"]),
                    (f"Wind: {w_speed} m/s, Waves: {w_height} m, Warnings: {warn_str}।", ["weather", "ocean", "marine_safety"])
                ])
            elif lang == "bn-Latn":
                return _make_segments([
                    (f"{loc_name}: {time_period} voyage is {clearance}. Reason: {reason}. ", ["safety_rules", "risk"]),
                    (f"Wind: {w_speed} m/s, Waves: {w_height} m, Warnings: {warn_str}.", ["weather", "ocean", "marine_safety"])
                ])
            elif lang == "hi-Latn":
                return _make_segments([
                    (f"{loc_name}: {time_period} safar ke liye condition {clearance} hai. Reason: {reason}. ", ["safety_rules", "risk"]),
                    (f"Wind: {w_speed} m/s, Waves: {w_height} m, Warnings: {warn_str}.", ["weather", "ocean", "marine_safety"])
                ])
            else:
                return _make_segments([
                    (f"{loc_name}: {time_period} voyage is {clearance}. Reason: {reason}. ", ["safety_rules", "risk"]),
                    (f"Wind is {w_speed} m/s, Waves are {w_height} m. Active warnings: {warn_str}.", ["weather", "ocean", "marine_safety"])
                ])
        else:
            if lang == "bn":
                return _make_segments([
                    (f"{loc_name}: {time_period} সমুদ্রযাত্রা {clearance}। কারণ: {reason}। ", ["safety_rules", "risk"]),
                    (f"Wind: {w_speed} m/s, Waves: {w_height} m, Warnings: {warn_str}। ", ["weather", "ocean", "marine_safety"]),
                    ("যাত্রা বাতিল করার পরামর্শ দেওয়া হচ্ছে।", ["safety_rules"])
                ])
            elif lang == "bn-Latn":
                return _make_segments([
                    (f"{loc_name}: {time_period} voyage is {clearance}. Reason: {reason}. ", ["safety_rules", "risk"]),
                    (f"Wind: {w_speed} m/s, Waves: {w_height} m, Warnings: {warn_str}. ", ["weather", "ocean", "marine_safety"]),
                    ("Jatra batil kora bhalo.", ["safety_rules"])
                ])
            elif lang == "hi-Latn":
                return _make_segments([
                    (f"{loc_name}: {time_period} safar ke liye condition {clearance} hai. Reason: {reason}. ", ["safety_rules", "risk"]),
                    (f"Wind: {w_speed} m/s, Waves: {w_height} m, Warnings: {warn_str}. ", ["weather", "ocean", "marine_safety"]),
                    ("Safar cancel karne ki salah di jati hai.", ["safety_rules"])
                ])
            else:
                return _make_segments([
                    (f"{loc_name}: {time_period} voyage is {clearance}. Reason: {reason}. ", ["safety_rules", "risk"]),
                    (f"Wind is {w_speed} m/s, Waves are {w_height} m. Active warnings: {warn_str}. ", ["weather", "ocean", "marine_safety"]),
                    ("It is highly recommended to cancel your voyage.", ["safety_rules"])
                ])

    if result_type == "FISHING_IMPACT_RESULT" and context:
        conds = recommendation_result.get("conditions", {})
        w_speed = conds.get("wind_speed_ms")
        sst = conds.get("sst_c")
        loc_name = recommendation_result.get("location", {}).get("name", "Target location") if isinstance(recommendation_result.get("location"), dict) else "Target location"
        
        if sst is not None and w_speed is not None and sst != "unknown" and w_speed != "unknown":
            if lang == "bn":
                return _make_segments([
                    ("এখানকার পরিবেশগত কারণে মাছ কম পাওয়া যেতে পারে। ", []),
                    (f"বর্তমানে SST {sst}°C এবং বাতাসের গতি {w_speed} m/s, ", ["ocean_state", "weather"]),
                    ("যা কিছু প্রজাতির জন্য অনুকূল নয়।", [])
                ])
            elif lang == "bn-Latn":
                return _make_segments([
                    ("Ekhankar poribeshgot karon mach kom pawa jete pare. ", []),
                    (f"Bortomane SST {sst}°C ebong batasher goti {w_speed} m/s ache, ", ["ocean_state", "weather"]),
                    ("ja kichhu projatir jonno onukul noy.", [])
                ])
            elif lang == "hi-Latn":
                return _make_segments([
                    ("Yahan environmental conditions ki wajah se machhli kam mil sakti hai. ", []),
                    (f"Abhi SST {sst}°C aur wind speed {w_speed} m/s hai, ", ["ocean_state", "weather"]),
                    ("jo kuch species ke liye theek nahi hai.", [])
                ])
            else:
                return _make_segments([
                    ("Environmental factors may be contributing to poor catch here. ", []),
                    (f"Current SST is {sst}°C and wind speed is {w_speed} m/s, ", ["ocean_state", "weather"]),
                    ("which may be unfavourable for certain species.", [])
                ])
        
    if result_type == "CONDITIONS_RESULT" and context:
        conds = recommendation_result.get("conditions", {})
        
        # Extract fields
        w_speed = conds.get("wind_speed_ms", "unknown")
        w_dir = conds.get("wind_direction", "unknown")
        w_height = conds.get("wave_height_m", "unknown")
        w_period = conds.get("wave_period_s", "unknown")
        sst = conds.get("sst_c", "unknown")
        current_vel = conds.get("surface_current_ms", "unknown")
        
        temp = conds.get("temperature_c", "unknown")
        cloud = conds.get("cloud_cover_pct", "unknown")
        precip = conds.get("precipitation_m", "unknown")
        
        tide_status = conds.get("tide_status", "UNAVAILABLE")
        tide = conds.get("tide")
        
        loc_name = recommendation_result.get("location", {}).get("name", "Target location") if isinstance(recommendation_result.get("location"), dict) else "Target location"
        
        if tide_status == "AVAILABLE" and tide:
            t_phase = tide.get("current_phase", "unknown")
            t_high_time = tide.get("next_high", {}).get("time", "unknown")
            t_high_m = tide.get("next_high", {}).get("height_m", "unknown")
            
            tide_en = f"Tide: {t_phase}, with the next high tide at {t_high_time} at {t_high_m} m. "
            
            # Map rising/falling to Bengali/Hindi terms
            t_phase_bn = "jowar uthchhe" if "rising" in t_phase.lower() else ("bhata porchhe" if "falling" in t_phase.lower() else t_phase)
            tide_bn = f"জোয়ার {t_phase_bn}, পরবর্তী হাই টাইড {t_high_time}-এ {t_high_m} m। "
            tide_bn_en = f"{t_phase_bn}, porer high tide {t_high_time}-e, height {t_high_m} m. "
            
            t_phase_hi = "badh raha" if "rising" in t_phase.lower() else ("ghat raha" if "falling" in t_phase.lower() else t_phase)
            tide_hi = f"Tide {t_phase_hi} hai, next high tide {t_high_time} par {t_high_m} m. "
        else:
            tide_en = "Tide data is currently unavailable. "
            tide_bn = "জোয়ারের তথ্য বর্তমানে উপলব্ধ নেই। "
            tide_bn_en = "Tide data ekhon available nei. "
            tide_hi = "Tide data abhi available nahi hai. "
            
        precip_str_en = "no precipitation" if precip == 0 or precip == 0.0 else f"{precip} m precipitation"
        precip_str_bn_en = "brishti nei" if precip == 0 or precip == 0.0 else f"{precip} m brishti"
        precip_str_hi = "barish nahi hai" if precip == 0 or precip == 0.0 else f"{precip} m barish"
        
        curr_str_en = "negligible" if current_vel == 0 or current_vel == 0.0 else f"{current_vel} m/s"
        curr_str_bn_en = "khub kom" if current_vel == 0 or current_vel == 0.0 else f"{current_vel} m/s"
        curr_str_hi = "negligible hai" if current_vel == 0 or current_vel == 0.0 else f"{current_vel} m/s hai"
        
        if lang == "bn-Latn":
            return _make_segments([
                (f"{loc_name}-r kachhakachhi aaj samudrer obostha\n\n", []),
                (tide_bn_en, ["tide"]),
                (f"Weather {temp}°C, cloud cover {cloud}%, {precip_str_bn_en}. Wind {w_speed} m/s, direction {w_dir}°. ", ["weather"]),
                (f"Wave {w_height} m, period {w_period} s. SST {sst}°C, surface current {curr_str_bn_en}.", ["ocean_state"])
            ])
        elif lang == "hi-Latn":
            return _make_segments([
                (f"{loc_name} ke paas aaj samundar ki haalat\n\n", []),
                (tide_hi, ["tide"]),
                (f"Weather {temp}°C, cloud cover {cloud}%, {precip_str_hi}. Wind {w_speed} m/s, direction {w_dir}°. ", ["weather"]),
                (f"Wave {w_height} m, period {w_period} s. SST {sst}°C, surface current {curr_str_hi}.", ["ocean_state"])
            ])
        else:
            return _make_segments([
                (f"Marine conditions near {loc_name}\n\n", []),
                (tide_en, ["tide"]),
                (f"Weather: {temp}°C, {cloud}% cloud cover, {precip_str_en}. Wind: {w_speed} m/s from {w_dir}°. ", ["weather"]),
                (f"Waves: {w_height} m, period {w_period} s. SST: {sst}°C, surface current {curr_str_en}.", ["ocean_state"])
            ])

    if result_type == "WEATHER_RESULT" and context:
        conds = recommendation_result.get("conditions", {})
        loc_name = recommendation_result.get("location", {}).get("name", "Target location") if isinstance(recommendation_result.get("location"), dict) else "Target location"
        
        w_speed = conds.get("wind_speed_ms", "unknown")
        w_dir = conds.get("wind_direction", "unknown")
        temp = conds.get("temperature_c", "unknown")
        cloud = conds.get("cloud_cover_pct", "unknown")
        precip = conds.get("precipitation_m", "unknown")
        
        precip_str_en = "no precipitation" if precip == 0 or precip == 0.0 else f"{precip} m precipitation"
        precip_str_bn_en = "brishti nei" if precip == 0 or precip == 0.0 else f"{precip} m brishti"
        precip_str_hi = "barish nahi hai" if precip == 0 or precip == 0.0 else f"{precip} m barish"
        
        if lang == "bn-Latn":
            return _make_segments([
                (f"{loc_name}-r kachhakachhi aaj abohawa\n\n", []),
                (f"Temperature {temp}°C, cloud cover {cloud}%, {precip_str_bn_en}. Wind {w_speed} m/s, direction {w_dir}°.", ["weather"])
            ])
        elif lang == "hi-Latn":
            return _make_segments([
                (f"{loc_name} ke paas aaj mausam\n\n", []),
                (f"Temperature {temp}°C, cloud cover {cloud}%, {precip_str_hi}. Wind {w_speed} m/s, direction {w_dir}°.", ["weather"])
            ])
        else:
            return _make_segments([
                (f"Weather near {loc_name}\n\n", []),
                (f"{temp}°C, {cloud}% cloud cover, {precip_str_en}. Wind: {w_speed} m/s from {w_dir}°.", ["weather"])
            ])
            
    if result_type == "PRODUCTIVITY_RESULT":
        loc_name = recommendation_result.get("reference_location", {}).get("name", "Target location")
        regions = recommendation_result.get("regions", [])
        
        if not regions:
            if lang == "bn" or lang == "bn-Latn":
                return _make_segments([(f"{loc_name}-r kachakachi kono productive region paowa jayni.", ["productivity"])])
            elif lang == "hi-Latn":
                return _make_segments([(f"{loc_name} ke paas koi productive region nahi mila.", ["productivity"])])
            else:
                return _make_segments([(f"No highly productive regions were found near {loc_name}.", ["productivity"])])
                
        best_region = regions[0]
        dist = best_region.get("distance_km", "unknown")
        dir_str = best_region.get("direction", "unknown")
        chlo = best_region.get("chlorophyll_mg_m3", "unknown")
        sst = best_region.get("sst_c", "unknown")
        
        if lang == "bn" or lang == "bn-Latn":
            return _make_segments([
                (f"{loc_name}-r kachakachi sobcheye favourable region holo praye {dist} km {dir_str}-e, ", ["productivity", "geospatial"]),
                (f"jekhane chlorophyll {chlo} mg/m³ ar SST {sst}°C.", ["productivity", "ocean_state"])
            ])
        elif lang == "hi-Latn":
            return _make_segments([
                (f"{loc_name} ke paas sabse favourable region lagbhag {dist} km {dir_str} mein hai, ", ["productivity", "geospatial"]),
                (f"jahan chlorophyll {chlo} mg/m³ aur SST {sst}°C hai.", ["productivity", "ocean_state"])
            ])
        else:
            return _make_segments([
                (f"Near {loc_name}, the most favourable region is approximately {dist} km {dir_str}, ", ["productivity", "geospatial"]),
                (f"with chlorophyll of {chlo} mg/m³ and SST of {sst}°C.", ["productivity", "ocean_state"])
            ])

    action_code = recommendation_result.get("decision", recommendation_result.get("action_code", "CLEAR_WEATHER_LOW_YIELD"))
    pfz_summary = recommendation_result.get("pfz_summary", {})
    pfz_prob = pfz_summary.get("pfz_probability")
    if pfz_prob is None:
        pfz_prob = 0.0
    pfz_prob_pct = int(round(pfz_prob * 100))

    raw_lat = recommendation_result.get("latitude")
    if raw_lat is None:
        raw_lat = recommendation_result.get("location", {}).get("latitude") if isinstance(recommendation_result.get("location"), dict) else None
    lat = float(raw_lat) if raw_lat is not None else 12.48

    raw_lon = recommendation_result.get("longitude")
    if raw_lon is None:
        raw_lon = recommendation_result.get("location", {}).get("longitude") if isinstance(recommendation_result.get("location"), dict) else None
    lon = float(raw_lon) if raw_lon is not None else 74.40

    # Handle geography distance to coast parameters
    raw_dist = recommendation_result.get("distance_km")
    if raw_dist is None:
        raw_dist = recommendation_result.get("why", {}).get("distance_km", 0.0)
    dist_km = float(raw_dist) if raw_dist is not None else 0.0

    nearest_coast = recommendation_result.get("nearest_coast_name")
    if not nearest_coast:
        nearest_coast = recommendation_result.get("why", {}).get("nearest_coast_name", "Coast")

    templates = RESPONSE_TEMPLATES.get(action_code, RESPONSE_TEMPLATES.get("CLEAR_WEATHER_LOW_YIELD"))
    template_str = templates.get(lang, templates["en"])

    try:
        base_response = template_str.format(
            pfz_prob_pct=pfz_prob_pct, lat=lat, lon=lon, dist_km=dist_km, nearest_coast=nearest_coast,
            recommendation_text=recommendation_result.get("recommendation_text", "")
        )
    except Exception:
        base_response = recommendation_result.get("recommendation_text", template_str)
        
    segs = [(base_response, ["rules"])]

    # Append top-ranked fishing candidate spot information if candidate spots exist
    ranked_spots = recommendation_result.get("ranked_candidate_spots", [])
    if ranked_spots and action_code not in ["CANCEL_VOYAGE", "UNSUPPORTED_LOCATION", "NEEDS_CLARIFICATION"]:
        spot_strings = []
        for s in ranked_spots[:3]:
            dist_km = s.get("distance_km", 0.0)
            direction = s.get("compass_direction", "CENTER")
            p = s.get("pfz_probability")
            b_prob_pct = int(round((p if p is not None else 0.0) * 100))
            rank = s.get("rank", 1)
            if dist_km == 0.0 and direction == "CENTER":
                continue
            if lang == "bn":
                spot_strings.append(f"স্পট #{rank}: {dist_km:.1f} কিমি {direction} ({b_prob_pct}%)")
            elif lang == "bn_en":
                spot_strings.append(f"Spot #{rank}: {dist_km:.1f} km {direction} ({b_prob_pct}%)")
            elif lang == "hi-Latn":
                spot_strings.append(f"Spot #{rank}: {dist_km:.1f} km {direction} ({b_prob_pct}%)")
            else:
                spot_strings.append(f"Spot #{rank}: {dist_km:.1f} km {direction} ({b_prob_pct}%)")

        if spot_strings:
            spot_text = ""
            if lang == "bn":
                spot_text = f" সেরা মাছের স্থান: {', '.join(spot_strings)}।"
            elif lang == "bn_en":
                spot_text = f" Top recommended spots: {', '.join(spot_strings)}."
            elif lang == "hi-Latn":
                spot_text = f" Sabse behtar spots: {', '.join(spot_strings)}."
            else:
                spot_text = f" Top recommended spots: {', '.join(spot_strings)}."
            segs.append((spot_text, ["pfz", "geospatial", "rules"]))

    # Phase 3: Add 'why' explanations
    why_data = recommendation_result.get("why", {})
    if why_data:
        primary_reason = why_data.get("primary_reason", "")
        if primary_reason:
            reason_text = ""
            if lang == "bn":
                reason_text = f" কারণ: {primary_reason}"
            elif lang == "bn_en":
                reason_text = f" Karon: {primary_reason}"
            elif lang == "hi-Latn":
                reason_text = f" Kaaran: {primary_reason}"
            else:
                reason_text = f" Reason: {primary_reason}"
            segs.append((reason_text, ["rules"]))

    return _make_segments(segs)

