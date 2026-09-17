# conversation/normalize.py

from schemas.extraction import ExtractionResult, Intent, LocationItem
import re
from location.resolver import LocationResolver

TEMPORAL_TOKENS = {
    "kal", "aaj", "ajke", "tomorrow", "yesterday", "today", "next week",
    "kal ke", "kaler", "parso", "subah", "raat", "din"
}

CONCEPT_TOKENS = {
    "pfz", "weather", "sea", "ocean", "samundar", "somundro", "mach",
    "fish", "fishing", "pani", "jol", "wave", "wind", "hawa",
    "safe", "unsafe", "nirapod", "bhalo", "kharap"
}

POSTPOSITIONS = [
    " te jawa", "te jawa", " er kache", " r kache", " kache", " kachhe",
    " theke", " er", " r", " e", " te", " ke paas", " mein"
]

def clean_location_text(text: str) -> str:
    if not text:
        return text
    clean_text = text.lower().strip()
    for pos in POSTPOSITIONS:
        if clean_text.endswith(pos):
            clean_text = clean_text[:-len(pos)].strip()
    # Basic title casing for the cleaned text, retaining original case if possible
    if len(clean_text) < len(text):
         return clean_text.title()
    return text.strip()


def normalize_extraction(result: ExtractionResult) -> ExtractionResult:
    """
    Normalizes raw LLM extraction before it hits the state or planner.
    - Preserves valid intents
    - Filters temporal words misclassified as locations
    - Cleans postpositions from location text
    """
    valid_locations = []
    found_temporal = None
    
    if hasattr(result, "locations") and result.locations:
        for loc in result.locations:
            text_lower = loc.text.lower().strip()
            
            # Temporal token filter
            matched_temporal = None
            for t in TEMPORAL_TOKENS:
                if t == text_lower or text_lower.startswith(t + " ") or text_lower.endswith(" " + t):
                    matched_temporal = t
                    break
                    
            if matched_temporal:
                if matched_temporal in ["kal", "tomorrow", "kaler", "kal ke", "parso"]:
                    found_temporal = "tomorrow"
                elif matched_temporal in ["aaj", "ajke", "today"]:
                    found_temporal = "today"
                
                # If the location is ONLY the temporal token, discard it immediately
                if len(text_lower.strip().split()) <= 2:
                    continue
                # Else it's a clause starting/ending with a temporal token, we will let strict validation handle discarding it
                
            # Concept token filter
            NON_LOCATION_CONCEPTS = {"pfz", "weather", "sea", "ocean", "wind", "wave", "fishing"}
            if any(t in text_lower.split() for t in NON_LOCATION_CONCEPTS) or any(t == text_lower for t in CONCEPT_TOKENS):
                continue
                
            # Cleanup postpositions
            cleaned = clean_location_text(loc.text)
            if not cleaned:
                continue

            # Strict location validation
            word_count = len(re.findall(r"\b\w+\b", cleaned))
            is_valid_location = False
            
            # If it's a massive clause, reject it immediately unless it's a known region
            if word_count <= 5 or any(r in cleaned.lower() for r in ["bay of bengal", "arabian sea", "indian ocean"]):
                resolver = LocationResolver()
                res = resolver.resolve(cleaned)
                if res.resolution_status != "NOT_FOUND":
                    is_valid_location = True
            
            if is_valid_location:
                # Retain original role
                valid_locations.append(LocationItem(text=cleaned, role=loc.role))
            else:
                # If rejected, scan the rejected text for temporal keywords to preserve time intent
                for t in TEMPORAL_TOKENS:
                    if t in text_lower:
                        if t in ["kal", "tomorrow", "kaler", "kal ke", "parso"]:
                            found_temporal = "tomorrow"
                        elif t in ["aaj", "ajke", "today"]:
                            found_temporal = "today"
                        
                # Scan for time (e.g. "4ter shomoy", "4 pm")
                time_match = re.search(r"\b(\d{1,2})\s*(?:tay|baje|ter|am|pm|o'clock)\b", text_lower)
                if time_match:
                    # We will store this implicitly by letting the router's extract_time_details pick it up from raw_query,
                    # but we definitely need to discard this as a location.
                    pass
                
    result.locations = valid_locations
    
    # If we recovered a time from a bad location, apply it
    if found_temporal:
        if not result.time_relative or result.time_relative == "none":
            result.time_relative = found_temporal

    return result
