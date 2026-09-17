import sys
import os
import asyncio

os.environ["ORCA_PFZ_SOURCE"] = "live"

from conversation.response_generator import generate_multilingual_response

recommendation_result = {
    "result_type": "PFZ_RESULT",
    "location": {"name": "Kochi"},
    "found": True,
    "qualified": True,
    "candidate": {
        "distance_km": 365.7,
        "bearing": "NW",
        "latitude": 12.7419,
        "longitude": 74.5247,
        "probability": None,
        "validity_window": None,
        "incois_distance_km_range": "35-40",
        "incois_depth_m_range": "49-54",
        "incois_direction": "W",
        "landing_center": "Kunzhathur"
    },
    "source": "INCOIS_LIVE",
    "decision_threshold": 0.85
}

for lang in ["en", "bn", "bn_en", "hi-Latn"]:
    text, seg = generate_multilingual_response(recommendation_result, language=lang, context={"pfz": {}})
    print(f"--- {lang} ---")
    print(text)

