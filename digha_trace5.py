import sys
import os
import asyncio

os.environ["ORCA_PFZ_SOURCE"] = "live"

from conversation.response_generator import generate_multilingual_response

recommendation_result = {
    "result_type": "PFZ_RESULT",
    "location": {"name": "Digha"},
    "found": True,
    "qualified": True,
    "candidate": {
        "distance_km": 25.6,
        "bearing": "S",
        "latitude": 21.4119,
        "longitude": 87.5983,
        "probability": None,
        "validity_window": None,
        "incois_distance_km_range": "22-27",
        "incois_depth_m_range": "6-11",
        "incois_direction": "SE"
    },
    "source": "INCOIS_LIVE",
    "decision_threshold": 0.85
}

for lang in ["en", "bn", "bn_en", "hi-Latn"]:
    text, seg = generate_multilingual_response(recommendation_result, language=lang, context={"pfz": {}})
    print(f"--- {lang} ---")
    print(text)

