import sys
import os
from pathlib import Path

# Fix sys.path for backend imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PARENT_ROOT = PROJECT_ROOT.parent
if str(PARENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARENT_ROOT))

import asyncio
from conversation.model import get_conversation_model
from conversation.router import llm_route, llm_route_stateful
from conversation.state import ConversationState
from location.resolver import extract_location
from orchestrator.engine import OrcaOrchestrator
from chat import build_engine

def main():
    print("Loading ORCA Models...")
    conv_model = get_conversation_model()
    engine = build_engine()

    queries = {
        "English": "Give me a safe route from Digha to coordinates 21.0, 88.0",
        "Bengali": "দীঘা থেকে ২১.০, ৮৮.০ স্থানাঙ্কে যাওয়ার একটি নিরাপদ রুট দিন",
        "Bengalish": "Digha theke 21.0, 88.0 e jaoar ekta safe route din",
        "Hinglish": "Digha se 21.0, 88.0 tak ek safe route batao"
    }

    results = {}

    for lang_name, query in queries.items():
        print(f"\nTesting {lang_name} query: {query}")
        
        state = ConversationState()
        action, plan, extraction, timings = llm_route_stateful(query, conv_model, extract_location, state)
        
        print(f"Detected Intent: {plan.intent}")
        print(f"Detected Language: {plan.language}")
        print(f"Resolved Location: {plan.location.name if plan.location else 'None'}")
        
        exec_result = engine.run(plan)
        
        rec_text = exec_result['recommendation'].get('recommendation_text', '')
        print(f"Recommendation Text: {rec_text}")
        
        results[lang_name] = {
            "intent": plan.intent,
            "location": plan.location.name if plan.location else None,
            "distance": exec_result['recommendation'].get('why', {}).get('routing_metrics', {}).get('total_distance_km'),
            "text": rec_text
        }

    print("\n--- Regression Test Summary ---")
    base = results["English"]
    passed = True
    for lang, res in results.items():
        if lang == "English":
            continue
        
        if res["intent"] != base["intent"]:
            print(f"FAIL: {lang} intent {res['intent']} != {base['intent']}")
            passed = False
        if res["location"] != base["location"]:
            print(f"FAIL: {lang} location {res['location']} != {base['location']}")
            passed = False
        if res["distance"] != base["distance"]:
            print(f"FAIL: {lang} route distance {res['distance']} != {base['distance']}")
            passed = False
            
        print(f"{lang} Text: {res['text']}")

    if passed:
        print("\nSUCCESS: All regressions passed! Semantic intent and geographic results remained invariant.")
    else:
        print("\nFAILED: Regressions broke semantic invariance.")

if __name__ == "__main__":
    main()
