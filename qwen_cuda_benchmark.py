import csv
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTO_DIR = PROJECT_ROOT / "Proto"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(PROTO_DIR))

from Proto.conversation.model import get_conversation_model
from Proto.conversation.router import llm_route
from qwen_location import extract_location

from tests.benchmark_cases import TEST_CASES


def run_case(case, model):
    start = time.perf_counter()

    try:
        action, plan, extraction = llm_route(
            case["query"],
            model,
            extract_location,
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000

        print()
        print("=" * 70)
        print("QWEN EXTRACTION ERROR")
        print("=" * 70)
        print(f"Query: {case['query']}")
        print(f"Error: {e}")
        print()

        if getattr(model, "last_extract_stats", None):
            print("RAW QWEN OUTPUT:")
            print(model.last_extract_stats.raw_output)
            print()
            print(
                f"Prompt tokens:     {model.last_extract_stats.prompt_tokens}"
            )
            print(
                f"Completion tokens: {model.last_extract_stats.completion_tokens}"
            )
            print(
                f"Qwen latency:      {model.last_extract_stats.latency_sec * 1000:.2f} ms"
            )

        print("=" * 70)
        print()

        return {
            "id": case["id"],
            "category": case["category"],
            "query": case["query"],
            "expected_intent": case["intent"],
            "predicted_intent": "ERROR",
            "intent_pass": False,
            "expected_location": case["location"],
            "predicted_location": None,
            "location_pass": False,
            "expected_language": case["language"],
            "predicted_language": None,
            "language_pass": False,
            "expected_time": case["time"],
            "predicted_time": None,
            "time_pass": False,
            "expected_activity": case["activity"],
            "predicted_activity": None,
            "activity_pass": False,
            "action": "ERROR",
            "agents": "",
            "latency_ms": round(latency_ms, 3),
            "overall_pass": False,
        }

    latency_ms = (time.perf_counter() - start) * 1000

    # plan may be None if action is CHAT (no location / unknown intent)
    if plan is not None:
        predicted_intent = plan.intent
        predicted_location = (
            plan.location.name
            if plan.location is not None
            else None
        )
        predicted_time = (
            plan.time.relative
            if plan.time is not None
            else None
        )
        predicted_language = plan.language
        predicted_activity = plan.activity
        predicted_agents = "|".join(plan.agents)
    else:
        predicted_intent = extraction.intent.value if hasattr(extraction.intent, "value") else str(extraction.intent)
        predicted_location = None
        predicted_time = None
        predicted_language = extraction.language.value if hasattr(extraction.language, "value") else str(extraction.language)
        predicted_activity = None if extraction.activity == "none" else extraction.activity
        predicted_agents = ""

    intent_pass = predicted_intent == case["intent"]

    location_pass = (
        predicted_location == case["location"]
    )

    language_pass = (
        case["language"] is None
        or predicted_language == case["language"]
    )

    time_pass = (
        case["time"] is None
        or predicted_time == case["time"]
    )

    activity_pass = (
        case["activity"] is None
        or predicted_activity == case["activity"]
    )

    overall_pass = all([
        intent_pass,
        location_pass,
        language_pass,
        time_pass,
        activity_pass,
    ])

    return {
        "id": case["id"],
        "category": case["category"],
        "query": case["query"],

        "expected_intent": case["intent"],
        "predicted_intent": predicted_intent,
        "intent_pass": intent_pass,

        "expected_location": case["location"],
        "predicted_location": predicted_location,
        "location_pass": location_pass,

        "expected_language": case["language"],
        "predicted_language": predicted_language,
        "language_pass": language_pass,

        "expected_time": case["time"],
        "predicted_time": predicted_time,
        "time_pass": time_pass,

        "expected_activity": case["activity"],
        "predicted_activity": predicted_activity,
        "activity_pass": activity_pass,

        "action": action,
        "agents": predicted_agents,

        "latency_ms": round(latency_ms, 3),
        "overall_pass": overall_pass,
    }

def main():
    print("Loading Qwen3-4B model (auto-detecting backend)...")
    model = get_conversation_model()
    backend = model.backend.upper()

    print("Running warm-up inference...")

    llm_route(
        "Digha te kal fishing safe hobe?",
        model,
        extract_location,
    )

    print()
    print(f"Running {len(TEST_CASES)} benchmark queries...")
    print()

    results = []

    for case in TEST_CASES:
        result = run_case(case, model)
        results.append(result)

        status = "PASS" if result["overall_pass"] else "FAIL"

        print(
            f"[{status}] {result['id']} "
            f"{result['latency_ms']:.2f} ms | "
            f"{result['category']}"
        )

        print(f"  Query: {result['query']}")
        print(
            f"  Intent: {result['expected_intent']} "
            f"-> {result['predicted_intent']}"
        )
        print(
            f"  Location: {result['expected_location']} "
            f"-> {result['predicted_location']}"
        )
        print(
            f"  Language: {result['expected_language']} "
            f"-> {result['predicted_language']}"
        )
        print(
            f"  Time: {result['expected_time']} "
            f"-> {result['predicted_time']}"
        )
        print(
            f"  Activity: {result['expected_activity']} "
            f"-> {result['predicted_activity']}"
        )
        print()

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    total = len(results)

    intent_accuracy = (
        sum(r["intent_pass"] for r in results) / total * 100
    )

    location_accuracy = (
        sum(r["location_pass"] for r in results) / total * 100
    )

    language_accuracy = (
        sum(r["language_pass"] for r in results) / total * 100
    )

    time_accuracy = (
        sum(r["time_pass"] for r in results) / total * 100
    )

    activity_accuracy = (
        sum(r["activity_pass"] for r in results) / total * 100
    )

    overall_accuracy = (
        sum(r["overall_pass"] for r in results) / total * 100
    )

    latencies = sorted(r["latency_ms"] for r in results)

    avg_latency = sum(latencies) / total

    p50_latency = latencies[int(0.50 * (total - 1))]

    p95_latency = latencies[int(0.95 * (total - 1))]

    print("=" * 70)
    print(f"QWEN3-4B {backend} BENCHMARK SUMMARY")
    print("=" * 70)

    print(f"Total queries:      {total}")
    print(f"Overall accuracy:   {overall_accuracy:.2f}%")
    print(f"Intent accuracy:    {intent_accuracy:.2f}%")
    print(f"Location accuracy:  {location_accuracy:.2f}%")
    print(f"Language accuracy:  {language_accuracy:.2f}%")
    print(f"Time accuracy:      {time_accuracy:.2f}%")
    print(f"Activity accuracy:  {activity_accuracy:.2f}%")
    print()

    print(f"Average latency:    {avg_latency:.2f} ms")
    print(f"P50 latency:        {p50_latency:.2f} ms")
    print(f"P95 latency:        {p95_latency:.2f} ms")
    print()

    # ---------------------------------------------------------
    # Save detailed results
    # ---------------------------------------------------------

    output_path = PROJECT_ROOT / "Proto" / "qwen_cuda_benchmark_results.csv"

    fieldnames = list(results[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()