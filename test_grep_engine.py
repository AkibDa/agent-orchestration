import re
with open("orchestrator/engine.py") as f:
    text = f.read()
    match = re.search(r"ROUTE_TO_FISHING_AREA", text)
    if match:
        start = max(0, match.start() - 500)
        end = min(len(text), match.end() + 1500)
        print(text[start:end])
