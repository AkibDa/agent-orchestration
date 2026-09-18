import re
with open("orchestrator/handlers.py") as f:
    text = f.read()
    match = re.search(r"class MarineConditionsHandler", text)
    if match:
        start = max(0, match.start() - 100)
        end = min(len(text), match.start() + 2000)
        print(text[start:end])
