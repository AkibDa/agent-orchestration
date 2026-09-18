import re
with open("orchestrator/handlers.py") as f:
    text = f.read()
    match = re.search(r"class MarineConditionsHandler", text)
    if match:
        start = match.start()
        print(text[start:start+4000])
