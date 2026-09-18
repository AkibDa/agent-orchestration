import re

with open("conversation/router.py", "r") as f:
    content = f.read()

replacement = """  if action == "CLARIFY":
      # Do not increment clarify rounds if we are still missing the destination for a route
      if not (plan.operation == "ROUTE_TO_FISHING_AREA" and plan.clarification_reason == "MISSING_DESTINATION"):
          state.clarify_rounds += 1"""

content = re.sub(
    r'  if action == "CLARIFY":\n      state.clarify_rounds \+= 1',
    replacement,
    content
)

with open("conversation/router.py", "w") as f:
    f.write(content)
