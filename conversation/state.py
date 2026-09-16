from dataclasses import dataclass, field
from typing import Optional, List


def _clean(value) -> Optional[str]:
  """Normalize None / "none" / "null" / "" all to real None.
  (Catches the literal string "null" bug seen in test 6's output —
  some outlines/schema paths emit the string "null" instead of JSON null.)"""
  if value is None:
    return None
  s = str(value).strip().lower()
  if s in ("none", "null", ""):
    return None
  return value


@dataclass
class ConversationState:
  language: Optional[str] = None
  intent: Optional[str] = None
  action_type: Optional[str] = None
  explanation_target: Optional[str] = None
  location_text: Optional[str] = None
  reference_location_text: Optional[str] = None
  target_location_text: Optional[str] = None
  comparison_location_texts: List[str] = field(default_factory=list)
  region_locations: List[str] = field(default_factory=list)
  location_role: Optional[str] = None
  activity: Optional[str] = None
  time_relative: Optional[str] = None
  time_offset_days: Optional[int] = None
  time_period: Optional[str] = None
  user_constraint: Optional[str] = None
  count: int = 1

  pending_clarification: Optional[str] = None
  clarification_context: Optional[dict] = None
  previous_query: Optional[str] = None
  previous_clarifications: List[str] = field(default_factory=list)
  clarify_rounds: int = 0

  MAX_CLARIFY_ROUNDS = 2  # hard cap so a demo never loops forever

  def known_summary(self) -> str:
    fields = {
      "location": self.location_text,
      "reference_location": self.reference_location_text,
      "target_location": self.target_location_text,
      "comparison_locations": self.comparison_location_texts,
      "region_locations": self.region_locations,
      "intent": self.intent,
      "activity": self.activity,
      "time_relative": self.time_relative,
      "time_period": self.time_period,
    }
    known = {k: v for k, v in fields.items() if _clean(v)}
    return ", ".join(f"{k}={v}" for k, v in known.items()) or "nothing yet"

  def merge(self, extraction) -> None:
    """Smart merge:
    - If answering a pending clarification, fill-forward missing fields.
    - If new query is marine_geography or chat/explain, reset operational fields (activity, time, constraints).
    - If intent shifts between operational domains without a pending clarification, reset unmentioned fields.
    """
    new_intent = _clean(getattr(extraction.intent, "value", extraction.intent))
    new_action_type = _clean(getattr(extraction, "action_type", None))
    is_geography = new_intent == "marine_geography" or (new_action_type == "EXPLAIN" and _clean(getattr(extraction, "explanation_target", None)) != "fishing_availability")
    intent_changed = new_intent and self.intent and new_intent != self.intent

    def pick(new, old):
      new = _clean(new)
      return new if new is not None else old

    # Check if this is an explicit follow-up answering clarification
    is_continuation = bool(self.pending_clarification)

    # Use Qwen's detected language if available, else fallback to deterministic
    if hasattr(extraction, "language") and extraction.language:
        self.language = getattr(extraction.language, "value", extraction.language)
        
    # Do not overwrite intent with unknown if it's a continuation answering a clarification
    if is_continuation and new_intent == "unknown":
        pass
    else:
        self.intent = new_intent if new_intent else self.intent
        
    if hasattr(extraction, "action_type") and extraction.action_type:
      self.action_type = getattr(extraction.action_type, "value", extraction.action_type)

    locs_list = getattr(extraction, "locations", [])
    if not locs_list and getattr(extraction, "location_text", None):
        locs_list = [{"text": extraction.location_text, "role": getattr(extraction, "location_role", None)}]
        
    # Categorize location semantics cleanly
    for loc_item in locs_list:
      loc_txt = _clean(loc_item.get("text") if isinstance(loc_item, dict) else getattr(loc_item, "text", None))
      if not loc_txt: continue
      
      loc_item_role = loc_item.get("role") if isinstance(loc_item, dict) else getattr(loc_item, "role", None)
      loc_role = getattr(loc_item_role, "value", loc_item_role)
      loc_role = str(loc_role).upper() if loc_role else "TARGET"
      l_low = loc_txt.lower()
      is_region = l_low in ("bay of bengal", "arabian sea", "indian ocean") or loc_role == "REGION"
      
      if is_region:
        if loc_txt.title() not in self.region_locations:
          self.region_locations.append(loc_txt.title())
        self.location_role = "REGION"
      elif loc_role == "REFERENCE" or l_low in ("mp", "madhya pradesh", "kolkata", "bengaluru", "pune", "ranchi", "howrah", "sealdah"):
        self.reference_location_text = loc_txt
        if not self.location_text:
          self.location_role = "REFERENCE"
      else:
        # If we receive multiple targets in the same turn, it's a comparison.
        # But if we receive a single explicit target, it MUST overwrite the previous one.
        if loc_txt != self.target_location_text:
            if len(locs_list) > 1:
                # Comparison mode
                if self.target_location_text and self.target_location_text not in self.comparison_location_texts:
                    self.comparison_location_texts.append(self.target_location_text)
                if loc_txt not in self.comparison_location_texts:
                    self.comparison_location_texts.append(loc_txt)
                self.target_location_text = loc_txt # Keep the latest as target too
            else:
                # Absolute override
                self.target_location_text = loc_txt
                self.comparison_location_texts = []
        else:
            self.target_location_text = loc_txt
        
        # Do not overwrite location_text if it's currently a REFERENCE
        if self.location_role != "REFERENCE":
            self.location_text = loc_txt
            self.location_role = loc_role

    # Activity, time, constraint handling
    if is_geography:
      # Reset operational contamination
      self.activity = None
      self.time_relative = None
      self.time_offset_days = None
      self.time_period = None
      self.user_constraint = None
      self.explanation_target = _clean(getattr(extraction, "explanation_target", None))
    elif intent_changed and not is_continuation:
      # Topic shifted: use only new extraction values
      self.activity = _clean(extraction.activity)
      self.time_relative = _clean(extraction.time_relative)
      self.time_offset_days = getattr(extraction, "time_offset_days", None)
      self.time_period = _clean(extraction.time_period)
      self.user_constraint = _clean(getattr(extraction, "user_constraint", None))
      self.explanation_target = _clean(getattr(extraction, "explanation_target", None))
      self.count = getattr(extraction, "count", 1)
    else:
      # Continuation or same intent: pick new if present, keep old if missing
      self.activity = pick(extraction.activity, self.activity)
      self.time_relative = pick(extraction.time_relative, self.time_relative)
      self.time_offset_days = pick(getattr(extraction, "time_offset_days", None), self.time_offset_days)
      self.time_period = pick(extraction.time_period, self.time_period)
      self.user_constraint = pick(getattr(extraction, "user_constraint", None), self.user_constraint)
      self.explanation_target = pick(getattr(extraction, "explanation_target", None), self.explanation_target)
      ext_count = getattr(extraction, "count", 1)
      if ext_count > 1:
          self.count = ext_count

  def is_routable(self) -> bool:
    return bool(self.location_text or self.reference_location_text or self.target_location_text or self.region_locations) and self.intent not in (None, "unknown")

  def clear(self) -> None:
    self.language = None
    self.intent = None
    self.action_type = None
    self.explanation_target = None
    self.location_text = None
    self.reference_location_text = None
    self.target_location_text = None
    self.comparison_location_texts = []
    self.region_locations = []
    self.location_role = None
    self.activity = None
    self.time_relative = None
    self.time_offset_days = None
    self.time_period = None
    self.user_constraint = None
    self.count = 1
    self.pending_clarification = None
    self.clarification_context = None
    self.previous_query = None
    self.previous_clarifications = []
    self.clarify_rounds = 0

  def update_from_plan(self, plan) -> None:
    """Synchronize ConversationState directly from the resolved canonical QueryPlan."""
    if getattr(plan, "intent", None):
      self.intent = getattr(plan, "intent")
    if getattr(plan, "action_type", None) and hasattr(plan.action_type, "value"):
      self.action_type = plan.action_type.value
    elif getattr(plan, "action_type", None):
      self.action_type = str(plan.action_type)
      
    if getattr(plan, "explanation_target", None):
      self.explanation_target = getattr(plan, "explanation_target")
      
    if getattr(plan, "location", None):
      self.location_text = plan.location.name
    if getattr(plan, "reference_location", None):
      self.reference_location_text = plan.reference_location.name
    if getattr(plan, "target_location", None):
      self.target_location_text = plan.target_location.name
      
    if getattr(plan, "time", None):
      self.time_relative = plan.time.relative
      self.time_offset_days = plan.time.offset_days
      self.time_period = plan.time.period
      
    if getattr(plan, "activity", None):
      self.activity = plan.activity