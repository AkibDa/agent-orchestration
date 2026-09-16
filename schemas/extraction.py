# schemas/extraction.py

from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator

class Action(str, Enum):
  CHAT = "CHAT"
  CLARIFY = "CLARIFY"
  ORCA_QUERY = "ORCA_QUERY"
  NON_COASTAL_ERROR = "NON_COASTAL_ERROR"

class ResultType(str, Enum):
  SAFETY_ASSESSMENT = "SAFETY_ASSESSMENT"
  PFZ_RESULT = "PFZ_RESULT"
  HAZARD_RESULT = "HAZARD_RESULT"
  CONDITIONS_RESULT = "CONDITIONS_RESULT"
  NEAREST_COAST_RESULT = "NEAREST_COAST_RESULT"
  GEOGRAPHY_RESULT = "GEOGRAPHY_RESULT"
  CLARIFICATION = "CLARIFICATION"

class Intent(str, Enum):
  marine_safety = "marine_safety"
  marine_conditions = "marine_conditions"
  pfz_search = "pfz_search"
  hazard_alert = "hazard_alert"
  nearest_coast = "nearest_coast"
  marine_geography = "marine_geography"
  nearest_pfz = "nearest_pfz"
  marine_safety_forecast = "marine_safety_forecast"
  fishing_zone_analysis = "fishing_zone_analysis"
  safe_route = "safe_route"
  productivity_analysis = "productivity_analysis"
  hazardous_zone_filter = "hazardous_zone_filter"
  unknown = "unknown"

class LocationRole(str, Enum):
  REFERENCE = "REFERENCE"
  TARGET = "TARGET"
  REGION = "REGION"
  COASTAL_POINT = "COASTAL_POINT"
  COMPARISON = "COMPARISON"
  UNKNOWN = "UNKNOWN"

class ActionType(str, Enum):
  ASSESS = "ASSESS"
  SEARCH = "SEARCH"
  COMPARE = "COMPARE"
  LOCATE = "LOCATE"
  EXPLAIN = "EXPLAIN"

class OperationType(str, Enum):
  MARINE_SAFETY = "MARINE_SAFETY"
  PFZ_SEARCH = "PFZ_SEARCH"
  FISHING_SAFETY_TRADEOFF = "FISHING_SAFETY_TRADEOFF"
  COMPARE_LOCATIONS = "COMPARE_LOCATIONS"
  ROUTE_SAFETY = "ROUTE_SAFETY"
  GENERAL_MARINE_QUERY = "GENERAL_MARINE_QUERY"
  SELECT_BEST_FISHING_OPTION = "SELECT_BEST_FISHING_OPTION"
  NEAREST_PFZ_SEARCH = "NEAREST_PFZ_SEARCH"
  SEARCH_PFZ = "SEARCH_PFZ"
  ASSESS_FISHING_SAFETY = "ASSESS_FISHING_SAFETY"
  ROUTE_TO_FISHING_AREA = "ROUTE_TO_FISHING_AREA"

class LocationItem(BaseModel):
  text: str
  role: LocationRole = LocationRole.TARGET

class Language(str, Enum):
  hi_latn = "hi-Latn"
  bn_latn = "bn-Latn"
  hi = "hi"
  bn = "bn"
  en = "en"

class ExtractionResult(BaseModel):
  action: Action = Action.ORCA_QUERY
  action_type: Optional[ActionType] = ActionType.ASSESS
  language: Language = Language.en
  intent: Intent = Intent.unknown
  locations: list[LocationItem] = Field(default_factory=list, description="Extracted locations with roles")
  activity: Optional[Literal["fishing", "sailing", "none"]] = "none"
  time_relative: Optional[Literal["today", "tomorrow", "day_after_tomorrow", "yesterday", "custom", "none"]] = "none"
  time_offset_days: Optional[int] = Field(None, description="Relative offset in days if specified, e.g. 3 for '3 din por'")
  time_period: Optional[Literal["morning", "afternoon", "evening", "night", "none"]] = "none"
  count: int = Field(1, description="Number of items or candidates requested, e.g. 'top three' -> 3. Default is 1.")
  spatial_distance_km: Optional[float] = Field(None, description="Extracted relative distance in km e.g. 200.0")
  spatial_direction: Optional[str] = Field(None, description="Extracted compass direction e.g. 'east', 'purbo', 'south'")
  clarify_question: Optional[str] = Field(None, description="Only set if action=CLARIFY")
  chat_reply: Optional[str] = Field(None, description="Only set if action=CHAT")
  explanation_target: Optional[str] = Field(None, description="E.g., 'fishing_availability'")

  @field_validator("action", mode="before")
  @classmethod
  def parse_action(cls, v):
    if isinstance(v, str):
      v_upper = v.upper().strip()
      if v_upper in ["CHAT", "CLARIFY", "ORCA_QUERY"]:
        return v_upper
      return "ORCA_QUERY"
    return "ORCA_QUERY"

  def ensure_non_null(self) -> "ExtractionResult":
    """Helper method to ensure sensible fallbacks if LLM outputs null/None."""
    # Intent preservation: if intent is a specific marine category but action was set to CHAT, promote to CLARIFY
    if self.intent != Intent.unknown and self.action == Action.CHAT:
      self.action = Action.CLARIFY

    if self.action == Action.CLARIFY and not self.clarify_question:
      pass # Planner handles clarification reasoning now
    elif self.action == Action.CHAT and not self.chat_reply:
      self.chat_reply = "I am ORCA, your marine safety assistant. How can I help you today?"
    return self