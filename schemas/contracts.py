# Proto/schemas/contracts.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class AgentStatus(str, Enum):
    PLANNED = "PLANNED"
    CALLED = "CALLED"
    SKIPPED = "SKIPPED"
    SUCCESS = "SUCCESS"
    DEGRADED = "DEGRADED"
    MOCKED = "MOCKED"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    ERROR = "ERROR"

class GeoLocation(BaseModel):
  latitude: float = Field(..., description="Latitude coordinate")
  longitude: float = Field(..., description="Longitude coordinate")
  name: Optional[str] = Field(None, description="Optional named location")

class QueryTime(BaseModel):
  relative: Optional[str] = Field(None, description="Relative time string")
  offset_days: Optional[int] = Field(None, description="Day offset count, e.g. 3")
  period: Optional[str] = Field(None, description="Time of day")
  hour: Optional[int] = Field(None, description="Hour of day (0-23)")
  minute: Optional[int] = Field(None, description="Minute of hour (0-59)")
  exact: Optional[datetime] = Field(None, description="Exact parsed datetime")

from schemas.extraction import LocationItem, LocationRole, ActionType

class CandidateSpot(BaseModel):
  id: str = Field("spot", description="Spot identifier")
  latitude: float = Field(..., description="Spot latitude")
  longitude: float = Field(..., description="Spot longitude")
  distance_km: float = Field(0.0, description="Distance from reference origin in km")
  bearing_deg: Optional[float] = Field(None, description="Bearing angle in degrees")
  compass_direction: str = Field("CENTER", description="Compass heading direction")
  pfz_probability: Optional[float] = Field(None, description="Predicted PFZ probability")
  productivity_score: Optional[float] = Field(None, description="Fish productivity score")
  weather_risk: str = Field("NORMAL", description="Weather risk level")
  marine_risk: str = Field("LOW", description="Marine hazard risk level")
  safety_clearance: str = Field("CLEARED", description="Safety clearance status")
  eligible: bool = Field(True, description="Safety guardrail eligibility flag")
  rejection_reason: Optional[str] = Field(None, description="Reason for rejection if ineligible")
  selection_reason: Optional[str] = Field(None, description="Selection justification text")
  composite_score: float = Field(0.0, description="Multi-objective ranking score")
  rank: Optional[int] = Field(None, description="Assigned rank index")

class SpatialConstraint(BaseModel):
  distance_km: Optional[float] = Field(None, description="Relative distance offset in km, e.g. 200.0")
  direction: Optional[str] = Field(None, description="Compass direction phrase e.g. 'east', 'south-west'")
  bearing_deg: Optional[float] = Field(None, description="Bearing angle in degrees (0-360)")
  radius_km: Optional[float] = Field(None, description="Search radius around spatial constraint in km")

class QueryPlan(BaseModel):
  query: str = Field(..., description="The original raw user query")
  intent: str = Field(..., description="Resolved intent")
  operation: Optional[str] = Field(None, description="Query operation: DISTANCE_TO_COAST, FIND_FISHING_SPOTS, ASSESS_HAZARD, SAFE_ALTERNATIVE_ZONE, TEMPORAL_PFZ_GUIDANCE, COMPARE_FISHING_REGIONS, FISHING_SAFETY_TRADEOFF, GENERAL_MARINE_QUERY, SELECT_BEST_FISHING_OPTION, NEAREST_PFZ_SEARCH, SEARCH_PFZ, ASSESS_FISHING_SAFETY, ROUTE_TO_FISHING_AREA")
  action_type: ActionType = Field(ActionType.ASSESS, description="Action: ASSESS, SEARCH, COMPARE, LOCATE, EXPLAIN")
  result_type: str = Field("SAFETY_ASSESSMENT", description="Result category: SAFETY_ASSESSMENT, PFZ_RESULT, HAZARD_RESULT, CONDITIONS_RESULT, NEAREST_COAST_RESULT, GEOGRAPHY_RESULT, FISHING_IMPACT_RESULT")
  response_mode: str = Field("LLM_SYNTHESIS", description="TEMPLATE or LLM_SYNTHESIS")
  readiness_status: str = Field("CAN_EXECUTE", description="CAN_EXECUTE | CONCEPTUAL_QUERY | NEED_LOCATION | NEED_TIME | NEED_COMPARISON_CANDIDATES | NEED_CLARIFICATION")
  language: str = Field(..., description="ISO language code")
  explanation_target: Optional[str] = Field(None, description="E.g., 'fishing_availability'")
  
  location: Optional[GeoLocation] = Field(None, description="Parsed primary spatial context (legacy)")
  reference_location: Optional[GeoLocation] = Field(None, description="Origin/start location where user is starting from")
  target_location: Optional[GeoLocation] = Field(None, description="Target/destination location being assessed")
  resolved_locations: List[GeoLocation] = Field(default_factory=list, description="All resolved geographic locations from the query")
  spatial_constraint: Optional[SpatialConstraint] = Field(None, description="Relative spatial constraint offset")
  location_required: bool = Field(True, description="Whether location context is required for operation")
  location_source: Optional[str] = Field(None, description="Location resolution source: GAZETTEER, METADATA")
  location_confidence: float = Field(1.0, ge=0.0, le=1.0, description="Location resolution confidence")
  search_radius_km: Optional[float] = Field(30.0, description="Search radius for candidate spot generation in km")
  count: int = Field(1, description="Number of items or candidates requested")
  compare_locations: List[GeoLocation] = Field(default_factory=list, description="Locations for comparison")
  region: Optional[str] = Field(None, description="Broad region context (e.g., Bay of Bengal)")
  clarification_reason: Optional[str] = Field(None, description="Reason for clarification if Action is CLARIFY")
  
  location_type: Optional[str] = Field("unknown", description="'coastal', 'inland', or 'unknown'")
  location_role: LocationRole = Field(LocationRole.TARGET, description="REFERENCE, TARGET, REGION, COASTAL_POINT, UNKNOWN")
  locations: List[LocationItem] = Field(default_factory=list, description="Extracted location items with roles")
  inland_name: Optional[str] = Field(None, description="Name of non-coastal inland place if detected")
  candidate_locations: List[str] = Field(default_factory=list, description="Other locations mentioned in query for comparative questions")
  time: QueryTime = Field(default_factory=QueryTime, description="Parsed temporal context")
  activity: Optional[str] = Field(None, description="Specific activity")
  user_constraint: Optional[str] = Field(None, description="Operational constraint e.g. must_fish_today")
  
  agents: List[str] = Field(default_factory=list, description="Capabilities requested")
  dependencies: List[str] = Field(default_factory=list, description="Explicit dependencies")
  constraints: List[Dict[str, Any]] = Field(default_factory=list, description="Conditions to validate")

class AgentAudit(BaseModel):
  inputs_used: Dict[str, Any] = Field(default_factory=dict, description="Variables actually consumed by this agent")
  outputs: Dict[str, Any] = Field(default_factory=dict, description="Meaningful outputs produced by this agent")
  output_reason: str = Field("No explanation provided.", description="Explanation of what this agent produced")
  sources: List[str] = Field(default_factory=list, description="Provenance sources or models used")
  score_source: Optional[str] = Field(None, description="Category of the score (e.g. MODEL_PREDICTION, MOCKED_DATA)")
  score_reason: Optional[str] = Field(None, description="Deterministic explanation of why this output was produced")

class AgentResult(BaseModel):
  agent: str = Field(..., description="Name of the agent")
  status: str = Field(..., description="Explicit execution status of the agent (e.g., SUCCESS, ERROR, NOT_AVAILABLE)")
  
  location: Optional[GeoLocation] = Field(None, description="Exact coordinates this applies to")
  timestamp: datetime = Field(..., description="Exact time this data applies to")
  
  data: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific payload")
  confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
  sources: List[str] = Field(default_factory=list, description="Data lineage")
  warnings: List[str] = Field(default_factory=list, description="Anomalies or flags")
  
  audit: Optional[AgentAudit] = Field(None, description="Structured execution audit metadata")

class BaseAgent(ABC):
  @property
  @abstractmethod
  def name(self) -> str:
    pass

  @abstractmethod
  def run(self, plan: QueryPlan, context: Dict[str, AgentResult]) -> AgentResult:
    pass

# ============================================================================
# PHASE 1: DATA LAYER CONTRACTS
# ============================================================================

class _UnknownSentinel:
    """Explicit UNKNOWN sentinel that evaluates to False in boolean contexts."""
    def __bool__(self): return False
    def __str__(self): return "UNKNOWN"
    def __repr__(self): return "UNKNOWN"
    def __eq__(self, other):
        if isinstance(other, str) and other == "UNKNOWN":
            return True
        return isinstance(other, _UnknownSentinel)

UNKNOWN = _UnknownSentinel()

class DataContract(BaseModel):
    source: str = Field(..., description="Provenance source (e.g., 'IMD', 'INCOIS_PFZ')")
    valid_time: Optional[datetime] = Field(None, description="When this data is valid for")
    retrieved_at: datetime = Field(default_factory=datetime.utcnow, description="When this data was fetched")
    quality: str = Field(..., description="'OBSERVED' or 'FORECAST'")
    confidence: float = Field(1.0, description="Confidence of the source data")

class Observation(DataContract):
    value: Any = Field(..., description="Observed value or UNKNOWN")
    unit: str = Field(..., description="Unit of measurement")

class Forecast(DataContract):
    value: Any = Field(..., description="Forecasted value or UNKNOWN")
    unit: str = Field(..., description="Unit of measurement")

class Advisory(DataContract):
    advisory_text: Any = Field(..., description="Advisory content or UNKNOWN")

class WarningContract(DataContract):
    severity: Any = Field(..., description="Warning severity level")
    warning_text: Any = Field(..., description="Warning content")

class PFZCandidate(DataContract):
    latitude: float = Field(...)
    longitude: float = Field(...)
    depth: Optional[float] = Field(None, description="Depth in meters if applicable")
    bearing_from_landmark: Optional[str] = Field(None)
    distance_from_landmark: Optional[float] = Field(None)
    advisory_date: Optional[datetime] = Field(None)
    validity_window: Optional[str] = Field(None)

class MarineCandidateIdentity(BaseModel):
    candidate_id: str
    latitude: float
    longitude: float
    display_name: str
    region: Optional[str]
    reference_landmark: Optional[str]
    distance_from_landmark_km: Optional[float]
    bearing_from_landmark: Optional[str]
    
    pfz_probability: Optional[float]
    pfz_present: bool = False
    pfz_signal_present: bool = False
    pfz_qualified: bool = False
    pfz_source: str = "UNKNOWN"
    pfz_threshold: Optional[float] = None
    pfz_description: Optional[str] = None
    
    safety_status: str = "UNKNOWN"
    weather_status: str = "UNKNOWN"
    ocean_status: str = "UNKNOWN"
    
    reachability_km: Optional[float] = None
    composite_score: Optional[float] = None
    score_source: str = "UNKNOWN"
    score_reason: Optional[str] = None
    source: str = "UNKNOWN"
    data_quality: str = "MODEL_ESTIMATE"
    
    eligible: bool = True
    rejection_reason: Optional[str] = None
    selection_reason: Optional[str] = None
    rank: Optional[int] = None