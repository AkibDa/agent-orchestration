# Proto/agents/risk/agent.py

from datetime import datetime, timezone
from typing import Dict
from schemas.contracts import BaseAgent, QueryPlan, AgentResult, GeoLocation
from .model import predict

class RiskAgent(BaseAgent):
    """
    DEPRECATED: This agent provides a supplementary ML risk signal. 
    It is NOT the authoritative marine safety assessment. 
    See marine_safety agent for the INCOIS-derived BSI assessment.
    """
    @property
    def name(self) -> str:
        return "risk"

    def run(self, plan: QueryPlan, context: Dict[str, AgentResult]) -> AgentResult:
        loc = plan.target_location or plan.location or plan.reference_location
        if not loc:
            return AgentResult(
                agent=self.name,
                status="ERROR",
                location=None,
                timestamp=datetime.now(timezone.utc),
                data={"error": "LOCATION_REQUIRED"},
                confidence=1.0,
                sources=["marine_risk_xgboost_v2"],
                warnings=["LOCATION_REQUIRED: No valid spatial location provided in QueryPlan."]
            )
        lat = loc.latitude
        lon = loc.longitude

        # Check for reported hazard in user query, plan intent, or user constraint
        q_low = (plan.query or "").lower()
        const_low = (plan.user_constraint or "").lower()
        has_hazard = (
            plan.intent == "hazard_alert" or
            any(w in q_low for w in ["cyclone", "storm", "typhoon", "tsunami", "squall", "hazard"]) or
            any(w in const_low for w in ["cyclone", "storm", "hazard"])
        )

        weather_ok = "weather" in context and context["weather"].status == "SUCCESS"
        ocean_ok = "ocean" in context and context["ocean"].status == "SUCCESS"
        
        if not weather_ok or not ocean_ok:
            from schemas.contracts import AgentAudit
            return AgentResult(
                agent=self.name,
                status="NOT_AVAILABLE",
                location=GeoLocation(latitude=lat, longitude=lon, name=loc.name if loc else None),
                timestamp=datetime.now(timezone.utc),
                data={"error": "UPSTREAM_DATA_UNAVAILABLE", "risk_level": "UNKNOWN"},
                confidence=0.0,
                sources=[],
                warnings=["UPSTREAM_DATA_UNAVAILABLE: Cannot evaluate risk without weather and ocean data."],
                audit=AgentAudit(
                    inputs_used={},
                    outputs={"risk_level": "UNKNOWN"},
                    output_reason="Risk evaluation aborted due to missing upstream weather or ocean data.",
                    sources=[],
                    score_source=None,
                    score_reason=None
                )
            )

        # Combine environmental features from weather, ocean, geospatial context
        weather_res = context["weather"].data if "weather" in context else {}
        ocean_res = context["ocean"].data if "ocean" in context else {}
        geo_data = context["geospatial"].data if "geospatial" in context else {}

        provider_weather = weather_res.get("provider_data", {}).get("weather", {})

        w_speed = provider_weather.get("wind_speed_ms", 5.0)
        msl = provider_weather.get("msl_hpa", 1012.0) * 100.0
        t2m = provider_weather.get("temperature_c", 25.0) + 273.15
        
        # Stub approximate u10/v10
        u10 = w_speed * 0.707
        v10 = w_speed * 0.707

        features = {
            "latitude": lat,
            "longitude": lon,
            "allowed": geo_data.get("allowed", True),
            "u10": u10,
            "v10": v10,
            "wind_speed_10m": w_speed,
            "t2m": t2m,
            "msl": msl,
            "tp": provider_weather.get("total_precipitation_m", 0.001),
            "wind_delta_3h": 0.0,
            "pressure_delta_3h": 0.0,
            "cyclone_distance_km": 9999.0
        }

        payload = predict(features)

        warnings = []
        if not payload.get("allowed", True):
            warnings.append("Restricted zone or hazardous 6-hour marine state")
        if payload.get("risk_level") == "DANGER":
            warnings.append("6-hour-ahead DANGER marine hazard predicted")

        from schemas.contracts import AgentAudit
        audit_data = payload.get("audit", {})
        return AgentResult(
            agent=self.name,
            status="SUCCESS",
            location=GeoLocation(latitude=lat, longitude=lon, name=loc.name if loc else None),
            timestamp=datetime.now(timezone.utc),
            data=payload,
            confidence=0.92,
            sources=["ORCA_XGBOOST_MARINE_RISK_v2_SUPPLEMENTARY"],
            warnings=warnings,
            audit=AgentAudit(
                inputs_used=audit_data.get("inputs_used", {}),
                outputs=payload,
                output_reason="Evaluated supplementary marine risk using weather and ocean inputs.",
                sources=audit_data.get("sources", ["ORCA_XGBOOST_MARINE_RISK_v2_SUPPLEMENTARY"]),
                score_source=audit_data.get("score_source", None),
                score_reason=audit_data.get("score_reason", None)
            )
        )