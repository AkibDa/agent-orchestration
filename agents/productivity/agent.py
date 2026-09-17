# agents/productivity/agent.py

from datetime import datetime, timezone
from typing import Dict, Any
from schemas.contracts import BaseAgent, QueryPlan, AgentResult, GeoLocation
from .model import predict_productivity

class FishProductivityAgent(BaseAgent):
    """
    Algorithmic Fish Productivity Agent (LSTM Model)

    Evaluates 12-month environmental spatio-temporal sequences to forecast fish catch-per-unit-effort
    (CPUE) and continuous productivity scores for marine location search.
    """
    @property
    def name(self) -> str:
        return "productivity"

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
                sources=["FISH_PRODUCTIVITY_LSTM_12M_ENV_v1"],
                warnings=["LOCATION_REQUIRED: No valid spatial location provided in QueryPlan."]
            )
        lat = loc.latitude
        lon = loc.longitude

        weather_res = context.get("weather")
        ocean_res = context.get("ocean")

        weather_data = weather_res.data if weather_res else {}
        ocean_data = ocean_res.data if ocean_res else {}
        w_features = weather_data.get("provider_data", {}).get("weather", {})
        ocean_state = ocean_data.get("incois_ocean_state", {})

        is_degraded = False
        
        req_w_speed = w_features.get("wind_speed_ms")
        req_temp = w_features.get("temperature_c")
        req_msl = w_features.get("msl_hpa")
        req_sst = ocean_state.get("sst")
        req_tp = w_features.get("total_precipitation_m")
        
        if req_w_speed is None or req_temp is None or req_msl is None or req_sst is None or req_tp is None:
            from schemas.contracts import AgentAudit
            audit_model = AgentAudit(
                inputs_used={},
                outputs={"productivity_score": None, "estimated_cpue": None},
                output_reason="Productivity score was not calculated because required live environmental inputs were unavailable.",
                sources=["FISH_PRODUCTIVITY_LSTM_12M_ENV_v1"],
                score_source="NOT_AVAILABLE",
                score_reason="Productivity score was not calculated because required live environmental inputs were unavailable."
            )
            return AgentResult(
                agent=self.name,
                status="NOT_AVAILABLE",
                location=GeoLocation(latitude=lat, longitude=lon, name=loc.name if loc else None),
                timestamp=datetime.now(timezone.utc),
                data={
                    "error": "LIVE_ENVIRONMENT_UNAVAILABLE", 
                    "productivity_score": None, 
                    "estimated_cpue": None,
                    "score_source": "NOT_AVAILABLE",
                    "score_reason": "Productivity score was not calculated because required live environmental inputs were unavailable."
                },
                confidence=0.0,
                sources=["FISH_PRODUCTIVITY_LSTM_12M_ENV_v1"],
                warnings=["LIVE_ENVIRONMENT_UNAVAILABLE: Aborting prediction. Model requires live environmental data. Synthetic inference is disabled."],
                audit=audit_model
            )

        features = {
            "latitude": lat,
            "longitude": lon,
            "wind_speed_10m": req_w_speed,
            "t2m": (req_temp + 273.15),
            "msl": (req_msl * 100.0),
            "sst": (req_sst + 273.15),
            "tp": req_tp,
            "month": datetime.now(timezone.utc).month
        }

        payload = predict_productivity(features)
        payload["latitude"] = lat
        payload["longitude"] = lon

        audit_data = payload.get("audit", {})
        from schemas.contracts import AgentAudit
        audit_model = AgentAudit(
            inputs_used=audit_data.get("inputs_used", {}),
            outputs=payload,
            output_reason="Evaluated 12-month environmental sequences to forecast fish productivity.",
            sources=audit_data.get("sources", ["FISH_PRODUCTIVITY_LSTM_12M_ENV_v1"]),
            score_source=audit_data.get("score_source", None),
            score_reason=audit_data.get("score_reason", None)
        )

        return AgentResult(
            agent=self.name,
            status="SUCCESS",
            location=GeoLocation(latitude=lat, longitude=lon, name=loc.name if loc else None),
            timestamp=datetime.now(timezone.utc),
            data=payload,
            confidence=0.92,
            sources=["FISH_PRODUCTIVITY_LSTM_12M_ENV_v1"],
            warnings=[],
            audit=audit_model
        )
