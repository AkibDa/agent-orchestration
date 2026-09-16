# Proto/agents/weather/agent.py

from datetime import datetime, timezone, timedelta
from typing import Dict
from schemas.contracts import BaseAgent, QueryPlan, AgentResult, GeoLocation
from .model import predict
import data_sources.weather_provider as provider

def resolve_plan_timestamp(query_time) -> datetime:
    now = datetime.now(timezone.utc)
    if not query_time:
        return now
    if query_time.exact:
        return query_time.exact
    offset_days = query_time.offset_days or (1 if query_time.relative == "tomorrow" else 0)
    hour = query_time.hour if query_time.hour is not None else now.hour
    minute = query_time.minute if query_time.minute is not None else 0
    target_dt = now + timedelta(days=offset_days)
    return target_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)

class WeatherAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "weather"

    def run(self, plan: QueryPlan, context: Dict[str, AgentResult]) -> AgentResult:
        loc = plan.target_location or plan.location or plan.reference_location
        if not loc:
            return AgentResult(
                agent=self.name,
                status="error",
                location=None,
                timestamp=datetime.now(timezone.utc),
                data={"error": "LOCATION_REQUIRED"},
                confidence=1.0,
                sources=["WEATHER_PROVIDER", "ORCA_XGBOOST_WEATHER_v4"],
                warnings=["LOCATION_REQUIRED: No valid spatial location provided in QueryPlan."]
            )
        lat = loc.latitude
        lon = loc.longitude
        loc_name = loc.name if loc.name else "Unknown"

        timestamp = resolve_plan_timestamp(plan.time)
        weather_ctx = provider.get_weather_context(loc_name, plan.intent, lat=lat, lon=lon)

        from schemas.contracts import UNKNOWN

        # 1. Query PostGIS DB for latest weather observations (Freshness-aware)
        obs = None
        try:
            if not context.get("_bypass_db_lookup"):
                from backend.db.marine.observations import get_latest_observations, upsert_observation
                radius = plan.search_radius_km if plan.search_radius_km else 30.0
                obs_list = get_latest_observations(lat, lon, radius)
                obs = obs_list[0] if obs_list else None

            if not obs:
                # 2. Attempt optional configured external provider
                from backend.services.data_fetchers.registry import get_weather_provider
                ext_provider = get_weather_provider()
                if ext_provider:
                    fetched_data = ext_provider.fetch_weather(lat, lon, timestamp)
                    if fetched_data:
                        try:
                            upsert_observation(fetched_data)
                        except Exception as e:
                            print(f"Failed to upsert weather observation: {e}")
                        obs = fetched_data
        except Exception as e:
            print(f"Weather DB/Provider Query failed: {e}")

        # Update context if DB/Ext Provider returned data
        if obs:
            from schemas.contracts import Observation
            weather_ctx["weather"] = Observation(source="MARINE_DB", quality="OBSERVED", value=obs)

        # Convert to primitive dicts for backwards compatibility where needed
        provider_data = {}
        for k, v in weather_ctx.items():
            if k in ["data_provenance", "cyclone_context"]:
                continue
            if hasattr(v, "value"): provider_data[k] = v.value if v.value is not UNKNOWN else None
            elif hasattr(v, "severity"): provider_data[k] = v.severity if v.severity is not UNKNOWN else None
            elif hasattr(v, "advisory_text"): provider_data[k] = v.advisory_text if v.advisory_text is not UNKNOWN else None

        # Determine if we have valid input data
        # Minimum required inputs: wind_speed_ms
        weather_dict = provider_data.get("weather")
        has_valid_inputs = False
        source_list = []
        if weather_dict and isinstance(weather_dict, dict):
            if "wind_speed_ms" in weather_dict:
                has_valid_inputs = True
                src = weather_ctx["weather"].source if hasattr(weather_ctx["weather"], "source") else "UNKNOWN_PROVIDER"
                source_list.extend([src, "ORCA_XGBOOST_WEATHER_v4"])

        if not has_valid_inputs:
            from schemas.contracts import AgentAudit
            return AgentResult(
                agent=self.name,
                status="NOT_AVAILABLE",
                location=GeoLocation(latitude=lat, longitude=lon, name=loc_name),
                timestamp=datetime.now(timezone.utc),
                data={"error": "WEATHER_DATA_UNAVAILABLE"},
                confidence=0.0,
                sources=[],
                warnings=["WEATHER_DATA_UNAVAILABLE: Could not fetch weather data from DB, IMD, or fallback providers."],
                audit=AgentAudit(
                    inputs_used={},
                    outputs={},
                    output_reason="Weather data unavailable.",
                    sources=[],
                    score_source=None,
                    score_reason=None
                )
            )

        cyclone_ctx = weather_ctx.get("cyclone_context")
        xgb_payload = predict(lat=lat, lon=lon, timestamp=timestamp, weather_context=weather_ctx, cyclone_context=cyclone_ctx)

        warnings = []
        fisherman_warning = weather_ctx.get("official_marine_warning")

        if fisherman_warning and fisherman_warning.severity not in (UNKNOWN, "NORMAL", "GREEN", "NONE"):
            warnings.append(f"IMD FISHERMAN WARNING: {fisherman_warning.warning_text}")

        status = "SUCCESS"
        if not xgb_payload.get("cyclone_data_available", True):
            status = "DEGRADED"
            warnings.append("CYCLONE_DATA_UNAVAILABLE: Cyclone features imputed to safe defaults. Predictions may be unreliable.")

        if weather_ctx.get("weather") and weather_ctx["weather"].value is UNKNOWN:
            status = "ERROR"
            warnings.append("WEATHER_SOURCE_FAILED: No primary weather data available.")

        provenance = weather_ctx.get("data_provenance", [])

        provenance_dict = {
            "model_version": "weather_risk_v4_hierarchical",
            "model_stages": "Stage1(NORMAL/HAZARD) -> Stage2(CAUTION/DANGER)",
            "feature_count": 43,
            "cyclone_data_available": xgb_payload.get("cyclone_data_available", True),
            "features_imputed": xgb_payload.get("features_imputed", [])
        }

        for prov in provenance:
            if prov["source"] == "IMD":
                provenance_dict["imd_status"] = prov["status"]
                provenance_dict["imd_latency_ms"] = prov.get("latency_ms", 0)
            elif prov["source"] == "OPEN_METEO":
                provenance_dict["weather_source"] = "OPEN_METEO"
                provenance_dict["weather_source_status"] = prov["status"]
                provenance_dict["weather_latency_ms"] = prov.get("latency_ms", 0)
                provenance_dict["features_with_real_values"] = prov.get("fields", [])
            elif prov["source"] == "JTWC":
                provenance_dict["cyclone_source"] = "JTWC"
                provenance_dict["cyclone_data_status"] = prov["status"]

        payload = {
            "provider_data": provider_data,
            "orca_xgboost": xgb_payload,
            "data_provenance": provenance_dict,
            "cyclone_context": weather_ctx.get("cyclone_context")
        }

        audit_data = xgb_payload.get("audit", {})
        from schemas.contracts import AgentAudit
        audit_model = AgentAudit(
            inputs_used=audit_data.get("inputs_used", {}),
            outputs=audit_data.get("outputs", {}),
            output_reason=audit_data.get("output_reason", "No explanation provided."),
            sources=audit_data.get("sources", source_list),
            score_source=audit_data.get("score_source", None),
            score_reason=audit_data.get("score_reason", None)
        )

        return AgentResult(
            agent=self.name,
            status=status,
            location=GeoLocation(latitude=lat, longitude=lon, name=loc_name),
            timestamp=datetime.now(timezone.utc),
            data=payload,
            confidence=xgb_payload.get("max_probability", 0.8),
            sources=source_list,
            warnings=warnings,
            audit=audit_model
        )
