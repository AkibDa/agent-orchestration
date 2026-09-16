# Proto/agents/ocean/agent.py

from datetime import datetime, timezone, timedelta
from typing import Dict
from schemas.contracts import BaseAgent, QueryPlan, AgentResult, GeoLocation
from .model import predict

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

class OceanAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "ocean"

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
                sources=["INCOIS_ROMS_FORECAST", "ORCA_OCEAN_SUITABILITY_XGBOOST_v1"],
                warnings=["LOCATION_REQUIRED: No valid spatial location provided in QueryPlan."]
            )
        lat = loc.latitude
        lon = loc.longitude

        timestamp = resolve_plan_timestamp(plan.time)

        from data_sources.weather_provider import get_ocean_context
        ocean_context = get_ocean_context(lat, lon, timestamp)

        from schemas.contracts import UNKNOWN

        # 1. Query PostGIS DB for latest ocean observations (Freshness-aware)
        obs = None
        try:
            if not context.get("_bypass_db_lookup"):
                from backend.db.marine.observations import get_latest_observations, upsert_observation
                radius = plan.search_radius_km if plan.search_radius_km else 30.0
                obs_list = get_latest_observations(lat, lon, radius)
                obs = obs_list[0] if obs_list else None

            if not obs:
                # 2. Attempt optional configured external provider
                from backend.services.data_fetchers.registry import get_ocean_provider
                ext_provider = get_ocean_provider()
                if ext_provider:
                    timestamp_exact = plan.time.exact if plan.time and plan.time.exact else datetime.now(timezone.utc)
                    import concurrent.futures
                    try:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(ext_provider.fetch_ocean, lat, lon, timestamp_exact)
                            fetched_data = future.result(timeout=1.0)
                    except concurrent.futures.TimeoutError:
                        print("Ocean provider timeout exceeded (1.0s). Failing fast.")
                        fetched_data = None
                    except Exception as e:
                        print(f"Ocean provider failed: {e}")
                        fetched_data = None

                    if fetched_data:
                        try:
                            upsert_observation(fetched_data)
                        except Exception as e:
                            print(f"Failed to upsert ocean observation: {e}")
                        obs = fetched_data
        except ImportError as e:
            print(f"Ocean DB/Provider Query bypassed: 'backend' module not available. Error: {e}")
        except Exception as e:
            print(f"Ocean DB/Provider Query failed: {e}")

        # Update context if DB/Ext Provider returned data
        if obs:
            from schemas.contracts import Observation
            ocean_context["sst"] = Observation(source="MARINE_DB", quality="OBSERVED", value=obs.get("sst", UNKNOWN))
            ocean_context["surface_current"] = Observation(source="MARINE_DB", quality="OBSERVED", value=obs.get("surface_current", UNKNOWN))

        has_valid_inputs = False
        source_list = []
        has_sst = ocean_context.get("sst") and hasattr(ocean_context["sst"], "value") and ocean_context["sst"].value is not UNKNOWN
        has_current = ocean_context.get("surface_current") and hasattr(ocean_context["surface_current"], "value") and ocean_context["surface_current"].value is not UNKNOWN

        if has_sst:
            has_valid_inputs = True
            src = ocean_context["sst"].source if hasattr(ocean_context["sst"], "source") else "UNKNOWN_PROVIDER"
            source_list.extend([src, "ORCA_OCEAN_SUITABILITY_XGBOOST_v1"])
            
            if not has_current:
                source_list.append("CURRENT_DATA_UNAVAILABLE")

        if not has_valid_inputs:
            from schemas.contracts import AgentAudit
            return AgentResult(
                agent=self.name,
                status="NOT_AVAILABLE",
                location=GeoLocation(latitude=lat, longitude=lon, name=loc.name if loc else None),
                timestamp=datetime.now(timezone.utc),
                data={"error": "OCEAN_DATA_UNAVAILABLE"},
                confidence=0.0,
                sources=[],
                warnings=["OCEAN_DATA_UNAVAILABLE: Could not fetch ocean state from DB, INCOIS, or fallback providers."],
                audit=AgentAudit(
                    inputs_used={},
                    outputs={},
                    output_reason="Ocean data unavailable.",
                    sources=[],
                    score_source=None,
                    score_reason=None
                )
            )

        payload = predict(lat=lat, lon=lon, timestamp=timestamp, ocean_context=ocean_context)

        warnings = []
        if payload.get("out_of_domain_warning"):
            warnings.append(payload["out_of_domain_warning"])

        status = "SUCCESS"
        if payload.get("feature_imputed"):
            status = "DEGRADED"
            warnings.append("DEGRADED: Using hardcoded synthetic ocean inputs due to missing live data.")

        audit_data = payload.get("audit", {})
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
            location=GeoLocation(latitude=lat, longitude=lon, name=loc.name if loc else None),
            timestamp=datetime.now(timezone.utc),
            data=payload,
            confidence=payload.get("orca_suitability", {}).get("confidence_score", 0.95),
            sources=source_list,
            warnings=warnings,
            audit=audit_model
        )
