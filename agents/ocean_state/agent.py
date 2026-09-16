from datetime import datetime, timezone
from typing import Dict
from schemas.contracts import BaseAgent, QueryPlan, AgentResult, GeoLocation, AgentAudit

# Re-use resolve_plan_timestamp from ocean.agent
from agents.ocean.agent import resolve_plan_timestamp

class OceanStateAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "ocean_state"

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
                sources=[],
                warnings=["LOCATION_REQUIRED: No valid spatial location provided in QueryPlan."]
            )
        lat = loc.latitude
        lon = loc.longitude

        timestamp = resolve_plan_timestamp(plan.time)

        from data_sources.weather_provider import get_ocean_context
        ocean_context = get_ocean_context(lat, lon, timestamp)

        from schemas.contracts import UNKNOWN
        
        # 1. Query PostGIS DB for latest ocean observations
        obs = None
        try:
            if not context.get("_bypass_db_lookup"):
                from backend.db.marine.observations import get_latest_observations, upsert_observation
                radius = plan.search_radius_km if plan.search_radius_km else 30.0
                obs_list = get_latest_observations(lat, lon, radius)
                obs = obs_list[0] if obs_list else None
            
            if not obs:
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
                        print("Ocean state provider timeout exceeded (1.0s). Failing fast.")
                        fetched_data = None
                    except Exception as e:
                        print(f"Ocean state provider failed: {e}")
                        fetched_data = None
                    if fetched_data:
                        try:
                            upsert_observation(fetched_data)
                        except Exception:
                            pass
                        obs = fetched_data
        except Exception:
            pass

        if obs:
            from schemas.contracts import Observation
            ocean_context["sst"] = Observation(source="MARINE_DB", quality="OBSERVED", value=obs.get("sst", UNKNOWN))
            ocean_context["surface_current"] = Observation(source="MARINE_DB", quality="OBSERVED", value=obs.get("surface_current", UNKNOWN))

        incois_data = {}
        if ocean_context:
            for k, v in ocean_context.items():
                incois_data[k] = v.value if hasattr(v, "value") and v.value is not UNKNOWN else None

        payload = {
            "incois_ocean_state": incois_data
        }

        return AgentResult(
            agent=self.name,
            status="SUCCESS",
            location=GeoLocation(latitude=lat, longitude=lon, name=loc.name if loc else None),
            timestamp=datetime.now(timezone.utc),
            data=payload,
            confidence=0.85,
            sources=["INCOIS_ROMS_FORECAST"],
            audit=AgentAudit(
                inputs_used={"lat": lat, "lon": lon},
                outputs=payload,
                output_reason="Retrieved marine observation data.",
                sources=["INCOIS_ROMS_FORECAST"],
                score_source=None,
                score_reason=None
            )
        )
