# agents/pfz/agent.py

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
    offset_days = query_time.offset_days or (1 if query_time.relative == "tomorrow" or query_time.relative == "kal" else 0)
    hour = query_time.hour if query_time.hour is not None else now.hour
    minute = query_time.minute if query_time.minute is not None else 0
    target_dt = now + timedelta(days=offset_days)
    return target_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)

class PFZAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "pfz"

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
                sources=["INCOIS_PFZ_XGBOOST_v1"],
                warnings=["LOCATION_REQUIRED: No valid spatial location provided in QueryPlan."]
            )

        from location.location_metadata import LOCATION_METADATA, INLAND_LOCATIONS

        loc_name_lower = loc.name.lower() if loc.name else ""
        coastal_access = True
        if loc_name_lower in LOCATION_METADATA:
            coastal_access = LOCATION_METADATA[loc_name_lower].get("coastal_access", True)
        elif loc_name_lower in INLAND_LOCATIONS:
            coastal_access = False

        lat = loc.latitude
        lon = loc.longitude

        nearest_lat, nearest_lon = lat, lon
        if not coastal_access:
            from location.gazetteer import GAZETTEER
            from agents.geospatial.distance import haversine_distance
            min_dist = float("inf")
            for g_key, (g_lat, g_lon, g_name) in GAZETTEER.items():
                if LOCATION_METADATA.get(g_key, {}).get("coastal_access", True) and g_key not in ["kolkata", "calcutta"]:
                    dist = haversine_distance(lat, lon, g_lat, g_lon)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_lat, nearest_lon = g_lat, g_lon

            if min_dist > 50.0:
                return AgentResult(
                    agent=self.name,
                    status="ERROR",
                    location=loc,
                    timestamp=datetime.now(timezone.utc),
                    data={"error": "INLAND_LOCATION_INVALID_FOR_PFZ"},
                    confidence=1.0,
                    sources=["INCOIS_PFZ_XGBOOST_v1"],
                    warnings=["INLAND_LOCATION: No marine candidate exists within max search radius."]
                )

        # Extract environmental data from context if provided by upstream data fetcher
        env_data = {}
        is_degraded = False
        if "ocean" in context and context["ocean"].data:
            if context["ocean"].status == "DEGRADED_SYNTHETIC":
                is_degraded = True
            env_data.update(context["ocean"].data)
        if "weather" in context and context["weather"].data:
            if context["weather"].status == "DEGRADED_SYNTHETIC":
                is_degraded = True
            env_data.update(context["weather"].data)

        from data_sources.incois import get_pfz_advisory

        candidates = get_pfz_advisory(lat, lon, count=plan.count)
        if candidates:
            # If we have official candidates near the user, we consider PFZ present
            prob = getattr(candidates[0], "confidence", None)
            is_qualified = prob >= 0.85 if prob is not None else True
            payload = {
                "pfz_present": True,
                "pfz_signal_present": True,
                "pfz_qualified": is_qualified,
                "pfz_probability": prob,
                "source": candidates[0].source,
                "score_source": "OFFICIAL_ADVISORY",
                "score_reason": "PFZ probability comes from the official INCOIS advisory for this location.",
                "candidates": candidates
            }
            from schemas.contracts import AgentAudit
            return AgentResult(
                agent=self.name,
                status="SUCCESS",
                location=GeoLocation(latitude=lat, longitude=lon, name=loc.name if loc else None),
                timestamp=datetime.now(timezone.utc),
                data=payload,
                confidence=1.0,
                sources=[candidates[0].source],
                warnings=[],
                audit=AgentAudit(
                    inputs_used={"lat": lat, "lon": lon},
                    outputs=payload,
                    output_reason="Found official PFZ advisory.",
                    sources=[candidates[0].source],
                    score_source="OFFICIAL_ADVISORY",
                    score_reason="PFZ probability comes from the official INCOIS advisory for this location."
                )
            )

        # 2. Fallback to XGBoost Model
        timestamp = resolve_plan_timestamp(plan.time)

        # 1. Query PostGIS database for known active PFZ zones
        try:
            from backend.db.marine.pfz import get_nearby_pfz_zones, upsert_pfz_zone
            radius = plan.search_radius_km if plan.search_radius_km else 30.0
            nearby_zones = get_nearby_pfz_zones(lat, lon, radius)

            if nearby_zones:
                best_zone = nearby_zones[0]
                prob = best_zone.get("confidence", 0.8)
                from schemas.contracts import AgentAudit
                payload = {
                    "pfz_probability": prob,
                    "pfz_present": prob > 0.5,
                    "pfz_signal_present": prob > 0.0,
                    "pfz_qualified": prob >= 0.85,
                    "suitability_score": best_zone.get("suitability_score", 0.8),
                    "db_zones": nearby_zones,
                    "score_source": "MARINE_DB_POSTGIS",
                    "score_reason": "PFZ probability is based on known active zones from the marine database."
                }
                return AgentResult(
                    agent=self.name,
                    status="SUCCESS",
                    location=GeoLocation(latitude=lat, longitude=lon, name=loc.name if loc else None),
                    timestamp=timestamp,
                    data=payload,
                    confidence=best_zone.get("confidence", 1.0),
                    sources=["MARINE_DB_POSTGIS"],
                    warnings=[],
                    audit=AgentAudit(
                        inputs_used={"lat": lat, "lon": lon},
                        outputs=payload,
                        output_reason="Retrieved PFZ probability from active marine database zones.",
                        sources=["MARINE_DB_POSTGIS"],
                        score_source="MARINE_DB_POSTGIS",
                        score_reason="PFZ probability is based on known active zones from the marine database."
                    )
                )
            else:
                # 2. Attempt optional configured external provider
                # from backend-ORCA.services.data_fetchers.registry import get_pfz_provider
                provider = None # get_pfz_provider()
                if provider:
                    fetched_data = provider.fetch_pfz(lat, lon, timestamp)
                    if fetched_data:
                        try:
                            upsert_pfz_zone(fetched_data)
                        except Exception as e:
                            print(f"Failed to upsert pfz zone: {e}")
                        prob = fetched_data.get("confidence", 0.8)
                        from schemas.contracts import AgentAudit
                        payload = {
                            "pfz_probability": prob,
                            "pfz_present": prob > 0.5,
                            "pfz_signal_present": prob > 0.0,
                            "pfz_qualified": prob >= 0.85,
                            "suitability_score": fetched_data.get("suitability_score", 0.8),
                            "db_zones": [fetched_data],
                            "score_source": "MODEL_PREDICTION",
                            "score_reason": "PFZ probability was fetched from an external provider's live forecast."
                        }
                        return AgentResult(
                            agent=self.name,
                            status="SUCCESS",
                            location=GeoLocation(latitude=lat, longitude=lon, name=loc.name if loc else None),
                            timestamp=timestamp,
                            data=payload,
                            confidence=fetched_data.get("confidence", 1.0),
                            sources=["LIVE_PROVIDER"],
                            warnings=[],
                            audit=AgentAudit(
                                inputs_used={"lat": lat, "lon": lon},
                                outputs=payload,
                                output_reason="Fetched PFZ probability from an external provider.",
                                sources=["LIVE_PROVIDER"],
                                score_source="MODEL_PREDICTION",
                                score_reason="PFZ probability was fetched from an external provider's live forecast."
                            )
                        )
        except ImportError as e:
            print(f"PFZ DB/Provider Query bypassed: 'backend' module not available. Error: {e}")
        except Exception as e:
            print(f"PFZ DB/Provider Query failed: {e}")
            # 3. Fallback to ML model

        # 2. Fallback to XGBoost model
        from agents.geospatial.grid import offset_coordinate
        best_payload = None
        best_prob = -1.0
        best_lat, best_lon = nearest_lat, nearest_lon

        try:
            from global_land_mask import globe
        except ImportError:
            globe = None

        # Project outward to generate marine candidates (e.g. 15km offshore in different bearings)
        for b in [0.0, 90.0, 180.0, 270.0]:
            cand_lat, cand_lon = offset_coordinate(nearest_lat, nearest_lon, 15.0, b)
            if globe and globe.is_land(cand_lat, cand_lon):
                continue
            payload = predict(lat=cand_lat, lon=cand_lon, timestamp=timestamp, env_data=env_data)
            prob = payload["pfz_probability"]
            if prob > best_prob:
                best_prob = prob
                best_payload = payload
                best_lat, best_lon = cand_lat, cand_lon

        payload = best_payload
        prob = best_prob
        lat, lon = best_lat, best_lon

        status = "DEGRADED" if is_degraded else "SUCCESS"
        warnings = []
        if not payload["pfz_present"]:
            warnings.append("No Potential Fishing Zone identified at location")
        if is_degraded:
            warnings.append("DEGRADED: PFZ calculation based on synthetic upstream inputs — not reliable.")
            
        payload["score_source"] = "DEFAULT_FALLBACK" if is_degraded else "MODEL_PREDICTION"
        payload["score_reason"] = "PFZ probability is the XGBoost model probability for the environmental conditions at this location."
        if is_degraded:
            payload["score_reason"] += " Note: Synthetic upstream inputs were used, so this is a degraded prediction."

        from schemas.contracts import AgentAudit
        audit_data = payload.get("audit", {})
        return AgentResult(
            agent=self.name,
            status=status,
            location=GeoLocation(latitude=lat, longitude=lon, name=loc.name if loc else None),
            timestamp=timestamp,
            data=payload,
            confidence=prob if payload["pfz_present"] else (1.0 - prob),
            sources=["INCOIS_PFZ_XGBOOST_v1"],
            warnings=warnings,
            audit=AgentAudit(
                inputs_used=audit_data.get("inputs_used", {}),
                outputs=payload,
                output_reason="Evaluated environmental conditions to predict potential fishing zone presence.",
                sources=audit_data.get("sources", ["INCOIS_PFZ_XGBOOST_v1"]),
                score_source=payload["score_source"],
                score_reason=payload["score_reason"]
            )
        )
