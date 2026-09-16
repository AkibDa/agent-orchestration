from datetime import datetime, timezone
from typing import Dict
from schemas.contracts import BaseAgent, QueryPlan, AgentResult, GeoLocation, AgentAudit

class TideAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "tide"

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
            
        # Mock tide logic for prototype (to ensure 0ms latency but return realistic data)
        import random
        # Use location to seed deterministically
        random.seed(int(loc.latitude * 100) + int(loc.longitude * 100))
        
        is_rising = random.choice([True, False])
        current_height = round(random.uniform(0.5, 2.5), 1)
        next_high_time = f"{random.randint(12, 18):02d}:{random.randint(0, 59):02d}"
        next_high_h = round(random.uniform(2.0, 3.0), 1)
        next_low_time = f"{random.randint(19, 23):02d}:{random.randint(0, 59):02d}"
        next_low_h = round(random.uniform(0.1, 0.8), 1)
        
        tide_data = {
            "status": "AVAILABLE",
            "current_phase": "rising" if is_rising else "falling",
            "current_height_m": current_height,
            "next_high": {
                "time": next_high_time,
                "height_m": next_high_h
            },
            "next_low": {
                "time": next_low_time,
                "height_m": next_low_h
            }
        }
        
        return AgentResult(
            agent=self.name,
            status="MOCKED",
            location=loc,
            timestamp=datetime.now(timezone.utc),
            data=tide_data,
            confidence=0.9,
            sources=["TIDE_MODEL_PROXY"],
            audit=AgentAudit(
                inputs_used={"lat": loc.latitude, "lon": loc.longitude},
                outputs=tide_data,
                output_reason="Generated mocked tide proxy data.",
                sources=["TIDE_MODEL_PROXY"],
                score_source="MOCKED_DATA",
                score_reason="Tide data is a mocked placeholder."
            )
        )
