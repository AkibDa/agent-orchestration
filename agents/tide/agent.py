import urllib.request
import json
import calendar
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
            with open("tide_real_error.log", "w") as f:
                f.write("ERROR: loc is None\n")
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
            
        target_time = None
        if hasattr(plan, "time") and hasattr(plan.time, "exact") and plan.time.exact:
            target_time = plan.time.exact
            
        if not target_time:
            target_time = datetime.now(timezone.utc)
            
        url = f"https://marine-api.open-meteo.com/v1/marine?latitude={loc.latitude}&longitude={loc.longitude}&hourly=sea_level_height_msl&timezone=auto"
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ORCA-Agent'})
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}")
                data = json.loads(response.read().decode())
                
                times = data["hourly"]["time"]
                levels = data["hourly"]["sea_level_height_msl"]
                
                dt_formats = "%Y-%m-%dT%H:%M"
                parsed_times = [datetime.strptime(t, dt_formats) for t in times]
                
                utc_offset = data.get("utc_offset_seconds", 0)
                target_timestamp = target_time.timestamp()
                
                timestamps = [calendar.timegm(pt.timetuple()) - utc_offset for pt in parsed_times]
                
                closest_idx = 0
                min_diff = float('inf')
                for i, ts in enumerate(timestamps):
                    diff = abs(ts - target_timestamp)
                    if diff < min_diff and levels[i] is not None:
                        min_diff = diff
                        closest_idx = i
                        
                current_height = levels[closest_idx]
                
                is_rising = True
                for i in range(closest_idx + 1, len(levels)):
                    if levels[i] is not None:
                        is_rising = levels[i] > current_height
                        break
                        
                next_high_time = None
                next_high_h = None
                next_low_time = None
                next_low_h = None
                
                for i in range(closest_idx + 1, len(levels) - 1):
                    if levels[i] is None: continue
                    
                    prev_l = levels[i-1]
                    curr_l = levels[i]
                    next_l = levels[i+1]
                    
                    if prev_l is not None and next_l is not None:
                        if curr_l > prev_l and curr_l >= next_l and next_high_time is None:
                            next_high_time = times[i]
                            next_high_h = curr_l
                        elif curr_l < prev_l and curr_l <= next_l and next_low_time is None:
                            next_low_time = times[i]
                            next_low_h = curr_l
                            
                    if next_high_time and next_low_time:
                        break
                        
                # Format to HH:MM to maintain existing expected format, but include day if needed.
                # Just HH:MM as the mockup had.
                def format_time(t_str):
                    if not t_str: return None
                    return t_str.split("T")[1]
                    
                tide_data = {
                    "status": "AVAILABLE",
                    "current_phase": "rising" if is_rising else "falling",
                    "current_height_m": current_height,
                    "next_high": {
                        "time": format_time(next_high_time),
                        "height_m": next_high_h
                    },
                    "next_low": {
                        "time": format_time(next_low_time),
                        "height_m": next_low_h
                    }
                }
                
                status_code = "SUCCESS"
                confidence = 0.95
                output_reason = "Tide data retrieved from OPEN_METEO."
                
        except Exception as e:
            import traceback
            with open("tide_real_error.log", "w") as f:
                f.write(f"Exception: {str(e)}\n")
                traceback.print_exc(file=f)
                f.write(f"target_time: {target_time}\n")
                if 'closest_idx' in locals():
                    f.write(f"closest_idx: {closest_idx}\n")
            
            tide_data = {
                "status": "UNAVAILABLE",
                "current_phase": None,
                "current_height_m": None,
                "next_high": {"time": None, "height_m": None},
                "next_low": {"time": None, "height_m": None}
            }
            status_code = "DEGRADED"
            confidence = 0.0
            output_reason = f"Failed to retrieve tide data: {str(e)}"

        return AgentResult(
            agent=self.name,
            status=status_code,
            location=loc,
            timestamp=datetime.now(timezone.utc),
            data=tide_data,
            confidence=confidence,
            sources=["OPEN_METEO"],
            audit=AgentAudit(
                inputs_used={"lat": loc.latitude, "lon": loc.longitude},
                outputs=tide_data,
                output_reason=output_reason,
                sources=["OPEN_METEO"],
                score_source="REAL_EXTERNAL_DATA",
                score_reason="Raw/current sea-level values: classification = REAL_EXTERNAL_DATA. Calculated high/low: classification = ORCA_DERIVED, derived_from = sea_level_height_msl. current_phase: classification = ORCA_DERIVED."
            )
        )
