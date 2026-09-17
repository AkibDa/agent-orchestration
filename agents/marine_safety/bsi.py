# agents/marine_safety/bsi.py
"""
ORCA implementation of the publicly described INCOIS BSI/SVAS structure.
Thresholds are ORCA assumptions unless otherwise noted.
"""

from typing import List, Dict, Optional
import math
from pydantic import BaseModel

class BoatProfile(BaseModel):
    max_hs_m: float = 2.0
    beam_width_m: float = 3.0
    length_m: float = 10.0

class SeaState(BaseModel):
    hs_m: float
    tz_s: float
    wave_dir_deg: float
    swell_dir_deg: Optional[float] = None
    wind_speed_ms: float

class FishingSafetyReport(BaseModel):
    bsi: float
    advisory: str
    hazards: List[str]
    is_safe: bool

def calculate_wavelength(tz: float) -> float:
    # Deep water wavelength approximation: L = (g * T^2) / (2 * pi)
    return (9.81 * tz**2) / (2 * math.pi)

def angle_diff(a: float, b: float) -> float:
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)

def evaluate_conditions(state: SeaState, boat: BoatProfile) -> FishingSafetyReport:
    hazards = []
    bsi = 0.0
    
    # 1. Boat-specific Hs cap
    if state.hs_m > boat.max_hs_m:
        hazards.append(f"WAVE_HEIGHT_CAP_EXCEEDED (Hs {state.hs_m:.1f}m > Max {boat.max_hs_m}m)")
        bsi += 100.0

    # 2. Steepness Index
    wavelength = calculate_wavelength(state.tz_s)
    steepness = state.hs_m / max(wavelength, 1.0)
    if steepness > 0.045: # ORCA assumption threshold
        hazards.append(f"DANGEROUS_STEEPNESS ({steepness:.3f})")
        bsi += 30.0
    elif steepness > 0.03:
        bsi += 10.0

    # 3. Crossing-sea index
    if state.swell_dir_deg is not None:
        crossing_angle = angle_diff(state.wave_dir_deg, state.swell_dir_deg)
        if crossing_angle > 30.0 and state.hs_m > 1.5: # ORCA assumption
            hazards.append(f"CROSSING_SEAS (Angle {crossing_angle:.1f} deg)")
            bsi += 20.0

    # 4. Rapid wind-sea growth (proxy using wind speed)
    if state.wind_speed_ms > 10.0: # ~20 knots
        hazards.append(f"HIGH_WIND ({state.wind_speed_ms:.1f} m/s)")
        bsi += 20.0

    advisory = "SAFE"
    is_safe = True
    
    if bsi >= 50.0:
        advisory = "DANGER"
        is_safe = False
    elif bsi >= 30.0:
        advisory = "WARNING"
        is_safe = False
    elif bsi >= 10.0:
        advisory = "CAUTION"
    
    return FishingSafetyReport(
        bsi=bsi,
        advisory=advisory,
        hazards=hazards,
        is_safe=is_safe
    )
