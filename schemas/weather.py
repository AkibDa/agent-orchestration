# schemas/weather.py

from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class WeatherForecast(BaseModel):
    source_tag: str
    district: str
    timestamp: datetime
    wind_speed_ms: float
    wind_direction: str
    temperature_c: float
    msl_hpa: float
    total_precipitation_m: float
    cloud_cover: str
    
class CoastalBulletin(BaseModel):
    source_tag: str
    sea_area: str
    timestamp: datetime
    wind_warning: Optional[str] = None
    state_of_sea: Optional[str] = None
    
class FishermanWarning(BaseModel):
    source_tag: str
    coast: str
    timestamp: datetime
    alert_level: str
    warning_text: str

class CycloneBulletin(BaseModel):
    source_tag: str
    active_cyclone: bool
    name: Optional[str] = None
    distance_km: Optional[float] = None
    intensity_rank: int = 0
    warning_text: Optional[str] = None
