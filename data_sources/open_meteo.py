# data_sources/open_meteo.py

import requests
import logging
from datetime import datetime, timezone
from location.gazetteer import GAZETTEER
from schemas.contracts import Observation, Advisory, WarningContract, UNKNOWN

logger = logging.getLogger(__name__)

def _get_coords(name: str) -> tuple[float, float]:
    name_lower = name.lower()
    for k, v in GAZETTEER.items():
        if k in name_lower or name_lower in k:
            return v[0], v[1]
    return 15.0, 75.0  # generic center fallback

def get_primary_weather(lat: float, lon: float, timeout: float = 10.0) -> dict:
    """
    Fetches comprehensive weather data from Open-Meteo, used when IMD is disabled.
    Includes past_hours=3 to compute deltas.
    """
    source = "OPEN_METEO"
    timestamp = datetime.now(timezone.utc)

    res = {
        "weather": Observation(source=source, quality="FORECAST", value=UNKNOWN, unit="complex"),
        "coastal_bulletin": Advisory(source=source, quality="FORECAST", advisory_text=UNKNOWN)
    }

    # Request current and hourly for the past 3 hours to compute deltas
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,surface_pressure,precipitation,wind_speed_10m,wind_direction_10m,cloud_cover&hourly=wind_speed_10m,surface_pressure&past_hours=3"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current", {})
        hourly = data.get("hourly", {})

        wind_kmh = current.get("wind_speed_10m", 0.0)
        wind_ms = wind_kmh * (1000 / 3600)
        pressure = current.get("surface_pressure", 1013.25)

        # Hourly data lists 3 hours of past data, then current hour, then future
        # past_hours=3 gives us index 0 as T-3h
        wind_delta_3h = 0.0
        pressure_delta_3h = 0.0

        if "wind_speed_10m" in hourly and len(hourly["wind_speed_10m"]) > 3:
            past_wind_kmh = hourly["wind_speed_10m"][0]
            if past_wind_kmh is not None and wind_kmh is not None:
                wind_delta_3h = (wind_kmh - past_wind_kmh) * (1000 / 3600)

        if "surface_pressure" in hourly and len(hourly["surface_pressure"]) > 3:
            past_pressure = hourly["surface_pressure"][0]
            if past_pressure is not None and pressure is not None:
                pressure_delta_3h = pressure - past_pressure

        res["weather"].value = {
            "wind_speed_ms": round(wind_ms, 2),
            "wind_direction": str(current.get("wind_direction_10m", 180)),
            "temperature_c": current.get("temperature_2m", 30.0),
            "msl_hpa": pressure,
            "total_precipitation_m": current.get("precipitation", 0.0) / 1000.0,  # mm to m
            "cloud_cover": current.get("cloud_cover", 0.0),
            "wind_delta_3h": round(wind_delta_3h, 2),
            "pressure_delta_3h": round(pressure_delta_3h, 2)
        }

        # Synthesize a coastal condition
        if wind_ms > 17.0:
            res["coastal_bulletin"].advisory_text = "Very Rough (Derived)"
        elif wind_ms > 10.0:
            res["coastal_bulletin"].advisory_text = "Rough (Derived)"
        else:
            res["coastal_bulletin"].advisory_text = "Slight to Moderate (Derived)"

    except Exception as e:
        logger.warning(f"Open-Meteo primary fetch failed: {e}")

    res["weather"].valid_time = timestamp
    res["coastal_bulletin"].valid_time = timestamp

    return res

def get_fallback_weather(location: str):
    """
    Legacy fallback using location name strings.
    """
    lat, lon = _get_coords(location)
    res = get_primary_weather(lat, lon)
    # Tag as fallback
    res["weather"].source = "OPEN_METEO_FALLBACK"
    res["coastal_bulletin"].source = "OPEN_METEO_FALLBACK"
    return res
