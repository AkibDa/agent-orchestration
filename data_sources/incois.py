# data_sources/incois.py

from datetime import datetime, timezone
from typing import Optional, List, Dict
import requests
import logging

from schemas.contracts import Observation, PFZCandidate, Advisory, UNKNOWN

logger = logging.getLogger(__name__)

def load_ocean_datasets():
    """No longer needed; real-time fetching is used."""
    pass

def get_pfz_advisory(lat: float, lon: float, count: int = 1) -> List[PFZCandidate]:
    """
    Algorithmic Real-Time PFZ Fetch (Prototype Proxy)
    In a production system, this would authenticate to INCOIS and parse the daily PFZ multi-lingual PDF/images.
    For this prototype, we synthesize a live candidate if current SST is favorable, using Open-Meteo as the data source.
    """
    try:
        url_wx = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m"
        resp_wx = requests.get(url_wx, timeout=3.0)
        resp_wx.raise_for_status()
        data_wx = resp_wx.json().get("current", {})

        temp = data_wx.get("temperature_2m")
        # Optimal SST range for typical pelagic fish in Indian Ocean (26C - 30C)
        if temp is not None and 26.0 <= temp <= 30.0:
            candidates = []
            import random
            random.seed(int(lat*100) + int(lon*100)) # Stable randomness for same location
            for i in range(count):
                d_lat = random.uniform(-0.1, 0.1)
                d_lon = random.uniform(-0.1, 0.1)
                depth = random.uniform(30.0, 80.0)
                dist = random.uniform(5.0, 25.0)
                bearing = random.choice(["NE", "SE", "NW", "SW", "N", "S", "E", "W"])
                conf = random.uniform(0.75, 0.92)

                candidates.append(
                    PFZCandidate(
                        source="INCOIS_PFZ_PROXY",
                        quality="FORECAST",
                        confidence=round(conf, 2),
                        latitude=round(lat + d_lat, 4),
                        longitude=round(lon + d_lon, 4),
                        depth=round(depth, 1),
                        bearing_from_landmark=bearing,
                        distance_from_landmark=round(dist, 1),
                        advisory_date=datetime.now(timezone.utc),
                        validity_window="48h"
                    )
                )
            return candidates
    except Exception as e:
        logger.warning(f"Error synthesizing PFZ data: {e}")

    return []

def get_ocean_state_forecast(lat: float, lon: float, timestamp: datetime) -> Dict[str, Observation]:
    source = "INCOIS_OSF"

    res = {
        "wave_height": Observation(source=source, quality="FORECAST", value=UNKNOWN, unit="m"),
        "wave_period": Observation(source=source, quality="FORECAST", value=UNKNOWN, unit="s"),
        "wind_speed": Observation(source=source, quality="FORECAST", value=UNKNOWN, unit="m/s"),
        "sst": Observation(source=source, quality="FORECAST", value=UNKNOWN, unit="C"),
        "surface_current": Observation(source=source, quality="FORECAST", value=UNKNOWN, unit="m/s")
    }

    def fetch_marine():
        url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&current=wave_height,wave_period,ocean_current_velocity,ocean_current_direction"
        r = requests.get(url, timeout=3.0)
        r.raise_for_status()
        return r.json().get("current", {})

    def fetch_wx():
        url_wx = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m"
        r = requests.get(url_wx, timeout=3.0)
        r.raise_for_status()
        return r.json().get("current", {})

    import concurrent.futures
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(fetch_marine)
            f2 = executor.submit(fetch_wx)

            data = f1.result()
            data_wx = f2.result()

            wh = data.get("wave_height")
            wp = data.get("wave_period")

            if wh is not None:
                res["wave_height"].value = wh
            if wp is not None:
                res["wave_period"].value = wp

            temp = data_wx.get("temperature_2m")
            wind = data_wx.get("wind_speed_10m")

            if temp is not None:
                res["sst"].value = temp
            if wind is not None:
                res["wind_speed"].value = round(wind * (1000/3600), 2)

            current_vel = data.get("ocean_current_velocity")
            current_dir = data.get("ocean_current_direction")
            
            if current_vel is not None:
                res["surface_current"].value = round(current_vel * (1000/3600), 2) # Assume km/h to m/s

    except Exception as e:
        logger.warning(f"Error extracting OSF data: {e}")

    for k in res.values():
        k.valid_time = timestamp

    return res

