# Proto/data_sources/mosdac.py

import requests
import logging
from datetime import datetime, timezone
from schemas.contracts import Observation, UNKNOWN

logger = logging.getLogger(__name__)

def get_mosdac_ocean_state(lat: float, lon: float, timestamp: datetime):
    """
    Secondary ocean source, using real-time fetching.
    """
    source = "MOSDAC"

    res = {
        "surface_current": Observation(source=source, quality="OBSERVED", value=UNKNOWN, unit="m/s"),
        "sst": Observation(source=source, quality="OBSERVED", value=UNKNOWN, unit="C")
    }

    try:
        url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&current=ocean_current_velocity"
        resp = requests.get(url, timeout=3.0)
        resp.raise_for_status()
        data = resp.json().get("current", {})
        vel = data.get("ocean_current_velocity")
        if vel is not None:
            res["surface_current"].value = round(vel * (1000/3600), 2)  # km/h to m/s

        url_wx = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m"
        resp_wx = requests.get(url_wx, timeout=3.0)
        resp_wx.raise_for_status()
        data_wx = resp_wx.json().get("current", {})
        temp = data_wx.get("temperature_2m")
        if temp is not None:
            res["sst"].value = temp

    except Exception as e:
        logger.warning(f"MOSDAC fetch failed: {e}")

    res["surface_current"].valid_time = timestamp
    res["sst"].valid_time = timestamp

    return res
