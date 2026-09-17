# data_sources/imd.py

import requests
from datetime import datetime, timezone
from schemas.contracts import Observation, Forecast, WarningContract, Advisory, UNKNOWN

IMD_IDS = {
    "digha": 42901, "kochi": 43351, "chennai": 43279,
    "mumbai": 43003, "goa": 43192, "puri": 42973,
    "vizag": 43149, "default": 43192
}

def _get_imd_id(name: str) -> int:
    name_lower = name.lower()
    for k, v in IMD_IDS.items():
        if k in name_lower:
            return v
    return IMD_IDS["default"]

def get_tier1(district: str, sea_area: str):
    """
    Tier 1: current_wx, districtwarning, coastalbulletin, seabulletin, portwarning.
    Returns a dictionary of contracts.
    Throws HTTPError on 401 to trigger circuit breaker upstream.
    """
    from config import IMD_ENABLED, IMD_DISABLED_REASON

    timestamp = datetime.now(timezone.utc)

    if not IMD_ENABLED:
        res = {
            "weather": Observation(source=IMD_DISABLED_REASON, quality="OBSERVED", value=UNKNOWN, unit="complex", valid_time=timestamp),
            "official_marine_warning": WarningContract(source=IMD_DISABLED_REASON, quality="FORECAST", severity=UNKNOWN, warning_text=UNKNOWN, valid_time=timestamp),
            "coastal_bulletin": Advisory(source=IMD_DISABLED_REASON, quality="FORECAST", advisory_text=UNKNOWN, valid_time=timestamp)
        }
        return res

    station_id = _get_imd_id(district)
    source = "IMD"

    import os
    api_key = os.environ.get("IMD_API_KEY")
    if not api_key:
        raise ValueError("NO_CREDENTIALS_CONFIGURED: IMD_API_KEY is not set.")

    headers = {"Authorization": f"Bearer {api_key}"}

    res = {
        "weather": Observation(source=source, quality="OBSERVED", value=UNKNOWN, unit="complex"),
        "official_marine_warning": WarningContract(source=source, quality="FORECAST", severity=UNKNOWN, warning_text=UNKNOWN),
        "coastal_bulletin": Advisory(source=source, quality="FORECAST", advisory_text=UNKNOWN)
    }

    # Weather
    url_wx = f"https://api.imd.gov.in/api/v1/current_wx?id={station_id}"
    try:
        resp = requests.get(url_wx, headers=headers, timeout=5)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        res["weather"].value = {
            "wind_speed_ms": float(data.get("wind_speed", 0.0)) * (1000/3600),
            "wind_direction": str(data.get("wind_dir", "VAR")),
            "temperature_c": float(data.get("temp", 0.0))
        }
    except Exception as e:
        if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 401:
            raise e # Pass up for circuit breaker

    # Warning
    url_warn = f"https://api.imd.gov.in/api/v1/districtwarning?id={station_id}"
    try:
        resp = requests.get(url_warn, headers=headers, timeout=5)
        resp.raise_for_status()
        warn_data = resp.json().get("data", [])
        if warn_data:
            day1 = warn_data[0]
            res["official_marine_warning"].severity = day1.get("colorcode", "GREEN").upper()
            res["official_marine_warning"].warning_text = day1.get("warning", "No warning")
    except Exception as e:
        if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 401:
            raise e

    # Coastal Bulletin
    url_coast = f"https://api.imd.gov.in/api/v1/coastalbulletin?id={station_id}"
    try:
        resp = requests.get(url_coast, headers=headers, timeout=5)
        resp.raise_for_status()
        c_data = resp.json().get("data", {})
        res["coastal_bulletin"].advisory_text = c_data.get("wind_warning", "No warning")
    except Exception as e:
        if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 401:
            raise e

    for v in res.values():
        v.valid_time = timestamp

    return res

def get_tier2():
    """
    Tier 2: cyclone_track
    """
    from config import IMD_ENABLED, IMD_DISABLED_REASON

    timestamp = datetime.now(timezone.utc)

    if not IMD_ENABLED:
        return {
            "cyclone": WarningContract(source=IMD_DISABLED_REASON, quality="OBSERVED", severity=UNKNOWN, warning_text=UNKNOWN, valid_time=timestamp)
        }

    source = "IMD"
    res = {
        "cyclone": WarningContract(source=source, quality="OBSERVED", severity=UNKNOWN, warning_text=UNKNOWN)
    }

    url = "https://api.imd.gov.in/api/v1/cyclone_track"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        tracks = resp.json().get("data", [])
        if tracks:
            latest = tracks[-1]
            res["cyclone"].severity = str(latest.get("category", 1))
            res["cyclone"].warning_text = f"Cyclone {latest.get('name', 'System')} active."
        else:
            res["cyclone"].severity = "NONE"
            res["cyclone"].warning_text = "No active cyclone."
    except Exception as e:
        if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 401:
            raise e

    res["cyclone"].valid_time = timestamp
    return res
