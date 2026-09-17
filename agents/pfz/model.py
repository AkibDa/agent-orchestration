# agents/pfz/model.py

import json
from pathlib import Path
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import xgboost as xgb
import threading

_PREDICT_LOCK = threading.Lock()

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "pfz" / "pfz_xgboost.json"
CONFIG_PATH = REPO_ROOT / "models" / "pfz" / "config.json"

EXPECTED_FEATURES = [
    "latitude", "longitude", "era_sst", "era_u10", "era_v10",
    "era_t2m", "era_msl", "era_tp", "wind_speed", "current_u",
    "current_v", "current_speed", "current_direction_rad", "month", "day_of_year"
]

_PFZ_MODEL = None
_DECISION_THRESHOLD = 0.85


def get_pfz_model() -> xgb.XGBClassifier:
    global _PFZ_MODEL, _DECISION_THRESHOLD
    if _PFZ_MODEL is None:
        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_PATH))
        _PFZ_MODEL = model
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
                _DECISION_THRESHOLD = float(cfg.get("decision_threshold", 0.85))
    return _PFZ_MODEL


def build_pfz_features(lat: float, lon: float, timestamp=None, env_data: dict = None) -> pd.DataFrame:
    """
    Constructs exact 15 features expected by pfz_xgboost.json from environmental data or reasonable defaults.
    """
    ts = pd.Timestamp(timestamp) if timestamp else pd.Timestamp.now()
    env = env_data or {}
    
    # Check if this is the wrapper payload from ocean agent
    # Determine which layer of context to pull from
    if "incois_ocean_state" in env:
        ocean_state = env.get("incois_ocean_state", {})
    else:
        ocean_state = env

    weather_state = env.get("weather", env)
    
    # Wind logic: use live wind speed/direction if available, otherwise fallback to era
    wind_speed_val = weather_state.get("wind_speed_10m")
    if wind_speed_val is None:
        wind_speed_val = ocean_state.get("wind_speed")
    if wind_speed_val is None:
        wind_speed_val = 3.87
    wind_speed = float(wind_speed_val)

    wind_dir_val = weather_state.get("wind_direction_10m")
    if wind_dir_val is None:
        wind_dir_val = 180.0
    wind_dir = float(wind_dir_val)

    # Convert speed/direction to u/v vectors
    u10 = float(wind_speed * np.sin(np.radians(wind_dir)))
    v10 = float(wind_speed * np.cos(np.radians(wind_dir)))

    # Surface current logic: use live ocean current if available
    curr_speed_val = ocean_state.get("current_speed")
    if curr_speed_val is None:
        curr_speed_val = ocean_state.get("surface_current")
    if curr_speed_val is None:
        curr_speed_val = 0.15
    curr_speed = float(curr_speed_val)

    curr_dir_val = ocean_state.get("current_direction")
    if curr_dir_val is None:
        curr_dir_val = 90.0
    curr_dir = float(curr_dir_val)

    curr_u = float(curr_speed * np.sin(np.radians(curr_dir)))
    curr_v = float(curr_speed * np.cos(np.radians(curr_dir)))

    curr_dir_rad = float(np.arctan2(curr_v, curr_u))

    # Safely extract SST
    sst_val = ocean_state.get("sst")
    if sst_val is None:
        sst_val = ocean_state.get("sea_surface_temperature_c")
    if sst_val is None:
        sst_val = ocean_state.get("era_sst")
    if sst_val is None:
        sst_val = 301.5
    era_sst = float(sst_val)

    # Safely extract temperature
    t2m_val = weather_state.get("t2m")
    if t2m_val is None:
        t2m_val = weather_state.get("temperature_2m")
    if t2m_val is None:
        t2m_val = weather_state.get("era_t2m")
    if t2m_val is None:
        t2m_val = 298.2
    era_t2m = float(t2m_val)

    # Safely extract MSL pressure
    msl_val = weather_state.get("msl")
    if msl_val is None:
        msl_val = weather_state.get("surface_pressure")
    if msl_val is None:
        msl_val = weather_state.get("era_msl")
    if msl_val is None:
        msl_val = 101200.0
    era_msl = float(msl_val)

    # Safely extract precipitation
    tp_val = weather_state.get("tp")
    if tp_val is None:
        tp_val = weather_state.get("precipitation")
    if tp_val is None:
        tp_val = weather_state.get("era_tp")
    if tp_val is None:
        tp_val = 0.001
    era_tp = float(tp_val)

    row = {
        "latitude": float(lat),
        "longitude": float(lon),
        "era_sst": era_sst,
        "era_u10": u10,
        "era_v10": v10,
        "era_t2m": era_t2m,
        "era_msl": era_msl,
        "era_tp": era_tp,
        "wind_speed": wind_speed,
        "current_u": curr_u,
        "current_v": curr_v,
        "current_speed": curr_speed,
        "current_direction_rad": curr_dir_rad,
        "month": int(ts.month),
        "day_of_year": int(ts.dayofyear)
    }

    return pd.DataFrame([[row[f] for f in EXPECTED_FEATURES]], columns=EXPECTED_FEATURES)


def predict(lat: float, lon: float, timestamp=None, env_data: dict = None) -> dict:
    model = get_pfz_model()
    X = build_pfz_features(lat, lon, timestamp, env_data)
    
    with _PREDICT_LOCK:
        probs = model.predict_proba(X)[0]
    
    pfz_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
    pfz_qualified = bool(pfz_prob >= _DECISION_THRESHOLD)
    pfz_signal_present = bool(pfz_prob > 0.0) # Any non-zero signal

    return {
        "latitude": float(lat),
        "longitude": float(lon),
        "pfz_probability": pfz_prob,
        "pfz_present": pfz_qualified, # Legacy compat
        "pfz_signal_present": pfz_signal_present,
        "pfz_qualified": pfz_qualified,
        "decision_threshold": _DECISION_THRESHOLD,
        "model_version": "v1_xgboost",
        "features": X.to_dict(orient="records")[0]
    }