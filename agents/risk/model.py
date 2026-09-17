# agents/risk/model.py

import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import threading

_PREDICT_LOCK = threading.Lock()

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "marine_risk" / "marine_risk_xgboost_v2.json"
METADATA_PATH = REPO_ROOT / "models" / "marine_risk" / "marine_risk_v2_model_metadata.json"

EXPECTED_FEATURES = [
    "latitude", "longitude", "hour", "month", "sin_hour", "cos_hour",
    "sin_month", "cos_month", "u10", "v10", "wind_speed_10m", "t2m",
    "msl", "tp", "wind_gradient", "pressure_gradient", "temperature_gradient",
    "pressure_local_anomaly", "wind_local_anomaly", "wind_delta_3h",
    "pressure_delta_3h", "cyclone_distance_km", "cyclone_wind_kt",
    "cyclone_pressure_hpa", "cyclone_pressure_drop_hpa", "cyclone_category_rank",
    "active_cyclone_flag"
]

CLASS_MAP = {0: "SAFE", 1: "CAUTION", 2: "DANGER"}

_MARINE_RISK_MODEL = None
_METADATA = None
_T_DANGER = 0.15
_T_CAUTION = 0.40


def get_marine_risk_model() -> xgb.XGBClassifier:
    global _MARINE_RISK_MODEL, _METADATA
    if _MARINE_RISK_MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Marine Risk v2 model not found at {MODEL_PATH}")
        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_PATH))
        _MARINE_RISK_MODEL = model
        if METADATA_PATH.exists():
            _METADATA = json.loads(METADATA_PATH.read_text())
    return _MARINE_RISK_MODEL


def build_marine_risk_features(lat: float, lon: float, timestamp=None, env_data: dict = None) -> pd.DataFrame:
    """
    Constructs exact 27 features expected by marine_risk_xgboost_v2.json.
    """
    ts = pd.Timestamp(timestamp) if timestamp else pd.Timestamp.now()
    cur = env_data or {}

    u10 = float(cur.get("u10", cur.get("u_wind_10m", 5.0)))
    v10 = float(cur.get("v10", cur.get("v_wind_10m", 3.0)))
    w_speed = float(cur.get("wind_speed_10m", np.sqrt(u10**2 + v10**2)))
    msl = float(cur.get("msl", cur.get("mean_sea_level_pressure_pa", 101300.0)))
    t2m = float(cur.get("t2m", cur.get("temp_2m_k", 300.0)))
    tp = float(cur.get("tp", cur.get("total_precipitation_m", 0.0)))

    hour = int(ts.hour)
    month = int(ts.month)

    sin_hour = float(np.sin(2 * np.pi * hour / 24.0))
    cos_hour = float(np.cos(2 * np.pi * hour / 24.0))
    sin_month = float(np.sin(2 * np.pi * (month - 1) / 12.0))
    cos_month = float(np.cos(2 * np.pi * (month - 1) / 12.0))

    # Cyclone parameters with safe ocean defaults if no cyclone active
    cyclone_dist = float(cur.get("cyclone_distance_km", 9999.0))
    cyclone_wind = float(cur.get("cyclone_wind_kt", 0.0))
    cyclone_press = float(cur.get("cyclone_pressure_hpa", 1013.0))
    cyclone_drop = float(cur.get("cyclone_pressure_drop_hpa", 0.0))
    cyclone_rank = float(cur.get("cyclone_category_rank", 0.0))
    active_cyclone = float(cur.get("active_cyclone_flag", 1.0 if cyclone_dist < 500.0 else 0.0))

    row = {
        "latitude": float(lat),
        "longitude": float(lon),
        "hour": hour,
        "month": month,
        "sin_hour": sin_hour,
        "cos_hour": cos_hour,
        "sin_month": sin_month,
        "cos_month": cos_month,
        "u10": u10,
        "v10": v10,
        "wind_speed_10m": w_speed,
        "t2m": t2m,
        "msl": msl,
        "tp": tp,
        "wind_gradient": float(cur.get("wind_gradient", 0.0)),
        "pressure_gradient": float(cur.get("pressure_gradient", 0.0)),
        "temperature_gradient": float(cur.get("temperature_gradient", 0.0)),
        "pressure_local_anomaly": float(cur.get("pressure_local_anomaly", 0.0)),
        "wind_local_anomaly": float(cur.get("wind_local_anomaly", 0.0)),
        "wind_delta_3h": float(cur.get("wind_delta_3h", 0.0)),
        "pressure_delta_3h": float(cur.get("pressure_delta_3h", 0.0)),
        "cyclone_distance_km": cyclone_dist,
        "cyclone_wind_kt": cyclone_wind,
        "cyclone_pressure_hpa": cyclone_press,
        "cyclone_pressure_drop_hpa": cyclone_drop,
        "cyclone_category_rank": cyclone_rank,
        "active_cyclone_flag": active_cyclone
    }

    return pd.DataFrame([[row[f] for f in EXPECTED_FEATURES]], columns=EXPECTED_FEATURES)


def predict(features_or_lat, lon=None, timestamp=None, env_data: dict = None) -> dict:
    """
    Evaluates 6-hour-ahead Marine Hazard state using marine_risk_xgboost_v2.json.
    Supports either passing a features dictionary or lat/lon coordinates.
    """
    if isinstance(features_or_lat, dict):
        cur_env = features_or_lat
        lat = cur_env.get("latitude", 12.48)
        lon = cur_env.get("longitude", 74.40)
    else:
        lat = features_or_lat
        cur_env = env_data or {}

    model = get_marine_risk_model()
    X = build_marine_risk_features(lat, lon, timestamp, cur_env)

    with _PREDICT_LOCK:
        probs = model.predict_proba(X)[0].tolist()
        
    p0, p1, p2 = probs[0], probs[1], probs[2]

    # Operational safety thresholding
    if p2 >= _T_DANGER or cur_env.get("cyclone_distance_km", 9999.0) < 150.0:
        pred_class = 2
    elif (p1 + p2) >= _T_CAUTION:
        pred_class = 1
    else:
        pred_class = 0

    risk_label = CLASS_MAP[pred_class]

    allowed = bool(pred_class != 2 and cur_env.get("allowed", True))

    return {
        "latitude": float(lat),
        "longitude": float(lon),
        "risk_level": risk_label,
        "predicted_class_id": pred_class,
        "prediction_horizon_hours": 6,
        "allowed": allowed,
        "class_probabilities": {
            "SAFE": p0,
            "CAUTION": p1,
            "DANGER": p2
        },
        "model_version": "v2_xgboost",
        "source": "ORCA_XGBOOST_SUPPLEMENTARY",
        "features": X.to_dict(orient="records")[0],
        "score_source": "MODEL_PREDICTION",
        "score_reason": f"Risk level was predicted as {risk_label} by the marine risk XGBoost model based on current weather, ocean conditions, and cyclone distance.",
        "audit": {
            "inputs_used": X.to_dict(orient="records")[0],
            "outputs": {
                "risk_level": risk_label,
                "allowed": allowed
            },
            "output_reason": "Evaluated environmental inputs to predict supplementary marine risk.",
            "sources": ["ORCA_XGBOOST_MARINE_RISK_v2_SUPPLEMENTARY"],
            "score_source": "MODEL_PREDICTION",
            "score_reason": f"Risk level was predicted as {risk_label} by the marine risk XGBoost model based on current weather, ocean conditions, and cyclone distance."
        }
    }