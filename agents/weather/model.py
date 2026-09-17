# agents/weather/model.py
"""SUPPLEMENTARY ORCA PREDICTION — not an official forecast."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import xgboost as xgb
import threading

_PREDICT_LOCK = threading.Lock()

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_STAGE1_PATH = REPO_ROOT / "models" / "weather_risk" / "weather_risk_v4_normal_hazard.json"
MODEL_STAGE2_PATH = REPO_ROOT / "models" / "weather_risk" / "weather_risk_v4_caution_danger.json"

EXPECTED_FEATURES = [
    "latitude", "longitude", "hour", "month", "sin_hour", "cos_hour",
    "sin_month", "cos_month", "u10", "v10", "wind_speed_10m",
    "t2m", "msl", "tp", "wind_gradient", "pressure_gradient",
    "temperature_gradient", "pressure_local_anomaly", "wind_local_anomaly",
    "wind_delta_3h", "pressure_delta_3h", "cyclone_distance_km",
    "cyclone_wind_kt", "cyclone_pressure_hpa", "cyclone_pressure_drop_hpa",
    "cyclone_category_rank", "cyclone_motion_speed_kmh", "cyclone_motion_direction",
    "cyclone_wind_change_6h", "cyclone_pressure_change_6h", "cyclone_active",
    "distance_km_capped", "distance_inverse", "distance_band_100km",
    "distance_band_200km", "distance_band_300km", "distance_band_500km",
    "category_wind_interaction", "category_distance_interaction",
    "wind_distance_interaction", "pressure_distance_interaction",
    "intensification_signal", "storm_severity_index"
]

CLASS_MAP = {0: "NORMAL", 1: "CAUTION", 2: "DANGEROUS"}

_STAGE1_MODEL = None
_STAGE2_MODEL = None

_T_HAZARD = 0.26
_T_DANGER = 0.47


def get_weather_model() -> tuple[xgb.XGBClassifier, xgb.XGBClassifier]:
    global _STAGE1_MODEL, _STAGE2_MODEL
    if _STAGE1_MODEL is None:
        _STAGE1_MODEL = xgb.XGBClassifier()
        _STAGE1_MODEL.load_model(str(MODEL_STAGE1_PATH))
    if _STAGE2_MODEL is None:
        _STAGE2_MODEL = xgb.XGBClassifier()
        _STAGE2_MODEL.load_model(str(MODEL_STAGE2_PATH))
    return _STAGE1_MODEL, _STAGE2_MODEL


def build_weather_features(lat: float, lon: float, timestamp=None, weather_context: dict = None, cyclone_context: dict = None) -> pd.DataFrame:
    """
    Constructs exact 43 features expected by weather_risk_v4.
    Features not available from IMD are imputed to 0.0 or safe defaults.
    """
    from schemas.contracts import UNKNOWN
    ts = pd.Timestamp(timestamp) if timestamp else pd.Timestamp.now()
    cur = weather_context or {}

    w_speed = 5.0
    msl_hpa = 1010.0
    temp_c = 30.0
    precip_m = 0.0
    wind_delta = 0.0
    pressure_delta = 0.0

    if cur.get("weather") and cur["weather"].value is not UNKNOWN:
        val = cur["weather"].value
        if isinstance(val, dict):
            w_speed = float(val.get("wind_speed_ms", 5.0))
            msl_hpa = float(val.get("msl_hpa", 1010.0))
            temp_c = float(val.get("temperature_c", 30.0))
            precip_m = float(val.get("total_precipitation_m", 0.0))
            wind_delta = float(val.get("wind_delta_3h", 0.0))
            pressure_delta = float(val.get("pressure_delta_3h", 0.0))

    u10 = w_speed * 0.707
    v10 = w_speed * 0.707
    msl = msl_hpa * 100.0 # to Pa
    t2m = temp_c + 273.15 # to K
    hour = int(ts.hour)
    month = int(ts.month)
    sin_hour = float(np.sin(2 * np.pi * hour / 24.0))
    cos_hour = float(np.cos(2 * np.pi * hour / 24.0))
    sin_month = float(np.sin(2 * np.pi * (month - 1) / 12.0))
    cos_month = float(np.cos(2 * np.pi * (month - 1) / 12.0))

    # Cyclone features
    c_dist = 9999.0
    c_wind = 0.0
    c_pres = 1013.0
    c_pdrop = 0.0
    c_rank = 0
    c_speed = 0.0
    c_dir = 0.0
    c_wchange = 0.0
    c_pchange = 0.0
    c_active = 0.0

    cat_to_rank = {"D": 1, "CS": 2, "SCS": 3, "VSCS": 4, "ESCS": 5, "SuCS": 6}

    if cyclone_context and cyclone_context.get("cyclone_data_status") == "ACTIVE_CYCLONE":
        active_list = cyclone_context.get("active_cyclones", [])
        if active_list:
            nearest = min(active_list, key=lambda x: x["distance_km"])
            c_dist = float(nearest["distance_km"])
            c_wind = float(nearest["max_wind_kt"])
            c_pres = float(nearest["central_pressure_hpa"])
            c_pdrop = max(0.0, 1013.0 - c_pres)
            c_rank = float(cat_to_rank.get(nearest["category"], 1))
            c_speed = float(nearest.get("motion_speed_kmh", 0.0))
            c_dir = float(nearest.get("motion_direction_deg", 0.0))
            c_active = 1.0

    d_cap = min(c_dist, 2000.0)
    d_inv = 1.0 / (c_dist + 1.0)

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
        "tp": precip_m,
        "wind_gradient": 0.0,
        "pressure_gradient": 0.0,
        "temperature_gradient": 0.0,
        "pressure_local_anomaly": 0.0,
        "wind_local_anomaly": 0.0,
        "wind_delta_3h": wind_delta,
        "pressure_delta_3h": pressure_delta,
        "cyclone_distance_km": c_dist,
        "cyclone_wind_kt": c_wind,
        "cyclone_pressure_hpa": c_pres,
        "cyclone_pressure_drop_hpa": c_pdrop,
        "cyclone_category_rank": c_rank,
        "cyclone_motion_speed_kmh": c_speed,
        "cyclone_motion_direction": c_dir,
        "cyclone_wind_change_6h": c_wchange,
        "cyclone_pressure_change_6h": c_pchange,
        "cyclone_active": c_active,
        "distance_km_capped": d_cap,
        "distance_inverse": d_inv,
        "distance_band_100km": float(c_dist <= 100),
        "distance_band_200km": float(c_dist <= 200),
        "distance_band_300km": float(c_dist <= 300),
        "distance_band_500km": float(c_dist <= 500),
        "category_wind_interaction": c_rank * c_wind,
        "category_distance_interaction": c_rank * d_inv,
        "wind_distance_interaction": w_speed * d_inv,
        "pressure_distance_interaction": c_pdrop * d_inv,
        "intensification_signal": c_wchange * c_rank,
        "storm_severity_index": (c_wind * c_rank) / max(c_dist, 1.0)
    }

    return pd.DataFrame([[row[f] for f in EXPECTED_FEATURES]], columns=EXPECTED_FEATURES)


def predict(lat: float, lon: float, timestamp=None, weather_context: dict = None, cyclone_context: dict = None) -> dict:
    model1, model2 = get_weather_model()
    X = build_weather_features(lat, lon, timestamp, weather_context, cyclone_context)

    with _PREDICT_LOCK:
        # Stage 1: Normal (0) vs Hazard (1)
        probs1 = model1.predict_proba(X)[0].tolist()
        p_hazard = probs1[1]
    
        # Stage 2: Caution (0) vs Danger (1)
        probs2 = model2.predict_proba(X)[0].tolist()
        p_danger_given_hazard = probs2[1]

    # Combine probabilities
    p0 = probs1[0]
    p1 = p_hazard * probs2[0]
    p2 = p_hazard * probs2[1]

    # Safety-aware operational thresholding based on metadata
    if p_hazard >= _T_HAZARD:
        if p_danger_given_hazard >= _T_DANGER:
            pred_class = 2
        else:
            pred_class = 1
    else:
        pred_class = 0

    risk_label = CLASS_MAP[pred_class]

    # Confidence metrics
    max_prob = max(p0, p1, p2)
    eps = 1e-12
    entropy = float(-np.sum([p * np.log2(p + eps) for p in [p0, p1, p2]]))

    cyclone_avail = True
    if cyclone_context and cyclone_context.get("cyclone_data_status") == "CYCLONE_DATA_UNAVAILABLE":
        cyclone_avail = False

    score_source = "MODEL_PREDICTION"
    score_reason = f"Weather risk was classified as {risk_label} by the hierarchical XGBoost model using available wind, pressure, temperature, precipitation, and cyclone features."
    if not cyclone_avail:
        score_reason += " Note: Cyclone data was unavailable and default/imputed values were used."

    inputs_used = {
        "wind_speed_10m": float(X["wind_speed_10m"].iloc[0]),
        "msl": float(X["msl"].iloc[0]),
        "t2m": float(X["t2m"].iloc[0]),
        "tp": float(X["tp"].iloc[0]),
        "cyclone_distance_km": float(X["cyclone_distance_km"].iloc[0]),
        "cyclone_wind_kt": float(X["cyclone_wind_kt"].iloc[0]),
        "cyclone_category_rank": float(X["cyclone_category_rank"].iloc[0])
    }

    audit = {
        "inputs_used": inputs_used,
        "outputs": {
            "risk_level": risk_label,
            "max_probability": max_prob
        },
        "output_reason": "Evaluated atmospheric and cyclone conditions to predict weather risk.",
        "sources": ["ORCA_WEATHER_RISK_XGBOOST_v4"],
        "score_source": score_source,
        "score_reason": score_reason
    }

    return {
        "latitude": float(lat),
        "longitude": float(lon),
        "risk_level": risk_label,
        "predicted_class_id": pred_class,
        "class_probabilities": {
            "NORMAL": p0,
            "CAUTION": p1,
            "DANGEROUS": p2
        },
        "safety_thresholds": {
            "t_hazard": _T_HAZARD,
            "t_danger_given_hazard": _T_DANGER
        },
        "max_probability": max_prob,
        "cyclone_data_available": cyclone_avail,
        "features_imputed": ["wind_gradient", "pressure_gradient", "temperature_gradient", "pressure_local_anomaly", "wind_local_anomaly"],
        "score_source": score_source,
        "score_reason": score_reason,
        "audit": audit
    }
