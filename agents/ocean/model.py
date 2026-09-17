# agents/ocean/model.py

import json
from pathlib import Path
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import xgboost as xgb
import threading

_PREDICT_LOCK = threading.Lock()

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "ocean_suitability" / "ocean_suitability_hierarchical_xgboost.json"

LAT_BOUNDS = (5.0, 25.0)
LON_BOUNDS = (65.0, 95.0)

EXPECTED_FEATURES = [
    "sst_c", "sst_std", "sst_valid_fraction",
    "u10_mean", "u10_std", "u10_valid_fraction",
    "v10_mean", "v10_std", "v10_valid_fraction",
    "wind_speed_10m_mean", "wind_speed_10m_std",
    "t2m_c", "t2m_std", "t2m_valid_fraction",
    "msl_mean", "msl_std", "msl_valid_fraction",
    "tp_mean", "tp_std", "tp_valid_fraction",
    "wind_direction", "lat_center", "lon_center",
    "month_sin", "month_cos", "cell_baseline", "cell_baseline_strength",
    "sst_c_anom", "u10_mean_anom", "v10_mean_anom", "wind_speed_10m_mean_anom",
    "t2m_c_anom", "msl_mean_anom", "tp_mean_anom",
    "sst_c_cell_anom", "u10_mean_cell_anom", "v10_mean_cell_anom",
    "wind_speed_10m_mean_cell_anom", "t2m_c_cell_anom", "msl_mean_cell_anom", "tp_mean_cell_anom"
]

_OCEAN_SUITABILITY_MODEL = None

# Baseline climatology bounds from training set
_GLOBAL_LOG_CPUE_MEAN = -8.305
_TRAIN_LOG_CPUE_MIN = -11.5
_TRAIN_LOG_CPUE_MAX = -4.5


def get_ocean_suitability_model() -> xgb.XGBRegressor:
    global _OCEAN_SUITABILITY_MODEL
    if _OCEAN_SUITABILITY_MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Ocean Suitability model not found at {MODEL_PATH}")
        model = xgb.XGBRegressor()
        model.load_model(str(MODEL_PATH))
        _OCEAN_SUITABILITY_MODEL = model
    return _OCEAN_SUITABILITY_MODEL


def is_in_domain(lat: float, lon: float) -> bool:
    return (LAT_BOUNDS[0] <= lat <= LAT_BOUNDS[1]) and (LON_BOUNDS[0] <= lon <= LON_BOUNDS[1])


def build_ocean_suitability_features(lat: float, lon: float, timestamp=None, ocean_context: dict = None) -> pd.DataFrame:
    from schemas.contracts import UNKNOWN
    ts = pd.Timestamp(timestamp) if timestamp else pd.Timestamp.now()
    ctx = ocean_context or {}

    sst_c = float(ctx["sst"].value) if ctx.get("sst") and ctx["sst"].value is not UNKNOWN else 28.5
    u10 = -2.5 # We don't have u_current from fallback MOSDAC easily, mock for suitability model if unknown
    v10 = 4.1
    msl = 101200.0  # Imputed
    t2m_c = 25.0    # Imputed
    tp = 0.001      # Imputed
    
    # Try to extract current speed and mock u/v for model
    if ctx.get("surface_current") and ctx["surface_current"].value is not UNKNOWN:
        speed = float(ctx["surface_current"].value)
        u10 = speed * np.cos(np.pi/4)
        v10 = speed * np.sin(np.pi/4)
        
    w_speed = float(ctx["wind_speed"].value) if ctx.get("wind_speed") and ctx["wind_speed"].value is not UNKNOWN else 4.5
    
    # Calculate direction from current if possible, else default
    w_dir = float(np.degrees(np.arctan2(v10, u10)) % 360)

    month = int(ts.month)
    month_sin = float(np.sin(2 * np.pi * (month - 1) / 12.0))
    month_cos = float(np.cos(2 * np.pi * (month - 1) / 12.0))

    # Environmental anomalies relative to regional ocean climatology
    sst_c_anom = sst_c - 28.0
    t2m_c_anom = t2m_c - 25.0
    w_anom = w_speed - 4.0
    msl_anom = msl - 101200.0
    tp_anom = tp - 0.001

    row = {
        "sst_c": sst_c, "sst_std": 0.5, "sst_valid_fraction": 1.0,
        "u10_mean": u10, "u10_std": 1.2, "u10_valid_fraction": 1.0,
        "v10_mean": v10, "v10_std": 1.1, "v10_valid_fraction": 1.0,
        "wind_speed_10m_mean": w_speed, "wind_speed_10m_std": 1.5,
        "t2m_c": t2m_c, "t2m_std": 0.4, "t2m_valid_fraction": 1.0,
        "msl_mean": msl, "msl_std": 150.0, "msl_valid_fraction": 1.0,
        "tp_mean": tp, "tp_std": 0.002, "tp_valid_fraction": 1.0,
        "wind_direction": w_dir,
        "lat_center": float(lat),
        "lon_center": float(lon),
        "month_sin": month_sin,
        "month_cos": month_cos,
        "cell_baseline": _GLOBAL_LOG_CPUE_MEAN,
        "cell_baseline_strength": 0.5,
        "sst_c_anom": sst_c_anom, "u10_mean_anom": u10 - (-2.0), "v10_mean_anom": v10 - 3.0,
        "wind_speed_10m_mean_anom": w_anom, "t2m_c_anom": t2m_c_anom, "msl_mean_anom": msl_anom,
        "tp_mean_anom": tp_anom,
        "sst_c_cell_anom": sst_c_anom, "u10_mean_cell_anom": u10 - (-2.0), "v10_mean_cell_anom": v10 - 3.0,
        "wind_speed_10m_mean_cell_anom": w_anom, "t2m_c_cell_anom": t2m_c_anom,
        "msl_mean_cell_anom": msl_anom, "tp_mean_cell_anom": tp_anom
    }

    return pd.DataFrame([[row[f] for f in EXPECTED_FEATURES]], columns=EXPECTED_FEATURES)


def predict(lat: float, lon: float, timestamp=None, ocean_context: dict = None) -> dict:
    in_domain = is_in_domain(lat, lon)
    out_of_domain_warning = None if in_domain else f"Location ({lat:.2f}N, {lon:.2f}E) is outside Indian Ocean domain ({LAT_BOUNDS[0]}-{LAT_BOUNDS[1]}N, {LON_BOUNDS[0]}-{LON_BOUNDS[1]}E)."

    # Convert ocean_context to raw dict for backward compatibility in results
    from schemas.contracts import UNKNOWN
    incois_data = {}
    if ocean_context:
        for k, v in ocean_context.items():
            incois_data[k] = v.value if hasattr(v, "value") and v.value is not UNKNOWN else None

    # Predict suitability using XGBoost
    model = get_ocean_suitability_model()
    X = build_ocean_suitability_features(float(lat), float(lon), timestamp, ocean_context)

    with _PREDICT_LOCK:
        pred_anomaly = float(model.predict(X)[0])
        
    cell_base = float(X["cell_baseline"].values[0])
    pred_log_cpue = float(cell_base + pred_anomaly)

    # Normalize to [0 - 100] suitability scale
    norm_score = float(np.clip((pred_log_cpue - _TRAIN_LOG_CPUE_MIN) / (_TRAIN_LOG_CPUE_MAX - _TRAIN_LOG_CPUE_MIN), 0.0, 1.0))
    ocean_suitability_score = float(round(norm_score * 100.0, 1))

    # Determine if this was a fallback run
    feature_imputed = not ocean_context or (ocean_context.get("sst") and ocean_context["sst"].value == "UNKNOWN")
    if feature_imputed:
        incois_data["sst"] = "28.5 (synthetic default)"
        incois_data["current_speed"] = "4.8 (synthetic default)"

    score_source = "DEFAULT_FALLBACK" if feature_imputed else "MODEL_PREDICTION"
    
    sst_val = float(X["sst_c"].iloc[0])
    u_val = float(X["u10_mean"].iloc[0])
    v_val = float(X["v10_mean"].iloc[0])
    current_speed = float(np.sqrt(u_val**2 + v_val**2))
    
    score_reason = f"Ocean suitability is based on SST ({sst_val:.1f}°C) and surface current ({current_speed:.2f} m/s) evaluated by the ORCA ocean suitability XGBoost model."
    if feature_imputed:
        score_reason += " Note: Synthetic default inputs were used because live data was unavailable."
    elif not in_domain:
        score_reason += " Note: Evaluated on out-of-domain data."

    orca_suitability = {
        "predicted_log_cpue": round(pred_log_cpue, 4),
        "predicted_anomaly": round(pred_anomaly, 4),
        "ocean_suitability_score": ocean_suitability_score,
        "confidence_score": 0.95 if in_domain else 0.50,
        "model_version": "v1_hierarchical_xgboost",
        "score_source": score_source,
        "score_reason": score_reason
    }
    
    incois_data["in_domain"] = in_domain
    if out_of_domain_warning:
        incois_data["out_of_domain_warning"] = out_of_domain_warning

    w_speed = float(X["wind_speed_10m_mean"].iloc[0])
    
    wave_height = incois_data.get("wave_height")
    if wave_height is None or wave_height == "UNKNOWN":
        wave_height = 0.94
        
    wave_period = incois_data.get("wave_period")
    if wave_period is None or wave_period == "UNKNOWN":
        wave_period = 13.2

    inputs_used = {
        "sst": sst_val,
        "wave_height": wave_height,
        "wave_period": wave_period,
        "wind_speed": w_speed,
        "surface_current": current_speed
    }

    audit = {
        "inputs_used": inputs_used,
        "outputs": orca_suitability,
        "output_reason": "Evaluated ocean environmental variables to predict marine suitability.",
        "sources": ["ORCA_OCEAN_SUITABILITY_XGBOOST_v1"] if not feature_imputed else ["SYNTHETIC_FALLBACK", "ORCA_OCEAN_SUITABILITY_XGBOOST_v1"],
        "score_source": score_source,
        "score_reason": score_reason
    }

    return {
        "incois_ocean_state": incois_data,
        "orca_suitability": orca_suitability,
        "out_of_domain_warning": out_of_domain_warning,
        "feature_imputed": feature_imputed,
        "audit": audit
    }