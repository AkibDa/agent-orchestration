# agents/productivity/model.py

import json
import os
import math
from pathlib import Path
import numpy as np

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# pyrefly: ignore [missing-import]
import keras

PROTO_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PROTO_DIR

MODEL_PATH = REPO_ROOT / "models" / "fish_productivity" / "fish_productivity_lstm_12m_env.keras"
METADATA_PATH = REPO_ROOT / "Datasets" / "processed" / "training" / "fish_productivity_12m_env" / "metadata.json"

_MODEL_SINGLETON = None
_METADATA_SINGLETON = None

def get_productivity_model():
    global _MODEL_SINGLETON, _METADATA_SINGLETON
    if _MODEL_SINGLETON is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Fish productivity model file not found at {MODEL_PATH}")
        _MODEL_SINGLETON = keras.models.load_model(str(MODEL_PATH), compile=False)
        
        if METADATA_PATH.exists():
            _METADATA_SINGLETON = json.loads(METADATA_PATH.read_text())
        else:
            _METADATA_SINGLETON = {
                "feature_scaler": {
                    "mean": [8.444, 16.748, -8.305, 0.00027, 0.00027, 0.00027, 0.00027, 0.184, 0.0, 0.0, 3.87, 299.77, 100924.1, 301.55, 0.004],
                    "scale": [0.714, 0.461, 0.465, 0.00012, 0.00012, 0.00011, 0.00010, 1.376, 0.707, 0.707, 1.484, 1.729, 312.56, 0.842, 0.0028]
                },
                "target_scaler": {"mean": [0.0002742269], "scale": [0.0001231400]}
            }
    return _MODEL_SINGLETON, _METADATA_SINGLETON


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 10.0), -10.0)))


def predict_productivity(features: dict) -> dict:
    """
    Constructs a 12-month x 15-feature sequence tensor using environmental features
    (wind_speed_10m, t2m, msl, sst, tp) extracted from NetCDF datasets alongside baseline CPUE metrics.
    
    Returns:
    {
       "estimated_cpue": float,
       "productivity_score": float,
       "raw_scaled_prediction": float
    }
    """
    model, metadata = get_productivity_model()

    f_mean = np.array(metadata["feature_scaler"]["mean"], dtype=np.float32)
    f_scale = np.array(metadata["feature_scaler"]["scale"], dtype=np.float32)

    w_speed = float(features.get("wind_speed_10m", 3.87))
    t2m = float(features.get("t2m", 299.77))
    msl = float(features.get("msl", 100924.1))
    sst = float(features.get("sst", 301.55))
    tp = float(features.get("tp", 0.004))
    month = int(features.get("month", 8))

    # Construct 12 monthly timesteps for input tensor (shape: 1, 12, 15)
    seq = np.zeros((12, 15), dtype=np.float32)

    for step in range(12):
        m_idx = (month - 11 + step) % 12 + 1
        m_sin = math.sin(2.0 * math.pi * m_idx / 12.0)
        m_cos = math.cos(2.0 * math.pi * m_idx / 12.0)

        # 15 features: log_catch, log_effort, log_cpue, cpue_lag1, cpue_lag3, cpue_roll3, cpue_roll12, cpue_yoy_change, month_sin, month_cos, wind_speed_10m, t2m, msl, sst, tp
        step_raw = np.array([
            f_mean[0], f_mean[1], f_mean[2], f_mean[3], f_mean[4],
            f_mean[5], f_mean[6], f_mean[7], m_sin, m_cos,
            w_speed, t2m, msl, sst, tp
        ], dtype=np.float32)

        # Scale features using z-score normalization
        step_scaled = (step_raw - f_mean) / np.maximum(f_scale, 1e-7)
        seq[step] = step_scaled

    tensor_input = np.expand_dims(seq, axis=0) # shape: (1, 12, 15)

    pred_scaled = float(model.predict(tensor_input, verbose=0)[0, 0])

    t_mean = float(metadata["target_scaler"]["mean"][0])
    t_scale = float(metadata["target_scaler"]["scale"][0])

    estimated_cpue = (pred_scaled * t_scale) + t_mean
    productivity_score = sigmoid(pred_scaled)

    score_source = "MODEL_PREDICTION"
    score_reason = f"Productivity was predicted using the 12-month environmental sequence (including wind speed {w_speed:.2f} m/s, temp {t2m:.2f}K, MSL {msl:.1f}Pa, SST {sst:.2f}K, precip {tp:.3f}m) supplied to the LSTM model."

    inputs_used = {
        "wind_speed_10m": w_speed,
        "t2m": t2m,
        "msl": msl,
        "sst": sst,
        "tp": tp
    }

    audit = {
        "inputs_used": inputs_used,
        "outputs": {
            "productivity_score": float(round(productivity_score, 4)),
            "estimated_cpue": float(round(estimated_cpue, 6))
        },
        "output_reason": "Evaluated 12-month environmental sequences to forecast fish productivity.",
        "sources": ["FISH_PRODUCTIVITY_LSTM_12M_ENV_v1"],
        "score_source": score_source,
        "score_reason": score_reason
    }

    return {
        "estimated_cpue": float(round(estimated_cpue, 6)),
        "productivity_score": float(round(productivity_score, 4)),
        "raw_scaled_prediction": float(round(pred_scaled, 4)),
        "score_source": score_source,
        "score_reason": score_reason,
        "audit": audit
    }
