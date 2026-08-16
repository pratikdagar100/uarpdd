"""Prediction assembly: calibrated probability, classification, confidence,
uncertainty level, conformal prediction set and a heuristic per-prediction
feature-influence explanation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AppConfig
from .uncertainty import (RAIN, NO_RAIN, prediction_set, confidence_label,
                          uncertainty_label)

FEATURE_LABELS = {
    "rainfall_today": "Rainfall today",
    "rainfall_lag_1": "Rainfall previous day",
    "rainfall_lag_2": "Rainfall 2 days ago",
    "rainfall_lag_3": "Rainfall 3 days ago",
    "rainfall_rolling_3": "Rainfall last 3 days",
    "rainfall_rolling_7": "Rainfall last 7 days",
    "rain_today_flag": "Rain today (yes/no)",
    "avg_temp": "Average temperature",
    "min_temp": "Minimum temperature",
    "max_temp": "Maximum temperature",
    "temp_range": "Temperature range",
    "temp_change_1d": "Temperature change vs yesterday",
    "wind_speed": "Wind speed",
    "air_pressure": "Air pressure",
    "pressure_change_1d": "Pressure change vs yesterday",
    "humidity": "Humidity",
    "cloud": "Cloud cover",
    "month_num": "Month",
    "doy_sin": "Time of year (seasonal cycle)",
    "doy_cos": "Time of year (seasonal cycle)",
    "season_code": "Season",
    "station_code": "Station identity",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "elevation": "Elevation",
}


def calibrated_rain_probability(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    X = X[bundle["feature_cols"]].astype(np.float32).fillna(bundle["train_medians"])
    p_raw = bundle["model"].predict_proba(X)[:, 1]
    return np.clip(bundle["calibrator"].predict(p_raw), 0.0, 1.0)


def predict_one(bundle: dict, row: pd.DataFrame, cfg: AppConfig) -> dict:
    """Full uncertainty-aware prediction for a single feature row."""
    p_rain = float(calibrated_rain_probability(bundle, row)[0])
    is_rain = p_rain >= cfg.classification_threshold
    label = RAIN if is_rain else NO_RAIN
    p_pred_class = p_rain if is_rain else 1.0 - p_rain
    pset = prediction_set(p_rain, bundle["qhat"])
    return {
        "label": label,
        "is_rain": is_rain,
        "p_rain": p_rain,
        "p_no_rain": 1.0 - p_rain,
        "p_pred_class": p_pred_class,
        "confidence": confidence_label(p_pred_class, cfg),
        "uncertainty": uncertainty_label(pset, p_pred_class, cfg),
        "prediction_set": pset,
        "ambiguous": len(pset) == 2,
        "coverage": cfg.conformal_coverage,
        "threshold": cfg.classification_threshold,
    }


def climatology_row(features: pd.DataFrame, station: str,
                    target_date: pd.Timestamp, window_days: int = 10
                    ) -> tuple[pd.DataFrame, dict]:
    """Feature row for a date beyond the dataset, built from the station's
    seasonal climatology.

    Historical observations within ±window_days of the target day-of-year
    (across all years) are averaged to form 'typical conditions for this time
    of year'; calendar features are set to the actual target date. The result
    is a CLIMATOLOGY-BASED estimate, not a forecast from current
    observations — callers must label it accordingly.
    """
    sdf = features[features["station"] == station]
    doy = int(target_date.dayofyear)
    d = sdf["date"].dt.dayofyear
    dist = (d - doy).abs()
    dist = np.minimum(dist, 365 - dist)          # wrap around new year
    window = sdf[dist <= window_days]
    if window.empty:
        window = sdf
    num = window.select_dtypes("number")
    row = num.mean().to_frame().T

    # Calendar features reflect the real target date
    row["month_num"] = float(target_date.month)
    row["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    row["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    if "season_code" in window.columns:
        row["season_code"] = float(window["season_code"].mode().iloc[0])
    if "station_code" in window.columns:
        row["station_code"] = float(window["station_code"].iloc[0])

    info = {
        "n_history_days": int(len(window)),
        "window_days": window_days,
        "history_years": sorted(window["date"].dt.year.unique().tolist()),
        "typical_rain_rate": float(window["RainTomorrow"].mean()),
    }
    return row, info


def feature_influence(bundle: dict, row: pd.Series, train_stats: pd.DataFrame,
                      top_n: int = 6) -> list[dict]:
    """Heuristic per-prediction 'model feature influence'.

    Combines the model's global feature importance with how unusual the
    current value is relative to the training distribution (|z-score|).
    This is an interpretability aid, not a causal attribution — the UI labels
    it accordingly.
    """
    model = bundle["model"]
    cols = bundle["feature_cols"]
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    else:
        imp = np.ones(len(cols)) / len(cols)

    mu = train_stats["mean"].reindex(cols).to_numpy()
    sd = train_stats["std"].reindex(cols).replace(0, np.nan).to_numpy()
    vals = row.reindex(cols).astype(float).to_numpy()
    z = np.abs((vals - mu) / sd)
    z = np.nan_to_num(z, nan=0.0)
    score = imp * (0.5 + np.minimum(z, 3.0))
    if score.sum() > 0:
        score = score / score.sum()

    order = np.argsort(score)[::-1]
    seen_labels, out = set(), []
    for i in order:
        lbl = FEATURE_LABELS.get(cols[i], cols[i])
        if lbl in seen_labels:
            continue
        seen_labels.add(lbl)
        s = score[i]
        band = "HIGH" if s >= 0.12 else ("MODERATE" if s >= 0.05 else "LOW")
        out.append({"feature": lbl, "column": cols[i],
                    "value": None if np.isnan(vals[i]) else float(vals[i]),
                    "influence_score": float(s), "influence": band})
        if len(out) >= top_n:
            break
    return out
