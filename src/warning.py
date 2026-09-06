"""Heavy-rainfall early-warning and inundation-risk logic.

Addresses SIH problem statement 26071 (MoES / India Meteorological
Department): an AI/ML-based integrated heavy rainfall early warning and
inundation prediction system.

The model provides a calibrated probability of significant next-day rain
plus a conformal prediction set. This module turns that into an IMD-style
colour-coded warning and a transparent inundation risk index:

* **Rainfall intensity categories** follow the IMD 24-hour definitions
  (light / moderate / heavy / very heavy / extremely heavy).
* **Warning level** (GREEN / YELLOW / ORANGE / RED, mirroring IMD's
  "No action / Be updated / Be prepared / Take action" colour code) is a
  documented decision matrix over the calibrated rain probability and the
  station's *heavy-rain potential* — how much rain a wet day at this
  station typically brings at this time of year.
* **Inundation risk** combines a soil-saturation proxy (recent 7-day
  accumulated rainfall, ranked against the station's own history), the
  rain probability and the heavy-rain potential.

All components are heuristic layers on top of the calibrated model output
and are labelled as such in the UI; the matrix and weights below are the
single source of truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AppConfig

# ------------------------------------------------- IMD 24-h rainfall bands
# (mm per day, IMD hydrometeorological definitions)
IMD_CATEGORIES = [
    ("EXTREMELY HEAVY", 204.5),
    ("VERY HEAVY", 115.6),
    ("HEAVY", 64.5),
    ("MODERATE", 15.6),
    ("LIGHT", 2.5),
    ("VERY LIGHT / DRY", 0.0),
]

GREEN, YELLOW, ORANGE, RED = "GREEN", "YELLOW", "ORANGE", "RED"

WARNING_META = {
    GREEN:  {"title": "No warning", "advice": "No action needed",
             "color": "#18a05a", "rank": 0},
    YELLOW: {"title": "Watch", "advice": "Be updated",
             "color": "#e8b40c", "rank": 1},
    ORANGE: {"title": "Alert", "advice": "Be prepared",
             "color": "#e2571b", "rank": 2},
    RED:    {"title": "Warning", "advice": "Take action",
             "color": "#cf3f2f", "rank": 3},
}

INUNDATION_META = {
    "LOW":      {"color": "#18a05a", "rank": 0},
    "MODERATE": {"color": "#e8b40c", "rank": 1},
    "HIGH":     {"color": "#e2571b", "rank": 2},
    "SEVERE":   {"color": "#cf3f2f", "rank": 3},
}


def imd_category(mm: float) -> str:
    """IMD 24-hour rainfall intensity category for an amount in mm."""
    for name, lo in IMD_CATEGORIES:
        if mm >= lo:
            return name
    return IMD_CATEGORIES[-1][0]


def heavy_rain_potential(features: pd.DataFrame, station: str,
                         target_date: pd.Timestamp, cfg: AppConfig,
                         window_days: int = 15) -> dict:
    """How much rain a wet day at this station typically brings around
    this time of year.

    Looks at historical days within ±window_days of the target
    day-of-year (all years), keeps the days that actually got significant
    next-day rain, and reports the median and 90th percentile amount plus
    the share of days reaching the IMD HEAVY band. The p90 is used as the
    'plausible upper intensity if it does rain' in the warning matrix.
    """
    sdf = features[features["station"] == station]
    amount_col = ("rain_tomorrow_mm" if "rain_tomorrow_mm" in sdf.columns
                  else "rainfall_today")
    doy = int(target_date.dayofyear)
    d = sdf["date"].dt.dayofyear
    dist = (d - doy).abs()
    dist = np.minimum(dist, 365 - dist)
    window = sdf[dist <= window_days]
    if window.empty:
        window = sdf
    amounts = window[amount_col].dropna()
    wet = amounts[amounts >= cfg.rain_threshold_mm]
    if wet.empty:
        return {"p50_mm": 0.0, "p90_mm": 0.0, "heavy_share": 0.0,
                "n_wet_days": 0, "n_window_days": int(len(window))}
    return {
        "p50_mm": float(wet.quantile(0.50)),
        "p90_mm": float(wet.quantile(0.90)),
        "heavy_share": float((amounts >= 64.5).mean()),
        "n_wet_days": int(len(wet)),
        "n_window_days": int(len(window)),
    }


def warning_level(p_rain: float, potential_p90_mm: float, ambiguous: bool,
                  cfg: AppConfig) -> str:
    """IMD-style colour-coded warning from the decision matrix below.

    p = calibrated probability of significant next-day rain;
    q90 = seasonal heavy-rain potential (p90 of wet-day amounts, mm).

      RED     p ≥ 0.70 and q90 ≥ 115.6 (very heavy plausible)
              or p ≥ 0.85 and q90 ≥ 64.5 (heavy plausible)
      ORANGE  p ≥ 0.60 and q90 ≥ 64.5
              or p ≥ 0.80 and q90 ≥ 15.6
      YELLOW  p ≥ classification threshold
              or the conformal set is ambiguous while q90 ≥ 64.5
      GREEN   otherwise
    """
    p, q90 = float(p_rain), float(potential_p90_mm)
    if (p >= 0.70 and q90 >= 115.6) or (p >= 0.85 and q90 >= 64.5):
        return RED
    if (p >= 0.60 and q90 >= 64.5) or (p >= 0.80 and q90 >= 15.6):
        return ORANGE
    if p >= cfg.classification_threshold or (ambiguous and q90 >= 64.5):
        return YELLOW
    return GREEN


def saturation_percentile(features: pd.DataFrame, station: str,
                          rolling7_mm: float) -> float:
    """Rank the current 7-day rainfall accumulation against the station's
    own history (0..1). Used as a soil-saturation / drainage-load proxy."""
    sdf = features[features["station"] == station]
    hist = sdf.get("rainfall_rolling_7")
    if hist is None or hist.dropna().empty:
        return 0.5
    hist = hist.dropna()
    return float((hist <= float(rolling7_mm)).mean())


def inundation_risk(sat_pct: float, p_rain: float,
                    potential_p90_mm: float) -> dict:
    """Inundation risk index in [0, 1] and band.

    score = 0.45 · saturation percentile (how unusually wet the last
            7 days already are for this station)
          + 0.35 · calibrated rain probability
          + 0.20 · heavy-rain potential scaled to the IMD extremely-heavy
            threshold (204.5 mm)

    Bands: ≥ 0.75 SEVERE · ≥ 0.55 HIGH · ≥ 0.35 MODERATE · else LOW.
    A screening indicator for drainage/response prioritisation, not a
    hydrological inundation model (no terrain or drainage-network input).
    """
    score = (0.45 * float(sat_pct)
             + 0.35 * float(p_rain)
             + 0.20 * min(float(potential_p90_mm) / 204.5, 1.0))
    if score >= 0.75:
        band = "SEVERE"
    elif score >= 0.55:
        band = "HIGH"
    elif score >= 0.35:
        band = "MODERATE"
    else:
        band = "LOW"
    return {"score": float(score), "band": band}


def assess_station(features: pd.DataFrame, station: str,
                   target_date: pd.Timestamp, pred: dict,
                   rolling7_mm: float, cfg: AppConfig) -> dict:
    """Full early-warning assessment for one station.

    `pred` is the output of predict.predict_one; `rolling7_mm` the 7-day
    rainfall accumulation from the same feature row the model saw.
    """
    potential = heavy_rain_potential(features, station, target_date, cfg)
    level = warning_level(pred["p_rain"], potential["p90_mm"],
                          pred["ambiguous"], cfg)
    sat = saturation_percentile(features, station, rolling7_mm)
    inun = inundation_risk(sat, pred["p_rain"], potential["p90_mm"])
    return {
        "station": station,
        "level": level,
        "meta": WARNING_META[level],
        "potential": potential,
        "intensity_category": imd_category(potential["p90_mm"]),
        "saturation_percentile": sat,
        "rolling7_mm": float(rolling7_mm),
        "inundation": inun,
        "pred": pred,
    }
