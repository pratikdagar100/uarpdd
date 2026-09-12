"""Target creation and leakage-safe feature engineering.

Leakage-prevention strategy
---------------------------
1. The target `RainTomorrow` is built exclusively by shifting the *observed*
   next-day rainfall backwards within each station (groupby-shift(-1)), so a
   row only ever sees its own station's next calendar record.
2. Rows whose next-day observed rainfall is missing (or is not actually the
   next record for that station) get a NaN target and are removed — never
   imputed.
3. Every predictor is computed from the current day or earlier: lags use
   shift(+k), rolling windows end at the current day. No feature is derived
   from the target or from any future observation.
4. The chronological split (train < validation < test by date) guarantees the
   evaluation simulates true forecasting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AppConfig

SEASON_ORDER = ["Winter", "Spring", "Summer", "Monsoon", "Autumn",
                "Post-Monsoon", "Pre-Monsoon"]


def build_features(clean: pd.DataFrame, cfg: AppConfig
                   ) -> tuple[pd.DataFrame, list[str], dict]:
    """Return (feature_df, feature_cols, report).

    feature_df keeps identification columns (date, station, rainfall_obs,
    rain_tomorrow_mm) alongside model features and the binary target
    `RainTomorrow`.
    """
    df = clean.sort_values(["station", "date"]).reset_index(drop=True).copy()
    g = df.groupby("station", sort=False)

    # ---------------- Target: next-day OBSERVED rainfall, same station ----
    df["rain_tomorrow_mm"] = g["rainfall_obs"].shift(-1)
    df["_next_date"] = g["date"].shift(-1)
    # Guard: "tomorrow" must be exactly the next calendar day. If the station
    # has a gap, the next record is not tomorrow -> target invalid.
    gap_ok = (df["_next_date"] - df["date"]).dt.days == 1
    df.loc[~gap_ok, "rain_tomorrow_mm"] = np.nan

    df["RainTomorrow"] = (df["rain_tomorrow_mm"] >= cfg.rain_threshold_mm
                          ).astype(float)
    df.loc[df["rain_tomorrow_mm"].isna(), "RainTomorrow"] = np.nan

    # ---------------- Leakage-safe predictors -----------------------------
    # Rainfall lags & rolling sums. 'rainfall' carries missing values through
    # from preprocessing (imputation is deferred to the training split), so
    # rows whose lags are undefined are dropped below and the remaining NaNs
    # are filled with train-split medians in train.py / predict.py.
    # The rolling sums deliberately skip missing days rather than going NaN:
    # they are accumulation proxies, computed identically here and in the
    # live feed, so a gap understates accumulation slightly but consistently.
    df["rainfall_today"] = df["rainfall"]
    for k in (1, 2, 3):
        df[f"rainfall_lag_{k}"] = g["rainfall"].shift(k)
    df["rainfall_rolling_3"] = g["rainfall"].transform(
        lambda s: s.rolling(3, min_periods=1).sum())
    df["rainfall_rolling_7"] = g["rainfall"].transform(
        lambda s: s.rolling(7, min_periods=1).sum())
    # NaN >= threshold is False, which would encode a day with no rainfall
    # observation as a confident "no rain today" while rainfall_today stays
    # missing for the imputer. Keep the pair honest: unknown rainfall today
    # means an unknown flag, imputed alongside it.
    df["rain_today_flag"] = np.where(
        df["rainfall"].isna(), np.nan,
        (df["rainfall"] >= cfg.rain_threshold_mm).astype(float))

    # Pressure change vs yesterday (a classic synoptic signal), if available
    if "air_pressure" in df.columns:
        df["pressure_change_1d"] = df["air_pressure"] - g["air_pressure"].shift(1)
    if "avg_temp" in df.columns:
        df["temp_change_1d"] = df["avg_temp"] - g["avg_temp"].shift(1)
    if {"max_temp", "min_temp"}.issubset(df.columns):
        df["temp_range"] = df["max_temp"] - df["min_temp"]

    # Calendar features
    df["month_num"] = df["date"].dt.month
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    if "season" in df.columns:
        df["season_code"] = df["season"].astype("category").cat.codes.astype(float)

    # Station identity (ordinal code; geography is also carried by
    # latitude/longitude/elevation)
    df["station_code"] = df["station"].astype("category").cat.codes.astype(float)

    # ---------------- Assemble feature list -------------------------------
    candidate_features = [
        "rainfall_today", "rainfall_lag_1", "rainfall_lag_2", "rainfall_lag_3",
        "rainfall_rolling_3", "rainfall_rolling_7", "rain_today_flag",
        "avg_temp", "min_temp", "max_temp", "temp_range", "temp_change_1d",
        "wind_speed", "air_pressure", "pressure_change_1d",
        "humidity", "cloud",
        "month_num", "doy_sin", "doy_cos", "season_code",
        "station_code", "latitude", "longitude", "elevation",
    ]
    feature_cols = [c for c in candidate_features if c in df.columns]

    report: dict = {"rows_before_target_filter": int(len(df))}

    # Remove rows with no valid next-day target (incl. each station's final
    # record) and rows whose lag features are undefined (first 3 days per
    # station).
    valid = df["RainTomorrow"].notna()
    for k in (1, 2, 3):
        valid &= df[f"rainfall_lag_{k}"].notna()
    if "pressure_change_1d" in df.columns:
        df["pressure_change_1d"] = df["pressure_change_1d"].fillna(0.0)
    if "temp_change_1d" in df.columns:
        df["temp_change_1d"] = df["temp_change_1d"].fillna(0.0)

    out = df[valid].copy()
    out["RainTomorrow"] = out["RainTomorrow"].astype(int)
    out = out.drop(columns=["_next_date"])

    report["rows_dropped_no_target_or_lags"] = int((~valid).sum())
    report["final_rows"] = int(len(out))
    report["positive_rate"] = float(out["RainTomorrow"].mean())
    report["feature_cols"] = feature_cols
    return out, feature_cols, report
