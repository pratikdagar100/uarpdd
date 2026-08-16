"""Live current-conditions feed for real-time forecasting.

The historical dataset is what the model LEARNS from; it ends well before
today. To forecast the real calendar tomorrow the model still needs today's
observations, so this module fetches recent daily observations for a
station's coordinates from the Open-Meteo API (free, no key) and assembles
exactly the same leakage-safe feature row the model was trained on.

Variables are chosen to match the training distribution:
  * pressure_msl_mean  — sea-level pressure, like the dataset's air_pressure
                         (~1009 hPa mean), NOT station-level pressure.
  * wind_speed_10m_mean — daily mean wind in km/h.
  * precipitation_sum   — daily total rainfall in mm.

If the network is unavailable the caller falls back to the station's
seasonal climatology (see predict.climatology_row), which is always
labelled as such in the UI.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import date, timedelta

import numpy as np
import pandas as pd

API_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS = [
    "temperature_2m_mean", "temperature_2m_min", "temperature_2m_max",
    "precipitation_sum", "wind_speed_10m_mean", "pressure_msl_mean",
]
TIMEZONE = "Asia/Kolkata"


class LiveWeatherError(RuntimeError):
    """Raised when live observations cannot be retrieved or are unusable."""


def fetch_recent_observations(lat: float, lon: float, past_days: int = 10,
                              timeout: int = 15) -> pd.DataFrame:
    """Daily observations for the last `past_days` days plus today.

    Returns a DataFrame indexed chronologically with dataset-style column
    names. Raises LiveWeatherError on any failure.
    """
    query = urllib.parse.urlencode({
        "latitude": round(float(lat), 4),
        "longitude": round(float(lon), 4),
        "daily": ",".join(DAILY_VARS),
        "past_days": int(past_days),
        "forecast_days": 1,
        "timezone": TIMEZONE,
    })
    try:
        with urllib.request.urlopen(f"{API_URL}?{query}", timeout=timeout) as r:
            payload = json.load(r)
    except Exception as exc:                       # network, DNS, HTTP, JSON
        raise LiveWeatherError(f"could not reach the live weather service "
                               f"({type(exc).__name__})") from exc

    daily = payload.get("daily")
    if not daily or not daily.get("time"):
        raise LiveWeatherError("live weather service returned no daily data")

    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "avg_temp": daily["temperature_2m_mean"],
        "min_temp": daily["temperature_2m_min"],
        "max_temp": daily["temperature_2m_max"],
        "rainfall": daily["precipitation_sum"],
        "wind_speed": daily["wind_speed_10m_mean"],
        "air_pressure": daily["pressure_msl_mean"],
    }).sort_values("date").reset_index(drop=True)

    # Forward/backward fill isolated gaps; drop days with no rainfall figure
    # at all (rainfall drives the lag features).
    for c in ["avg_temp", "min_temp", "max_temp", "wind_speed", "air_pressure"]:
        df[c] = df[c].ffill().bfill()
    df = df[df["rainfall"].notna()].reset_index(drop=True)
    if len(df) < 4:
        raise LiveWeatherError("live weather service returned too few days")
    return df


def station_constants(features: pd.DataFrame, station: str,
                      target_date: pd.Timestamp) -> dict:
    """Station identity / geography / season codes taken from the training
    data, so live rows are encoded exactly like training rows."""
    sdf = features[features["station"] == station]
    if sdf.empty:
        raise LiveWeatherError(f"station {station!r} is not in the dataset")
    last = sdf.iloc[-1]
    consts = {}
    for col in ["station_code", "latitude", "longitude", "elevation"]:
        if col in sdf.columns:
            consts[col] = float(last[col])
    if "season_code" in sdf.columns:
        same_month = sdf[sdf["date"].dt.month == target_date.month]
        source = same_month if not same_month.empty else sdf
        consts["season_code"] = float(source["season_code"].mode().iloc[0])
    return consts


def build_live_feature_row(features: pd.DataFrame, station: str,
                           feature_cols: list[str], obs: pd.DataFrame,
                           target_date: pd.Timestamp,
                           rain_threshold: float) -> tuple[pd.DataFrame, dict]:
    """Assemble the model's feature row for `target_date` (tomorrow) from
    live observations ending on the day before it (today).

    Mirrors feature_engineering.build_features exactly: every predictor is
    a function of `today` or earlier.
    """
    today = target_date - pd.Timedelta(days=1)
    hist = obs[obs["date"] <= today].sort_values("date").reset_index(drop=True)
    if hist.empty:
        raise LiveWeatherError("no live observations up to today")
    if (today - hist["date"].iloc[-1]).days > 1:
        raise LiveWeatherError("live observations are stale")

    rain = hist["rainfall"].to_numpy(dtype=float)
    cur = hist.iloc[-1]

    def lag(k: int) -> float:
        return float(rain[-1 - k]) if len(rain) > k else float("nan")

    row: dict[str, float] = {
        "rainfall_today": float(rain[-1]),
        "rainfall_lag_1": lag(1),
        "rainfall_lag_2": lag(2),
        "rainfall_lag_3": lag(3),
        "rainfall_rolling_3": float(rain[-3:].sum()),
        "rainfall_rolling_7": float(rain[-7:].sum()),
        "rain_today_flag": float(rain[-1] >= rain_threshold),
        "avg_temp": float(cur["avg_temp"]),
        "min_temp": float(cur["min_temp"]),
        "max_temp": float(cur["max_temp"]),
        "wind_speed": float(cur["wind_speed"]),
        "air_pressure": float(cur["air_pressure"]),
        "temp_range": float(cur["max_temp"] - cur["min_temp"]),
        "temp_change_1d": (float(cur["avg_temp"] - hist["avg_temp"].iloc[-2])
                           if len(hist) > 1 else 0.0),
        "pressure_change_1d": (
            float(cur["air_pressure"] - hist["air_pressure"].iloc[-2])
            if len(hist) > 1 else 0.0),
        "month_num": float(today.month),
    }
    doy = int(today.dayofyear)
    row["doy_sin"] = float(np.sin(2 * np.pi * doy / 365.25))
    row["doy_cos"] = float(np.cos(2 * np.pi * doy / 365.25))
    row.update(station_constants(features, station, today))

    frame = pd.DataFrame([{c: row.get(c, np.nan) for c in feature_cols}])
    info = {
        "observation_date": today.date(),
        "target_date": target_date.date(),
        "n_days_fetched": int(len(hist)),
        "history_start": hist["date"].iloc[0].date(),
        "conditions": {
            "avg_temp": row["avg_temp"], "min_temp": row["min_temp"],
            "max_temp": row["max_temp"], "wind_speed": row["wind_speed"],
            "air_pressure": row["air_pressure"],
            "rainfall_today": row["rainfall_today"],
            "rainfall_rolling_3": row["rainfall_rolling_3"],
            "rainfall_rolling_7": row["rainfall_rolling_7"],
        },
        "recent": hist[["date", "rainfall", "avg_temp", "air_pressure"]].copy(),
    }
    return frame, info


def live_forecast_row(features: pd.DataFrame, station: str,
                      feature_cols: list[str], target_date: pd.Timestamp,
                      rain_threshold: float) -> tuple[pd.DataFrame, dict]:
    """Convenience wrapper: fetch + assemble in one call."""
    consts = station_constants(features, station, target_date)
    lat, lon = consts.get("latitude"), consts.get("longitude")
    if lat is None or lon is None:
        raise LiveWeatherError("station has no coordinates in the dataset")
    obs = fetch_recent_observations(lat, lon)
    return build_live_feature_row(features, station, feature_cols, obs,
                                  target_date, rain_threshold)
