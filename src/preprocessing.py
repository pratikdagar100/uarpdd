"""Cleaning pipeline: standardise columns, remove duplicates, repair
impossible values, impute — while preserving the *observed* rainfall series
separately so the prediction target is never built from imputed values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AppConfig, PHYSICAL_BOUNDS


def preprocess(df: pd.DataFrame, mapping: dict[str, str | None],
               cfg: AppConfig) -> tuple[pd.DataFrame, dict]:
    """Return (clean_df, report).

    clean_df has standardised role names as columns:
      date, station, rainfall_obs (observed, NaN preserved),
      rainfall (imputed, feature use only), plus mapped numeric/categorical
      roles.
    """
    report: dict = {"original_rows": int(len(df))}
    missing_before = int(df.isna().sum().sum())

    out = pd.DataFrame(index=df.index)

    # --- Required roles ---
    date_col = mapping.get("date")
    if date_col is None:
        raise ValueError("No date column detected. Set the mapping manually "
                         "in the Training & Config tab.")
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")

    station_col = mapping.get("station")
    if station_col is not None:
        out["station"] = df[station_col].astype(str).str.strip()
    else:
        out["station"] = "ALL"

    rain_col = mapping.get("rainfall")
    if rain_col is None:
        raise ValueError("No rainfall column detected. Set the mapping "
                         "manually in the Training & Config tab.")
    out["rainfall_obs"] = pd.to_numeric(df[rain_col], errors="coerce")

    # --- Optional roles ---
    for role in ["avg_temp", "min_temp", "max_temp", "wind_speed",
                 "air_pressure", "humidity", "cloud",
                 "latitude", "longitude", "elevation"]:
        col = mapping.get(role)
        if col is not None:
            out[role] = pd.to_numeric(df[col], errors="coerce")
    for role in ["state", "district", "season"]:
        col = mapping.get(role)
        if col is not None:
            out[role] = df[col].astype(str).str.strip()

    # --- Drop rows with unusable dates ---
    n_bad_dates = int(out["date"].isna().sum())
    out = out[out["date"].notna()].copy()
    report["rows_dropped_bad_date"] = n_bad_dates

    # --- Physically impossible values -> NaN (repair, do not delete rows) ---
    bounds_fixed = {}
    for role, (lo, hi) in PHYSICAL_BOUNDS.items():
        key = "rainfall_obs" if role == "rainfall" else role
        if key in out.columns:
            bad = (out[key] < lo) | (out[key] > hi)
            n_bad = int(bad.sum())
            if n_bad:
                out.loc[bad, key] = np.nan
                bounds_fixed[key] = n_bad
    report["values_nullified_out_of_range"] = bounds_fixed

    # --- min_temp > max_temp inconsistency: swap ---
    if {"min_temp", "max_temp"}.issubset(out.columns):
        swap = out["min_temp"] > out["max_temp"]
        n_swap = int(swap.sum())
        if n_swap:
            lo_v = out.loc[swap, "max_temp"].copy()
            out.loc[swap, "max_temp"] = out.loc[swap, "min_temp"]
            out.loc[swap, "min_temp"] = lo_v
        report["min_max_temp_swapped"] = n_swap

    # --- Duplicate (station, date) records: aggregate ---
    dup_mask = out.duplicated(["station", "date"], keep=False)
    report["station_date_duplicate_rows"] = int(dup_mask.sum())
    if dup_mask.any():
        num_cols = [c for c in out.columns
                    if c not in ("station", "date")
                    and pd.api.types.is_numeric_dtype(out[c])]
        cat_cols = [c for c in out.columns
                    if c not in ("station", "date") and c not in num_cols]
        agg = {c: "mean" for c in num_cols} | {c: "first" for c in cat_cols}
        out = (out.groupby(["station", "date"], as_index=False, sort=False)
                  .agg(agg))
    report["rows_after_dedup"] = int(len(out))

    # --- Feature-use rainfall: impute; target-use rainfall stays observed ---
    out = out.sort_values(["station", "date"]).reset_index(drop=True)
    out["rainfall"] = out["rainfall_obs"]

    # Statistical imputation is DEFERRED to train.py to prevent temporal data leakage.
    # We do not compute global/future medians here. The train split will compute
    # training-only medians which are then applied to validation, test, and live inference.
    impute_cols = [c for c in ["rainfall", "avg_temp", "min_temp", "max_temp",
                               "wind_speed", "air_pressure", "humidity",
                               "cloud"] if c in out.columns]
    imputed_counts = {}
    for c in impute_cols:
        # Record missing count for the dataset report, but leave NaNs intact for now.
        imputed_counts[c] = int(out[c].isna().sum())
    report["imputed_values"] = imputed_counts

    # Static geo columns: forward-fill within station then median
    for c in ["latitude", "longitude", "elevation"]:
        if c in out.columns and out[c].isna().any():
            out[c] = out.groupby("station")[c].transform(
                lambda s: s.fillna(s.median()))
            out[c] = out[c].fillna(out[c].median())


    report["missing_values_before"] = missing_before
    report["missing_values_after"] = int(
        out.drop(columns=["rainfall_obs"]).isna().sum().sum())
    report["missing_observed_rainfall"] = int(out["rainfall_obs"].isna().sum())
    report["final_rows"] = int(len(out))
    return out, report
