"""Dataset loading, inspection and automatic column mapping.

The project must not assume column names: this module inspects whatever file
is placed in data/ and maps its real columns onto the roles the pipeline
needs. A manual override is exposed in the UI (Training & Config tab).
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Roles the pipeline understands. Only `date`, `rainfall` are strictly
# required; `station` falls back to a single pseudo-station.
ROLE_PATTERNS: dict[str, list[str]] = {
    "date":       [r"^date", r"date$", r"^time$", r"datetime", r"day_of_record"],
    "station":    [r"station", r"^location$", r"^city$", r"^place$", r"^site"],
    "state":      [r"^state"],
    "district":   [r"district"],
    "rainfall":   [r"^rainfall$", r"^rain_?mm$", r"precip", r"^rain$", r"rainfall"],
    "avg_temp":   [r"^avg_?temp", r"^mean_?temp", r"^temp(erature)?$", r"^t2m$"],
    "min_temp":   [r"^min_?temp", r"temp.*min"],
    "max_temp":   [r"^max_?temp", r"temp.*max"],
    "wind_speed": [r"wind_?speed", r"^wind$", r"^ws$"],
    "air_pressure": [r"pressure", r"^slp$", r"^msl"],
    "humidity":   [r"humid", r"^rh$"],
    "cloud":      [r"cloud"],
    "latitude":   [r"^lat"],
    "longitude":  [r"^lon", r"^lng"],
    "elevation":  [r"elev", r"altitude"],
    "season":     [r"^season$"],
    "month":      [r"^month$"],
    "rain_tomorrow": [r"rain_?tomorrow"],
}

SPREADSHEET_SUFFIXES = (".xlsx", ".xls")

NUMERIC_ROLES = [
    "rainfall", "avg_temp", "min_temp", "max_temp", "wind_speed",
    "air_pressure", "humidity", "cloud", "latitude", "longitude", "elevation",
]


def find_dataset(data_dir: Path) -> Path | None:
    """Return the first usable dataset file in data/."""
    if not data_dir.exists():
        return None
    exts = {".csv", ".parquet", ".xlsx", ".xls"}
    preferred = ["dataset.csv", "dataset.parquet", "dataset.xlsx"]
    for name in preferred:
        p = data_dir / name
        if p.exists():
            return p
    for p in sorted(data_dir.iterdir()):
        if p.suffix.lower() in exts and p.is_file():
            return p
    return None


def dataset_fingerprint(path: Path) -> str:
    """Cheap fingerprint: size + mtime + hash of the first megabyte."""
    h = hashlib.md5()
    stat = path.stat()
    h.update(f"{path.name}|{stat.st_size}|{int(stat.st_mtime)}".encode())
    with open(path, "rb") as f:
        h.update(f.read(1_048_576))
    return h.hexdigest()


def _read_any(path: Path, suffix: str) -> pd.DataFrame:
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in SPREADSHEET_SUFFIXES:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported dataset format: {suffix}")


def _raw_cache_path(cache_dir: Path, fingerprint: str) -> Path:
    return cache_dir / f"raw_cache_{fingerprint[:16]}.parquet"


def _discard(path: Path) -> None:
    """Delete a cache file, tolerating a refusal.

    Windows refuses to unlink a file another process still has open, and a
    cache we merely failed to tidy up is never worth failing a run over.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _prune_raw_cache(cache_dir: Path, keep: Path) -> None:
    """One sidecar at a time.

    Sidecars for older datasets are dead weight, and the glob deliberately
    covers the `.tmp` staging files too: a process killed mid-write leaves
    one behind, and nothing else would ever collect it.
    """
    for old in cache_dir.glob("raw_cache_*"):
        if old != keep:
            _discard(old)


def load_raw(path: Path, cache_dir: Path | None = None,
             fingerprint: str | None = None) -> pd.DataFrame:
    """Read the dataset, optionally through a Parquet sidecar cache.

    A spreadsheet is decompressed and parsed cell by cell: this project's
    62 MB / 970k-row dataset.xlsx takes ~41 s to read, against ~2 s for the
    same frame written as Parquet. When a cache directory and fingerprint
    are given, the first read of a spreadsheet is mirrored to a sidecar and
    every later read comes from there.

    The sidecar is a cache, not a dataset: `dataset_fingerprint` still
    hashes the original file, so downstream cache invalidation is unchanged,
    and `find_dataset` never sees it because it lives outside data/.
    """
    suffix = path.suffix.lower()
    use_cache = (cache_dir is not None and fingerprint is not None
                 and suffix in SPREADSHEET_SUFFIXES)
    cached = _raw_cache_path(cache_dir, fingerprint) if use_cache else None

    if cached is not None:
        # Prune on every call, not just on a write: a hit would otherwise
        # let a superseded dataset's sidecar sit there indefinitely.
        _prune_raw_cache(cache_dir, keep=cached)

    if cached is not None and cached.exists():
        try:
            return pd.read_parquet(cached)
        except Exception:
            # A truncated or unreadable sidecar must never be fatal --
            # drop it and fall through to the real file.
            _discard(cached)

    df = _read_any(path, suffix)

    if cached is not None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            # PID-tagged staging name: two app instances starting together
            # must not write the same temporary file, which on Windows fails
            # outright rather than interleaving.
            tmp = cached.with_name(f"{cached.name}.{os.getpid()}.tmp")
            df.to_parquet(tmp)
            tmp.replace(cached)  # atomic, so a crash cannot leave a half file
        except Exception:
            pass  # caching is best-effort; a failure only costs time
    return df


def auto_map_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Map dataset columns to pipeline roles using name patterns.

    Returns {role: column_name_or_None}. First match wins; each column is
    used for at most one role.
    """
    mapping: dict[str, str | None] = {}
    used: set[str] = set()
    cols = list(df.columns)
    for role, patterns in ROLE_PATTERNS.items():
        found = None
        for pat in patterns:
            for col in cols:
                if col in used:
                    continue
                if re.search(pat, str(col).strip().lower()):
                    found = col
                    break
            if found:
                break
        mapping[role] = found
        if found:
            used.add(found)
    return mapping


def inspect_dataset(df: pd.DataFrame, mapping: dict[str, str | None]) -> dict:
    """Structured inspection report shown in the UI and used downstream."""
    report: dict = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": {c: str(df[c].dtype) for c in df.columns},
        "missing_per_column": {c: int(df[c].isna().sum()) for c in df.columns},
        "duplicate_full_rows": int(df.duplicated().sum()),
    }
    date_col, station_col = mapping.get("date"), mapping.get("station")
    if date_col is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        report["date_range"] = (
            str(dates.min().date()) if pd.notna(dates.min()) else None,
            str(dates.max().date()) if pd.notna(dates.max()) else None,
        )
        report["unparseable_dates"] = int(dates.isna().sum())
    if station_col is not None:
        report["n_stations"] = int(df[station_col].nunique())
        if date_col is not None:
            report["station_date_duplicates"] = int(
                df.duplicated([station_col, date_col]).sum()
            )
    rain_col = mapping.get("rainfall")
    if rain_col is not None and pd.api.types.is_numeric_dtype(df[rain_col]):
        rain = df[rain_col]
        report["rainfall_stats"] = {
            "missing": int(rain.isna().sum()),
            "negative": int((rain < 0).sum()),
            "max": float(rain.max()) if rain.notna().any() else None,
            "share_ge_2_5mm": float((rain >= 2.5).mean()) if rain.notna().any() else None,
        }
    # Flag suspicious values against physical bounds
    from .config import PHYSICAL_BOUNDS
    suspicious = {}
    for role, (lo, hi) in PHYSICAL_BOUNDS.items():
        col = mapping.get(role)
        if col is not None and pd.api.types.is_numeric_dtype(df[col]):
            n_bad = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n_bad:
                suspicious[col] = n_bad
    report["out_of_physical_range"] = suspicious
    report["numeric_columns"] = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    report["categorical_columns"] = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    report["mapping"] = dict(mapping)
    return report
