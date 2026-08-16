"""Central configuration for the Uncertainty-Aware Rainfall Prediction project.

All scientifically meaningful knobs live here so the dashboard, training page
and README stay consistent. Values can be overridden from the Training tab and
are persisted to models/config.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DB_PATH = DATA_DIR / "study.db"
DATASET_CANDIDATES = [
    DATA_DIR / "dataset.csv",
    DATA_DIR / "dataset.parquet",
    DATA_DIR / "dataset.xlsx",
]
CONFIG_PATH = MODELS_DIR / "config.json"

# Physically plausible bounds for Indian surface weather observations.
# Values outside these ranges are treated as sensor/entry errors -> NaN.
PHYSICAL_BOUNDS = {
    "avg_temp": (-40.0, 55.0),
    "min_temp": (-45.0, 50.0),
    "max_temp": (-35.0, 55.0),
    "wind_speed": (0.0, 120.0),
    "air_pressure": (850.0, 1090.0),
    "rainfall": (0.0, 1000.0),
    "humidity": (0.0, 100.0),
}


@dataclass
class AppConfig:
    # --- Problem definition ---
    rain_threshold_mm: float = 2.5      # rainfall >= threshold -> RAIN
    classification_threshold: float = 0.50

    # --- Chronological split (fractions of rows ordered by date) ---
    train_frac: float = 0.70
    val_frac: float = 0.15              # validation = calibration + conformal
    # test_frac is the remainder (0.15 by default)

    # --- Random Forest ---
    n_estimators: int = 200
    max_depth: int = 18
    min_samples_split: int = 10
    min_samples_leaf: int = 5

    # --- Uncertainty ---
    conformal_coverage: float = 0.90    # target coverage of the prediction set

    # --- User-facing confidence bands (calibrated probability of the
    #     predicted class; distance from 0.5). Configurable, heuristic labels.
    conf_low_max: float = 0.60          # 50-59% -> LOW
    conf_moderate_max: float = 0.75     # 60-74% -> MODERATE
    conf_high_max: float = 0.90         # 75-89% -> HIGH  (>= -> VERY HIGH)

    # --- Study ---
    scenarios_per_participant: int = 10
    random_state: int = 42

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "AppConfig":
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                known = {f for f in cls.__dataclass_fields__}
                return cls(**{k: v for k, v in data.items() if k in known})
            except Exception:
                pass
        return cls()

    def model_key(self) -> str:
        """Hashable string of everything that requires retraining when changed."""
        relevant = [
            self.rain_threshold_mm, self.train_frac, self.val_frac,
            self.n_estimators, self.max_depth, self.min_samples_split,
            self.min_samples_leaf, self.conformal_coverage, self.random_state,
        ]
        return "|".join(str(v) for v in relevant)
