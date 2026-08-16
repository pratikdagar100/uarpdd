"""Model training: chronological split, model comparison, probability
calibration and conformal calibration. Persists a single bundle to models/.

Validation-block usage (no test-set peeking anywhere):
  - train (first 70% of the date span):     fit candidate models
  - validation (next 15% of the date span): interleaved day-parity halves:
      * half A -> model selection + probability calibration
      * half B -> conformal calibration (+ calibrator choice by Brier)
  - test (final 15% of the date span):      touched only in evaluation.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .config import AppConfig, MODELS_DIR
from .uncertainty import conformal_qhat, empirical_coverage, avg_set_size

BUNDLE_PATH = MODELS_DIR / "rainfall_bundle.joblib"


# ---------------------------------------------------------------- calibration
class PlattCalibrator:
    """Platt scaling: logistic regression on the logit of the raw probability."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e6, solver="lbfgs")

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p_raw, y):
        self.lr.fit(self._logit(p_raw), y)
        return self

    def predict(self, p_raw):
        return self.lr.predict_proba(self._logit(p_raw))[:, 1]


class IsotonicCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, p_raw, y):
        self.iso.fit(p_raw, y)
        return self

    def predict(self, p_raw):
        return np.clip(self.iso.predict(p_raw), 1e-6, 1 - 1e-6)


class IdentityCalibrator:
    def fit(self, p_raw, y):
        return self

    def predict(self, p_raw):
        return np.asarray(p_raw, dtype=float)


# ------------------------------------------------------------------- splitting
def chronological_split(df: pd.DataFrame, cfg: AppConfig) -> dict:
    """Row masks + date ranges for train/val/test ordered globally by date.

    The split fractions are applied to the DATE SPAN, not to row counts:
    Indian rainfall is strongly seasonal, and a row-count split can produce a
    validation window without any monsoon months (station density grows over
    time), which silently miscalibrates both the probabilities and the
    conformal quantile. A span split keeps every block ≈ seasonally complete
    while every station's test period remains strictly after training.
    """
    dmin, dmax = df["date"].min(), df["date"].max()
    span = dmax - dmin
    t_end = dmin + span * cfg.train_frac
    v_end = dmin + span * (cfg.train_frac + cfg.val_frac)

    train_mask = df["date"] <= t_end
    val_mask = (df["date"] > t_end) & (df["date"] <= v_end)
    test_mask = df["date"] > v_end
    return {
        "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
        "ranges": {
            "train": (str(df.loc[train_mask, "date"].min().date()),
                      str(t_end.date())),
            "validation": (str(df.loc[val_mask, "date"].min().date()),
                           str(v_end.date())),
            "test": (str(df.loc[test_mask, "date"].min().date()),
                     str(df.loc[test_mask, "date"].max().date())),
        },
        "counts": {
            "train": int(train_mask.sum()),
            "validation": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
    }


# -------------------------------------------------------------------- training
def _candidate_models(cfg: AppConfig) -> dict:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_split=cfg.min_samples_split,
            min_samples_leaf=cfg.min_samples_leaf,
            n_jobs=-1,
            random_state=cfg.random_state,
            class_weight="balanced_subsample",
        ),
        "Logistic Regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, C=1.0,
                                      class_weight="balanced")),
        ]),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=None, max_iter=300, learning_rate=0.08,
            random_state=cfg.random_state,
        ),
    }


def train_pipeline(features: pd.DataFrame, feature_cols: list[str],
                   cfg: AppConfig, fingerprint: str,
                   progress=None) -> dict:
    """Full training procedure. Returns the persisted bundle (also saved to
    models/rainfall_bundle.joblib)."""
    def _report(pct, msg):
        if progress:
            progress(pct, msg)

    split = chronological_split(features, cfg)
    X = features[feature_cols].astype(np.float32)
    y = features["RainTomorrow"].to_numpy()

    tr, va, te = split["train_mask"], split["val_mask"], split["test_mask"]
    X_tr, y_tr = X[tr], y[tr.to_numpy()]
    X_va, y_va = X[va], y[va.to_numpy()]

    # RF cannot handle NaN; median-impute using TRAIN statistics only.
    medians = X_tr.median()
    X_tr = X_tr.fillna(medians)
    X_va = X_va.fillna(medians)

    # Validation halves interleaved by day parity so BOTH halves span the
    # same seasons (a chronological half-split would put e.g. only winter
    # into the conformal half and miscalibrate the quantile).
    va_dates = features.loc[va, "date"]
    cal_a = (va_dates.dt.dayofyear % 2 == 0).to_numpy()  # selection + prob. calibration
    cal_b = ~cal_a                                       # conformal calibration

    # --- Fit candidates, select on validation-A ROC-AUC ---
    results = {}
    fitted = {}
    models = _candidate_models(cfg)
    for i, (name, model) in enumerate(models.items()):
        _report(0.15 + 0.5 * i / len(models), f"Training {name}…")
        t0 = time.time()
        model.fit(X_tr, y_tr)
        p_va = model.predict_proba(X_va)[:, 1]
        results[name] = {
            "val_roc_auc": float(roc_auc_score(y_va[cal_a], p_va[cal_a])),
            "val_brier_raw": float(brier_score_loss(y_va[cal_a], p_va[cal_a])),
            "fit_seconds": round(time.time() - t0, 1),
        }
        fitted[name] = model

    best_name = max(results, key=lambda k: results[k]["val_roc_auc"])
    best = fitted[best_name]
    _report(0.7, f"Selected {best_name}. Calibrating probabilities…")

    # --- Probability calibration on validation-A; choose by Brier on
    #     validation-B (kept separate from selection data) ---
    p_va_raw = best.predict_proba(X_va)[:, 1]
    calibrators = {
        "platt": PlattCalibrator().fit(p_va_raw[cal_a], y_va[cal_a]),
        "isotonic": IsotonicCalibrator().fit(p_va_raw[cal_a], y_va[cal_a]),
        "none": IdentityCalibrator(),
    }
    briers = {name: float(brier_score_loss(y_va[cal_b],
                                           c.predict(p_va_raw[cal_b])))
              for name, c in calibrators.items()}
    cal_name = min(briers, key=briers.get)
    calibrator = calibrators[cal_name]

    # --- Conformal calibration on validation-B with calibrated probs ---
    _report(0.85, "Conformal calibration…")
    p_cal_b = calibrator.predict(p_va_raw[cal_b])
    qhat = conformal_qhat(p_cal_b, y_va[cal_b], cfg.conformal_coverage)
    conformal_diag = {
        "qhat": qhat,
        "coverage_target": cfg.conformal_coverage,
        "coverage_on_calibration": empirical_coverage(p_cal_b, y_va[cal_b], qhat),
        "avg_set_size_on_calibration": avg_set_size(p_cal_b, qhat),
        "n_calibration": int(cal_b.sum()),
    }

    bundle = {
        "model": best,
        "model_name": best_name,
        "model_comparison": results,
        "calibrator": calibrator,
        "calibrator_name": cal_name,
        "calibrator_briers": briers,
        "qhat": qhat,
        "conformal": conformal_diag,
        "feature_cols": feature_cols,
        "train_medians": medians,
        "split_ranges": split["ranges"],
        "split_counts": split["counts"],
        "config_key": cfg.model_key(),
        "fingerprint": fingerprint,
        "trained_at": pd.Timestamp.now().isoformat(),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    _report(1.0, "Done.")
    return bundle


def load_bundle(fingerprint: str, cfg: AppConfig) -> dict | None:
    """Load the saved bundle if it matches the current dataset + config."""
    if not BUNDLE_PATH.exists():
        return None
    try:
        bundle = joblib.load(BUNDLE_PATH)
    except Exception:
        return None
    if bundle.get("fingerprint") != fingerprint:
        return None
    if bundle.get("config_key") != cfg.model_key():
        return None
    return bundle
