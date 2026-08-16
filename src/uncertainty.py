"""Uncertainty quantification: split conformal prediction for binary
classification, plus the user-facing confidence/uncertainty labelling.

Method (split conformal, LAC / inverse-probability score)
---------------------------------------------------------
On a held-out conformal-calibration set (chronologically after training data)
compute the nonconformity score of the TRUE class:

    s_i = 1 - p_hat(y_i | x_i)

qhat is the ceil((n+1)(1-alpha))/n empirical quantile of {s_i}. At prediction
time, class c enters the prediction set iff 1 - p_hat(c|x) <= qhat.

Guarantee: marginal coverage P(y in set) >= 1 - alpha over exchangeable data.
It is NOT "this individual forecast is correct with 90% probability" — the
UI copy is explicit about this.
"""
from __future__ import annotations

import math

import numpy as np

from .config import AppConfig

RAIN, NO_RAIN = "RAIN", "NO SIGNIFICANT RAIN"


def conformal_qhat(p_rain_cal: np.ndarray, y_cal: np.ndarray,
                   coverage: float) -> float:
    """qhat from calibrated rain probabilities and true labels."""
    p_true = np.where(y_cal == 1, p_rain_cal, 1.0 - p_rain_cal)
    scores = 1.0 - p_true
    n = len(scores)
    alpha = 1.0 - coverage
    q_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


def prediction_set(p_rain: float, qhat: float) -> list[str]:
    """Conformal prediction set for one example."""
    s = []
    if 1.0 - p_rain <= qhat:
        s.append(RAIN)
    if p_rain <= qhat:
        s.append(NO_RAIN)
    if not s:
        # qhat can be small enough that no class qualifies; fall back to the
        # most likely class so the set is never empty (standard practice).
        s.append(RAIN if p_rain >= 0.5 else NO_RAIN)
    return s


def empirical_coverage(p_rain: np.ndarray, y: np.ndarray, qhat: float) -> float:
    """Fraction of examples whose true label is inside the prediction set."""
    in_set = np.where(
        y == 1, (1.0 - p_rain) <= qhat, p_rain <= qhat)
    # account for the never-empty fallback
    fallback = ((1.0 - p_rain) > qhat) & (p_rain > qhat)
    pred_pos = p_rain >= 0.5
    in_set = in_set | (fallback & (pred_pos == (y == 1)))
    return float(np.mean(in_set))


def avg_set_size(p_rain: np.ndarray, qhat: float) -> float:
    size = ((1.0 - p_rain) <= qhat).astype(int) + (p_rain <= qhat).astype(int)
    size = np.maximum(size, 1)
    return float(size.mean())


def confidence_label(p_pred_class: float, cfg: AppConfig) -> str:
    """User-facing confidence indicator from the calibrated probability of
    the PREDICTED class (>= 0.5 by construction). Heuristic, configurable."""
    if p_pred_class < cfg.conf_low_max:
        return "LOW"
    if p_pred_class < cfg.conf_moderate_max:
        return "MODERATE"
    if p_pred_class < cfg.conf_high_max:
        return "HIGH"
    return "VERY HIGH"


def uncertainty_label(pred_set: list[str], p_pred_class: float,
                      cfg: AppConfig) -> str:
    """User-facing uncertainty indicator combining the conformal set and the
    probability margin."""
    if len(pred_set) == 2:
        return "HIGH"
    if p_pred_class < cfg.conf_moderate_max:
        return "MODERATE"
    if p_pred_class < cfg.conf_high_max:
        return "LOW-MODERATE"
    return "LOW"
