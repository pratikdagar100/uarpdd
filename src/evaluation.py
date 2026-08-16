"""Model evaluation on the held-out chronological test set.

Everything reported in the Model Performance tab is computed here from real
predictions — nothing is hard-coded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             brier_score_loss, roc_curve,
                             classification_report)

from .config import AppConfig
from .predict import calibrated_rain_probability
from .uncertainty import empirical_coverage, avg_set_size


def evaluate_bundle(bundle: dict, features: pd.DataFrame, test_mask: pd.Series,
                    cfg: AppConfig, perm_sample: int = 20000) -> dict:
    cols = bundle["feature_cols"]
    test = features[test_mask]
    X_te = test[cols].astype(np.float32).fillna(bundle["train_medians"])
    y_te = test["RainTomorrow"].to_numpy()

    p_rain = calibrated_rain_probability(bundle, X_te)
    y_pred = (p_rain >= cfg.classification_threshold).astype(int)

    ev: dict = {
        "n_test": int(len(y_te)),
        "positive_rate_test": float(y_te.mean()),
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "recall": float(recall_score(y_te, y_pred, zero_division=0)),
        "f1": float(f1_score(y_te, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_te, p_rain)),
        "brier": float(brier_score_loss(y_te, p_rain)),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        "classification_report": classification_report(
            y_te, y_pred, target_names=["No significant rain", "Rain"],
            zero_division=0),
    }

    # ROC curve (downsampled for plotting)
    fpr, tpr, _ = roc_curve(y_te, p_rain)
    idx = np.linspace(0, len(fpr) - 1, min(400, len(fpr))).astype(int)
    ev["roc_curve"] = {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}

    # Calibration curve: 10 equal-width probability bins
    bins = np.linspace(0, 1, 11)
    binned = np.digitize(p_rain, bins[1:-1])
    cal_rows = []
    for b in range(10):
        m = binned == b
        if m.sum() >= 20:
            cal_rows.append({
                "bin_mid": float((bins[b] + bins[b + 1]) / 2),
                "mean_predicted": float(p_rain[m].mean()),
                "observed_frequency": float(y_te[m].mean()),
                "count": int(m.sum()),
            })
    ev["calibration_curve"] = cal_rows

    # Conformal diagnostics on the test set
    qhat = bundle["qhat"]
    ev["conformal_test"] = {
        "empirical_coverage": empirical_coverage(p_rain, y_te, qhat),
        "avg_set_size": avg_set_size(p_rain, qhat),
        "share_ambiguous": float(
            (((1.0 - p_rain) <= qhat) & (p_rain <= qhat)).mean()),
        "target_coverage": cfg.conformal_coverage,
    }

    # Feature importance: model-native + permutation on a test sample
    if hasattr(bundle["model"], "feature_importances_"):
        ev["model_importance"] = dict(zip(
            cols, [float(v) for v in bundle["model"].feature_importances_]))

    rng = np.random.RandomState(cfg.random_state)
    n = len(X_te)
    samp = rng.choice(n, size=min(perm_sample, n), replace=False)
    pi = permutation_importance(
        bundle["model"], X_te.iloc[samp], y_te[samp],
        n_repeats=3, random_state=cfg.random_state,
        scoring="roc_auc", n_jobs=-1)
    ev["permutation_importance"] = {
        c: {"mean": float(m), "std": float(s)}
        for c, m, s in zip(cols, pi.importances_mean, pi.importances_std)
    }

    ev["class_distribution"] = {
        "train_positive_rate": float(
            features.loc[~test_mask, "RainTomorrow"].mean()),
        "test_positive_rate": float(y_te.mean()),
    }
    return ev
