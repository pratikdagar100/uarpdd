"""User-study machinery: scenario generation from held-out test data,
balanced within-subject condition assignment, result aggregation and
statistical analysis.

Experimental design (documented for the report)
-----------------------------------------------
WITHIN-SUBJECT, counterbalanced: each participant completes N scenarios;
half are shown as POINT (prediction + probability only) and half as
UNCERTAINTY (adds confidence, uncertainty level and the 90% conformal
prediction set). Which scenario receives which condition, and the
presentation order, are randomised per participant, so condition is not
confounded with scenario difficulty or order. The actual outcome is revealed
only AFTER the decision is logged.

Decision-quality ground truth: planning an outdoor event is scored as
appropriate iff no significant rain (>= threshold) actually occurred the
next day; declining to plan is appropriate iff significant rain occurred.
This is a simplification (real event decisions weigh asymmetric costs) and
is documented as such in the README.
"""
from __future__ import annotations

import json
import uuid

import numpy as np
import pandas as pd
from scipy import stats

from .config import AppConfig
from .predict import calibrated_rain_probability
from .uncertainty import (RAIN, NO_RAIN, prediction_set, confidence_label,
                          uncertainty_label)

MIN_DECISIONS_PER_CONDITION = 30   # below this the analysis is labelled exploratory
MIN_PARTICIPANTS = 5


# ------------------------------------------------------------------ scenarios
def generate_scenarios(bundle: dict, features: pd.DataFrame,
                       test_mask: pd.Series, cfg: AppConfig,
                       n_scenarios: int = 40) -> pd.DataFrame:
    """Build the scenario pool from the chronological test set.

    Stratified so the pool contains certain and ambiguous predictions as
    well as rain and no-rain actual outcomes.
    """
    rng = np.random.RandomState(cfg.random_state)
    test = features[test_mask].copy()
    p = calibrated_rain_probability(bundle, test)
    test["p_rain"] = p
    qhat = bundle["qhat"]
    test["ambiguous"] = ((1 - p) <= qhat) & (p <= qhat)

    test["actual_rain"] = test["RainTomorrow"] == 1
    strata = test.groupby([test["ambiguous"], test["actual_rain"]],
                          group_keys=False)
    per_stratum = max(1, n_scenarios // 4)
    pool = strata.apply(
        lambda g: g.sample(min(per_stratum, len(g)), random_state=rng),
        include_groups=True)
    if len(pool) < n_scenarios:
        extra = test.drop(pool.index).sample(
            min(n_scenarios - len(pool), len(test) - len(pool)),
            random_state=rng)
        pool = pd.concat([pool, extra])
    pool = pool.sample(frac=1.0, random_state=rng).head(n_scenarios)

    rows = []
    for _, r in pool.iterrows():
        p_rain = float(r["p_rain"])
        is_rain = p_rain >= cfg.classification_threshold
        p_pred = p_rain if is_rain else 1 - p_rain
        pset = prediction_set(p_rain, qhat)
        rows.append({
            "scenario_id": f"S{r.name}",
            "station": r["station"],
            "date": str(pd.Timestamp(r["date"]).date()),
            "avg_temp": r.get("avg_temp"),
            "min_temp": r.get("min_temp"),
            "max_temp": r.get("max_temp"),
            "wind_speed": r.get("wind_speed"),
            "air_pressure": r.get("air_pressure"),
            "rainfall_today": r.get("rainfall_today"),
            "rainfall_rolling_3": r.get("rainfall_rolling_3"),
            "rainfall_rolling_7": r.get("rainfall_rolling_7"),
            "prediction": RAIN if is_rain else NO_RAIN,
            "p_rain": p_rain,
            "confidence": confidence_label(p_pred, cfg),
            "uncertainty": uncertainty_label(pset, p_pred, cfg),
            "prediction_set": " / ".join(pset),
            "ambiguous": bool(r["ambiguous"]),
            "actual_rain_mm": float(r["rain_tomorrow_mm"]),
            "actual_outcome": RAIN if r["RainTomorrow"] == 1 else NO_RAIN,
        })
    return pd.DataFrame(rows)


def assign_participant_plan(scenarios: pd.DataFrame, cfg: AppConfig,
                            participant_seed: int | None = None) -> pd.DataFrame:
    """Within-subject counterbalanced plan for one participant."""
    rng = np.random.RandomState(participant_seed)
    n = min(cfg.scenarios_per_participant, len(scenarios))
    chosen = scenarios.sample(n, random_state=rng).reset_index(drop=True)
    conditions = np.array(["POINT", "UNCERTAINTY"] * (n // 2 + 1))[:n]
    rng.shuffle(conditions)
    chosen["condition"] = conditions
    return chosen.sample(frac=1.0, random_state=rng).reset_index(drop=True)


def decision_is_appropriate(user_decision: str, actual_outcome: str) -> int:
    """EVENT is appropriate iff no significant rain occurred."""
    if user_decision == "EVENT":
        return int(actual_outcome == NO_RAIN)
    return int(actual_outcome == RAIN)


# -------------------------------------------------------------------- results
def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize_conditions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, g in df.groupby("condition"):
        k = int(g["decision_correct"].sum())
        n = len(g)
        lo, hi = _wilson_ci(k, n)
        rows.append({
            "condition": cond,
            "n_decisions": n,
            "n_participants": g["participant_id"].nunique(),
            "decision_accuracy": k / n if n else np.nan,
            "accuracy_ci_low": lo, "accuracy_ci_high": hi,
            "avg_decision_time_s": g["decision_time_seconds"].mean(),
            "median_decision_time_s": g["decision_time_seconds"].median(),
            "avg_user_confidence": g["user_confidence"].mean(),
        })
    return pd.DataFrame(rows)


def overconfidence(df: pd.DataFrame) -> pd.DataFrame:
    """Mean stated confidence (rescaled 1-5 -> 0-1) minus accuracy, per
    condition. Positive = overconfident."""
    rows = []
    for cond, g in df.groupby("condition"):
        conf01 = (g["user_confidence"] - 1) / 4.0
        rows.append({
            "condition": cond,
            "stated_confidence_0_1": conf01.mean(),
            "accuracy": g["decision_correct"].mean(),
            "overconfidence": conf01.mean() - g["decision_correct"].mean(),
        })
    return pd.DataFrame(rows)


def decision_changes(df: pd.DataFrame) -> dict:
    """How often decisions differ between conditions for the same scenario.

    Uses scenarios that received decisions under BOTH conditions (across
    participants): compares the EVENT-rate per condition per scenario.
    """
    piv = (df.groupby(["scenario_id", "condition"])["user_decision"]
             .apply(lambda s: (s == "EVENT").mean()).unstack())
    if piv.shape[1] < 2 or piv.dropna().empty:
        return {"n_paired_scenarios": 0}
    paired = piv.dropna()
    diff = (paired["UNCERTAINTY"] - paired["POINT"]).abs()
    return {
        "n_paired_scenarios": int(len(paired)),
        "mean_abs_event_rate_change": float(diff.mean()),
        "share_scenarios_majority_flipped": float(
            ((paired["POINT"] >= 0.5) != (paired["UNCERTAINTY"] >= 0.5)).mean()),
    }


# ------------------------------------------------------------------ statistics
def statistical_analysis(df: pd.DataFrame) -> dict:
    """Compare POINT vs UNCERTAINTY. Test choice is data-driven and the
    rationale is included in the output."""
    out: dict = {"notes": []}
    a = df[df["condition"] == "POINT"]
    b = df[df["condition"] == "UNCERTAINTY"]
    n_a, n_b = len(a), len(b)
    out["n_point"], out["n_uncertainty"] = n_a, n_b
    n_participants = df["participant_id"].nunique()
    out["n_participants"] = int(n_participants)

    out["exploratory"] = (min(n_a, n_b) < MIN_DECISIONS_PER_CONDITION
                          or n_participants < MIN_PARTICIPANTS)
    if out["exploratory"]:
        out["notes"].append(
            f"Sample is small (<{MIN_DECISIONS_PER_CONDITION} decisions per "
            f"condition or <{MIN_PARTICIPANTS} participants): results are "
            "EXPLORATORY and should not be treated as confirmatory evidence.")

    if n_a == 0 or n_b == 0:
        out["notes"].append("Need decisions in both conditions.")
        return out

    # --- Decision accuracy: 2x2 contingency table ---
    k_a, k_b = int(a["decision_correct"].sum()), int(b["decision_correct"].sum())
    table = np.array([[k_a, n_a - k_a], [k_b, n_b - k_b]])
    expected = stats.contingency.expected_freq(table)
    if (expected < 5).any():
        odds, p_val = stats.fisher_exact(table)
        test_name = "Fisher's exact test (an expected cell count < 5)"
    else:
        chi2, p_val, _, _ = stats.chi2_contingency(table, correction=True)
        test_name = "Chi-square test with Yates' correction (all expected counts >= 5)"
    p1, p2 = k_a / n_a, k_b / n_b
    h = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))  # Cohen's h
    se_diff = np.sqrt(p1 * (1 - p1) / n_a + p2 * (1 - p2) / n_b)
    diff = p2 - p1
    out["accuracy"] = {
        "point": p1, "uncertainty": p2, "difference": diff,
        "diff_ci95": (diff - 1.96 * se_diff, diff + 1.96 * se_diff),
        "test": test_name, "p_value": float(p_val),
        "effect_size_cohens_h": float(h),
        "point_ci95": _wilson_ci(k_a, n_a),
        "uncertainty_ci95": _wilson_ci(k_b, n_b),
    }

    # --- Decision time: check normality, choose test accordingly ---
    t_a = a["decision_time_seconds"].to_numpy()
    t_b = b["decision_time_seconds"].to_numpy()
    normal = False
    if min(len(t_a), len(t_b)) >= 8:
        sh_a = stats.shapiro(t_a[:5000]).pvalue
        sh_b = stats.shapiro(t_b[:5000]).pvalue
        normal = sh_a > 0.05 and sh_b > 0.05
    if normal:
        t_stat, p_t = stats.ttest_ind(t_a, t_b, equal_var=False)
        time_test = "Welch's t-test (Shapiro-Wilk consistent with normality)"
        pooled_sd = np.sqrt((t_a.var(ddof=1) + t_b.var(ddof=1)) / 2)
        eff = (t_b.mean() - t_a.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        eff_name = "Cohen's d"
    else:
        u_stat, p_t = stats.mannwhitneyu(t_a, t_b, alternative="two-sided")
        time_test = ("Mann-Whitney U (decision times are typically "
                     "right-skewed; normality rejected or n too small)")
        eff = 1 - 2 * u_stat / (len(t_a) * len(t_b))  # rank-biserial r
        eff_name = "rank-biserial correlation"
    out["time"] = {
        "point_mean": float(t_a.mean()), "uncertainty_mean": float(t_b.mean()),
        "point_median": float(np.median(t_a)),
        "uncertainty_median": float(np.median(t_b)),
        "test": time_test, "p_value": float(p_t),
        "effect_size": float(eff), "effect_size_name": eff_name,
    }

    # --- Stated confidence (ordinal 1-5): Mann-Whitney U ---
    c_a = a["user_confidence"].to_numpy()
    c_b = b["user_confidence"].to_numpy()
    if len(np.unique(np.concatenate([c_a, c_b]))) > 1:
        u_c, p_c = stats.mannwhitneyu(c_a, c_b, alternative="two-sided")
        out["confidence"] = {
            "point_mean": float(c_a.mean()),
            "uncertainty_mean": float(c_b.mean()),
            "test": "Mann-Whitney U (ordinal 1-5 scale)",
            "p_value": float(p_c),
            "effect_size_rank_biserial": float(
                1 - 2 * u_c / (len(c_a) * len(c_b))),
        }
    else:
        out["confidence"] = {"point_mean": float(c_a.mean()),
                             "uncertainty_mean": float(c_b.mean()),
                             "test": "not applicable (no variance)",
                             "p_value": None}
    return out


# ---------------------------------------------------------------- demo data
def generate_demo_data(scenarios: pd.DataFrame, cfg: AppConfig,
                       n_participants: int = 12) -> list[dict]:
    """SYNTHETIC demo decisions, clearly flagged is_demo=1.

    Only for demonstrating the results dashboard; excluded from real analysis
    by default and deletable with one click. The generator plants a plausible
    behavioural pattern but the records are entirely artificial.
    """
    rng = np.random.RandomState(cfg.random_state)
    records = []
    for pi in range(n_participants):
        participant = f"demo-{uuid.uuid4().hex[:8]}"
        plan = assign_participant_plan(scenarios, cfg,
                                       participant_seed=1000 + pi)
        for _, s in plan.iterrows():
            follow_model = rng.rand() < (0.80 if s["condition"] == "UNCERTAINTY"
                                         and not s["ambiguous"] else 0.65)
            model_says_event = s["prediction"] == NO_RAIN
            decision = "EVENT" if (model_says_event == follow_model) else "NO_EVENT"
            conf = int(np.clip(rng.normal(
                3.6 if s["condition"] == "POINT" else 3.2, 0.9), 1, 5))
            t = float(np.clip(rng.lognormal(
                2.2 if s["condition"] == "POINT" else 2.45, 0.4), 2, 90))
            records.append({
                "participant_id": participant,
                "session_id": f"demo-session-{pi}",
                "scenario_id": s["scenario_id"],
                "condition": s["condition"],
                "prediction": s["prediction"],
                "rain_probability": s["p_rain"],
                "confidence": s["confidence"],
                "uncertainty_level": s["uncertainty"],
                "prediction_set": s["prediction_set"],
                "user_decision": decision,
                "user_confidence": conf,
                "decision_time_seconds": round(t, 2),
                "actual_outcome": s["actual_outcome"],
                "decision_correct": decision_is_appropriate(
                    decision, s["actual_outcome"]),
                "is_demo": 1,
            })
    return records
