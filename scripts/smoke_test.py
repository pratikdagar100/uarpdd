"""End-to-end pipeline smoke test (no Streamlit). Run from project root:
    python scripts/smoke_test.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AppConfig
from src.pipeline import run_pipeline
from src.predict import predict_one, feature_influence
from src.study import (generate_scenarios, assign_participant_plan,
                       generate_demo_data, statistical_analysis,
                       summarize_conditions, overconfidence, decision_changes)
import pandas as pd

t0 = time.time()
cfg = AppConfig.load()
state = run_pipeline(cfg, progress=lambda p, m: print(f"[{p:5.0%}] {m}", flush=True))
assert "error" not in state, state
print(f"\n== pipeline done in {time.time()-t0:.0f}s ==")
print("model:", state["bundle"]["model_name"],
      "| calibrator:", state["bundle"]["calibrator_name"],
      "| qhat:", round(state["bundle"]["qhat"], 4))
print("split:", state["bundle"]["split_ranges"])
print("comparison:", state["bundle"]["model_comparison"])
ev = state["evaluation"]
print({k: round(ev[k], 4) for k in
       ["accuracy", "precision", "recall", "f1", "roc_auc", "brier"]})
print("conformal test:", ev["conformal_test"])
print("confusion:", ev["confusion_matrix"])

# --- single predictions across certainty spectrum ---
feats = state["features"]
test = feats[state["split"]["test_mask"]]
for name, q in [("low-p", 0.02), ("mid-p", 0.5), ("high-p", 0.98)]:
    from src.predict import calibrated_rain_probability
    import numpy as np
    p = calibrated_rain_probability(state["bundle"], test)
    idx = np.argsort(p)[int(q * (len(p) - 1))]
    row = test.iloc[[idx]]
    pred = predict_one(state["bundle"], row, cfg)
    print(name, {k: (round(v, 3) if isinstance(v, float) else v)
                 for k, v in pred.items()})
    infl = feature_influence(state["bundle"], row.iloc[0], state["train_stats"])
    print("  influence:", [(i["feature"], i["influence"]) for i in infl[:3]])

# --- live real-time forecast (falls back to climatology when offline) ---
import pandas as _pd
from src.live_weather import live_forecast_row, LiveWeatherError
from src.predict import climatology_row
_tom = _pd.Timestamp(_pd.Timestamp.today().date()) + _pd.Timedelta(days=1)
_station = feats["station"].iloc[0]
try:
    _row, _info = live_forecast_row(feats, _station, state["bundle"]["feature_cols"],
                                    _tom, cfg.rain_threshold_mm)
    _p = predict_one(state["bundle"], _row, cfg)
    print(f"\nlive forecast {_station} obs {_info['observation_date']} -> "
          f"{_info['target_date']}: {_p['label']} {_p['p_rain']:.0%} "
          f"{_p['confidence']}/{_p['uncertainty']} {_p['prediction_set']}")
except LiveWeatherError as e:
    _row, _ci = climatology_row(feats, _station, _tom)
    _p = predict_one(state["bundle"], _row, cfg)
    print(f"\nlive feed unavailable ({e}); climatology fallback: "
          f"{_p['label']} {_p['p_rain']:.0%}")

# --- study machinery ---
pool = generate_scenarios(state["bundle"], feats, state["split"]["test_mask"],
                          cfg, n_scenarios=40)
print("\nscenarios:", len(pool), "| ambiguous:", int(pool.ambiguous.sum()),
      "| rain outcomes:", int((pool.actual_outcome == "RAIN").sum()))
plan = assign_participant_plan(pool, cfg, participant_seed=7)
print("plan conditions:", plan.condition.value_counts().to_dict())

demo = pd.DataFrame(generate_demo_data(pool, cfg, n_participants=12))
print("demo records:", len(demo))
print(summarize_conditions(demo).to_string())
print(overconfidence(demo).to_string())
print("decision changes:", decision_changes(demo))
res = statistical_analysis(demo)
print("stats exploratory:", res["exploratory"])
print("accuracy stats:", {k: v for k, v in res["accuracy"].items()})
print("time stats:", res["time"])
print("confidence stats:", res["confidence"])
print("\nALL OK")
