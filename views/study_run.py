"""User study — design screens 05/06/07.

Within-subject, counterbalanced design: each participant answers the
decision task for N scenarios; half rendered as POINT, half as UNCERTAINTY,
in randomised order. Outcome is revealed only after the decision is logged.
"""
from __future__ import annotations

import time
import uuid

import pandas as pd
import streamlit as st

from src.config import AppConfig
from src.database import log_decision, counts
from src.study import (generate_scenarios, assign_participant_plan,
                       decision_is_appropriate)
from src.uncertainty import RAIN
from . import theme as T


def _scenario_pool(state: dict, cfg: AppConfig) -> pd.DataFrame:
    if "scenario_pool" not in st.session_state:
        st.session_state["scenario_pool"] = generate_scenarios(
            state["bundle"], state["features"], state["split"]["test_mask"],
            cfg, n_scenarios=40)
    return st.session_state["scenario_pool"]


def render(state: dict, cfg: AppConfig) -> None:
    T.page_header("User study")

    pool = _scenario_pool(state, cfg)

    # ---------------- Start screen ----------------
    if "study" not in st.session_state:
        c = counts()
        left, right = st.columns([1.9, 1])
        with left:
            st.markdown(f"""
<div class="uar-card">
  <div style="font-family:Poppins,sans-serif;font-size:1.35rem;
    font-weight:600">Ready to take part?</div>
  <div style="color:{T.INK_2};font-size:0.9rem;margin:0.3rem 0 0.9rem">
    {cfg.scenarios_per_participant} scenarios, about five minutes. No account
    or personal data required.</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">
    <div class="step-card"><div class="step-k">Step 1</div>
      <div class="step-t">Read the forecast</div>
      <div class="step-d">A station, a date, today's conditions and the
        model's forecast for tomorrow.</div></div>
    <div class="step-card"><div class="step-k">Step 2</div>
      <div class="step-t">Decide</div>
      <div class="step-d">Would you plan an outdoor event tomorrow? YES or
        NO, plus how confident you are (1–5).</div></div>
    <div class="step-card"><div class="step-k">Step 3</div>
      <div class="step-t">Learn the outcome</div>
      <div class="step-d">After saving, you see what actually happened.
        Then the next scenario.</div></div>
  </div>
</div>""", unsafe_allow_html=True)
            b1, b2 = st.columns([1, 2.2])
            with b1:
                start = st.button("Start a study session", type="primary",
                                  use_container_width=True)
            with b2:
                st.markdown(
                    f'<div style="font-size:0.82rem;color:{T.MUTED};'
                    f'padding-top:0.55rem">Decisions recorded so far: '
                    f'<b>{c["real_decisions"]} real</b> (from '
                    f'{c["real_participants"]} participants), '
                    f'<b>{c["demo_decisions"]} demo</b>.</div>',
                    unsafe_allow_html=True)
        with right:
            ex = pool.iloc[0]
            st.markdown(f"""
<div class="uar-card">
  <b>What you will see</b>
  <div style="font-size:0.72rem;letter-spacing:0.1em;color:{T.MUTED};
    margin:0.7rem 0 0.3rem">POINT CONDITION</div>
  <div style="border:1px solid {T.GRID};border-radius:10px;
    padding:0.6rem 0.9rem">
    <b>Rain tomorrow: {"YES" if ex['prediction'] == RAIN else "NO"}</b><br>
    <span style="font-size:0.82rem;color:{T.INK_2}">Probability of rain
      <b>{ex['p_rain']:.0%}</b></span></div>
  <div style="font-size:0.72rem;letter-spacing:0.1em;color:{T.MUTED};
    margin:0.8rem 0 0.3rem">UNCERTAINTY CONDITION</div>
  <div style="border:1px solid {T.GRID};border-radius:10px;
    padding:0.6rem 0.9rem">
    <b>Rain tomorrow: {"YES" if ex['prediction'] == RAIN else "NO"}</b><br>
    <span style="font-size:0.82rem;color:{T.INK_2}">Probability of rain
      <b>{ex['p_rain']:.0%}</b></span><br>
    <div style="margin-top:0.4rem;margin-left:-0.35rem">
    {T.pill("Confidence · " + ex['confidence'], T.CONF_PILL[ex['confidence']])}
    {T.pill("Uncertainty · " + ex['uncertainty'], T.UNC_PILL[ex['uncertainty']])}
    {T.pill(f"{cfg.conformal_coverage:.0%} set · {ex['prediction_set']}",
            "orange" if ex['ambiguous'] else "blue")}</div></div>
</div>""", unsafe_allow_html=True)
        if start:
            pid = uuid.uuid4().hex[:12]
            st.session_state["study"] = {
                "participant_id": pid,
                "session_id": uuid.uuid4().hex[:12],
                "plan": assign_participant_plan(
                    pool, cfg, participant_seed=int(pid[:6], 16) % (2**31)),
                "index": 0,
                "shown_at": None,
                "last_result": None,
            }
            st.rerun()
        return

    study = st.session_state["study"]
    plan: pd.DataFrame = study["plan"]
    i = study["index"]

    # ---------------- Finished ----------------
    if i >= len(plan):
        st.markdown('<div class="strip strip-good"><b>Session complete '
                    '— thank you!</b> Your decisions have been logged.</div>',
                    unsafe_allow_html=True)
        if st.button("Start another session (new participant)"):
            del st.session_state["study"]
            st.rerun()
        return

    s = plan.iloc[i]
    condition = s["condition"]
    st.progress(i / len(plan), text=f"Scenario {i + 1} of {len(plan)}")

    # ---------------- Previous outcome strip ----------------
    if study["last_result"] is not None:
        lr = study["last_result"]
        cls = "strip-good" if lr["correct"] else "strip-bad"
        verdict = "appropriate" if lr["correct"] else "inappropriate"
        st.markdown(
            f'<div class="strip {cls}"><b>Previous scenario:</b> actual '
            f'outcome was <b>{lr["actual"]}</b> ({lr["mm"]:.1f} mm). Your '
            f'decision (<b>{lr["decision"].replace("_", " ")}</b>) was '
            f'{verdict}.</div>', unsafe_allow_html=True)

    # ---------------- Scenario card ----------------
    chips = []
    for label, key, unit in [("Avg temp", "avg_temp", "°C"),
                             ("Wind", "wind_speed", "km/h"),
                             ("Pressure", "air_pressure", "hPa"),
                             ("Rain today", "rainfall_today", "mm"),
                             ("Rain last 7 days", "rainfall_rolling_7", "mm")]:
        v = s.get(key)
        if v is not None and not pd.isna(v):
            chips.append(T.chip(label, f"{v:.1f} {unit}"))

    icon = T.RAIN_SVG if s["prediction"] == RAIN else T.SUN_SVG
    ans = "YES" if s["prediction"] == RAIN else "NO"
    ans_color = T.BLUE if s["prediction"] == RAIN else T.INK
    pills = ""
    if condition == "UNCERTAINTY":
        pills = ('<div style="margin-top:0.55rem">'
                 + T.pill("Confidence · " + s["confidence"],
                          T.CONF_PILL[s["confidence"]])
                 + T.pill("Uncertainty · " + s["uncertainty"],
                          T.UNC_PILL[s["uncertainty"]])
                 + T.pill(f"{cfg.conformal_coverage:.0%} set · "
                          f"{s['prediction_set']}",
                          "orange" if s["ambiguous"] else "blue")
                 + "</div>")
    st.markdown(f"""
<div class="uar-card" style="text-align:center;padding:1.4rem 1.5rem">
  <div style="font-size:0.78rem;color:{T.MUTED}">{s['station']} ·
    {s['date']} · scenario {s['scenario_id']} · condition
    <b>{condition}</b></div>
  <div style="margin:0.6rem 0 0.7rem">{''.join(chips)}</div>
  <div style="display:flex;align-items:center;justify-content:center;gap:12px">
    {icon}
    <span style="font-family:Poppins,sans-serif;font-size:1.8rem;
      font-weight:600">Rain tomorrow:
      <span style="color:{ans_color}">{ans}</span></span>
  </div>
  <div style="color:{T.INK_2};margin-top:0.2rem">Probability of rain
    <b>{s['p_rain']:.0%}</b></div>
  {pills}
</div>""", unsafe_allow_html=True)

    if study["shown_at"] is None:
        study["shown_at"] = time.time()

    # ---------------- Decision task ----------------
    with st.form(f"decision_{i}"):
        st.markdown("**Would you plan an outdoor event tomorrow?**")
        decision = st.radio(
            "Your decision",
            ["YES — Plan the event", "NO — Do not plan the event"],
            label_visibility="collapsed", horizontal=True, index=None)
        conf = st.slider("How confident are you in your decision? "
                         "(1 = not at all, 5 = extremely)", 1, 5, 3)
        cA, cB = st.columns([1, 5])
        submitted = cB.form_submit_button("Submit decision", type="primary")
        abandoned = cA.form_submit_button("Abandon session")

    if abandoned:
        del st.session_state["study"]
        st.rerun()

    if submitted:
        if decision is None:
            st.warning("Please choose YES or NO first.")
            return
        elapsed = time.time() - study["shown_at"]
        user_decision = "EVENT" if decision.startswith("YES") else "NO_EVENT"
        correct = decision_is_appropriate(user_decision, s["actual_outcome"])
        log_decision({
            "participant_id": study["participant_id"],
            "session_id": study["session_id"],
            "scenario_id": s["scenario_id"],
            "condition": condition,
            "prediction": s["prediction"],
            "rain_probability": float(s["p_rain"]),
            "confidence": s["confidence"] if condition == "UNCERTAINTY" else None,
            "uncertainty_level": s["uncertainty"] if condition == "UNCERTAINTY" else None,
            "prediction_set": s["prediction_set"] if condition == "UNCERTAINTY" else None,
            "user_decision": user_decision,
            "user_confidence": int(conf),
            "decision_time_seconds": round(elapsed, 3),
            "actual_outcome": s["actual_outcome"],
            "decision_correct": int(correct),
            "is_demo": 0,
        })
        study["last_result"] = {
            "actual": s["actual_outcome"], "mm": s["actual_rain_mm"],
            "decision": user_decision, "correct": bool(correct)}
        study["index"] = i + 1
        study["shown_at"] = None
        st.rerun()
