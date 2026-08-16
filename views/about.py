"""About — design screen 11: research-question banner, pipeline chips, key
definitions, limitations, future improvements and quick-reference cards."""
from __future__ import annotations

import streamlit as st

from src.config import AppConfig
from . import theme as T

PIPELINE = [
    "Historical weather dataset", "Cleaning & imputation",
    "Threshold ≥ 2.5 mm → RAIN", "Same-station next-day target",
    "Leakage-safe features", "Chronological split",
    "Model comparison (RF · LR · GB)", "Probability calibration",
    "Split conformal (90%)", "Interactive dashboard",
    "Human decision experiment", "SQLite logging", "Statistical analysis",
]

LIMITATIONS = [
    "Decision ground truth ignores asymmetric costs of cancelling vs. "
    "getting rained out.",
    "Decisions are treated as independent; mixed-effects models would "
    "handle within-participant correlation.",
    "Conformal coverage is marginal, not conditional per station or season.",
    "Feature influence is a heuristic, not SHAP-style attribution.",
    "Convenience sampling; small samples are flagged exploratory.",
    "Station identity is label-encoded; unseen stations need retraining.",
]

FUTURE = [
    "Mondrian (per-station / per-season) conformal prediction",
    "Mixed-effects logistic regression for the study analysis",
    "Cost-sensitive, expected-utility decision support",
    "SHAP-based local explanations",
    "Rain-amount intervals via conformalized quantile regression",
]


def render(state: dict, cfg: AppConfig) -> None:
    T.page_header("About this project")

    st.markdown(f"""
<div style="background:linear-gradient(120deg,{T.NAVY} 0%,#1c3f77 100%);
  border-radius:12px;padding:1.5rem 1.8rem;margin-bottom:1rem">
  <div style="font-size:0.7rem;letter-spacing:0.16em;color:#8fa1b8;
    font-weight:700">RESEARCH QUESTION</div>
  <div style="font-family:Poppins,sans-serif;font-size:1.45rem;
    font-weight:600;color:#ffffff;line-height:1.35;margin:0.35rem 0 0.6rem">
    Does communicating uncertainty in rainfall predictions improve human
    decision-making compared with presenting only a point prediction?</div>
  <div style="font-size:0.82rem;color:#c3d0e0">
    <b style="color:#fff">Primary outcome</b> — decision accuracy
    &nbsp;&nbsp; <b style="color:#fff">Secondary</b> — decision time ·
    stated confidence · overconfidence · decision changes</div>
</div>""", unsafe_allow_html=True)

    chips = (' <span style="color:' + T.MUTED + '">→</span> ').join(
        f'<span class="chip" style="margin:0.2rem 0.1rem">{p}</span>'
        for p in PIPELINE)
    st.markdown(f'<div class="uar-card"><b>Pipeline</b>'
                f'<div style="margin-top:0.5rem;line-height:2.1">{chips}'
                f'</div></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        defs = [
            ("Probability",
             f"The calibrated model estimate that tomorrow's rainfall is "
             f"≥ {cfg.rain_threshold_mm} mm."),
            ("Classification threshold",
             f"Probability ≥ {cfg.classification_threshold:.0%} ⇒ prediction "
             f"RAIN (configurable)."),
            ("Confidence / uncertainty labels",
             "User-facing indicators derived from the calibrated probability "
             "and the conformal set; heuristic, not a formal guarantee."),
            (f"{cfg.conformal_coverage:.0%} prediction set",
             f"Split conformal prediction: across many days the true outcome "
             f"falls inside the set about {cfg.conformal_coverage:.0%} of "
             f"the time. This is not a {cfg.conformal_coverage:.0%} "
             f"correctness probability for an individual forecast."),
        ]
        body = "".join(
            f'<div style="padding:0.55rem 0;border-bottom:1px solid {T.GRID}">'
            f'<b style="font-size:0.9rem">{k}</b>'
            f'<div style="font-size:0.83rem;color:{T.INK_2};'
            f'margin-top:0.15rem">{v}</div></div>' for k, v in defs)
        st.markdown(f'<div class="uar-card" style="height:100%">'
                    f'<b>Key definitions</b>{body}</div>',
                    unsafe_allow_html=True)
    with c2:
        lim = "".join(f'<li style="margin:0.3rem 0">{l}</li>'
                      for l in LIMITATIONS)
        fut = "".join(f'<li style="margin:0.3rem 0">{f}</li>' for f in FUTURE)
        st.markdown(
            f'<div class="uar-card"><b>Limitations</b>'
            f'<ul style="font-size:0.83rem;color:{T.INK_2};margin:0.4rem 0 0;'
            f'padding-left:1.1rem">{lim}</ul></div>'
            f'<div class="uar-card"><b>Future improvements</b>'
            f'<ul style="font-size:0.83rem;color:{T.INK_2};margin:0.4rem 0 0;'
            f'padding-left:1.1rem">{fut}</ul></div>',
            unsafe_allow_html=True)

    insp = state["inspection"]
    q1, q2, q3, q4 = st.columns(4)
    cards = [
        ("Dataset", "Kaggle — Indian Rainfall and Weather Data",
         f"{insp['n_rows']:,} daily observations · "
         f"{insp.get('n_stations', 1)} stations"),
        ("Reproduce", "pip install -r requirements.txt<br>streamlit run app.py",
         "", True),
        ("Headless check", "python scripts/smoke_test.py", "", True),
        ("Privacy", "Participant / session UUIDs only.",
         "No accounts, no PII. Demo records flagged and deletable."),
    ]
    for col, card in zip([q1, q2, q3, q4], cards):
        label, main, foot = card[0], card[1], card[2]
        mono = len(card) > 3
        main_html = (f'<div class="mono" style="margin-top:0.35rem;'
                     f'padding:0.5rem 0.7rem">{main}</div>' if mono else
                     f'<div style="font-weight:650;font-size:0.88rem;'
                     f'margin-top:0.25rem">{main}</div>')
        foot_html = (f'<div style="font-size:0.76rem;color:{T.MUTED};'
                     f'margin-top:0.25rem">{foot}</div>' if foot else "")
        with col:
            st.markdown(
                f'<div class="uar-card" style="height:100%">'
                f'<div style="font-size:0.72rem;color:{T.MUTED}">{label}'
                f'</div>{main_html}{foot_html}</div>',
                unsafe_allow_html=True)
