"""Study results — design screens 08/09: condition comparison cards,
charts, decision-change card, statistical-analysis cards, demo-data
management and the raw decision log. Real decisions only by default."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import AppConfig
from src.database import (fetch_decisions, counts, delete_demo_data,
                          delete_all_data, log_decision)
from src.study import (summarize_conditions, overconfidence, decision_changes,
                       statistical_analysis, generate_demo_data,
                       MIN_DECISIONS_PER_CONDITION, MIN_PARTICIPANTS)
from . import theme as T

NAMES = {"POINT": "Point prediction", "UNCERTAINTY": "Uncertainty-aware"}
COLORS = {"POINT": "#aebdcd", "UNCERTAINTY": T.BLUE}


def render(state: dict, cfg: AppConfig) -> None:
    T.page_header(
        "Study results",
        "Decisions recorded so far, compared between the point-forecast "
        "and uncertainty conditions.")

    c = counts()
    include_demo = False
    if c["demo_decisions"] > 0:
        colA, colB = st.columns([2.6, 1])
        with colA:
            st.markdown(
                f'<div class="strip strip-amber">The database contains '
                f'<b>{c["demo_decisions"]} SYNTHETIC demo decisions</b>. '
                f'They are excluded from the analysis unless explicitly '
                f'included — never use them for real conclusions.</div>',
                unsafe_allow_html=True)
        with colB:
            include_demo = st.toggle("Include synthetic demo data",
                                     value=False)

    df = fetch_decisions(include_demo=include_demo)

    if df.empty:
        _empty_state(state, cfg)
        return

    if include_demo and (df["is_demo"] == 1).any():
        st.markdown('<div class="strip strip-amber">Synthetic demo records '
                    'are included below — these numbers do not represent '
                    'real human behaviour.</div>', unsafe_allow_html=True)

    # ---------------- Condition comparison cards ----------------
    summary = summarize_conditions(df)
    n_participants = df["participant_id"].nunique()
    src = "demo" if include_demo and (df["is_demo"] == 1).all() else "real"
    st.markdown(f"### Condition comparison "
                f"<span style='font-size:0.76rem;color:{T.MUTED};"
                f"font-weight:400'>{len(df)} decisions · {n_participants} "
                f"participants</span>", unsafe_allow_html=True)
    if len(summary) < 2:
        st.markdown('<div class="strip">Both conditions need decisions '
                    'before a comparison is possible. Keep running study '
                    'sessions.</div>', unsafe_allow_html=True)

    cols = st.columns(max(len(summary), 1))
    for col, (_, row) in zip(cols, summary.iterrows()):
        accent = (f"border-top: 3px solid {T.BLUE};"
                  if row["condition"] == "UNCERTAINTY" else "")
        with col:
            st.markdown(f"""
<div class="uar-card" style="{accent}">
  <div style="font-size:0.8rem;font-weight:650;color:{T.INK_2}">
    {NAMES.get(row['condition'], row['condition'])}</div>
  <div style="font-size:var(--t-data);letter-spacing:-0.025em;
    font-weight:600;line-height:1.2">{row['decision_accuracy']:.0%}</div>
  <div style="font-size:0.76rem;color:{T.MUTED}">decision accuracy (95% CI
    {row['accuracy_ci_low']:.0%}–{row['accuracy_ci_high']:.0%})</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;
    margin-top:0.7rem;border-top:1px solid {T.GRID};padding-top:0.6rem">
    <div><div style="font-size:var(--t-xs);color:{T.MUTED}">Decisions</div>
      <div style="font-weight:650">{row['n_decisions']}</div></div>
    <div><div style="font-size:var(--t-xs);color:{T.MUTED}">Avg decision time</div>
      <div style="font-weight:650">{row['avg_decision_time_s']:.1f} s</div></div>
    <div><div style="font-size:var(--t-xs);color:{T.MUTED}">Avg stated
      confidence</div>
      <div style="font-weight:650">{row['avg_user_confidence']:.2f} / 5</div></div>
  </div>
</div>""", unsafe_allow_html=True)

    # ---------------- Charts ----------------
    c1, c2 = st.columns(2)
    order = ["POINT", "UNCERTAINTY"]
    with c1:
        fig = go.Figure()
        for cond in order:
            row = summary[summary["condition"] == cond]
            if row.empty:
                continue
            r = row.iloc[0]
            fig.add_bar(x=[NAMES[cond].replace(" prediction", "")],
                        y=[r["decision_accuracy"]],
                        marker_color=COLORS[cond],
                        marker_line=dict(color=T.SURFACE, width=2),
                        error_y=dict(
                            type="data", symmetric=False,
                            array=[r["accuracy_ci_high"] - r["decision_accuracy"]],
                            arrayminus=[r["decision_accuracy"] - r["accuracy_ci_low"]],
                            color=T.INK_2),
                        text=[f"{r['decision_accuracy']:.0%}"],
                        textposition="outside", showlegend=False,
                        hovertemplate=f"{NAMES[cond]}: %{{y:.1%}}<extra></extra>")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.08])
        fig.update_layout(title="Decision accuracy by condition (95% CI)")
        T.apply_layout(fig, height=310)
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False})
    with c2:
        fig = go.Figure()
        for cond in order:
            g = df[df["condition"] == cond]
            if g.empty:
                continue
            fig.add_box(y=g["decision_time_seconds"],
                        name=NAMES[cond].replace(" prediction", ""),
                        marker_color=COLORS[cond], boxmean=True,
                        showlegend=False)
        fig.update_layout(title="Decision time (seconds)")
        T.apply_layout(fig, height=310)
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False})

    # ---------------- Confidence calibration & decision change ----------
    st.markdown("### Confidence calibration & decision changes")
    oc = overconfidence(df)
    dc = decision_changes(df)
    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure()
        for cond in order:
            row = oc[oc["condition"] == cond]
            if row.empty:
                continue
            r = row.iloc[0]
            fig.add_bar(x=["Stated confidence", "Actual accuracy"],
                        y=[r["stated_confidence_0_1"], r["accuracy"]],
                        name=NAMES[cond].replace(" prediction", ""),
                        marker_color=COLORS[cond],
                        marker_line=dict(color=T.SURFACE, width=2),
                        text=[f"{r['stated_confidence_0_1']:.0%}",
                              f"{r['accuracy']:.0%}"],
                        textposition="outside",
                        hovertemplate="%{x}: %{y:.0%}<extra></extra>")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.12])
        fig.update_layout(barmode="group",
                          title="Stated confidence (1–5 → 0–1) vs accuracy")
        T.apply_layout(fig, height=320)
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False})
        notes = []
        for _, r in oc.iterrows():
            direction = ("overconfident" if r["overconfidence"] > 0
                         else "underconfident")
            notes.append(f"{NAMES[r['condition']].replace(' prediction', '')}: "
                         f"{abs(r['overconfidence']):.0%} {direction}")
        st.caption(" · ".join(notes) + ".")
    with c4:
        if dc.get("n_paired_scenarios", 0) > 0:
            def kv(k, v):
                return (f'<div style="display:flex;justify-content:'
                        f'space-between;align-items:center;padding:0.55rem 0;'
                        f'border-bottom:1px solid {T.GRID}">'
                        f'<span style="font-size:0.84rem;color:{T.INK_2}">'
                        f'{k}</span><span style="font-size:var(--t-xl);'
                        f'font-weight:650;letter-spacing:-0.02em">'
                        f'{v}</span></div>')
            st.markdown(
                f'<div class="uar-card"><b>Decision change with uncertainty '
                f'information</b><div style="height:0.4rem"></div>'
                + kv("Scenarios decided under both conditions "
                     "(across participants)", dc["n_paired_scenarios"])
                + kv('Mean absolute change in "plan the event" rate',
                     f"{dc['mean_abs_event_rate_change']:.0%}")
                + kv("Scenarios where the majority decision flipped",
                     f"{dc['share_scenarios_majority_flipped']:.0%}")
                + "</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="strip">Decision-change analysis needs '
                        'scenarios that received decisions under both '
                        'conditions.</div>', unsafe_allow_html=True)

    # ---------------- Statistical analysis ----------------
    st.markdown("### Statistical analysis")
    res = statistical_analysis(df)
    if res.get("exploratory"):
        st.markdown(
            f'<div class="strip strip-amber"><b>Exploratory only.</b> Fewer '
            f'than {MIN_DECISIONS_PER_CONDITION} decisions per condition or '
            f'fewer than {MIN_PARTICIPANTS} participants '
            f'({res.get("n_point", 0)} POINT / '
            f'{res.get("n_uncertainty", 0)} UNCERTAINTY decisions from '
            f'{res.get("n_participants", 0)} participants). Do not treat '
            f'p-values as confirmatory evidence.</div>',
            unsafe_allow_html=True)

    cards = []
    if "accuracy" in res:
        a = res["accuracy"]
        cards.append((
            "Primary outcome — decision accuracy",
            f"Point <b>{a['point']:.1%}</b> (95% CI "
            f"{a['point_ci95'][0]:.1%}–{a['point_ci95'][1]:.1%}) vs "
            f"uncertainty-aware <b>{a['uncertainty']:.1%}</b> (95% CI "
            f"{a['uncertainty_ci95'][0]:.1%}–{a['uncertainty_ci95'][1]:.1%})."
            f"<br>Difference <b>{a['difference']:+.1%}</b> (95% CI "
            f"{a['diff_ci95'][0]:+.1%} to {a['diff_ci95'][1]:+.1%}), "
            f"Cohen's h <b>{a['effect_size_cohens_h']:.2f}</b>.<br>"
            f"{a['test']}: <b>p = {a['p_value']:.4f}</b>."))
    if "time" in res:
        t = res["time"]
        cards.append((
            "Decision time",
            f"Point mean <b>{t['point_mean']:.1f} s</b> (median "
            f"{t['point_median']:.1f}) vs uncertainty-aware mean "
            f"<b>{t['uncertainty_mean']:.1f} s</b> (median "
            f"{t['uncertainty_median']:.1f}).<br>{t['test']}: "
            f"<b>p = {t['p_value']:.4f}</b>, {t['effect_size_name']} = "
            f"{t['effect_size']:.2f}."))
    if "confidence" in res and res["confidence"].get("p_value") is not None:
        cf = res["confidence"]
        cards.append((
            "Stated confidence",
            f"Point <b>{cf['point_mean']:.2f}/5</b> vs uncertainty-aware "
            f"<b>{cf['uncertainty_mean']:.2f}/5</b>.<br>{cf['test']}: "
            f"<b>p = {cf['p_value']:.4f}</b>, rank-biserial r = "
            f"{cf['effect_size_rank_biserial']:.2f}."))
    if cards:
        cc = st.columns(len(cards))
        for col, (title, body) in zip(cc, cards):
            with col:
                st.markdown(
                    f'<div class="uar-card" style="height:100%">'
                    f'<b>{title}</b><div style="font-size:0.83rem;'
                    f'color:{T.INK_2};margin-top:0.4rem;line-height:1.55">'
                    f'{body}</div></div>', unsafe_allow_html=True)
    for note in res.get("notes", []):
        st.caption(note)

    # ---------------- Demo data management ----------------
    st.markdown("### Demo data (synthetic)")
    b1, b2, b3 = st.columns(3)
    if b1.button("Generate demo data (12 synthetic participants)",
                 width="stretch"):
        from .study_run import _scenario_pool
        pool = _scenario_pool(state, cfg)
        for rec in generate_demo_data(pool, cfg):
            log_decision(rec)
        st.rerun()
    if b2.button("Delete all demo data", width="stretch"):
        delete_demo_data()
        st.rerun()
    if b3.button("Delete ALL study data", width="stretch",
                 help="Removes real AND demo decisions."):
        delete_all_data()
        st.rerun()

    # ---------------- Raw log ----------------
    with st.expander(f"Raw decision log ({len(df)} rows)"):
        st.dataframe(df.drop(columns=["id"]), width="stretch",
                     hide_index=True, height=300)
        st.download_button("Download decision log (CSV)",
                           df.to_csv(index=False).encode("utf-8"),
                           "study_decisions.csv", "text/csv")


def _empty_state(state: dict, cfg: AppConfig) -> None:
    st.markdown(f"""
<div class="uar-card" style="text-align:center;padding:2.6rem 2rem">
  <svg width="52" height="52" viewBox="0 0 24 24" fill="none"
    stroke="{T.MUTED}" stroke-width="1.6" stroke-linecap="round">
    <path d="M4 20h16M7 20V10M12 20V4M17 20v-9"/></svg>
  <div style="font-size:var(--t-xl);letter-spacing:-0.02em;
    font-weight:600;margin-top:0.6rem">No study data collected yet</div>
  <div style="color:{T.INK_2};font-size:0.9rem;max-width:520px;
    margin:0.4rem auto 1rem">Run sessions in the <b>User study</b> page;
    results are computed automatically from real decisions.</div>
</div>""", unsafe_allow_html=True)
    b1, b2, _sp = st.columns([1, 1.2, 2])
    if b1.button("Go to User study", type="primary",
                 width="stretch"):
        st.session_state["nav"] = "User study"
        st.rerun()
    if b2.button("Generate synthetic demo data", width="stretch"):
        from .study_run import _scenario_pool
        pool = _scenario_pool(state, cfg)
        for rec in generate_demo_data(pool, cfg):
            log_decision(rec)
        st.rerun()

    m1, m2, m3 = st.columns(3)
    for col, (k, t, d) in zip(
            [m1, m2, m3],
            [("Primary outcome", "Decision accuracy",
              "Fisher's exact / χ² (Yates), Cohen's h, Wilson 95% CIs"),
             ("Secondary", "Decision time",
              "Welch's t or Mann–Whitney U, chosen from Shapiro–Wilk"),
             ("Secondary", "Stated confidence & overconfidence",
              "Mann–Whitney U on the ordinal 1–5 scale")]):
        with col:
            st.markdown(
                f'<div class="uar-card" style="height:100%">'
                f'<div style="font-size:var(--t-xs);color:{T.MUTED}">{k}</div>'
                f'<b>{t}</b><div style="font-size:0.8rem;color:{T.INK_2};'
                f'margin-top:0.2rem">{d}</div></div>',
                unsafe_allow_html=True)
