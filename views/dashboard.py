"""Prediction dashboard — design screens 01/02.

Two forecast modes:

* **Live (default)** — a real forecast for the actual calendar tomorrow.
  The model is trained on the historical dataset; today's conditions come
  from a live observation feed for the station's coordinates, assembled
  into exactly the same leakage-safe feature row used in training.
* **Historical backtest** — any date in the dataset, predicting the day
  after it from the conditions actually observed then, with the real
  outcome revealed.

If the live feed is unreachable the app falls back to the station's
seasonal climatology and says so.
"""
from __future__ import annotations

import datetime as _dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import AppConfig
from src.live_weather import LiveWeatherError, live_forecast_row
from src.predict import predict_one, feature_influence, climatology_row
from src.uncertainty import RAIN
from . import theme as T

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
          "Oct", "Nov", "Dec"]


@st.cache_data(ttl=1800, show_spinner=False)
def _live_row(station: str, target_date, feature_cols: tuple,
              rain_threshold: float, _features):
    """Cached live feature row (30-minute TTL). Returns (row, info)."""
    return live_forecast_row(_features, station, list(feature_cols),
                             pd.Timestamp(target_date), rain_threshold)


def render(state: dict, cfg: AppConfig) -> None:
    features: pd.DataFrame = state["features"]
    bundle = state["bundle"]

    T.page_header(
        "Prediction dashboard",
        "Tomorrow's rainfall forecast for one station, with the "
        "calibrated probability and what the model can and cannot "
        "rule out.")

    c1, c2 = st.columns([2.2, 1])
    stations = sorted(features["station"].unique())
    default_ix = stations.index("New Delhi / Safdarjung") \
        if "New Delhi / Safdarjung" in stations else 0
    with c1:
        station = st.selectbox("Location / station", stations,
                               index=default_ix)
    sdf = features[features["station"] == station].sort_values("date")
    dates = sdf["date"].dt.date

    today = pd.Timestamp(_dt.date.today())
    tomorrow = today + pd.Timedelta(1, "D")
    with c2:
        mode = st.selectbox(
            "Forecast mode",
            [f"Tomorrow — {tomorrow.strftime('%d %b %Y')}",
             "Historical backtest"], index=0)
    live_mode = mode.startswith("Tomorrow")

    if live_mode:
        _render_live(state, cfg, station, sdf, today, tomorrow)
        return

    sel_date = st.selectbox("Observation date (predicts the following day)",
                            list(dates)[::-1], index=0)

    row = sdf[dates == sel_date].iloc[[0]]
    pred = predict_one(bundle, row, cfg)
    r = row.iloc[0]
    actual_mm = r.get("rain_tomorrow_mm")
    outcome = ("significant rain" if r.get("RainTomorrow") == 1
               else "no significant rain")
    st.markdown(
        f'<div class="strip">Actual outcome on '
        f'{sel_date + _dt.timedelta(days=1)}: <b>{outcome}</b> '
        f'({actual_mm:.1f} mm).</div>', unsafe_allow_html=True)

    # ---------------- Hero card ----------------
    _hero_card(station,
               f"{station} &middot; prediction for the day after {sel_date}",
               pred, cfg)

    # ---------------- Class probabilities + conditions ----------------
    col_a, col_b = st.columns([1.05, 1])
    with col_a:
        st.markdown(
            f'<div class="uar-card"><div style="display:flex;'
            f'justify-content:space-between"><b>Class probabilities</b></div>'
            + _prob_bars(pred, cfg)
            + f'<div style="font-size:0.76rem;color:{T.MUTED};'
              f'margin:0.6rem 0 0.2rem">Model confidence</div>'
            + T.confidence_scale_html(pred["confidence"])
            + '</div>', unsafe_allow_html=True)
    with col_b:
        tiles = []
        for label, key, unit in [("Avg temperature", "avg_temp", " °C"),
                                 ("Min temperature", "min_temp", " °C"),
                                 ("Max temperature", "max_temp", " °C"),
                                 ("Wind speed", "wind_speed", " km/h"),
                                 ("Air pressure", "air_pressure", " hPa"),
                                 ("Rainfall today", "rainfall_today", " mm"),
                                 ("Rain last 3 days", "rainfall_rolling_3", " mm"),
                                 ("Rain last 7 days", "rainfall_rolling_7", " mm")]:
            v = r.get(key)
            if v is not None and not pd.isna(v):
                tiles.append(T.tile(label, f"{v:,.1f}{unit}"))
        grid = ('<div style="display:grid;grid-template-columns:1fr 1fr;'
                'gap:10px;margin-top:0.6rem">' + "".join(tiles) + "</div>")
        st.markdown(
            f'<div class="uar-card"><div style="display:flex;'
            f'justify-content:space-between"><b>Conditions</b></div>{grid}</div>', unsafe_allow_html=True)

    # ---------------- Feature influence ----------------
    _influence_card(bundle, row.iloc[0], state["train_stats"])

    _history_charts(sdf, cfg, station, sel_date)


def _history_charts(sdf: pd.DataFrame, cfg: AppConfig, station: str,
                    window_end) -> None:
    """Station history: rainfall, temperature, pressure and climatology."""
    st.markdown("### Weather history")
    lookback = st.slider("History window (days)", 30, 730, 180, step=30)
    hist = sdf[sdf["date"] <= pd.Timestamp(window_end)].tail(lookback)

    fig = go.Figure()
    fig.add_bar(x=hist["date"], y=hist["rainfall_today"], name="Rainfall (mm)",
                marker_color=T.BLUE,
                hovertemplate="%{x|%d %b %Y}: %{y:.1f} mm<extra></extra>")
    fig.add_hline(y=cfg.rain_threshold_mm, line_dash="dot",
                  line_color=T.ORANGE,
                  annotation_text=f"{cfg.rain_threshold_mm} mm threshold",
                  annotation_font_color=T.ORANGE)
    fig.update_layout(title="Daily rainfall (mm)")
    T.apply_layout(fig, height=250)
    st.plotly_chart(fig, width="stretch",
                    config={"displayModeBar": False})

    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure()
        if "min_temp" in hist.columns:
            fig.add_scatter(x=hist["date"], y=hist["min_temp"], name="Min",
                            mode="lines", line=dict(color=T.BLUE, width=1.4))
        if "avg_temp" in hist.columns:
            fig.add_scatter(x=hist["date"], y=hist["avg_temp"], name="Avg",
                            mode="lines", line=dict(color=T.ORANGE, width=2))
        if "max_temp" in hist.columns:
            fig.add_scatter(x=hist["date"], y=hist["max_temp"], name="Max",
                            mode="lines", line=dict(color=T.AQUA, width=1.4))
        fig.update_layout(title="Temperature (°C)")
        T.apply_layout(fig, height=250)
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False})
    with c4:
        fig = go.Figure()
        if "air_pressure" in hist.columns:
            fig.add_scatter(x=hist["date"], y=hist["air_pressure"],
                            name="Pressure (hPa)", mode="lines",
                            line=dict(color=T.BLUE, width=2))
            fig.update_layout(title="Air pressure (hPa)")
        elif "wind_speed" in hist.columns:
            fig.add_scatter(x=hist["date"], y=hist["wind_speed"],
                            name="Wind (km/h)", mode="lines",
                            line=dict(color=T.BLUE, width=2))
            fig.update_layout(title="Wind speed (km/h)")
        T.apply_layout(fig, height=250)
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False})

    c5, c6 = st.columns(2)
    with c5:
        monthly = (sdf.assign(m=sdf["date"].dt.month)
                      .groupby("m")["rainfall_today"].mean())
        fig = go.Figure(go.Bar(
            x=MONTHS, y=[monthly.get(m, 0) for m in range(1, 13)],
            marker_color=T.SEQ_BLUE[7],
            marker_line=dict(color=T.SURFACE, width=2),
            hovertemplate="%{x}: %{y:.1f} mm/day<extra></extra>"))
        fig.update_layout(title="Average daily rainfall by month (mm)")
        T.apply_layout(fig, height=250)
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False})
    with c6:
        freq = (sdf.assign(m=sdf["date"].dt.month)
                   .groupby("m")["RainTomorrow"].mean())
        fig = go.Figure(go.Bar(
            x=MONTHS, y=[freq.get(m, 0) for m in range(1, 13)],
            marker_color=T.GREEN,
            marker_line=dict(color=T.SURFACE, width=2),
            hovertemplate="%{x}: %{y:.0%} of days<extra></extra>"))
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(
            title=f"Share of days with ≥ {cfg.rain_threshold_mm} mm next-day rain")
        T.apply_layout(fig, height=250)
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False})


def _hero_card(station: str, context: str, pred: dict, cfg: AppConfig,
               extra_strip: str = "") -> None:
    icon = T.RAIN_SVG if pred["is_rain"] else T.SUN_SVG
    answer = "YES" if pred["is_rain"] else "NO"
    ans_color = T.BLUE if pred["is_rain"] else T.INK
    pset_txt = " / ".join(pred["prediction_set"])
    pset_kind = "orange" if pred["ambiguous"] else (
        "blue" if pred["is_rain"] else "green")
    amb = ""
    if pred["ambiguous"]:
        amb = ('<div class="strip strip-warn" style="margin:0.7rem 0 0">'
               '<b>Ambiguous</b> — the model cannot rule out either '
               'outcome.</div>')
    st.markdown(f"""
<div class="uar-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px">
    <div style="font-size:0.78rem;color:{T.MUTED}">{context}</div>
    <div>
      <span class="chip">Confidence <b>{pred['confidence']}</b></span>
      <span class="chip">Uncertainty <b>{pred['uncertainty']}</b></span>
      <span class="chip">{cfg.conformal_coverage:.0%} set <b>{pset_txt}</b></span>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:14px;margin-top:0.5rem">
    <div>{icon}</div>
    <div>
      <div style="font-size:var(--t-hero);font-weight:650;
        letter-spacing:-0.025em;
        line-height:1.12;color:{T.INK}">Rain tomorrow:
        <span style="color:{ans_color}">{answer}</span></div>
      <div style="color:{T.INK_2};font-size:0.95rem">Rain probability
        <b>{pred['p_rain']:.0%}</b> &middot; No-rain probability
        <b>{pred['p_no_rain']:.0%}</b></div>
    </div>
  </div>
  {T.probability_track(pred['p_rain'], cfg.classification_threshold)}
  <div style="font-size:0.76rem;color:{T.MUTED};margin-top:0.5rem">
    {cfg.conformal_coverage:.0%} prediction set</div>
  {T.pset_boxes(pred, cfg.rain_threshold_mm)}
  {extra_strip}{amb}
</div>""", unsafe_allow_html=True)


def _render_live(state: dict, cfg: AppConfig, station: str,
                 sdf: pd.DataFrame, today: pd.Timestamp,
                 tomorrow: pd.Timestamp) -> None:
    """Forecast for the real calendar tomorrow from live conditions."""
    features, bundle = state["features"], state["bundle"]
    last_obs = sdf["date"].max()

    row = info = None
    source = "live"
    try:
        with st.spinner("Fetching current conditions…"):
            row, info = _live_row(station, tomorrow.date(),
                                  tuple(bundle["feature_cols"]),
                                  cfg.rain_threshold_mm, features)
    except LiveWeatherError as exc:
        source = "climatology"
        reason = str(exc)
        row, clim_info = climatology_row(features, station, tomorrow)

    pred = predict_one(bundle, row, cfg)

    if source == "live":
        badge = '<span class="chip">Source <b>Live weather</b></span>'
        context = (f"{station} · forecast for tomorrow, "
                   f"{tomorrow.strftime('%d %B %Y')}")
        strip = (
            f'<div class="strip strip-note" style="margin:0.7rem 0 0">'
            f'Live conditions from {info["observation_date"]} · model '
            f'trained on data through {last_obs.date()}.</div>')
        cond = info["conditions"]
        recent = info["recent"]
    else:
        badge = '<span class="chip">Source <b>Seasonal average</b></span>'
        context = (f"{station} · estimate for tomorrow, "
                   f"{tomorrow.strftime('%d %B %Y')}")
        strip = (
            f'<div class="strip strip-note" style="margin:0.7rem 0 0">'
            f'Live conditions unavailable ({reason}) — using this '
            f'station\'s seasonal average for this date.</div>')
        c0 = row.iloc[0]
        cond = {k: float(c0[k]) for k in
                ["avg_temp", "min_temp", "max_temp", "wind_speed",
                 "air_pressure", "rainfall_today", "rainfall_rolling_3",
                 "rainfall_rolling_7"] if k in row.columns}
        recent = None

    st.markdown(f'<div style="margin-bottom:0.4rem">{badge}</div>',
                unsafe_allow_html=True)
    _hero_card(station, context, pred, cfg, extra_strip=strip)

    col_a, col_b = st.columns([1.05, 1])
    with col_a:
        st.markdown(
            f'<div class="uar-card"><div style="display:flex;'
            f'justify-content:space-between"><b>Class probabilities</b></div>'
            + _prob_bars(pred, cfg)
            + f'<div style="font-size:0.76rem;color:{T.MUTED};'
              f'margin:0.6rem 0 0.2rem">Model confidence</div>'
            + T.confidence_scale_html(pred["confidence"])
            + '</div>', unsafe_allow_html=True)
    with col_b:
        label_map = [("Avg temperature", "avg_temp", " °C"),
                     ("Min temperature", "min_temp", " °C"),
                     ("Max temperature", "max_temp", " °C"),
                     ("Wind speed", "wind_speed", " km/h"),
                     ("Air pressure", "air_pressure", " hPa"),
                     ("Rainfall today", "rainfall_today", " mm"),
                     ("Rain last 3 days", "rainfall_rolling_3", " mm"),
                     ("Rain last 7 days", "rainfall_rolling_7", " mm")]
        tiles = [T.tile(lbl, f"{cond[k]:,.1f}{u}")
                 for lbl, k, u in label_map if k in cond
                 and cond[k] == cond[k]]
        head = ("Conditions today" if source == "live"
                else "Typical conditions for this time of year")
        sub = (str(info["observation_date"]) if source == "live"
               else "seasonal average")
        st.markdown(
            f'<div class="uar-card"><div style="display:flex;'
            f'justify-content:space-between"><b>{head}</b>'
            f'<span style="font-size:var(--t-xs);color:{T.MUTED}">{sub}</span>'
            f'</div><div style="display:grid;grid-template-columns:1fr 1fr;'
            f'gap:10px;margin-top:0.6rem">' + "".join(tiles)
            + "</div></div>", unsafe_allow_html=True)

    # ---------------- Feature influence ----------------
    _influence_card(bundle, row.iloc[0], state["train_stats"])

    # ---------------- Live recent-days chart ----------------
    if recent is not None and len(recent) > 1:
        fig = go.Figure()
        fig.add_bar(x=recent["date"], y=recent["rainfall"],
                    name="Rainfall (mm)", marker_color=T.BLUE,
                    hovertemplate="%{x|%d %b}: %{y:.1f} mm<extra></extra>")
        fig.add_hline(y=cfg.rain_threshold_mm, line_dash="dot",
                      line_color=T.ORANGE,
                      annotation_text=f"{cfg.rain_threshold_mm} mm threshold",
                      annotation_font_color=T.ORANGE)
        fig.update_layout(title="Recent rainfall (mm)")
        T.apply_layout(fig, height=240)
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False})

    _history_charts(sdf, cfg, station, last_obs.date())


def _influence_card(bundle, row_series, train_stats) -> None:
    infl = feature_influence(bundle, row_series, train_stats, top_n=8)
    max_score = max(i["influence_score"] for i in infl) or 1.0
    rows_html = []
    for item in infl:
        w = max(4, item["influence_score"] / max_score * 100)
        color = {"HIGH": T.BLUE, "MODERATE": "#86b6ef",
                 "LOW": "#d5dde8"}[item["influence"]]
        val = "" if item["value"] is None else f' <code>{item["value"]:.1f}</code>'
        rows_html.append(
            f'<div class="infl-row"><div class="infl-name">{item["feature"]}'
            f'{val}</div><div class="infl-track"><div class="infl-fill" '
            f'style="width:{w:.0f}%;background:{color}"></div></div>'
            f'<div class="infl-band">{item["influence"]}</div></div>')
    st.markdown(
        f'<div class="uar-card"><div style="display:flex;'
        f'justify-content:space-between"><b>Model feature influence</b>'
        f'</div>'
        f'<div style="height:0.4rem"></div>' + "".join(rows_html)
        + "</div>", unsafe_allow_html=True)


def _prob_bars(pred: dict, cfg: AppConfig) -> str:
    def bar(label, p, color):
        return (f'<div style="display:flex;align-items:center;gap:10px;'
                f'margin:8px 0"><div style="width:150px;font-size:0.8rem;'
                f'color:{T.INK_2}">{label}</div>'
                f'<div style="flex:1;height:14px;background:#eef1f5;'
                f'border-radius:4px;position:relative">'
                f'<div style="width:{p * 100:.0f}%;height:14px;'
                f'background:{color};border-radius:4px"></div>'
                f'<div style="position:absolute;top:-4px;bottom:-4px;'
                f'left:{cfg.classification_threshold * 100:.0f}%;width:1.5px;'
                f'border-left:2px dotted {T.INK_2}"></div></div>'
                f'<div style="width:44px;text-align:right;font-weight:650">'
                f'{p:.0%}</div></div>')
    return ('<div style="margin-top:0.6rem">'
            + bar("NO SIGNIFICANT RAIN", pred["p_no_rain"], "#aebdcd")
            + bar("RAIN", pred["p_rain"], T.BLUE)
            + f'<div style="font-size:var(--t-xs);color:{T.MUTED};'
              f'text-align:center">threshold '
              f'{cfg.classification_threshold:.0%}</div></div>')
