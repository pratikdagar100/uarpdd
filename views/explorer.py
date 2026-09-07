"""Dataset explorer — design screen 03: stat tiles, mapping + preprocessing
cards, split strip, record browser and charts."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import AppConfig
from . import theme as T


def render(state: dict, cfg: AppConfig) -> None:
    features: pd.DataFrame = state["features"]
    insp = state["inspection"]
    prep = state["prep_report"]
    feat = state["feat_report"]

    T.page_header(
        "Dataset explorer",
        "What the pipeline found in the source data, how it mapped the "
        "columns, and what cleaning removed before training.")

    dr = insp.get("date_range", ("—", "—"))
    rain_days = int(features["RainTomorrow"].sum())
    tiles = [
        T.tile("Rows (raw)", f"{insp['n_rows']:,}"),
        T.tile("Columns", str(insp["n_cols"])),
        T.tile("Date range", f"{dr[0]}", f"→ {dr[1]}"),
        T.tile("Stations", str(insp.get("n_stations", 1))),
        T.tile("Rain days (target = 1)", f"{rain_days:,}",
               f"{rain_days / len(features):.1%} of modelling rows"),
        T.tile("No-rain days", f"{len(features) - rain_days:,}"),
    ]
    st.markdown('<div style="display:grid;grid-template-columns:repeat(6,1fr);'
                'gap:10px;margin-bottom:0.9rem">' + "".join(tiles) + "</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        mapping = insp["mapping"]
        rows = "".join(
            f'<tr><td style="padding:5px 8px;color:{T.MUTED};'
            f'font-size:0.78rem">{role}</td>'
            f'<td style="padding:5px 8px;font-family:var(--mono);'
            f'font-size:0.82rem;color:{T.INK}">'
            f'{col if col else "&mdash; not present &mdash;"}</td></tr>'
            for role, col in mapping.items())
        st.markdown(
            f'<div class="uar-card"><div style="display:flex;'
            f'justify-content:space-between"><b>Automatic column mapping</b>'
            f'</div><table style="width:100%;border-collapse:collapse;'
            f'margin-top:0.5rem"><tr>'
            f'<th style="text-align:left;padding:5px 8px;font-size:0.7rem;'
            f'letter-spacing:0.08em;color:{T.MUTED}">PIPELINE ROLE</th>'
            f'<th style="text-align:left;padding:5px 8px;font-size:0.7rem;'
            f'letter-spacing:0.08em;color:{T.MUTED}">DETECTED DATASET COLUMN'
            f'</th></tr>{rows}</table>'
            f'</div>',
            unsafe_allow_html=True)
    with c2:
        steps = [
            ("Original rows", f"{prep['original_rows']:,}"),
            ("Rows dropped (unparseable date)",
             f"{prep.get('rows_dropped_bad_date', 0):,}"),
            ("Duplicate (station, date) rows merged",
             f"{prep.get('station_date_duplicate_rows', 0):,}"),
            ("Rows after deduplication", f"{prep.get('rows_after_dedup', 0):,}"),
            ("Missing values before preprocessing",
             f"{prep['missing_values_before']:,}"),
            ("Missing values after preprocessing",
             f"{prep['missing_values_after']:,}"),
            ("Observed rainfall left missing (never imputed for target)",
             f"{prep.get('missing_observed_rainfall', 0):,}"),
            ("Rows dropped (no valid next-day target / warm-up lags)",
             f"{feat['rows_dropped_no_target_or_lags']:,}"),
            ("Final modelling rows", f"{feat['final_rows']:,}"),
        ]
        rows = "".join(
            f'<tr><td style="padding:5px 8px;font-size:0.82rem;'
            f'color:{T.INK_2}">{k}</td>'
            f'<td style="padding:5px 8px;text-align:right;font-weight:650;'
            f'font-size:0.85rem">{v}</td></tr>' for k, v in steps)
        extras = []
        if prep.get("values_nullified_out_of_range"):
            extras.append("Values outside physical bounds set to missing: "
                          + " · ".join(f"{k}: {v:,}" for k, v in
                                       prep["values_nullified_out_of_range"].items()))
        if prep.get("imputed_values"):
            extras.append("Imputed (station-month median): "
                          + " · ".join(f"{k} {v:,}" for k, v in
                                       prep["imputed_values"].items() if v))
        extra_html = (f'<div style="font-size:var(--t-xs);color:{T.MUTED};'
                      f'margin-top:0.5rem">{" — ".join(extras)}</div>'
                      if extras else "")
        st.markdown(
            f'<div class="uar-card"><div style="display:flex;'
            f'justify-content:space-between"><b>Preprocessing report</b>'
            f'</div><table style="width:100%;'
            f'border-collapse:collapse;margin-top:0.5rem"><tr>'
            f'<th style="text-align:left;padding:5px 8px;font-size:0.7rem;'
            f'letter-spacing:0.08em;color:{T.MUTED}">STEP</th>'
            f'<th style="text-align:right;padding:5px 8px;font-size:0.7rem;'
            f'letter-spacing:0.08em;color:{T.MUTED}">VALUE</th></tr>{rows}'
            f'</table>{extra_html}</div>', unsafe_allow_html=True)

    split = state["split"]
    rg, ct = split["ranges"], split["counts"]
    st.markdown(f"""
<div class="split-strip">
  <div class="split-cell"><div class="s-name">Train</div>
    <div class="s-range">{rg['train'][0]} → {rg['train'][1]}</div>
    <div class="s-n">{ct['train']:,} rows</div></div>
  <div class="split-cell"><div class="s-name">Validation / calibration</div>
    <div class="s-range">{rg['validation'][0]} → {rg['validation'][1]}</div>
    <div class="s-n">{ct['validation']:,} rows</div></div>
  <div class="split-cell"><div class="s-name">Test (held out)</div>
    <div class="s-range">{rg['test'][0]} → {rg['test'][1]}</div>
    <div class="s-n">{ct['test']:,} rows</div></div>
</div>""", unsafe_allow_html=True)

    # ---------------- Browse ----------------
    st.markdown("### Browse cleaned records")
    c3, c4, c5 = st.columns([1.6, 1.4, 1])
    stations = ["All stations"] + sorted(features["station"].unique())
    with c3:
        sel = st.selectbox("Station", stations, key="explorer_station")
    dmin, dmax = features["date"].min().date(), features["date"].max().date()
    with c4:
        rng = st.date_input("Date range", (dmin, dmax),
                            min_value=dmin, max_value=dmax)

    view = features
    if sel != "All stations":
        view = view[view["station"] == sel]
    if isinstance(rng, tuple) and len(rng) == 2:
        view = view[(view["date"].dt.date >= rng[0])
                    & (view["date"].dt.date <= rng[1])]

    show_cols = [c for c in ["date", "station", "state", "avg_temp",
                             "min_temp", "max_temp", "wind_speed",
                             "air_pressure", "rainfall_today",
                             "rain_tomorrow_mm", "RainTomorrow"]
                 if c in view.columns]
    with c5:
        st.markdown("<div style='height:1.72rem'></div>",
                    unsafe_allow_html=True)
        st.download_button("Download filtered CSV",
                           view[show_cols].to_csv(index=False).encode("utf-8"),
                           file_name="processed_rainfall_dataset.csv",
                           mime="text/csv", width="stretch")

    st.dataframe(view[show_cols].head(2000), width="stretch",
                 hide_index=True, height=320)

    c6, c7 = st.columns(2)
    with c6:
        counts = view["RainTomorrow"].value_counts()
        fig = go.Figure(go.Bar(
            x=["No significant rain", "Rain"],
            y=[int(counts.get(0, 0)), int(counts.get(1, 0))],
            marker_color=["#aebdcd", T.BLUE],
            marker_line=dict(color=T.SURFACE, width=2),
            text=[f"{int(counts.get(0, 0)):,}", f"{int(counts.get(1, 0)):,}"],
            textposition="outside",
            hovertemplate="%{x}: %{y:,}<extra></extra>"))
        fig.update_layout(title="Target distribution (filtered selection)")
        T.apply_layout(fig, height=300)
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False})
    with c7:
        if sel == "All stations":
            top = (features.groupby("station")["rainfall_today"].mean()
                   .sort_values(ascending=False).head(12).sort_values())
            fig = go.Figure(go.Bar(
                x=top.values, y=top.index, orientation="h",
                marker_color=T.BLUE,
                marker_line=dict(color=T.SURFACE, width=2),
                hovertemplate="%{y}: %{x:.1f} mm/day<extra></extra>"))
            fig.update_layout(title="Wettest stations (mean daily rainfall, mm)")
        else:
            yearly = (view.assign(y=view["date"].dt.year)
                      .groupby("y")["rainfall_today"].sum())
            fig = go.Figure(go.Bar(
                x=yearly.index.astype(str), y=yearly.values,
                marker_color=T.BLUE,
                marker_line=dict(color=T.SURFACE, width=2),
                hovertemplate="%{x}: %{y:,.0f} mm<extra></extra>"))
            fig.update_layout(title=f"Yearly rainfall — {sel} (mm)")
        T.apply_layout(fig, height=300)
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False})
