"""Early warning board — heavy rainfall warnings and inundation risk.

The headline page for SIH problem statement 26071 (MoES / IMD): sweeps the
monitored stations, forecasts tomorrow for each one from live conditions
(NWP-driven feed, seasonal fallback), and issues an IMD-style colour-coded
warning plus an inundation risk band per station. Organised as interactive
tabs: Overview, Warning map, Station detail, Data table, Method.
"""
from __future__ import annotations

import datetime as _dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import AppConfig
from src.live_weather import LiveWeatherError, live_forecast_row
from src.predict import predict_one, climatology_row
from src.warning import WARNING_META, INUNDATION_META, assess_station
from . import theme as T

MAX_DEFAULT_STATIONS = 12
LEVEL_ORDER = ["RED", "ORANGE", "YELLOW", "GREEN"]


@st.cache_data(ttl=1800, show_spinner=False)
def _live_row(station: str, target_date, feature_cols: tuple,
              rain_threshold: float, _features):
    """Cached live feature row (30-minute TTL). Returns (row, info)."""
    return live_forecast_row(_features, station, list(feature_cols),
                             pd.Timestamp(target_date), rain_threshold)


def _assess(state: dict, cfg: AppConfig, station: str,
            tomorrow: pd.Timestamp) -> dict:
    """Forecast + warning assessment for one station (live weather, with
    seasonal-average fallback)."""
    features, bundle = state["features"], state["bundle"]
    source = "live"
    try:
        row, _info = _live_row(station, tomorrow.date(),
                               tuple(bundle["feature_cols"]),
                               cfg.rain_threshold_mm, features)
    except LiveWeatherError:
        source = "seasonal"
        row, _info = climatology_row(features, station, tomorrow)

    pred = predict_one(bundle, row, cfg)
    r7 = row.iloc[0].get("rainfall_rolling_7")
    r7 = 0.0 if r7 is None or pd.isna(r7) else float(r7)
    out = assess_station(features, station, tomorrow, pred, r7, cfg)
    out["source"] = source
    sdf = features[features["station"] == station]
    for col in ("latitude", "longitude"):
        if col in sdf.columns:
            out[col] = float(sdf[col].iloc[-1])
    return out


def _range_text(a: dict) -> str:
    """Plain-language reading of the conformal prediction set."""
    if a["pred"]["ambiguous"]:
        return "either outcome possible"
    return ("rain expected" if a["pred"]["is_rain"]
            else "no significant rain expected")


def render(state: dict, cfg: AppConfig) -> None:
    features: pd.DataFrame = state["features"]

    tomorrow = pd.Timestamp(_dt.date.today()) + pd.Timedelta(days=1)
    T.page_header(
        "Early warning board",
        f'Rainfall warnings and inundation risk for '
        f'<b>{tomorrow.strftime("%A, %d %B %Y")}</b>, issued per station on '
        f'the IMD colour scale.')

    # ------------------------------------------------------ sweep controls
    stations = sorted(features["station"].unique())
    c1, c2 = st.columns([3, 1])
    with c1:
        chosen = st.multiselect("Stations to monitor", stations,
                                default=stations[:MAX_DEFAULT_STATIONS])
    with c2:
        n_more = st.number_input("Quick-add first N stations", 0,
                                 len(stations), 0, step=5)
    if n_more:
        chosen = sorted(set(chosen) | set(stations[:int(n_more)]))
    if not chosen:
        st.info("Select at least one station to run the warning sweep.")
        return

    results, prog = [], st.progress(0.0, text="Running warning sweep…")
    for i, stn in enumerate(chosen):
        prog.progress((i + 1) / len(chosen), text=f"Assessing {stn}…")
        try:
            results.append(_assess(state, cfg, stn, tomorrow))
        except Exception:
            continue
    prog.empty()
    if not results:
        st.error("No station could be assessed — check the dataset.")
        return

    results.sort(key=lambda a: (-a["meta"]["rank"],
                                -a["inundation"]["score"],
                                -a["pred"]["p_rain"]))

    tab_over, tab_map, tab_detail, tab_table, tab_method = st.tabs(
        ["Overview", "Warning map", "Station detail", "Data table",
         "Method"])

    with tab_over:
        _tab_overview(results, cfg)
    with tab_map:
        _tab_map(results)
    with tab_detail:
        _tab_detail(results, features, cfg)
    with tab_table:
        _tab_table(results, tomorrow)
    with tab_method:
        _tab_method(cfg)


# ------------------------------------------------------------- Overview tab
def _tab_overview(results: list[dict], cfg: AppConfig) -> None:
    counts = {lvl: sum(1 for a in results if a["level"] == lvl)
              for lvl in LEVEL_ORDER}
    live_n = sum(1 for a in results if a["source"] == "live")

    # A level with no stations is a fact worth showing, but it should not
    # compete with one that needs action -- so an empty count keeps the
    # neutral ground and drops its colour.
    cells = []
    for lvl in LEVEL_ORDER:
        sev = T.SEVERITY[lvl]
        n = counts[lvl]
        live = n > 0
        ground = (f"background:{sev['tint']};"
                  f"box-shadow:inset 0 0 0 1px {sev['fill']}40" if live else "")
        value_col = sev["ink"] if live else T.MUTED
        cells.append(
            f'<div class="tile" style="{ground}">'
            f'<div style="display:flex;align-items:center;gap:6px">'
            f'{T.severity_glyph(lvl) if live else ""}'
            f'<span class="t-label" style="margin:0;color:'
            f'{sev["ink"] if live else T.MUTED}">{lvl} &middot; '
            f'{sev["title"]}</span></div>'
            f'<div class="t-value" style="color:{value_col}">{n}</div>'
            f'<div class="t-foot" style="color:'
            f'{sev["ink"] if live else T.MUTED}">{sev["advice"]}</div></div>')

    acting = counts["RED"] + counts["ORANGE"]
    lede = (f"<b>{acting}</b> of {len(results)} stations need action today"
            if acting else
            f"No station is above Watch across {len(results)} assessed")
    st.markdown(
        '<div style="display:grid;grid-template-columns:'
        'repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:0.2rem 0 0.7rem">'
        + "".join(cells) + "</div>"
        f'<div style="font-size:var(--t-sm);color:{T.MUTED};'
        f'margin-bottom:0.6rem">{lede} &middot; {live_n} forecast from live '
        f'weather, {len(results) - live_n} from seasonal averages.</div>',
        unsafe_allow_html=True)

    show_levels = st.multiselect(
        "Show warning levels", LEVEL_ORDER,
        default=[l for l in LEVEL_ORDER if counts[l]],
        key="ov_levels")
    filtered = [a for a in results if a["level"] in show_levels]
    max_cards = st.slider("Stations shown", 1, max(len(filtered), 1),
                          min(10, max(len(filtered), 1)),
                          key="ov_max") if len(filtered) > 1 else 1
    for a in filtered[:max_cards]:
        _station_card(a)
    if len(filtered) > max_cards:
        st.caption(f"{len(filtered) - max_cards} more in the Data table tab.")


# ------------------------------------------------------------------ Map tab
def _tab_map(results: list[dict]) -> None:
    mapped = [a for a in results if "latitude" in a and "longitude" in a]
    if len(mapped) < 2:
        st.info("The dataset has no station coordinates to map.")
        return
    c1, c2 = st.columns(2)
    with c1:
        size_by = st.radio("Marker size shows", ["Same size",
                                                 "Waterlogging risk",
                                                 "Chance of rain"],
                           horizontal=True, key="map_size")
    with c2:
        show_names = st.checkbox("Show station names", True, key="map_names")

    if size_by == "Waterlogging risk":
        sizes = [8 + 16 * a["inundation"]["score"] for a in mapped]
    elif size_by == "Chance of rain":
        sizes = [8 + 16 * a["pred"]["p_rain"] for a in mapped]
    else:
        sizes = [13] * len(mapped)

    fig = go.Figure(go.Scattergeo(
        lat=[a["latitude"] for a in mapped],
        lon=[a["longitude"] for a in mapped],
        mode="markers+text" if show_names else "markers",
        text=[a["station"].split("/")[0].strip() for a in mapped],
        textposition="top center",
        textfont=dict(size=10, color=T.INK_2),
        marker=dict(size=sizes, color=[a["meta"]["color"] for a in mapped],
                    line=dict(color="white", width=1.5)),
        customdata=[[a["level"], f'{a["pred"]["p_rain"]:.0%}',
                     a["inundation"]["band"]] for a in mapped],
        hovertemplate="<b>%{text}</b><br>Warning: %{customdata[0]}<br>"
                      "Chance of rain: %{customdata[1]}<br>"
                      "Waterlogging: %{customdata[2]}<extra></extra>"))
    fig.update_geos(
        scope="asia", fitbounds="locations", resolution=50,
        showcountries=True, countrycolor=T.BASELINE,
        landcolor="#f4f6f9", showland=True, bgcolor=T.SURFACE)
    fig.update_layout(title="Colour = warning level")
    T.apply_layout(fig, height=480)
    st.plotly_chart(fig, width="stretch",
                    config={"displayModeBar": False})


# --------------------------------------------------------------- Detail tab
def _tab_detail(results: list[dict], features: pd.DataFrame,
                cfg: AppConfig) -> None:
    names = [a["station"] for a in results]
    pick = st.selectbox("Station", names, key="detail_station")
    a = next(r for r in results if r["station"] == pick)
    _station_card(a, big=True)

    pot, pred = a["potential"], a["pred"]
    tiles = [
        T.tile("Chance of rain", f"{pred['p_rain']:.0%}", "tomorrow"),
        T.tile("If it rains", f"up to ~{pot['p90_mm']:.0f} mm",
               a["intensity_category"].lower()),
        T.tile("Last 7 days", f"{a['rolling7_mm']:.0f} mm",
               f"wetter than {a['saturation_percentile']:.0%} of history"),
        T.tile("Waterlogging risk", a["inundation"]["band"],
               f"index {a['inundation']['score']:.2f}"),
    ]
    st.markdown('<div style="display:grid;grid-template-columns:repeat(4,'
                '1fr);gap:10px;margin:0.6rem 0">' + "".join(tiles)
                + "</div>", unsafe_allow_html=True)

    sdf = features[features["station"] == pick].sort_values("date")
    lookback = st.slider("History window (days)", 30, 365, 90, step=30,
                         key="detail_lookback")
    hist = sdf.tail(lookback)
    fig = go.Figure()
    fig.add_bar(x=hist["date"], y=hist["rainfall_today"],
                name="Daily rainfall (mm)", marker_color=T.BLUE,
                hovertemplate="%{x|%d %b %Y}: %{y:.1f} mm<extra></extra>")
    for mm, name, color in [(64.5, "heavy", T.ORANGE),
                            (115.6, "very heavy", T.CONF_COLORS["LOW"])]:
        if float(hist["rainfall_today"].max() or 0) >= mm * 0.5:
            fig.add_hline(y=mm, line_dash="dot", line_color=color,
                          annotation_text=f"{name} ≥ {mm} mm",
                          annotation_font_color=color)
    fig.update_layout(title=f"Recent rainfall — {pick}")
    T.apply_layout(fig, height=260)
    st.plotly_chart(fig, width="stretch",
                    config={"displayModeBar": False})


# ---------------------------------------------------------------- Table tab
def _tab_table(results: list[dict], tomorrow: pd.Timestamp) -> None:
    df = pd.DataFrame([{
        "Station": a["station"],
        "Warning": a["level"],
        "Advice": a["meta"]["advice"],
        "Chance of rain": round(a["pred"]["p_rain"], 2),
        "Outlook": _range_text(a),
        "If it rains (mm, p90)": round(a["potential"]["p90_mm"], 1),
        "Intensity band": a["intensity_category"],
        "Last 7 days (mm)": round(a["rolling7_mm"], 1),
        "Wetness percentile": round(a["saturation_percentile"], 2),
        "Waterlogging": a["inundation"]["band"],
        "Risk index": round(a["inundation"]["score"], 2),
        "Source": ("Live weather" if a["source"] == "live"
                   else "Seasonal average"),
    } for a in results])
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button(
        "Download warnings (CSV)",
        df.to_csv(index=False).encode(),
        file_name=f"warnings_{tomorrow.date()}.csv", mime="text/csv")


# --------------------------------------------------------------- Method tab
def _tab_method(cfg: AppConfig) -> None:
    st.markdown(f"""
**Data sources (integrated per problem statement 26071)**
- *Observational weather* — the historical station dataset the forecast
  model is trained on (temperature, wind, pressure, rainfall).
- *NWP model data* — live current conditions come from the Open-Meteo
  feed, which serves output of operational numerical weather prediction
  models for each station's coordinates; if unreachable, the station's
  seasonal average is used and labelled.
- *Satellite (INSAT-3D/3DR) and Doppler weather radar* — planned ingest
  channels; the feature pipeline accepts extra columns without changes to
  the model layer.

**Warning level** — IMD colour code (Green *No action* · Yellow
*Be updated* · Orange *Be prepared* · Red *Take action*) from a decision
matrix over the probability *p* of ≥ {cfg.rain_threshold_mm} mm next-day
rain and the station's seasonal *heavy-rain potential* q90 (90th
percentile of wet-day rainfall within ±15 days of this date across all
years; IMD 24-h bands: heavy ≥ 64.5 mm, very heavy ≥ 115.6 mm,
extremely heavy ≥ 204.5 mm):

| Level | Condition |
|---|---|
| RED | p ≥ 0.70 & q90 ≥ 115.6 mm, or p ≥ 0.85 & q90 ≥ 64.5 mm |
| ORANGE | p ≥ 0.60 & q90 ≥ 64.5 mm, or p ≥ 0.80 & q90 ≥ 15.6 mm |
| YELLOW | p ≥ {cfg.classification_threshold:.0%}, or ambiguous outlook & q90 ≥ 64.5 mm |
| GREEN | otherwise |

**Waterlogging / inundation risk** = 0.45 × wetness percentile (current
7-day rainfall accumulation ranked against this station's own history — a
soil-saturation / drainage-load proxy) + 0.35 × p + 0.20 × q90 scaled to
the extremely-heavy threshold. Bands: ≥ 0.75 SEVERE · ≥ 0.55 HIGH ·
≥ 0.35 MODERATE · else LOW. A screening indicator for response
prioritisation, not a terrain-based hydrological flood model.

The probability is calibrated on held-out data and every forecast carries
a {cfg.conformal_coverage:.0%}-coverage prediction set (see *Model
performance* for the evaluation). The matrix and weights live in
`src/warning.py`.""")


# ------------------------------------------------------------ station card
# Waterlogging bands map onto the same four-step severity ramp as the IMD
# levels, so the two scales stay visually consistent without competing: the
# IMD badge is solid, this one is a tint.
_INUN_LEVEL = {"LOW": "GREEN", "MODERATE": "YELLOW",
               "HIGH": "ORANGE", "SEVERE": "RED"}


def _station_card(a: dict, big: bool = False) -> None:
    """One station's warning.

    Severity is carried four ways at once -- ground tint, badge colour, glyph
    silhouette and rank ticks -- because the red/green ends of the IMD scale
    are indistinguishable to a red-green colour-blind reader, and this is the
    screen a response decision is made from.
    """
    pred, inun = a["pred"], a["inundation"]
    lvl = a["level"]
    sev = T.SEVERITY[lvl]
    pot = a["potential"]

    # An acting level lifts its own ground so the eye lands on it first;
    # quiet levels stay on plain white and recede.
    ground = (f"background:{sev['tint']};"
              f"box-shadow:inset 0 0 0 1px {sev['fill']}40"
              if lvl in ("RED", "ORANGE") else "")

    inun_sev = T.SEVERITY[_INUN_LEVEL[inun["band"]]]
    source_label = ("Live weather" if a["source"] == "live"
                    else "Seasonal estimate")
    name_size = "var(--t-xl)" if big else "var(--t-lg)"

    st.markdown(f"""
<div class="stn" style="{ground}">
  <div class="stn-head">
    {T.severity_badge(lvl)}
    <span class="stn-name" style="font-size:{name_size}">{a['station']}</span>
    <span style="flex:1"></span>
    <span class="chip">{source_label}</span>
    <span class="chip" style="background:{inun_sev['tint']};
      color:{inun_sev['ink']};border-color:{inun_sev['fill']}40">
      Waterlogging <b style="color:inherit">{inun['band']}</b></span>
  </div>
  <div class="stn-facts">
    <span class="stn-fact">Chance of rain <b>{pred['p_rain']:.0%}</b></span>
    <span class="stn-fact">Outlook <b>{_range_text(a)}</b></span>
    <span class="stn-fact">If it rains
      <b>up to ~{pot['p90_mm']:.0f} mm</b>
      ({a['intensity_category'].lower()})</span>
    <span class="stn-fact">Last 7 days
      <b>{a['rolling7_mm']:.0f} mm</b></span>
    <span class="stn-fact">Advice <b>{sev['advice']}</b></span>
  </div>
</div>""", unsafe_allow_html=True)
