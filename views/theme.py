"""Shared visual language for the Rainfall Uncertainty Lab UI.

Implements the approved UI design proposal: dark navy sidebar shell, light
content plane, stat tiles, tinted pills, prediction-set boxes and info
strips. Chart palette follows the validated categorical ordering
(blue -> orange -> aqua).
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------- palette
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA = SERIES[0], SERIES[1], SERIES[2]
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
GREEN = "#18a05a"

NAVY = "#0e1726"            # sidebar / dark chrome
NAVY_2 = "#16233a"
SURFACE = "#ffffff"
PAGE = "#eef2f7"
INK = "#16202e"
INK_2 = "#4c5a6b"
MUTED = "#8494a7"
GRID = "#e3e8ef"
BASELINE = "#c6cfda"

# Solid pill colors per design (right side of hero cards)
PILL = {
    "navy": "#1c3f77", "blue": "#2a78d6", "teal": "#0e9f6e",
    "green": "#18a05a", "amber": "#e8930c", "orange": "#e2571b",
    "red": "#cf3f2f", "gray": "#5b6b7a",
}

CONF_PILL = {"LOW": "orange", "MODERATE": "amber", "HIGH": "blue",
             "VERY HIGH": "navy"}
UNC_PILL = {"HIGH": "orange", "MODERATE": "amber", "LOW-MODERATE": "blue",
            "LOW": "teal"}
CONF_COLORS = {"LOW": SERIOUS, "MODERATE": WARNING, "HIGH": "#2a78d6",
               "VERY HIGH": "#1c3f77"}

SUN_SVG = """<svg width="34" height="34" viewBox="0 0 24 24" fill="none"
 stroke="#e8930c" stroke-width="2" stroke-linecap="round">
 <circle cx="12" cy="12" r="4" fill="#f6b93b" stroke="#e8930c"/>
 <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>"""
RAIN_SVG = """<svg width="34" height="34" viewBox="0 0 24 24" fill="none"
 stroke="#2a78d6" stroke-width="2" stroke-linecap="round">
 <path d="M17.5 17a4.5 4.5 0 0 0 0-9 6 6 0 0 0-11.6 1.6A4 4 0 0 0 7 17.5z"
  fill="#dbeafe" stroke="#2a78d6"/>
 <path d="M8 19.5l-1 2M12 19.5l-1 2M16 19.5l-1 2"/></svg>"""


def apply_layout(fig: go.Figure, height: int = 320, **kwargs) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family='"Segoe UI", system-ui, sans-serif', color=INK_2,
                  size=12.5),
        title_font=dict(size=13.5, color=INK),
        margin=dict(l=10, r=10, t=40 if fig.layout.title.text else 12, b=10),
        height=height,
        hoverlabel=dict(bgcolor="white", font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1,
                    xanchor="right", font=dict(size=11)),
        **kwargs,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=BASELINE,
                     linecolor=BASELINE, tickfont=dict(color=MUTED, size=11))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=BASELINE,
                     linecolor=BASELINE, tickfont=dict(color=MUTED, size=11))
    return fig


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

/* ---------- chrome ---------- */
#MainMenu, footer, header[data-testid="stHeader"] {{ display: none !important; }}
div[data-testid="stToolbar"], div[data-testid="stDecoration"] {{ display: none !important; }}
.stApp {{ background: {PAGE}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1150px; }}
html, body, [class*="css"] {{
  font-family: "Segoe UI", system-ui, -apple-system, sans-serif; color: {INK};
}}
h1, h2, h3, h4 {{ font-family: Poppins, "Segoe UI", sans-serif; color: {INK}; }}
h2 {{ font-size: 1.45rem; font-weight: 600; letter-spacing: -0.01em; }}
h3 {{ font-size: 1.08rem; font-weight: 600; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{
  background: {NAVY}; border-right: none; min-width: 250px; max-width: 250px;
}}
section[data-testid="stSidebar"] * {{ color: #cdd7e4; }}
section[data-testid="stSidebar"] .stButton > button {{
  width: 100%; text-align: left; justify-content: flex-start;
  background: transparent; color: #b9c6d6; border: none;
  padding: 0.42rem 0.8rem; border-radius: 8px; font-size: 0.92rem;
  font-weight: 500; box-shadow: none;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
  background: {NAVY_2}; color: #ffffff;
}}
section[data-testid="stSidebar"] .stButton button {{
  justify-content: flex-start !important;
}}
section[data-testid="stSidebar"] .stButton button div,
section[data-testid="stSidebar"] .stButton button span,
section[data-testid="stSidebar"] .stButton button p {{
  width: 100% !important; text-align: left !important;
}}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
  background: #1b2c47; color: #ffffff; font-weight: 600;
  border-left: 3px solid {BLUE}; border-radius: 6px;
}}
.sb-brand {{
  display: flex; align-items: center; gap: 10px; padding: 0.2rem 0.3rem 0.9rem;
}}
.sb-logo {{
  width: 34px; height: 34px; border-radius: 9px; background: {BLUE};
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; color: #fff; font-family: Poppins, sans-serif;
}}
.sb-name {{ font-family: Poppins, sans-serif; font-weight: 600;
  color: #fff !important; line-height: 1.25; font-size: 0.95rem; }}
.sb-stats {{
  border: 1px solid #22314d; border-radius: 8px; padding: 0.55rem 0.7rem;
  font-family: Consolas, monospace; font-size: 0.72rem; color: #8fa1b8;
  line-height: 1.7; margin-bottom: 0.9rem;
}}
.sb-foot {{
  font-family: Consolas, monospace; font-size: 0.68rem; color: #64748b;
  line-height: 1.6; padding-top: 1rem;
}}

/* ---------- page header ---------- */
.pg-kicker {{
  font-size: 0.7rem; letter-spacing: 0.16em; text-transform: uppercase;
  color: {MUTED}; font-weight: 700; margin-bottom: 0.15rem;
}}
.pg-title {{
  font-family: Poppins, "Segoe UI", sans-serif; font-size: 1.7rem;
  font-weight: 600; color: {INK}; letter-spacing: -0.01em; line-height: 1.2;
}}
.pg-sub {{ color: {INK_2}; font-size: 0.92rem; margin-top: 0.2rem; }}
.pg-meta {{
  text-align: right; font-size: 0.78rem; color: {MUTED}; line-height: 1.55;
}}
.pg-meta b {{ color: {INK_2}; }}

/* ---------- cards & tiles ---------- */
.uar-card {{
  background: {SURFACE}; border: 1px solid {GRID}; border-radius: 12px;
  padding: 1.05rem 1.25rem; margin-bottom: 0.9rem;
  box-shadow: 0 1px 2px rgba(22,32,46,0.04);
}}
.tile {{
  background: {SURFACE}; border: 1px solid {GRID}; border-radius: 10px;
  padding: 0.7rem 0.95rem;
}}
.tile .t-label {{ font-size: 0.74rem; color: {MUTED}; margin-bottom: 0.15rem; }}
.tile .t-value {{ font-family: Poppins, sans-serif; font-size: 1.35rem;
  font-weight: 600; color: {INK}; line-height: 1.2; }}
.tile .t-foot {{ font-size: 0.72rem; color: {MUTED}; margin-top: 0.1rem; }}

/* ---------- pills ---------- */
.pill {{
  display: inline-block; padding: 0.22rem 0.75rem; border-radius: 999px;
  font-size: 0.75rem; font-weight: 700; color: #fff; letter-spacing: 0.02em;
  margin-left: 0.35rem; white-space: nowrap;
}}
.chip {{
  display: inline-block; padding: 0.18rem 0.6rem; border-radius: 999px;
  font-size: 0.78rem; color: {INK_2}; border: 1px solid {GRID};
  background: {SURFACE}; margin: 0.15rem 0.3rem 0.15rem 0;
}}
.chip b {{ color: {INK}; font-family: Consolas, monospace; }}

/* ---------- prediction-set boxes ---------- */
.pset-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
  margin: 0.5rem 0; }}
.pset-box {{
  border: 1px solid {GRID}; border-radius: 10px; padding: 0.65rem 0.9rem;
  background: #f6f8fa;
}}
.pset-box .p-name {{ font-weight: 700; font-size: 0.85rem; color: {INK_2}; }}
.pset-box .p-sub {{ font-size: 0.75rem; color: {MUTED}; margin-top: 0.1rem; }}
.pset-in-green {{ background: #e9f9f0; border-color: #9fdcbb; }}
.pset-in-green .p-name {{ color: #0b6b3a; }}
.pset-in-blue {{ background: #e9f1fc; border-color: #a9c9ef; }}
.pset-in-blue .p-name {{ color: #1c4f8f; }}
.pset-in-amber {{ background: #fdf0e4; border-color: #f2c39a; }}
.pset-in-amber .p-name {{ color: #8f4a10; }}

/* ---------- strips ---------- */
.strip {{
  font-size: 0.84rem; color: {INK_2}; background: #eaf2fd;
  border-left: 3px solid {BLUE}; padding: 0.65rem 0.95rem;
  border-radius: 0 8px 8px 0; margin: 0.7rem 0; line-height: 1.55;
}}
.strip-warn {{ background: #fdf0e4; border-left-color: {SERIOUS};
  color: #7a4a20; }}
.strip-good {{ background: #e9f9f0; border-left-color: {GREEN};
  color: #14532d; }}
.strip-bad  {{ background: #fdecea; border-left-color: {CRITICAL};
  color: #7f1d1d; }}
.strip-amber {{ background: #fdf6e3; border-left-color: {WARNING};
  color: #6b4c00; }}

/* ---------- probability track ---------- */
.ptrack {{ position: relative; height: 10px; border-radius: 999px;
  background: linear-gradient(to right, #dfe7f0, #dfe7f0); margin: 1.1rem 0 0.2rem; }}
.ptrack .fill {{ position: absolute; left: 0; top: 0; bottom: 0;
  border-radius: 999px; background: {BLUE}; }}
.ptrack .thr {{ position: absolute; top: -5px; bottom: -5px; width: 2px;
  background: {INK}; }}
.ptrack .dot {{ position: absolute; top: 50%; transform: translate(-50%,-50%);
  width: 18px; height: 18px; border-radius: 50%; background: #fff;
  border: 3px solid {BLUE}; box-shadow: 0 1px 3px rgba(0,0,0,0.25); }}
.ptrack-labels {{ display: flex; justify-content: space-between;
  font-family: Consolas, monospace; font-size: 0.7rem; color: {MUTED};
  margin-bottom: 0.4rem; }}

/* ---------- confidence scale ---------- */
.conf-scale {{ display: flex; gap: 6px; margin: 0.35rem 0 0.4rem 0; }}
.conf-step {{
  flex: 1; text-align: center; padding: 0.4rem 0.2rem; border-radius: 7px;
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em;
  color: {MUTED}; background: #eef1f5; border: 1px solid transparent;
}}
.conf-step.active {{ color: #fff; }}

/* ---------- influence rows ---------- */
.infl-row {{ display: flex; align-items: center; gap: 12px; margin: 7px 0; }}
.infl-name {{ width: 260px; font-size: 0.85rem; color: {INK_2};
  flex-shrink: 0; }}
.infl-name code {{ font-family: Consolas, monospace; color: {INK};
  background: #f1f4f8; padding: 0 4px; border-radius: 4px; font-size: 0.78rem; }}
.infl-track {{ flex: 1; height: 8px; background: #eef1f5; border-radius: 999px; }}
.infl-fill {{ height: 8px; border-radius: 999px; }}
.infl-band {{ width: 78px; font-size: 0.72rem; font-weight: 700;
  color: {MUTED}; text-align: right; flex-shrink: 0; }}

/* ---------- split summary ---------- */
.split-strip {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0;
  background: {SURFACE}; border: 1px solid {GRID}; border-radius: 10px;
  overflow: hidden; margin: 0.6rem 0 1rem; }}
.split-cell {{ padding: 0.7rem 1rem; border-left: 3px solid {BLUE}; }}
.split-cell + .split-cell {{ border-left: 1px solid {GRID}; }}
.split-cell .s-name {{ font-size: 0.74rem; color: {MUTED}; }}
.split-cell .s-range {{ font-weight: 650; font-size: 0.9rem; color: {INK}; }}
.split-cell .s-n {{ font-size: 0.74rem; color: {MUTED}; }}

/* ---------- confusion matrix tiles ---------- */
.cm-grid {{ display: grid; grid-template-columns: 110px 1fr 1fr; gap: 8px;
  align-items: stretch; margin-top: 0.4rem; }}
.cm-h {{ font-size: 0.76rem; color: {MUTED}; align-self: center;
  text-align: center; }}
.cm-cell {{ border-radius: 10px; text-align: center; padding: 1.05rem 0.4rem; }}
.cm-cell .cm-v {{ font-family: Poppins, sans-serif; font-weight: 600;
  font-size: 1.25rem; }}
.cm-cell .cm-t {{ font-size: 0.7rem; opacity: 0.75; }}

/* ---------- steps (study intro) ---------- */
.step-card {{ border: 1px solid {GRID}; border-radius: 10px;
  padding: 0.8rem 1rem; height: 100%; background: {SURFACE}; }}
.step-k {{ color: {BLUE}; font-weight: 700; font-size: 0.78rem; }}
.step-t {{ font-weight: 650; margin: 0.15rem 0; }}
.step-d {{ font-size: 0.8rem; color: {INK_2}; line-height: 1.45; }}

/* mono table */
.mono {{ font-family: Consolas, monospace; font-size: 0.8rem; color: {INK_2};
  background: #f6f8fa; border: 1px solid {GRID}; border-radius: 8px;
  padding: 0.8rem 1rem; white-space: pre; overflow-x: auto; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def pill(text: str, kind: str) -> str:
    return f'<span class="pill" style="background:{PILL[kind]}">{text}</span>'


def chip(label: str, value: str) -> str:
    return f'<span class="chip">{label} <b>{value}</b></span>'


def tile(label: str, value: str, foot: str = "") -> str:
    foot_html = f'<div class="t-foot">{foot}</div>' if foot else ""
    return (f'<div class="tile"><div class="t-label">{label}</div>'
            f'<div class="t-value">{value}</div>{foot_html}</div>')


def page_header(title: str) -> None:
    st.markdown(f'<div class="pg-title">{title}</div>'
                "<div style='height:0.7rem'></div>", unsafe_allow_html=True)


def confidence_scale_html(active: str) -> str:
    steps = ["LOW", "MODERATE", "HIGH", "VERY HIGH"]
    cells = []
    for s in steps:
        if s == active:
            cells.append(f'<div class="conf-step active" '
                         f'style="background:{CONF_COLORS[s]}">{s}</div>')
        else:
            cells.append(f'<div class="conf-step">{s}</div>')
    return '<div class="conf-scale">' + "".join(cells) + "</div>"


def probability_track(p_rain: float, threshold: float) -> str:
    pct = p_rain * 100
    return f"""
<div class="ptrack">
  <div class="fill" style="width:{pct:.1f}%"></div>
  <div class="thr" style="left:{threshold * 100:.0f}%"></div>
  <div class="dot" style="left:{pct:.1f}%"></div>
</div>
<div class="ptrack-labels">
  <span>0% &middot; no significant rain</span>
  <span><b>threshold {threshold:.0%}</b></span>
  <span>rain &middot; 100%</span>
</div>"""


def pset_boxes(pred: dict, thr_mm: float) -> str:
    from src.uncertainty import RAIN, NO_RAIN
    in_rain = RAIN in pred["prediction_set"]
    in_no = NO_RAIN in pred["prediction_set"]
    both = in_rain and in_no
    rain_cls = ("pset-in-amber" if both else
                ("pset-in-blue" if in_rain else ""))
    no_cls = ("pset-in-amber" if both else
              ("pset-in-green" if in_no else ""))
    rain_sub = "in set" if in_rain else "ruled out"
    no_sub = "in set" if in_no else "ruled out"
    return f"""
<div class="pset-grid">
  <div class="pset-box {rain_cls}">
    <div class="p-name">RAIN</div>
    <div class="p-sub">&ge; {thr_mm} mm tomorrow &middot; {rain_sub}</div>
  </div>
  <div class="pset-box {no_cls}">
    <div class="p-name">NO SIGNIFICANT RAIN</div>
    <div class="p-sub">&lt; {thr_mm} mm tomorrow &middot; {no_sub}</div>
  </div>
</div>"""
