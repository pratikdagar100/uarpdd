"""Shared visual language for the Heavy Rainfall Early Warning System.

The system follows PRODUCT.md's first principle: **colour is data**. The IMD
scale (Green / Yellow / Orange / Red) is a regulated vocabulary, so every
decorative use of colour elsewhere steals signal from it. Chrome is therefore
neutral -- a graphite sidebar and a cool grey/white content plane -- and the
only saturated things on screen are severity and the single accent reserved
for primary actions, current selection and focus.

Two consequences worth knowing before editing:

* **Every text colour here is contrast-verified**, not chosen by eye. Body
  text clears 4.5:1 against both `SURFACE` and `PAGE`; the ratios are recorded
  beside each token. The previous muted grey sat at 3.10:1, which is why
  readability is treated as correctness in this file and not as taste.
* **Severity is never encoded in colour alone.** `severity_badge` emits a
  distinct glyph silhouette and a 4-step rank alongside the colour, because
  the red/green ends of the IMD scale are indistinguishable under the most
  common forms of colour blindness.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# ============================================================ neutral ramp
# Contrast ratios are against SURFACE (#fff) / PAGE (#f1f4f8).
INK = "#0f141a"        # 18.50 / 16.77 -- primary text, headings, data
INK_2 = "#414d5c"      #  8.60 /  7.80 -- secondary text, labels
MUTED = "#5c6a7a"      #  5.53 /  5.01 -- tertiary text; the AA floor here
SURFACE = "#ffffff"
PAGE = "#f1f4f8"
SUNKEN = "#f7f9fb"     # inset panels inside a surface
GRID = "#dfe4ec"       # hairlines, borders
BASELINE = "#b9c2ce"   # chart axis lines

# Chrome. Deliberately near-neutral rather than navy: a blue sidebar would
# spend the one thing severity needs.
NAVY = "#151a21"       # sidebar ground (name kept: used across views)
NAVY_2 = "#1e252f"     # sidebar hover
SIDEBAR_INK = "#e8edf3"   # 14.85 on sidebar
SIDEBAR_DIM = "#9aa7b6"   #  7.14 on sidebar
SIDEBAR_FAINT = "#7b8896"  # 4.83 on sidebar

# ======================================================= accent (actions only)
ACCENT = "#1c5fb0"       # 6.34 / 5.75 on light; white label on it is 6.34
ACCENT_HOVER = "#174e93"
ACCENT_TINT = "#e8f0fb"
ACCENT_INK = "#12447f"
BLUE = ACCENT            # legacy alias, still used for chart series

# ============================================================= IMD severity
# `fill` carries `on_fill` as its label colour; `tint`/`ink` are the card
# pairing; `text` is safe as text on both light grounds. All AA-verified.
SEVERITY = {
    "RED": {
        "fill": "#b3261e", "on_fill": "#ffffff", "tint": "#fce9e7",
        "ink": "#8c1d17", "text": "#b3261e", "rank": 4,
        "title": "Warning", "advice": "Take action",
    },
    "ORANGE": {
        "fill": "#bd5411", "on_fill": "#ffffff", "tint": "#fceadd",
        "ink": "#8a3b06", "text": "#a5470b", "rank": 3,
        "title": "Alert", "advice": "Be prepared",
    },
    "YELLOW": {
        # Yellow can never carry white text; it takes dark ink on the fill,
        # which is also what printed IMD warning graphics do.
        "fill": "#e3b019", "on_fill": "#3a2b00", "tint": "#fbf1d4",
        "ink": "#5a4000", "text": "#5a4000", "rank": 2,
        "title": "Watch", "advice": "Be updated",
    },
    "GREEN": {
        "fill": "#17794c", "on_fill": "#ffffff", "tint": "#e6f4ec",
        "ink": "#0d5a34", "text": "#17794c", "rank": 1,
        "title": "No warning", "advice": "No action needed",
    },
}

# Distinct silhouettes, not just distinct colours: a circle, a ring, a
# triangle and an octagon stay separable in greyscale and under deuteranopia.
_GLYPH = {
    "GREEN": '<circle cx="8" cy="8" r="6.5"/>'
             '<path d="M5 8.2l2.1 2.1L11 6.4" fill="none" stroke="{on}" '
             'stroke-width="1.8" stroke-linecap="round" '
             'stroke-linejoin="round"/>',
    "YELLOW": '<circle cx="8" cy="8" r="6.5"/>'
              '<circle cx="8" cy="8" r="2.6" fill="{on}"/>',
    "ORANGE": '<path d="M8 1.4l6.6 12.2H1.4z"/>'
              '<path d="M8 6.1v3.4M8 11.4v.1" fill="none" stroke="{on}" '
              'stroke-width="1.7" stroke-linecap="round"/>',
    "RED": '<path d="M5.4 1.5h5.2L14.5 5.4v5.2L10.6 14.5H5.4L1.5 10.6V5.4z"/>'
           '<path d="M8 4.6v4.1M8 11.2v.1" fill="none" stroke="{on}" '
           'stroke-width="1.8" stroke-linecap="round"/>',
}


def severity_glyph(level: str, color: str | None = None,
                   on: str | None = None, size: int = 16) -> str:
    """Inline SVG mark for an IMD level -- the non-colour half of the code."""
    s = SEVERITY[level]
    fill = color or s["fill"]
    inner = on or s["on_fill"]
    body = _GLYPH[level].format(on=inner)
    return (f'<svg class="sev-glyph" width="{size}" height="{size}" '
            f'viewBox="0 0 16 16" fill="{fill}" aria-hidden="true">'
            f'{body}</svg>')


def severity_rank(level: str, on_tint: bool = False) -> str:
    """Four ticks, n filled -- a third, purely positional encoding."""
    s = SEVERITY[level]
    on = s["ink"] if on_tint else s["fill"]
    off = "rgba(15,20,26,0.16)"
    ticks = "".join(
        f'<i style="background:{on if i < s["rank"] else off}"></i>'
        for i in range(4))
    return (f'<span class="sev-rank" role="img" '
            f'aria-label="severity {s["rank"]} of 4">{ticks}</span>')


def severity_badge(level: str, *, solid: bool = True,
                   show_rank: bool = True) -> str:
    """Full redundant encoding: colour + glyph + text label + rank."""
    s = SEVERITY[level]
    rank = severity_rank(level, on_tint=not solid) if show_rank else ""
    if solid:
        style = f"background:{s['fill']};color:{s['on_fill']}"
        glyph = severity_glyph(level, color=s["on_fill"], on=s["fill"])
    else:
        style = (f"background:{s['tint']};color:{s['ink']};"
                 f"box-shadow:inset 0 0 0 1px {s['fill']}33")
        glyph = severity_glyph(level)
    return (f'<span class="sev-badge" style="{style}">{glyph}'
            f'<span class="sev-text">{level} &middot; '
            f'{s["title"].upper()}</span>{rank}</span>')


# ===================================================== data-viz series
# Ordering validated for categorical separation; kept stable so saved charts
# stay comparable across releases.
SERIES = ["#1c5fb0", "#bd5411", "#0e8f6f", "#8a5cd6", "#b3261e",
          "#6b7280", "#0f6f8f", "#9a6b00"]
ORANGE, AQUA = SERIES[1], SERIES[2]
SEQ_BLUE = ["#dce9f8", "#c6dbf3", "#aecdee", "#95bee9", "#7cafe3", "#649fdc",
            "#4b8ed3", "#3a7dc6", "#2f6db4", "#275ea0", "#1f4f8b", "#184075",
            "#12325e"]

# Qualitative outcome colours (model quality, not IMD severity).
GOOD = SEVERITY["GREEN"]["fill"]
WARNING = SEVERITY["YELLOW"]["fill"]
SERIOUS = SEVERITY["ORANGE"]["fill"]
CRITICAL = SEVERITY["RED"]["fill"]
GREEN = GOOD

PILL = {
    "navy": "#2b3543", "blue": ACCENT, "teal": "#0e7c66",
    "green": SEVERITY["GREEN"]["fill"], "amber": "#9a6b00",
    "orange": SEVERITY["ORANGE"]["fill"], "red": SEVERITY["RED"]["fill"],
    "gray": "#4c5866",
}

CONF_PILL = {"LOW": "orange", "MODERATE": "amber", "HIGH": "blue",
             "VERY HIGH": "navy"}
UNC_PILL = {"HIGH": "orange", "MODERATE": "amber", "LOW-MODERATE": "blue",
            "LOW": "teal"}
CONF_COLORS = {"LOW": "#5f7086", "MODERATE": "#3d6da8",
               "HIGH": ACCENT, "VERY HIGH": "#123f75"}  # all white-safe

SUN_SVG = """<svg width="30" height="30" viewBox="0 0 24 24" fill="none"
 stroke="#9a6b00" stroke-width="1.8" stroke-linecap="round" aria-hidden="true">
 <circle cx="12" cy="12" r="4" fill="#fbf1d4" stroke="#9a6b00"/>
 <path d="M12 2.5v2M12 19.5v2M5.2 5.2l1.4 1.4M17.4 17.4l1.4 1.4M2.5 12h2M19.5 12h2M5.2 18.8l1.4-1.4M17.4 6.6l1.4-1.4"/></svg>"""
RAIN_SVG = """<svg width="30" height="30" viewBox="0 0 24 24" fill="none"
 stroke="#1c5fb0" stroke-width="1.8" stroke-linecap="round" aria-hidden="true">
 <path d="M17.5 17a4.5 4.5 0 0 0 0-9 6 6 0 0 0-11.6 1.6A4 4 0 0 0 7 17.5z"
  fill="#e8f0fb" stroke="#1c5fb0"/>
 <path d="M8 19.5l-1 2M12 19.5l-1 2M16 19.5l-1 2"/></svg>"""

FONT_STACK = ('Inter, "SF Pro Text", -apple-system, BlinkMacSystemFont, '
              '"Segoe UI", Roboto, system-ui, sans-serif')
MONO_STACK = ('ui-monospace, SFMono-Regular, "SF Mono", "JetBrains Mono", '
              'Menlo, Consolas, "Liberation Mono", monospace')


def apply_layout(fig: go.Figure, height: int = 320, **kwargs) -> go.Figure:
    """House style for every Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT_STACK, color=INK_2, size=12.5),
        title_font=dict(size=13.5, color=INK),
        margin=dict(l=8, r=8, t=40 if fig.layout.title.text else 10, b=8),
        height=height,
        hoverlabel=dict(bgcolor=INK, font_size=12.5, font_color="#ffffff",
                        bordercolor=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1,
                    xanchor="right", font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)"),
        **kwargs,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=BASELINE,
                     linecolor=BASELINE, tickfont=dict(color=MUTED, size=11),
                     title_font=dict(color=MUTED, size=11.5))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=BASELINE,
                     linecolor=BASELINE, tickfont=dict(color=MUTED, size=11),
                     title_font=dict(color=MUTED, size=11.5))
    return fig


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
  --ink: {INK}; --ink-2: {INK_2}; --muted: {MUTED};
  --surface: {SURFACE}; --page: {PAGE}; --sunken: {SUNKEN};
  --grid: {GRID}; --baseline: {BASELINE};
  --accent: {ACCENT}; --accent-hover: {ACCENT_HOVER};
  --accent-tint: {ACCENT_TINT}; --accent-ink: {ACCENT_INK};
  --nav: {NAVY}; --nav-2: {NAVY_2};
  --sans: {FONT_STACK};
  --mono: {MONO_STACK};

  /* Fixed rem scale, ratio ~1.2 -- product UI, not fluid display type. */
  --t-micro: 0.6875rem; --t-xs: 0.75rem;  --t-sm: 0.8125rem;
  --t-md: 0.875rem;     --t-base: 0.9375rem;
  --t-lg: 1.0625rem;    --t-xl: 1.3125rem; --t-2xl: 1.625rem;
  --t-data: 2rem;       --t-hero: 2.375rem;

  --r-sm: 6px; --r-md: 9px; --r-lg: 12px; --r-pill: 999px;

  /* One soft elevation and one lifted; anything more is noise here. */
  --sh-1: 0 1px 2px rgba(15,20,26,0.05), 0 0 0 1px rgba(15,20,26,0.04);
  --sh-2: 0 4px 14px rgba(15,20,26,0.08), 0 0 0 1px rgba(15,20,26,0.05);

  --dur: 180ms;
  --ease: cubic-bezier(0.22, 1, 0.36, 1);  /* ease-out-quart */

  --z-sticky: 100; --z-dropdown: 200; --z-modal: 400; --z-toast: 600;
}}

/* ---------- chrome ---------- */
#MainMenu, footer, header[data-testid="stHeader"],
div[data-testid="stToolbar"], div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"] {{ display: none !important; }}

.stApp {{ background: var(--page); }}
.block-container {{
  padding-top: 1.75rem; padding-bottom: 4rem; max-width: 1240px;
}}
html, body, [class*="css"], .stApp, button, input, select, textarea {{
  font-family: {FONT_STACK};
  /* Tabular figures: columns of numbers must not jitter between reruns. */
  font-feature-settings: "tnum" 1, "cv05" 1;
}}
html, body, .stApp {{ color: var(--ink); }}
body {{ -webkit-font-smoothing: antialiased; }}

h1, h2, h3, h4 {{
  font-family: {FONT_STACK}; color: var(--ink);
  letter-spacing: -0.012em; text-wrap: balance;
}}
h2 {{ font-size: var(--t-xl); font-weight: 600; margin: 0 0 0.15rem; }}
h3 {{ font-size: var(--t-lg); font-weight: 600; }}
p, li {{ color: var(--ink-2); font-size: var(--t-md); line-height: 1.6; }}
code, .mono, kbd {{ font-family: {MONO_STACK}; }}

/* Visible focus on every interactive thing -- keyboard users first. */
:where(a, button, input, select, textarea, [role="button"],
       [tabindex]):focus-visible {{
  outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px;
}}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{
  background: var(--nav); border-right: none;
  min-width: 254px; max-width: 254px;
}}
section[data-testid="stSidebar"] * {{ color: {SIDEBAR_DIM}; }}
section[data-testid="stSidebar"] .stButton > button {{
  width: 100%; text-align: left; justify-content: flex-start;
  background: transparent; color: {SIDEBAR_DIM}; border: none;
  padding: 0.5rem 0.75rem; border-radius: var(--r-sm);
  font-size: var(--t-md); font-weight: 500; box-shadow: none;
  transition: background var(--dur) var(--ease), color var(--dur) var(--ease);
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
  background: var(--nav-2); color: {SIDEBAR_INK};
}}
section[data-testid="stSidebar"] .stButton button {{
  justify-content: flex-start !important;
}}
section[data-testid="stSidebar"] .stButton button div,
section[data-testid="stSidebar"] .stButton button span,
section[data-testid="stSidebar"] .stButton button p {{
  width: 100% !important; text-align: left !important;
}}
/* Selected nav item: a filled ground, not a coloured edge. */
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
  background: #273241; color: #ffffff; font-weight: 600;
}}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
  background: #2f3c4d;
}}
.sb-brand {{
  display: flex; align-items: center; gap: 10px;
  padding: 0.1rem 0.2rem 1rem;
}}
.sb-logo {{
  width: 32px; height: 32px; border-radius: 8px; background: var(--accent);
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; color: #fff; font-size: var(--t-sm);
  letter-spacing: 0.02em; flex-shrink: 0;
}}
.sb-name {{
  font-weight: 600; color: #fff !important; line-height: 1.3;
  font-size: var(--t-md); letter-spacing: -0.01em;
}}
.sb-stats {{
  border: 1px solid #2a3543; border-radius: var(--r-sm);
  padding: 0.6rem 0.7rem; font-family: {MONO_STACK};
  font-size: var(--t-micro); color: {SIDEBAR_DIM};
  line-height: 1.75; margin-bottom: 1rem;
}}
.sb-foot {{
  font-family: {MONO_STACK}; font-size: var(--t-micro);
  color: {SIDEBAR_FAINT}; line-height: 1.65; padding-top: 1.1rem;
}}

/* ---------- page header ---------- */
.pg-kicker {{
  font-size: var(--t-xs); color: var(--muted); font-weight: 600;
  margin-bottom: 0.2rem;
}}
.pg-title {{
  font-size: var(--t-2xl); font-weight: 650; color: var(--ink);
  letter-spacing: -0.02em; line-height: 1.18;
}}
.pg-sub {{
  color: var(--ink-2); font-size: var(--t-base); margin-top: 0.3rem;
  max-width: 68ch; line-height: 1.55;
}}
.pg-meta {{
  text-align: right; font-size: var(--t-sm); color: var(--muted);
  line-height: 1.6;
}}
.pg-meta b {{ color: var(--ink-2); font-weight: 600; }}
.pg-rule {{
  height: 1px; background: var(--grid); margin: 0.9rem 0 1.25rem;
}}

/* ---------- surfaces ---------- */
.uar-card {{
  background: var(--surface); border-radius: var(--r-lg);
  padding: 1.15rem 1.35rem; margin-bottom: 0.9rem; box-shadow: var(--sh-1);
}}
.tile {{
  background: var(--surface); border-radius: var(--r-md);
  padding: 0.8rem 1rem; box-shadow: var(--sh-1); height: 100%;
}}
.tile .t-label {{
  font-size: var(--t-xs); color: var(--muted); margin-bottom: 0.25rem;
  font-weight: 500;
}}
.tile .t-value {{
  font-size: var(--t-xl); font-weight: 650; color: var(--ink);
  line-height: 1.15; letter-spacing: -0.02em;
}}
.tile .t-foot {{
  font-size: var(--t-xs); color: var(--muted); margin-top: 0.25rem;
}}

/* ---------- severity encoding ---------- */
.sev-badge {{
  display: inline-flex; align-items: center; gap: 7px;
  padding: 0.3rem 0.7rem; border-radius: var(--r-pill);
  font-size: var(--t-xs); font-weight: 700; letter-spacing: 0.03em;
  white-space: nowrap; vertical-align: middle;
}}
.sev-glyph {{ flex-shrink: 0; display: block; }}
.sev-text {{ line-height: 1; }}
.sev-rank {{ display: inline-flex; gap: 2px; align-items: center;
  margin-left: 2px; }}
.sev-rank i {{ width: 3px; height: 9px; border-radius: 1px; display: block; }}

/* ---------- pills & chips ---------- */
.pill {{
  display: inline-block; padding: 0.26rem 0.7rem; border-radius: var(--r-pill);
  font-size: var(--t-xs); font-weight: 650; color: #fff;
  letter-spacing: 0.02em; margin-left: 0.35rem; white-space: nowrap;
}}
.chip {{
  display: inline-block; padding: 0.22rem 0.62rem; border-radius: var(--r-pill);
  font-size: var(--t-sm); color: var(--ink-2); border: 1px solid var(--grid);
  background: var(--surface); margin: 0.15rem 0.3rem 0.15rem 0;
}}
.chip b {{ color: var(--ink); font-family: {MONO_STACK};
  font-size: var(--t-xs); }}

/* ---------- prediction-set boxes ---------- */
.pset-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 12px; margin: 0.6rem 0;
}}
.pset-box {{
  border-radius: var(--r-md); padding: 0.75rem 1rem; background: var(--sunken);
  box-shadow: inset 0 0 0 1px var(--grid);
}}
.pset-box .p-name {{ font-weight: 700; font-size: var(--t-sm);
  color: var(--muted); letter-spacing: 0.02em; }}
.pset-box .p-sub {{ font-size: var(--t-xs); color: var(--muted);
  margin-top: 0.2rem; }}
.pset-in-green {{ background: {SEVERITY["GREEN"]["tint"]};
  box-shadow: inset 0 0 0 1px #9fd4b7; }}
.pset-in-green .p-name, .pset-in-green .p-sub {{
  color: {SEVERITY["GREEN"]["ink"]}; }}
.pset-in-blue {{ background: var(--accent-tint);
  box-shadow: inset 0 0 0 1px #aac6ea; }}
.pset-in-blue .p-name, .pset-in-blue .p-sub {{ color: var(--accent-ink); }}
.pset-in-amber {{ background: {SEVERITY["ORANGE"]["tint"]};
  box-shadow: inset 0 0 0 1px #eabc98; }}
.pset-in-amber .p-name, .pset-in-amber .p-sub {{
  color: {SEVERITY["ORANGE"]["ink"]}; }}

/* ---------- notes ---------- */
/* Full-bordered tinted note with a leading mark -- no side stripes. */
.strip {{
  display: flex; gap: 0.6rem; align-items: flex-start;
  font-size: var(--t-md); color: var(--accent-ink);
  background: var(--accent-tint); border: 1px solid #c3d8f2;
  padding: 0.7rem 0.95rem; border-radius: var(--r-md);
  margin: 0.75rem 0; line-height: 1.55;
}}
.strip::before {{
  content: ""; flex-shrink: 0; width: 8px; height: 8px; margin-top: 0.42rem;
  border-radius: 50%; background: currentColor; opacity: 0.65;
}}
.strip b {{ color: inherit; font-weight: 650; }}
.strip-note {{ background: var(--page); border-color: var(--grid);
  color: var(--ink-2); }}
.strip-warn {{ background: {SEVERITY["ORANGE"]["tint"]};
  border-color: #f0c39d; color: {SEVERITY["ORANGE"]["ink"]}; }}
.strip-good {{ background: {SEVERITY["GREEN"]["tint"]};
  border-color: #a6d8bd; color: {SEVERITY["GREEN"]["ink"]}; }}
.strip-bad {{ background: {SEVERITY["RED"]["tint"]};
  border-color: #f0b8b3; color: {SEVERITY["RED"]["ink"]}; }}
.strip-amber {{ background: {SEVERITY["YELLOW"]["tint"]};
  border-color: #e8d18f; color: {SEVERITY["YELLOW"]["ink"]}; }}

/* ---------- probability track ---------- */
.ptrack {{
  position: relative; height: 10px; border-radius: var(--r-pill);
  background: #e2e8f0; margin: 1.15rem 0 0.3rem;
}}
.ptrack .fill {{
  position: absolute; left: 0; top: 0; bottom: 0;
  border-radius: var(--r-pill); background: var(--accent);
  transition: width var(--dur) var(--ease);
}}
.ptrack .thr {{
  position: absolute; top: -6px; bottom: -6px; width: 2px;
  background: var(--ink); border-radius: 1px;
}}
.ptrack .dot {{
  position: absolute; top: 50%; transform: translate(-50%,-50%);
  width: 18px; height: 18px; border-radius: 50%; background: #fff;
  box-shadow: 0 0 0 3px var(--accent), 0 1px 4px rgba(15,20,26,0.3);
  transition: left var(--dur) var(--ease);
}}
.ptrack-labels {{
  display: flex; justify-content: space-between; font-family: {MONO_STACK};
  font-size: var(--t-micro); color: var(--muted); margin-bottom: 0.4rem;
}}
.ptrack-labels b {{ color: var(--ink-2); }}

/* ---------- confidence scale ---------- */
.conf-scale {{ display: flex; gap: 5px; margin: 0.4rem 0 0.45rem; }}
.conf-step {{
  flex: 1; text-align: center; padding: 0.42rem 0.2rem;
  border-radius: var(--r-sm); font-size: var(--t-micro); font-weight: 700;
  letter-spacing: 0.04em; color: var(--muted); background: #e9edf3;
  transition: background var(--dur) var(--ease);
}}
.conf-step.active {{ color: #fff; }}

/* ---------- influence rows ---------- */
.infl-row {{ display: flex; align-items: center; gap: 12px; margin: 8px 0; }}
.infl-name {{ width: 250px; font-size: var(--t-md); color: var(--ink-2);
  flex-shrink: 0; }}
.infl-name code {{ color: var(--ink); background: var(--sunken);
  padding: 1px 5px; border-radius: 4px; font-size: var(--t-xs); }}
.infl-track {{ flex: 1; height: 8px; background: #e9edf3;
  border-radius: var(--r-pill); overflow: hidden; }}
.infl-fill {{ height: 8px; border-radius: var(--r-pill);
  transition: width var(--dur) var(--ease); }}
.infl-band {{ width: 82px; font-size: var(--t-xs); font-weight: 650;
  color: var(--muted); text-align: right; flex-shrink: 0; }}

/* ---------- split summary ---------- */
.split-strip {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1px; background: var(--grid); border-radius: var(--r-md);
  overflow: hidden; margin: 0.7rem 0 1.1rem; box-shadow: var(--sh-1);
}}
.split-cell {{ padding: 0.8rem 1.05rem; background: var(--surface); }}
.split-cell .s-name {{ font-size: var(--t-xs); color: var(--muted);
  font-weight: 500; }}
.split-cell .s-range {{ font-weight: 600; font-size: var(--t-md);
  color: var(--ink); margin: 0.15rem 0 0.1rem; }}
.split-cell .s-n {{ font-size: var(--t-xs); color: var(--muted);
  font-family: {MONO_STACK}; }}

/* ---------- confusion matrix ---------- */
.cm-grid {{ display: grid; grid-template-columns: 116px 1fr 1fr; gap: 8px;
  align-items: stretch; margin-top: 0.5rem; }}
.cm-h {{ font-size: var(--t-xs); color: var(--muted); align-self: center;
  text-align: center; }}
.cm-cell {{ border-radius: var(--r-md); text-align: center;
  padding: 1.1rem 0.4rem; }}
.cm-cell .cm-v {{ font-weight: 650; font-size: var(--t-xl);
  letter-spacing: -0.02em; }}
.cm-cell .cm-t {{ font-size: var(--t-micro); opacity: 0.8;
  letter-spacing: 0.04em; }}

/* ---------- steps ---------- */
.step-card {{ border-radius: var(--r-md); padding: 0.9rem 1.05rem;
  height: 100%; background: var(--sunken);
  box-shadow: inset 0 0 0 1px var(--grid); }}
.step-k {{ color: var(--accent); font-weight: 700; font-size: var(--t-xs);
  letter-spacing: 0.04em; }}
.step-t {{ font-weight: 650; margin: 0.2rem 0; color: var(--ink);
  font-size: var(--t-base); }}
.step-d {{ font-size: var(--t-sm); color: var(--ink-2); line-height: 1.5; }}

/* ---------- mono block ---------- */
.mono {{
  font-family: {MONO_STACK}; font-size: var(--t-sm); color: var(--ink-2);
  background: var(--sunken); border: 1px solid var(--grid);
  border-radius: var(--r-md); padding: 0.85rem 1.05rem; white-space: pre;
  overflow-x: auto;
}}

/* ---------- station row (early warning board) ---------- */
.stn {{
  background: var(--surface); border-radius: var(--r-md);
  padding: 0.9rem 1.1rem; margin-bottom: 0.6rem; box-shadow: var(--sh-1);
  transition: box-shadow var(--dur) var(--ease),
              transform var(--dur) var(--ease);
}}
.stn:hover {{ box-shadow: var(--sh-2); transform: translateY(-1px); }}
.stn-head {{ display: flex; align-items: center; gap: 0.6rem;
  flex-wrap: wrap; }}
.stn-name {{ font-size: var(--t-lg); font-weight: 650; color: var(--ink);
  letter-spacing: -0.01em; }}
.stn-facts {{ display: flex; flex-wrap: wrap; gap: 0.3rem 1.5rem;
  margin-top: 0.55rem; }}
.stn-fact {{ font-size: var(--t-sm); color: var(--muted); }}
.stn-fact b {{ color: var(--ink); font-weight: 600; }}

/* ---------- streamlit control surfaces ---------- */
.stButton > button {{
  border-radius: var(--r-sm); font-weight: 550; font-size: var(--t-md);
  transition: background var(--dur) var(--ease),
              box-shadow var(--dur) var(--ease);
}}
.stButton > button[kind="primary"] {{
  background: var(--accent); border-color: var(--accent); color: #fff;
}}
.stButton > button[kind="primary"] * {{ color: #fff; }}
.stButton > button[kind="secondary"] {{ color: var(--ink); }}
.stButton > button[kind="primary"]:hover {{
  background: var(--accent-hover); border-color: var(--accent-hover);
}}
.stButton > button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
div[data-testid="stMetricValue"] {{ font-size: var(--t-xl);
  font-weight: 650; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid var(--grid); }}
.stTabs [data-baseweb="tab"] {{ font-size: var(--t-md); font-weight: 550;
  color: var(--muted); }}
.stTabs [aria-selected="true"] {{ color: var(--ink) !important; }}

/* ---------- motion ---------- */
@keyframes uar-rise {{
  from {{ opacity: 0; transform: translateY(6px); }}
  to   {{ opacity: 1; transform: none; }}
}}
.uar-rise {{ animation: uar-rise 240ms var(--ease) both; }}

@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }}
  .stn:hover {{ transform: none; }}
}}

/* ---------- responsive ---------- */
@media (max-width: 900px) {{
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
  .infl-name {{ width: 150px; }}
  .cm-grid {{ grid-template-columns: 84px 1fr 1fr; }}
  .pg-meta {{ text-align: left; margin-top: 0.5rem; }}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ================================================================= helpers
def pill(text: str, kind: str) -> str:
    return f'<span class="pill" style="background:{PILL[kind]}">{text}</span>'


def chip(label: str, value: str) -> str:
    return f'<span class="chip">{label} <b>{value}</b></span>'


def tile(label: str, value: str, foot: str = "") -> str:
    foot_html = f'<div class="t-foot">{foot}</div>' if foot else ""
    return (f'<div class="tile"><div class="t-label">{label}</div>'
            f'<div class="t-value">{value}</div>{foot_html}</div>')


def page_header(title: str, sub: str = "", meta: str = "") -> None:
    """Page title, optional one-line orientation, and a closing rule."""
    sub_html = f'<div class="pg-sub">{sub}</div>' if sub else ""
    meta_html = f'<div class="pg-meta">{meta}</div>' if meta else ""
    st.markdown(
        f'<div class="pg-title">{title}</div>{sub_html}{meta_html}'
        f'<div class="pg-rule"></div>', unsafe_allow_html=True)


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
<div class="ptrack" role="img"
     aria-label="rain probability {pct:.0f} percent against a
                 {threshold:.0%} decision threshold">
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
