"""Uncertainty-Aware Rainfall Prediction & Decision Dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src.config import AppConfig, DATA_DIR
from src.pipeline import run_pipeline
from views import theme as T
from views import (dashboard, explorer, performance, training, study_run,
                   study_results, about)

st.set_page_config(page_title="Uncertainty-Aware Rainfall Prediction",
                   layout="wide", initial_sidebar_state="expanded")
T.inject_css()

cfg = AppConfig.load()

PAGES = [
    ("Prediction dashboard", dashboard.render),
    ("Dataset explorer", explorer.render),
    ("Model performance", performance.render),
    ("User study", study_run.render),
    ("Study results", study_results.render),
    ("Training & config", training.render),
    ("About", about.render),
]


def _bootstrap() -> dict:
    """First-run pipeline with a progress display; cached in session state
    (heavy artifacts are additionally cached on disk in models/)."""
    if "pipeline_state" in st.session_state:
        return st.session_state["pipeline_state"]

    force = st.session_state.pop("force_retrain", False)
    holder = st.empty()
    with holder.container():
        st.markdown('<div class="pg-kicker">First run</div>'
                    '<div class="pg-title">Building the pipeline</div>'
                    '<div class="pg-sub">Runs once for a new dataset or '
                    'config; a few minutes for ~1M rows. All artifacts are '
                    'cached in <code>models/</code> and later starts load in '
                    'seconds.</div>', unsafe_allow_html=True)
        bar = st.progress(0.0, text="Starting pipeline…")

    def progress(pct, msg):
        bar.progress(min(1.0, pct), text=msg)

    state = run_pipeline(cfg, progress=progress, force_retrain=force)
    holder.empty()
    st.session_state["pipeline_state"] = state
    return state


def _sidebar(state: dict | None) -> str:
    with st.sidebar:
        st.markdown("""
<div class="sb-brand">
  <div class="sb-logo">RU</div>
  <div class="sb-name">Rainfall<br>Uncertainty Lab</div>
</div>""", unsafe_allow_html=True)

        if state and "features" in state:
            insp = state["inspection"]
            dr = insp.get("date_range", ("—", "—"))
            n_rows = state["feat_report"]["final_rows"]
            model = state["bundle"]["model_name"]
            st.markdown(
                f'<div class="sb-stats">{insp.get("n_stations", 1)} stations '
                f'&middot; India<br>{dr[0]} &rarr; {dr[1]}<br>'
                f'{n_rows:,} modelling days<br>model &middot; {model}</div>',
                unsafe_allow_html=True)

        current = st.session_state.get("nav", PAGES[0][0])
        for name, _ in PAGES:
            if st.button(name, key=f"nav_{name}", use_container_width=True,
                         type="primary" if name == current else "secondary"):
                if st.session_state.get("nav") != name:
                    st.session_state["nav"] = name
                    st.rerun()

        if state and "bundle" in state:
            fp = state["bundle"]["fingerprint"][:10]
            st.markdown(f'<div class="sb-foot">Pipeline cached<br>'
                        f'models/rainfall_bundle.joblib<br>'
                        f'fingerprint {fp}…</div>', unsafe_allow_html=True)
    return st.session_state.get("nav", PAGES[0][0])


# Deep-linking: ?page=<name> selects a page (e.g. ?page=Model performance)
_qp = st.query_params.get("page")
if _qp and _qp in dict(PAGES) and st.session_state.get("nav") != _qp:
    st.session_state["nav"] = _qp

state = _bootstrap()

if state.get("error") == "no_dataset":
    _sidebar(None)
    T.page_header("No dataset found", "Add a historical weather dataset",
                  "The pipeline inspects the columns automatically — it "
                  "needs at least a date column and a rainfall column; a "
                  "station column is strongly recommended.", None, cfg)
    st.markdown(f'<div class="strip">Place a file at '
                f'<code>{DATA_DIR / "dataset.csv"}</code> (CSV, Parquet or '
                f'XLSX) or upload it below.</div>', unsafe_allow_html=True)
    up = st.file_uploader("Upload dataset", type=["csv", "xlsx", "parquet"])
    if up is not None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        suffix = up.name.rsplit(".", 1)[-1].lower()
        (DATA_DIR / f"dataset.{suffix}").write_bytes(up.getbuffer())
        st.session_state.pop("pipeline_state", None)
        st.rerun()
    st.stop()

page = _sidebar(state)
dict(PAGES)[page](state, cfg)
