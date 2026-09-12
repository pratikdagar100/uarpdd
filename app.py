"""Uncertainty-Aware Rainfall Prediction & Decision Dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

# Must precede every numpy/scikit-learn/streamlit import: OpenMP reads
# its thread count once, when the runtime loads.
from src.runtime import configure_runtime

configure_runtime()

import streamlit as st  # noqa: E402

from src.config import AppConfig, DATA_DIR  # noqa: E402
from src.pipeline import dataset_cache_key, run_pipeline  # noqa: E402
from views import theme as T  # noqa: E402
from views import (warning, dashboard, explorer, performance, training,  # noqa: E402
                   study_run, study_results, about)

st.set_page_config(page_title="Heavy Rainfall Early Warning System",
                   layout="wide", initial_sidebar_state="expanded")
T.inject_css()

cfg = AppConfig.load()

PAGES = [
    ("Early warning board", warning.render),
    ("Prediction dashboard", dashboard.render),
    ("Dataset explorer", explorer.render),
    ("Model performance", performance.render),
    ("User study", study_run.render),
    ("Study results", study_results.render),
    ("Training & config", training.render),
    ("About", about.render),
]


@st.cache_resource(show_spinner=False)
def _get_global_state(config_key: str, dataset_key: str) -> dict:
    """Loads the pipeline state globally across all user sessions.
    If the disk cache (models/) is hot, this reads in ~1-2 seconds.

    Both keys matter: config_key invalidates the cache when the training
    configuration changes, dataset_key when the file in data/ is replaced or
    uploaded. Without the latter a newly uploaded dataset is never picked up,
    because an uploaded file changes nothing about the config.
    """
    # A fresh config is loaded in case it changed on disk; config_key is what
    # decides whether this result is reused.
    return run_pipeline(AppConfig.load(), progress=None, force_retrain=False)


def _bootstrap() -> dict:
    """First-run pipeline with a progress display."""
    force = st.session_state.pop("force_retrain", False)

    if force:
        # Clear the global memory cache so it picks up the new disk artifacts later
        st.cache_resource.clear()
        holder = st.empty()
        with holder.container():
            st.markdown(
                '<div style="max-width:52ch;margin:14vh auto 0">'
                '<div class="pg-title">Building the forecast pipeline</div>'
                '<div class="pg-sub">This runs once for a new dataset or '
                'configuration &mdash; a few minutes for around a million rows. '
                'Everything is cached in <code>models/</code>, so later starts '
                'take a second.</div></div>', unsafe_allow_html=True)
            bar = st.progress(0.0, text="Starting pipeline…")

        def progress(pct, msg):
            bar.progress(min(1.0, pct), text=msg)

        state = run_pipeline(cfg, progress=progress, force_retrain=True)
        holder.empty()
        return state

    # Normal load: use the globally shared cached state to prevent memory leaks
    return _get_global_state(cfg.model_key(), dataset_cache_key())


def _sidebar(state: dict | None) -> str:
    with st.sidebar:
        st.markdown("""
<div class="sb-brand">
  <div class="sb-logo">EW</div>
  <div class="sb-name">Heavy Rainfall<br>Early Warning</div>
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
            if st.button(name, key=f"nav_{name}", width="stretch",
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
    T.page_header("No dataset found")
    st.markdown("Add a historical weather dataset. The pipeline inspects "
                "the columns automatically — it needs at least a date "
                "column and a rainfall column; a station column is "
                "strongly recommended.")
    st.markdown(f'<div class="strip">Place a file at '
                f'<code>{DATA_DIR / "dataset.csv"}</code> (CSV, Parquet or '
                f'XLSX) or upload it below.</div>', unsafe_allow_html=True)
    up = st.file_uploader("Upload dataset", type=["csv", "xlsx", "parquet"])
    if up is not None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        suffix = up.name.rsplit(".", 1)[-1].lower()
        (DATA_DIR / f"dataset.{suffix}").write_bytes(up.getbuffer())
        st.rerun()
    st.stop()

page = _sidebar(state)
dict(PAGES)[page](state, cfg)
