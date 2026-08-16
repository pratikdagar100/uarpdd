"""Training & config — design screen 10: thresholds, split fractions,
hyperparameters, confidence-band preview, column-mapping override and the
trained-artifacts table."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import AppConfig
from src.data_loader import ROLE_PATTERNS
from src.pipeline import load_mapping_override, save_mapping_override
from . import theme as T


def render(state: dict, cfg: AppConfig) -> None:
    T.page_header("Training & config")

    with st.form("config_form"):
        st.markdown("#### Problem definition")
        c1, c2, c3 = st.columns(3)
        rain_thr = c1.number_input("Rainfall threshold (mm) — RAIN if ≥",
                                   0.1, 50.0, cfg.rain_threshold_mm, 0.1)
        cls_thr = c2.slider("Classification threshold (probability)",
                            0.05, 0.95, cfg.classification_threshold, 0.05)
        coverage = c3.slider("Conformal coverage target", 0.70, 0.99,
                             cfg.conformal_coverage, 0.01)

        st.markdown("#### Chronological split")
        c4, c5, c6 = st.columns([1.2, 1.2, 1])
        train_frac = c4.slider("Training fraction", 0.5, 0.85,
                               cfg.train_frac, 0.05)
        val_frac = c5.slider("Validation / calibration fraction", 0.05, 0.3,
                             cfg.val_frac, 0.05)
        test_frac = 1 - train_frac - val_frac
        with c6:
            st.markdown(f"""
<div class="tile"><div class="t-label">Test fraction (most recent data)</div>
<div class="t-value">{test_frac:.0%}</div>
<div style="display:flex;height:8px;border-radius:999px;overflow:hidden;
  margin-top:0.4rem">
  <div style="width:{train_frac * 100:.0f}%;background:{T.BLUE}"></div>
  <div style="width:{val_frac * 100:.0f}%;background:{T.WARNING}"></div>
  <div style="width:{test_frac * 100:.0f}%;background:{T.SERIOUS}"></div>
</div></div>""", unsafe_allow_html=True)

        st.markdown("#### Random Forest hyperparameters")
        c7, c8, c9, c10 = st.columns(4)
        n_est = c7.number_input("Trees", 50, 1000, cfg.n_estimators, 50)
        max_depth = c8.number_input("Max depth", 4, 40, cfg.max_depth, 1)
        min_split = c9.number_input("Min samples split", 2, 100,
                                    cfg.min_samples_split, 1)
        min_leaf = c10.number_input("Min samples leaf", 1, 50,
                                    cfg.min_samples_leaf, 1)

        st.markdown("#### Confidence bands")
        c11, c12, c13 = st.columns(3)
        low_max = c11.slider("LOW below", 0.51, 0.7, cfg.conf_low_max, 0.01)
        mod_max = c12.slider("MODERATE below", 0.6, 0.85,
                             cfg.conf_moderate_max, 0.01)
        high_max = c13.slider("HIGH below", 0.8, 0.99, cfg.conf_high_max, 0.01)
        lo_w = (low_max - 0.5) * 200
        mo_w = (mod_max - low_max) * 200
        hi_w = (high_max - mod_max) * 200
        vh_w = (1.0 - high_max) * 200
        st.markdown(f"""
<div style="display:flex;height:26px;border-radius:8px;overflow:hidden;
  font-size:0.68rem;font-weight:700;color:#fff;text-align:center;
  line-height:26px;margin:0.3rem 0 0.6rem">
  <div style="width:{lo_w:.0f}%;background:{T.SERIOUS}">LOW 50–{low_max:.0%}</div>
  <div style="width:{mo_w:.0f}%;background:{T.WARNING}">MODERATE –{mod_max:.0%}</div>
  <div style="width:{hi_w:.0f}%;background:{T.BLUE}">HIGH –{high_max:.0%}</div>
  <div style="width:{vh_w:.0f}%;background:#1c3f77">VERY HIGH ≥ {high_max:.0%}</div>
</div>""", unsafe_allow_html=True)

        colA, colB, _sp = st.columns([1.3, 1, 2.2])
        apply_light = colA.form_submit_button("Apply (no retraining needed)",
                                              use_container_width=True)
        apply_retrain = colB.form_submit_button("Apply & retrain",
                                                type="primary",
                                                use_container_width=True)

    if apply_light or apply_retrain:
        new_cfg = AppConfig(
            rain_threshold_mm=rain_thr, classification_threshold=cls_thr,
            train_frac=train_frac, val_frac=val_frac,
            n_estimators=int(n_est), max_depth=int(max_depth),
            min_samples_split=int(min_split), min_samples_leaf=int(min_leaf),
            conformal_coverage=coverage,
            conf_low_max=low_max, conf_moderate_max=mod_max,
            conf_high_max=high_max,
            scenarios_per_participant=cfg.scenarios_per_participant,
            random_state=cfg.random_state)
        new_cfg.save()
        needs_retrain = new_cfg.model_key() != cfg.model_key()
        if apply_retrain or needs_retrain:
            st.session_state["force_retrain"] = True
        st.session_state.pop("pipeline_state", None)
        st.rerun()

    # ---------------- Column mapping override ----------------
    st.markdown("### Column mapping override")
    mapping = dict(state["inspection"]["mapping"])
    override = load_mapping_override()
    raw_columns = ["— none —"] + list(state["inspection"]["columns"].keys())

    with st.form("mapping_form"):
        new_map = {}
        roles = list(ROLE_PATTERNS.keys())
        cols = st.columns(3)
        for i, role in enumerate(roles):
            current = override.get(role, mapping.get(role)) or "— none —"
            idx = raw_columns.index(current) if current in raw_columns else 0
            new_map[role] = cols[i % 3].selectbox(role, raw_columns, index=idx)
        _sp, colS = st.columns([3, 1])
        save_map = colS.form_submit_button("Save mapping & rebuild",
                                           type="primary",
                                           use_container_width=True)
    if save_map:
        save_mapping_override({r: (None if v == "— none —" else v)
                               for r, v in new_map.items()})
        st.session_state["force_retrain"] = True
        st.session_state.pop("pipeline_state", None)
        st.rerun()

    # ---------------- Current artifacts ----------------
    st.markdown("### Current trained artifacts")
    b = state["bundle"]
    rows = [
        ("Selected model", b["model_name"]),
        ("Probability calibration", b["calibrator_name"]),
        ("Calibrator Brier scores (validation-B)",
         ", ".join(f"{k}: {v:.4f}" for k, v in b["calibrator_briers"].items())),
        ("Conformal q̂", f"{b['qhat']:.4f}"),
        ("Coverage on calibration split",
         f"{b['conformal']['coverage_on_calibration']:.1%} "
         f"(target {b['conformal']['coverage_target']:.0%})"),
        ("Avg. set size on calibration split",
         f"{b['conformal']['avg_set_size_on_calibration']:.2f}"),
        ("Trained at", b["trained_at"]),
        ("Dataset fingerprint", b["fingerprint"][:16] + "…"),
    ]
    html = "".join(
        f'<tr><td style="padding:7px 10px;font-size:0.82rem;'
        f'color:{T.INK_2};width:280px">{k}</td>'
        f'<td style="padding:7px 10px;font-family:Consolas,monospace;'
        f'font-size:0.82rem;color:{T.INK}">{v}</td></tr>' for k, v in rows)
    st.markdown(f'<div class="uar-card"><table style="width:100%;'
                f'border-collapse:collapse">{html}</table></div>',
                unsafe_allow_html=True)
