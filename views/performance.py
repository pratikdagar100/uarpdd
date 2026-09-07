"""Model performance — design screen 04: stat tiles, confusion-matrix
tiles, ROC/calibration charts, conformal diagnostics, model comparison and
feature importance. Every number is computed on the held-out test period."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import AppConfig
from src.predict import FEATURE_LABELS
from . import theme as T


def render(state: dict, cfg: AppConfig) -> None:
    ev = state["evaluation"]
    bundle = state["bundle"]

    T.page_header(
        "Model performance",
        "How the selected model scores on the held-out test period, and "
        "whether its stated probabilities and prediction sets hold up.")

    tiles = [
        T.tile("Accuracy", f"{ev['accuracy']:.1%}"),
        T.tile("Precision", f"{ev['precision']:.1%}"),
        T.tile("Recall", f"{ev['recall']:.1%}"),
        T.tile("F1 score", f"{ev['f1']:.1%}"),
        T.tile("ROC-AUC", f"{ev['roc_auc']:.3f}"),
        T.tile("Brier score", f"{ev['brier']:.3f}"),
    ]
    st.markdown('<div style="display:grid;grid-template-columns:repeat(6,1fr);'
                'gap:10px;margin-bottom:0.9rem">' + "".join(tiles) + "</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        cm = np.array(ev["confusion_matrix"])
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        def cell(v, tag, bg, fg):
            return (f'<div class="cm-cell" style="background:{bg};color:{fg}">'
                    f'<div class="cm-v">{v:,}</div>'
                    f'<div class="cm-t">{tag}</div></div>')
        st.markdown(f"""
<div class="uar-card"><div style="display:flex;justify-content:space-between">
<b>Confusion matrix</b></div>
<div class="cm-grid">
  <div></div><div class="cm-h">Pred: No sig. rain</div>
  <div class="cm-h">Pred: Rain</div>
  <div class="cm-h">Actual: No sig. rain</div>
  {cell(tn, "TN", T.SEQ_BLUE[7], "#ffffff")}
  {cell(fp, "FP", "#dbe7f6", T.INK)}
  <div class="cm-h">Actual: Rain</div>
  {cell(fn, "FN", "#dbe7f6", T.INK)}
  {cell(tp, "TP", "#b7d3f6", T.INK)}
</div></div>""", unsafe_allow_html=True)
    with c2:
        roc = ev["roc_curve"]
        fig = go.Figure()
        fig.add_scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                        name=f"Model (AUC {ev['roc_auc']:.3f})",
                        line=dict(color=T.BLUE, width=2))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                        showlegend=False,
                        line=dict(color=T.BASELINE, width=1.5, dash="dot"))
        fig.update_layout(title="ROC curve",
                          xaxis_title="False positive rate",
                          yaxis_title="True positive rate")
        T.apply_layout(fig, height=330)
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False})

    c3, c4 = st.columns(2)
    with c3:
        cal = pd.DataFrame(ev["calibration_curve"])
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                        line=dict(color=T.BASELINE, width=1.5, dash="dot"))
        if not cal.empty:
            fig.add_scatter(x=cal["mean_predicted"],
                            y=cal["observed_frequency"],
                            mode="lines+markers", name="Model",
                            marker=dict(size=8, color=T.BLUE,
                                        line=dict(color=T.SURFACE, width=2)),
                            line=dict(color=T.BLUE, width=2),
                            customdata=cal["count"],
                            hovertemplate="Predicted %{x:.0%} → observed "
                                          "%{y:.0%} (n=%{customdata:,})"
                                          "<extra></extra>")
        fig.update_layout(title="Calibration curve (reliability diagram)",
                          xaxis_title="Mean predicted rain probability",
                          yaxis_title="Observed rain frequency")
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        T.apply_layout(fig, height=330)
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False})
    with c4:
        cd = ev["class_distribution"]
        fig = go.Figure(go.Bar(
            x=["Train", "Test"],
            y=[cd["train_positive_rate"], cd["test_positive_rate"]],
            marker_color=[T.BLUE, T.ORANGE],
            marker_line=dict(color=T.SURFACE, width=2),
            text=[f"{cd['train_positive_rate']:.1%}",
                  f"{cd['test_positive_rate']:.1%}"],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1%} rain days<extra></extra>"))
        fig.update_yaxes(tickformat=".0%", range=[0, max(
            cd["train_positive_rate"], cd["test_positive_rate"]) * 1.4])
        fig.update_layout(title="Class distribution — share of RainTomorrow = 1")
        T.apply_layout(fig, height=330)
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False})

    # ---------------- Conformal diagnostics ----------------
    st.markdown("### Uncertainty layer diagnostics")
    ct = ev["conformal_test"]
    tiles = [
        T.tile("Target coverage", f"{ct['target_coverage']:.0%}"),
        T.tile("Empirical test coverage", f"{ct['empirical_coverage']:.1%}"),
        T.tile("Avg. prediction-set size", f"{ct['avg_set_size']:.2f}"),
        T.tile("Ambiguous predictions", f"{ct['share_ambiguous']:.1%}"),
    ]
    st.markdown('<div style="display:grid;grid-template-columns:repeat(4,1fr);'
                'gap:10px;margin-bottom:0.9rem">' + "".join(tiles) + "</div>",
                unsafe_allow_html=True)

    # ---------------- Comparison + classification report ----------------
    c5, c6 = st.columns(2)
    with c5:
        st.markdown("#### Model comparison (validation)")
        rows = []
        for name, res in bundle["model_comparison"].items():
            sel = ('<span style="background:#e9f9f0;color:#0b6b3a;'
                   'border-radius:6px;padding:1px 8px;font-size:var(--t-xs);'
                   'font-weight:700">selected</span>'
                   if name == bundle["model_name"] else "")
            rows.append(f'<tr><td style="padding:6px 8px">{name}</td>'
                        f'<td style="padding:6px 8px;text-align:right">'
                        f'{res["val_roc_auc"]:.4f}</td>'
                        f'<td style="padding:6px 8px;text-align:right">'
                        f'{res["val_brier_raw"]:.4f}</td>'
                        f'<td style="padding:6px 8px;text-align:right">'
                        f'{res["fit_seconds"]}</td>'
                        f'<td style="padding:6px 8px;text-align:center">{sel}'
                        f'</td></tr>')
        st.markdown(
            f'<div class="uar-card"><table style="width:100%;'
            f'border-collapse:collapse;font-size:0.85rem"><tr>'
            f'<th style="text-align:left;padding:6px 8px;font-size:0.7rem;'
            f'letter-spacing:0.08em;color:{T.MUTED}">MODEL</th>'
            f'<th style="text-align:right;padding:6px 8px;font-size:0.7rem;'
            f'color:{T.MUTED}">VAL. ROC-AUC</th>'
            f'<th style="text-align:right;padding:6px 8px;font-size:0.7rem;'
            f'color:{T.MUTED}">VAL. BRIER (RAW)</th>'
            f'<th style="text-align:right;padding:6px 8px;font-size:0.7rem;'
            f'color:{T.MUTED}">FIT TIME (S)</th><th></th></tr>'
            + "".join(rows) + "</table></div>", unsafe_allow_html=True)
    with c6:
        st.markdown("#### Full classification report")
        st.markdown(f'<div class="mono">{ev["classification_report"]}</div>',
                    unsafe_allow_html=True)

    # ---------------- Feature importance ----------------
    st.markdown("### Feature importance")
    c7, c8 = st.columns(2)
    with c7:
        if "model_importance" in ev:
            imp = pd.Series(ev["model_importance"]).sort_values().tail(12)
            fig = go.Figure(go.Bar(
                x=imp.values,
                y=[FEATURE_LABELS.get(i, i) for i in imp.index],
                orientation="h", marker_color=T.BLUE,
                marker_line=dict(color=T.SURFACE, width=2),
                hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
            fig.update_layout(
                title=f"{bundle['model_name']} impurity importance")
            T.apply_layout(fig, height=380)
            st.plotly_chart(fig, width="stretch",
                            config={"displayModeBar": False})
    with c8:
        pi = pd.DataFrame(ev["permutation_importance"]).T
        pi = pi.sort_values("mean").tail(12)
        fig = go.Figure(go.Bar(
            x=pi["mean"], y=[FEATURE_LABELS.get(i, i) for i in pi.index],
            orientation="h", marker_color=T.ORANGE,
            marker_line=dict(color=T.SURFACE, width=2),
            error_x=dict(array=pi["std"], color=T.INK_2, thickness=1),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
        fig.update_layout(title="Permutation importance (ΔROC-AUC, test sample)")
        T.apply_layout(fig, height=380)
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False})
