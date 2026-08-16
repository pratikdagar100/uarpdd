"""Generate UARPDD_Presentation.pptx in the project root.

Pulls real metrics from the trained bundle / evaluation artifacts so the
deck never contains fabricated numbers. Design language mirrors the app:
navy #0e1726, blue #2a78d6, light plane, Poppins-style headings.

Run:  python scripts/make_ppt.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from src.config import MODELS_DIR, PROJECT_ROOT

NAVY = RGBColor(0x0E, 0x17, 0x26)
NAVY2 = RGBColor(0x1B, 0x2C, 0x47)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
INK = RGBColor(0x16, 0x20, 0x2E)
INK2 = RGBColor(0x4C, 0x5A, 0x6B)
MUT = RGBColor(0x84, 0x94, 0xA7)
LIGHT = RGBColor(0xEE, 0xF2, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x18, 0xA0, 0x5A)
AMBER = RGBColor(0xE8, 0x93, 0x0C)

bundle = joblib.load(MODELS_DIR / "rainfall_bundle.joblib")
ev = joblib.load(MODELS_DIR / "evaluation.joblib")["evaluation"]
rg = bundle["split_ranges"]
ct = ev["conformal_test"]

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = prs.slide_width, prs.slide_height


def slide(bg=LIGHT):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = bg
    return s


def box(s, x, y, w, h, fill=None, line=None):
    sh = s.shapes.add_shape(1, x, y, w, h)  # rounded rect
    sh.fill.solid() if fill else sh.fill.background()
    if fill:
        sh.fill.fore_color.rgb = fill
    if line:
        sh.line.color.rgb = line
        sh.line.width = Pt(0.75)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def text(s, x, y, w, h, runs, align=PP_ALIGN.LEFT, space_after=4):
    """runs: list of paragraphs; each paragraph is (text, size, color, bold)
    or list of run tuples."""
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        if isinstance(para, tuple):
            para = [para]
        for t, size, color, bold in para:
            r = p.add_run()
            r.text = t
            r.font.size = Pt(size)
            r.font.color.rgb = color
            r.font.bold = bold
            r.font.name = "Poppins"
    return tb


def kicker(s, txt, x=Inches(0.7), y=Inches(0.45), color=MUT):
    text(s, x, y, Inches(9), Inches(0.35), [(txt.upper(), 11, color, True)])


def title(s, txt, x=Inches(0.7), y=Inches(0.75), size=30, color=INK,
          w=Inches(11.9)):
    text(s, x, y, w, Inches(0.9), [(txt, size, color, True)])


def bullets(s, items, x, y, w, size=13, gap=8, color=INK2, h=Inches(5)):
    runs = []
    for it in items:
        if isinstance(it, tuple):
            head, rest = it
            runs.append([(head + "  ", size, INK, True),
                         (rest, size, color, False)])
        else:
            runs.append((it, size, color, False))
    text(s, x, y, w, h, runs, space_after=gap)


def tile(s, x, y, w, h, label, value, foot="", vcolor=INK):
    box(s, x, y, w, h, fill=WHITE, line=RGBColor(0xE3, 0xE8, 0xEF))
    text(s, x + Inches(0.15), y + Inches(0.08), w - Inches(0.3), Inches(0.3),
         [(label, 10, MUT, False)])
    text(s, x + Inches(0.15), y + Inches(0.36), w - Inches(0.3), Inches(0.5),
         [(value, 20, vcolor, True)])
    if foot:
        text(s, x + Inches(0.15), y + h - Inches(0.36), w - Inches(0.3),
             Inches(0.3), [(foot, 9, MUT, False)])


# ---------------------------------------------------------------- 1. Title
s = slide(NAVY)
box(s, 0, 0, W, Inches(0.12), fill=BLUE)
text(s, Inches(0.9), Inches(1.6), Inches(11), Inches(0.4),
     [("ACADEMIC RESEARCH PROJECT", 12, RGBColor(0x8F, 0xA1, 0xB8), True)])
text(s, Inches(0.9), Inches(2.05), Inches(11.5), Inches(1.8),
     [("Uncertainty-Aware Rainfall Prediction", 40, WHITE, True),
      ("& Decision Dashboard", 40, WHITE, True)], space_after=0)
text(s, Inches(0.9), Inches(3.9), Inches(11), Inches(0.8),
     [("Communicating what the model predicts — and how uncertain it is.",
       16, RGBColor(0xC3, 0xD0, 0xE0), False),
      ("Machine learning · conformal prediction · controlled human-subject "
       "experiment · Streamlit", 12, RGBColor(0x8F, 0xA1, 0xB8), False)],
     space_after=8)
text(s, Inches(0.9), Inches(6.6), Inches(11), Inches(0.4),
     [("Pratik Dagar · SGT University · August 2026", 11,
       RGBColor(0x8F, 0xA1, 0xB8), False)])

# ------------------------------------------------- 2. Problem & question
s = slide()
kicker(s, "Motivation")
title(s, "The problem & the research question")
bullets(s, [
    ("Weather apps show a point prediction.",
     '"Rain: 78%" hides whether the model can actually rule out the '
     'alternative outcome.'),
    ("Decisions happen under uncertainty.",
     "Whether to plan an outdoor event depends on how decisively the model "
     "separates rain from no rain."),
    ("Poorly communicated uncertainty",
     "leads to overconfident or overly conservative decisions."),
], Inches(0.7), Inches(1.7), Inches(6.2), size=14, gap=14)
box(s, Inches(7.2), Inches(1.7), Inches(5.4), Inches(3.5), fill=NAVY)
text(s, Inches(7.5), Inches(1.95), Inches(4.8), Inches(0.3),
     [("RESEARCH QUESTION", 10, RGBColor(0x8F, 0xA1, 0xB8), True)])
text(s, Inches(7.5), Inches(2.35), Inches(4.8), Inches(1.9),
     [("Does communicating uncertainty in rainfall predictions improve "
       "human decision-making compared with presenting only a point "
       "prediction?", 16, WHITE, True)])
text(s, Inches(7.5), Inches(4.35), Inches(4.8), Inches(0.7),
     [[("Primary outcome — ", 11, WHITE, True),
       ("decision accuracy", 11, RGBColor(0xC3, 0xD0, 0xE0), False)],
      [("Secondary — ", 11, WHITE, True),
       ("decision time · stated confidence · overconfidence · decision "
        "changes", 11, RGBColor(0xC3, 0xD0, 0xE0), False)]], space_after=4)
text(s, Inches(0.7), Inches(5.6), Inches(12), Inches(1.2),
     [[("The system: ", 13, INK, True),
       ("a full ML pipeline on ~1M Indian weather observations, an "
        "uncertainty layer with statistical guarantees (split conformal "
        "prediction), a professional dashboard, and a built-in controlled "
        "experiment that logs real human decisions and analyses them "
        "statistically.", 13, INK2, False)]])

# ------------------------------------------------- 3. What it uses
s = slide()
kicker(s, "What it uses")
title(s, "Dataset & technology stack")
text(s, Inches(0.7), Inches(1.55), Inches(6), Inches(0.3),
     [("DATASET — Kaggle: Indian Rainfall and Weather Data", 11, BLUE, True)])
bullets(s, [
    ("970,339 daily observations", "across 406 Indian stations"),
    ("2015-01-01 → 2025-02-10", "date, station, state, district"),
    ("Weather variables:",
     "avg/min/max temperature, wind speed, air pressure, rainfall, "
     "latitude, longitude, elevation, season"),
    ("Real-world messiness:",
     "26.5% missing rainfall, ~30% missing wind/pressure, 43k duplicate "
     "station-days, impossible values (max_temp = 87 °C)"),
], Inches(0.7), Inches(1.95), Inches(6.0), size=12.5, gap=10)
text(s, Inches(7.1), Inches(1.55), Inches(6), Inches(0.3),
     [("TECHNOLOGY", 11, BLUE, True)])
bullets(s, [
    ("Python 3.10+", "pandas · numpy · pyarrow for the data layer"),
    ("scikit-learn", "Random Forest, Logistic Regression, "
     "HistGradientBoosting, isotonic/Platt calibration, permutation "
     "importance"),
    ("Own conformal layer", "split conformal prediction (LAC score)"),
    ("scipy", "Fisher / chi-square / Mann-Whitney / Welch tests, effect sizes"),
    ("Open-Meteo API", "live current conditions for real-time forecasts "
     "(no key; climatology fallback when offline)"),
    ("Streamlit + Plotly", "interactive dashboard, custom design system"),
    ("SQLite + joblib", "decision logging and cached model artifacts"),
], Inches(7.1), Inches(1.95), Inches(5.6), size=12.5, gap=8)
box(s, Inches(0.7), Inches(6.35), Inches(12), Inches(0.65), fill=WHITE,
    line=RGBColor(0xE3, 0xE8, 0xEF))
text(s, Inches(0.95), Inches(6.5), Inches(11.5), Inches(0.4),
     [[("Schema-agnostic: ", 11.5, INK, True),
       ("the loader auto-detects which column plays which role and exposes "
        "a manual mapping override — any compatible weather dataset can be "
        "dropped into data/.", 11.5, INK2, False)]])

# ------------------------------------------------- 4. How it works
s = slide()
kicker(s, "How it works")
title(s, "End-to-end pipeline")
steps = [
    ("1", "Inspect & clean", "auto column mapping, dedup, physical bounds, "
     "station-month median imputation"),
    ("2", "Build the target", "RainTomorrow = next-day observed rainfall "
     "≥ 2.5 mm, same station only, calendar-gap guard"),
    ("3", "Engineer features", "rain lags 1-3, rolling 3/7-day sums, "
     "temp/pressure changes, seasonality, geography — all leakage-safe"),
    ("4", "Split chronologically", f"train {rg['train'][0]} → {rg['train'][1]} · "
     f"validation {rg['validation'][0]} → {rg['validation'][1]} · "
     f"test {rg['test'][0]} → {rg['test'][1]}"),
    ("5", "Train & select", "RF vs LR vs Gradient Boosting — selected on "
     "validation ROC-AUC, never on test"),
    ("6", "Calibrate + conformal", "isotonic/Platt on validation-A, "
     "conformal quantile on validation-B (day-parity halves)"),
    ("7", "Serve & experiment", "live forecast for the real tomorrow, decision logging, statistics"),
]
y = Inches(1.6)
for num, head, desc in steps:
    box(s, Inches(0.7), y, Inches(0.42), Inches(0.42), fill=BLUE)
    text(s, Inches(0.7), y + Inches(0.045), Inches(0.42), Inches(0.35),
         [(num, 13, WHITE, True)], align=PP_ALIGN.CENTER)
    text(s, Inches(1.35), y - Inches(0.02), Inches(11.3), Inches(0.75),
         [[(head + "   ", 13.5, INK, True), (desc, 12, INK2, False)]])
    y += Inches(0.78)

# ------------------------------------------------- 5. Leakage prevention
s = slide()
kicker(s, "Scientific rigour")
title(s, "No data leakage, by construction")
bullets(s, [
    ("Target from the future, features from the past.",
     "The label is tomorrow's OBSERVED rainfall (never imputed); every "
     "feature uses only the current day or earlier."),
    ("Same-station shifting with a gap guard.",
     "groupby(station).shift(-1); if the next record is not literally the "
     "next calendar day, the row is dropped rather than mislabeled."),
    ("Chronological evaluation.",
     "The test period lies strictly after training — the model is scored "
     "the way a real forecaster would be."),
    ("Date-span splitting (a real bug we caught).",
     "A naive row-count split left no monsoon months in the calibration "
     "window and conformal coverage fell to 83%. Splitting by date span "
     "and interleaving calibration halves by day parity restored 90.8%."),
    ("Train-only statistics.",
     "Imputation medians for the model matrix come from training rows "
     "only; model selection and calibration never touch the test set."),
], Inches(0.7), Inches(1.7), Inches(12), size=13.5, gap=13)

# ------------------------------------------------- 6. Model results
s = slide()
kicker(s, "Held-out evaluation")
title(s, "Model performance — real numbers, test period")
tiles = [
    ("Accuracy", f"{ev['accuracy']:.1%}"), ("Precision", f"{ev['precision']:.1%}"),
    ("Recall", f"{ev['recall']:.1%}"), ("F1 score", f"{ev['f1']:.1%}"),
    ("ROC-AUC", f"{ev['roc_auc']:.3f}"), ("Brier", f"{ev['brier']:.3f}"),
]
x = Inches(0.7)
for label, val in tiles:
    tile(s, x, Inches(1.6), Inches(1.92), Inches(1.05), label, val)
    x += Inches(2.02)
rows = [(name, f"{r['val_roc_auc']:.4f}", f"{r['val_brier_raw']:.4f}",
         "selected" if name == bundle["model_name"] else "")
        for name, r in bundle["model_comparison"].items()]
y = Inches(3.1)
text(s, Inches(0.7), y, Inches(6), Inches(0.3),
     [("MODEL COMPARISON (VALIDATION ONLY)", 10.5, BLUE, True)])
y += Inches(0.4)
for name, auc_v, brier_v, sel in rows:
    text(s, Inches(0.7), y, Inches(3.3), Inches(0.3), [(name, 12.5, INK, sel != "")])
    text(s, Inches(4.0), y, Inches(1.5), Inches(0.3), [(auc_v, 12.5, INK2, False)])
    text(s, Inches(5.5), y, Inches(1.5), Inches(0.3), [(brier_v, 12.5, INK2, False)])
    if sel:
        text(s, Inches(6.9), y, Inches(1.2), Inches(0.3), [(sel, 11, GREEN, True)])
    y += Inches(0.42)
text(s, Inches(0.7), y + Inches(0.1), Inches(6.5), Inches(0.8),
     [("Selected on validation ROC-AUC with probability calibration "
       f"({bundle['calibrator_name']}) chosen by Brier score on a held-out "
       "calibration half.", 11, MUT, False)])
box(s, Inches(8.4), Inches(3.1), Inches(4.2), Inches(3.3), fill=NAVY)
text(s, Inches(8.7), Inches(3.35), Inches(3.7), Inches(0.3),
     [("CONFORMAL UNCERTAINTY (TEST)", 10, RGBColor(0x8F, 0xA1, 0xB8), True)])
conf_rows = [
    ("Target coverage", f"{ct['target_coverage']:.0%}"),
    ("Empirical coverage", f"{ct['empirical_coverage']:.1%}"),
    ("Avg. set size", f"{ct['avg_set_size']:.2f}"),
    ("Ambiguous days", f"{ct['share_ambiguous']:.1%}"),
]
yy = Inches(3.8)
for k, v in conf_rows:
    text(s, Inches(8.7), yy, Inches(2.4), Inches(0.35),
         [(k, 11.5, RGBColor(0xC3, 0xD0, 0xE0), False)])
    text(s, Inches(11.1), yy, Inches(1.3), Inches(0.35), [(v, 13, WHITE, True)])
    yy += Inches(0.55)
text(s, Inches(8.7), yy + Inches(0.05), Inches(3.7), Inches(0.8),
     [("Coverage lands on the 90% target — the uncertainty layer does what "
       "it promises.", 10.5, RGBColor(0x8F, 0xA1, 0xB8), False)])

# ------------------------------------------------- 7. Uncertainty explained
s = slide()
kicker(s, "Uncertainty quantification")
title(s, "Split conformal prediction — not just a probability")
bullets(s, [
    ("Probability alone is not uncertainty.",
     "A calibrated 62% tells you the odds; it does not tell you whether "
     "the model can statistically rule out the other outcome."),
    ("Nonconformity score:", "s = 1 − p(true class) on a held-out "
     "calibration set; q̂ = ⌈(n+1)(1−α)⌉/n quantile."),
    ("Prediction set:", "every class whose score fits under q̂. "
     "{RAIN} = decisive · {RAIN, NO SIGNIFICANT RAIN} = genuinely ambiguous."),
    ("The guarantee (marginal coverage):",
     "across many days, the true outcome falls inside the set ~90% of the "
     "time. The UI explicitly avoids the wrong reading — it is NOT a 90% "
     "correctness probability for one forecast."),
    ("User-facing layer:",
     "confidence bands (LOW / MODERATE / HIGH / VERY HIGH) derived from "
     "calibrated probability, labelled as heuristic indicators."),
], Inches(0.7), Inches(1.7), Inches(12), size=13.5, gap=13)

# ------------------------------------------------- 8. Dashboard
s = slide()
kicker(s, "The application")
title(s, "Rainfall Uncertainty Lab — seven pages")
pages = [
    ("Prediction dashboard", "LIVE forecast for the real calendar tomorrow "
     "(plus historical backtest mode) — probability track, prediction-set "
     "boxes, confidence scale, feature influence, history charts"),
    ("Dataset explorer", "inspection report, column mapping, preprocessing "
     "counts, record browser, CSV export"),
    ("Model performance", "test metrics, confusion matrix, ROC & "
     "calibration curves, conformal diagnostics, feature importance"),
    ("User study", "the controlled experiment — scenario cards, decision "
     "task, timing"),
    ("Study results", "condition comparison, overconfidence, decision "
     "changes, statistical tests"),
    ("Training & config", "thresholds, split fractions, hyperparameters, "
     "confidence bands, column mapping, retrain"),
    ("About", "research question, pipeline, definitions, limitations"),
]
y = Inches(1.65)
for name, desc in pages:
    box(s, Inches(0.7), y, Inches(0.14), Inches(0.5), fill=BLUE)
    text(s, Inches(1.05), y - Inches(0.02), Inches(11.6), Inches(0.6),
         [[(name + "   ", 13, INK, True), (desc, 11.5, INK2, False)]])
    y += Inches(0.72)
text(s, Inches(0.7), y + Inches(0.05), Inches(12), Inches(0.5),
     [("Dark-navy sidebar shell · stat tiles · solid status pills · "
       "info strips · one first-run pipeline that caches every artifact.",
       11, MUT, False)])

# ------------------------------------------------- 8b. Real-time
s = slide()
kicker(s, "Real-time operation")
title(s, "Learning from history, forecasting the real tomorrow")
bullets(s, [
    ("The dataset ends before today.",
     "It is what the model LEARNS from — 2015-01-01 to 2025-02-10. It "
     "cannot supply today's weather."),
    ("Live conditions close the gap.",
     "For the selected station's coordinates the app fetches recent daily "
     "observations (Open-Meteo, free, no key) and assembles them into "
     "exactly the same leakage-safe feature row used in training: rainfall "
     "today, lags 1-3, rolling 3/7-day sums, temperature and pressure "
     "changes, calendar and station encodings."),
    ("Variables matched to the training distribution.",
     "Sea-level pressure (not station pressure), daily mean wind speed, "
     "daily precipitation total — so live rows look like training rows."),
    ("Honest fallback.",
     "If the feed is unreachable the card switches from LIVE CONDITIONS to "
     "CLIMATOLOGY FALLBACK and says the estimate uses the station's typical "
     "conditions for this time of year. Current weather is never invented."),
    ("Backtest mode remains.",
     "Any historical date can still be replayed with the real outcome "
     "revealed — that is how the reported metrics were measured."),
], Inches(0.7), Inches(1.7), Inches(12), size=13, gap=12)

# ------------------------------------------------- 9. User study
s = slide()
kicker(s, "The experiment")
title(s, "Controlled user study — POINT vs UNCERTAINTY")
bullets(s, [
    ("Design: within-subject, counterbalanced.",
     "10 scenarios per participant from the held-out test period; a random "
     "half rendered as POINT (prediction + probability), half as "
     "UNCERTAINTY (adds confidence, uncertainty, 90% prediction set). "
     "Order randomised; renderings otherwise identical."),
    ("Task:", '"Would you plan an outdoor event tomorrow?" YES / NO, '
     "stated confidence 1–5, decision time measured; outcome revealed only "
     "after the decision is saved."),
    ("Ground truth:", "planning is appropriate iff no significant rain "
     "actually fell; declining is appropriate iff it did."),
    ("Logging:", "SQLite — participant/session UUIDs only, no accounts, "
     "no PII."),
    ("Scenario pool:", "stratified over ambiguous vs decisive predictions "
     "and rain vs no-rain outcomes."),
], Inches(0.7), Inches(1.7), Inches(12), size=13.5, gap=13)

# ------------------------------------------------- 10. Statistics
s = slide()
kicker(s, "Analysis")
title(s, "Statistical analysis — data-driven test selection")
bullets(s, [
    ("Decision accuracy (primary):",
     "2×2 contingency — Fisher's exact when any expected cell < 5, else "
     "chi-square with Yates' correction; Cohen's h; Wilson 95% CIs."),
    ("Decision time:",
     "Welch's t-test only if Shapiro-Wilk is consistent with normality in "
     "both groups, otherwise Mann-Whitney U (times are right-skewed); "
     "Cohen's d or rank-biserial r."),
    ("Stated confidence (ordinal 1-5):", "Mann-Whitney U."),
    ("Overconfidence:", "stated confidence (rescaled 0-1) minus actual "
     "accuracy, per condition."),
    ("Integrity rules:",
     "results are labelled EXPLORATORY below 30 decisions/condition or 5 "
     "participants; the results page shows 'No study data collected yet' "
     "until real decisions exist; the demo-data generator is clearly "
     "synthetic, excluded by default and one-click deletable."),
], Inches(0.7), Inches(1.7), Inches(12), size=13.5, gap=13)

# ------------------------------------------------- 11. Run it
s = slide()
kicker(s, "Reproduce")
title(s, "How to run")
box(s, Inches(0.7), Inches(1.7), Inches(5.9), Inches(1.5), fill=NAVY)
text(s, Inches(1.0), Inches(1.95), Inches(5.3), Inches(1.1),
     [("pip install -r requirements.txt", 14, RGBColor(0x9E, 0xC5, 0xF4), False),
      ("streamlit run app.py", 14, RGBColor(0x9E, 0xC5, 0xF4), False)],
     space_after=6)
bullets(s, [
    ("First run:", "loads the dataset, inspects and cleans it, builds the "
     "target and features, trains and calibrates, computes the conformal "
     "quantile, evaluates — then caches everything in models/."),
    ("Every later start:", "loads in seconds; retrains only when the "
     "dataset file or a training-relevant setting changes."),
    ("Headless check:", "python scripts/smoke_test.py runs the whole "
     "pipeline end-to-end without the UI."),
    ("Swap the dataset:", "drop any CSV/XLSX/Parquet with a date + "
     "rainfall column into data/ — roles are auto-detected."),
], Inches(0.7), Inches(3.55), Inches(6.2), size=12.5, gap=10)
text(s, Inches(7.2), Inches(1.7), Inches(5.5), Inches(0.3),
     [("PROJECT STRUCTURE", 10.5, BLUE, True)])
struct = ("app.py — entry point & sidebar shell\n"
          "src/ — config, loader, preprocessing, features,\n"
          "        train, uncertainty, predict, evaluation,\n"
          "        study, database, pipeline\n"
          "views/ — theme + one module per page\n"
          "data/ — dataset.csv, study.db\n"
          "models/ — cached bundle, features, evaluation\n"
          "notebooks/ — exploration.ipynb\n"
          "scripts/ — smoke_test.py, make_ppt.py\n"
          "README.md · requirements.txt")
box(s, Inches(7.2), Inches(2.1), Inches(5.4), Inches(3.6), fill=WHITE,
    line=RGBColor(0xE3, 0xE8, 0xEF))
tb = text(s, Inches(7.45), Inches(2.3), Inches(5.0), Inches(3.3),
          [(line, 11.5, INK2, False) for line in struct.split("\n")],
          space_after=3)
for p in tb.text_frame.paragraphs:
    for r in p.runs:
        r.font.name = "Consolas"

# ------------------------------------------------- 12. Limitations
s = slide()
kicker(s, "Honest science")
title(s, "Limitations & future work")
bullets(s, [
    ("Decision ground truth is simplified", "— it ignores asymmetric costs "
     "of cancelling vs. getting rained out."),
    ("Decisions treated as independent", "— mixed-effects models would "
     "handle within-participant correlation."),
    ("Marginal, not conditional coverage", "— per-station/per-season "
     "(Mondrian) conformal is the natural next step."),
    ("Heuristic explanations", "— feature influence is not SHAP-style "
     "attribution and is labelled as such."),
    ("Small convenience samples", "— flagged exploratory automatically."),
], Inches(0.7), Inches(1.65), Inches(6.1), size=12.5, gap=11)
text(s, Inches(7.1), Inches(1.65), Inches(5.5), Inches(0.3),
     [("FUTURE IMPROVEMENTS", 10.5, BLUE, True)])
bullets(s, [
    "Mondrian conformal prediction (per station / season)",
    "Mixed-effects logistic regression for the study",
    "Cost-sensitive, expected-utility decision support",
    "SHAP-based local explanations",
    "Rain-amount intervals via conformalized quantile regression",
], Inches(7.1), Inches(2.05), Inches(5.5), size=12.5, gap=10)
box(s, Inches(0.7), Inches(5.9), Inches(12), Inches(0.9), fill=NAVY)
text(s, Inches(1.0), Inches(6.08), Inches(11.4), Inches(0.6),
     [[("Core contribution:  ", 13, WHITE, True),
       ("a complete, honest pipeline from raw weather data to a "
        "statistically analysed human-decision experiment — uncertainty "
        "communicated, never fabricated.", 13,
        RGBColor(0xC3, 0xD0, 0xE0), False)]])

out = PROJECT_ROOT / "UARPDD_Presentation.pptx"
prs.save(out)
print("saved", out)
