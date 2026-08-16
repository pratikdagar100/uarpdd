# Uncertainty-Aware Rainfall Prediction & Decision Dashboard

> Communicating what the model predicts — and how uncertain it is.

An academic research project that (1) predicts whether **significant rainfall
(≥ 2.5 mm)** will occur **tomorrow** at Indian weather stations, (2) quantifies
the uncertainty of every prediction with **split conformal prediction**, and
(3) runs a **controlled human-subject experiment** testing whether showing
uncertainty changes user decisions and improves decision quality.

## Problem statement & motivation

Weather apps usually show a single point prediction ("Rain: 78%"). Decision
makers, however, act under uncertainty: whether to plan an outdoor event
depends not only on the most likely outcome but on how strongly the model can
rule out the alternative. Poorly communicated uncertainty leads to
overconfident or overly conservative decisions. This project builds an
interface that communicates prediction uncertainty rather than only a point
prediction, and evaluates its effect on human decisions in a controlled study.

## Research question

> **Does communicating uncertainty in rainfall predictions improve human
> decision-making compared with presenting only a point prediction?**

- **Primary outcome:** decision accuracy (was the event decision appropriate
  given the actual next-day rainfall?).
- **Secondary outcomes:** decision time, stated confidence (1–5),
  overconfidence (stated confidence vs. actual correctness), decision changes
  between conditions.

## Dataset

`data/dataset.csv` — historical Indian weather/rainfall data
(source: [Kaggle — Indian Rainfall and Weather Data](https://www.kaggle.com/datasets/ameydilipmorye/indian-rainfall-and-weather-data)).

Detected structure (reported live in the Dataset Explorer tab):

| Property | Value |
|---|---|
| Rows | 970,339 daily observations |
| Stations | 406 across India |
| Date range | 2015-01-01 → 2025-02-10 |
| Columns | date_of_record, month, season, station_name, state, district, avg_temp, min_temp, max_temp, wind_speed, air_pressure, elevation, latitude, longitude, rainfall |

The pipeline **does not assume these column names**: `src/data_loader.py`
auto-detects which column plays which role via name patterns, and the
Training & Config tab exposes a manual mapping override. To use another
weather dataset, drop a CSV/XLSX/Parquet with at least a date column and a
rainfall column (a station/location column is strongly recommended) into
`data/` — everything downstream adapts automatically.

## Preprocessing (`src/preprocessing.py`)

- Unparseable dates → rows dropped (counted and reported).
- Duplicate (station, date) records → aggregated (mean for numerics).
- Physically impossible values (e.g. max_temp = 87 °C, pressure < 850 hPa)
  → set to missing, not deleted.
- `min_temp > max_temp` → swapped.
- Missing weather features → imputed with per-(station, month) median,
  falling back to per-month then global median.
- **Observed rainfall is kept separately (`rainfall_obs`) with missing values
  preserved** — the prediction target is never built from imputed rainfall.
- Every step's row/value counts are reported in the Dataset Explorer tab.

## Target creation (2.5 mm threshold)

Per station, chronologically sorted:

```
rain_tomorrow_mm = rainfall_obs shifted back by one record (same station only)
valid only when the next record is exactly the next calendar day
RainTomorrow     = 1 if rain_tomorrow_mm >= 2.5 mm else 0
```

Rows without a valid next-day observation (station's final record, date gaps,
missing next-day rainfall) are removed. Rainfall is never shifted across
stations.

## Leakage prevention

1. The target uses only the *next-day observed* rainfall of the *same
   station*; a calendar-gap guard rejects "tomorrows" that are not actually
   the next day.
2. All predictors are functions of the current day or earlier: lags use
   `shift(+k)`, rolling windows end at the current day.
3. No feature is derived from the target or any future observation.
4. Chronological train/validation/test split: the test period lies strictly
   after training; imputation statistics for the model matrix use training
   rows only.
5. Model selection, probability calibration and conformal calibration all
   happen on validation data — the test set is only touched in evaluation.

## Feature engineering (`src/feature_engineering.py`)

Rainfall today, lags 1–3, rolling 3/7-day sums, rain-today flag; avg/min/max
temperature, temperature range and 1-day change; wind speed; air pressure and
1-day pressure change; month, day-of-year (sin/cos), season; station identity
code, latitude, longitude, elevation. Humidity/cloud cover are used
automatically **if present** in the dataset (they are not in this one — none
are fabricated).

## ML methodology (`src/train.py`)

- **Chronological split:** first 70% of the date span → train, next 15% →
  validation/calibration, final 15% → test (configurable). Fractions are
  applied to the date span rather than row counts so each block stays
  seasonally complete (a row-count split left the validation window without
  monsoon months, which miscalibrated the conformal quantile). Date ranges
  are displayed in the app.
- **Models compared:** Random Forest (primary), Logistic Regression baseline,
  Gradient Boosting (`HistGradientBoostingClassifier`). The dashboard model
  is selected on **validation** ROC-AUC — never on test performance.
- **Probability calibration:** the validation block is split into two
  seasonally-interleaved halves (by day-of-year parity). On half A, Platt
  scaling and isotonic regression are fitted; the calibrator with the lower
  Brier score on half B is used (identity is also a candidate). Calibration quality is shown as a
  reliability diagram + Brier score in the Model Performance tab.
- Artifacts are persisted (`models/rainfall_bundle.joblib`) keyed by a
  dataset fingerprint + config hash, so the app retrains only when the data
  or training-relevant settings change.

## Uncertainty methodology (`src/uncertainty.py`)

**Split conformal prediction** (LAC / inverse-probability score) on
validation half B, at a configurable coverage level (default **90%**):

```
score s_i = 1 − p̂(true class | x_i)          on the conformal calibration set
q̂ = ⌈(n+1)(1−α)⌉/n empirical quantile of s
prediction set = { class c : 1 − p̂(c|x) ≤ q̂ }
```

- **{RAIN}** or **{NO SIGNIFICANT RAIN}** → strong prediction.
- **{RAIN, NO SIGNIFICANT RAIN}** → ambiguous: the model cannot rule out
  either outcome at the target coverage.

The guarantee is **marginal coverage**: across many days, the true outcome
falls inside the set ≈ 90% of the time. The UI explicitly avoids the wrong
reading "this individual forecast is 90% likely to be correct". Empirical
test-set coverage and average set size are reported in Model Performance.

Separately, a **user-facing confidence indicator** is derived from the
calibrated probability of the predicted class (defaults: 50–59% LOW,
60–74% MODERATE, 75–89% HIGH, ≥90% VERY HIGH — configurable), clearly
labelled as a heuristic indicator, not a formal guarantee.

## Dashboard — "Rainfall Uncertainty Lab"

`streamlit run app.py` opens the app: a dark-navy sidebar shell (brand,
dataset stats, navigation, cached-pipeline footer) with seven pages,
following the approved UI design proposal. Pages can be deep-linked with
`?page=<name>` (e.g. `?page=Model performance`).

1. **Prediction dashboard** — two forecast modes:
   * **Tomorrow (default)** — a genuine forecast for the real calendar
     tomorrow. The model *learns* from the historical dataset; today's
     conditions are fetched live from the [Open-Meteo](https://open-meteo.com)
     daily API for the selected station's coordinates (`src/live_weather.py`)
     and assembled into exactly the same leakage-safe feature row used in
     training — rainfall today, lags 1–3, rolling 3/7-day sums,
     temperature/pressure changes, calendar and station encodings. Variables
     are matched to the training distribution (sea-level pressure, daily mean
     wind, daily precipitation total). Results are cached for 30 minutes.
     If the feed is unreachable the app falls back to the station's seasonal
     climatology and labels the card **CLIMATOLOGY FALLBACK** instead of
     **LIVE CONDITIONS** — it never silently invents current weather.
   * **Historical backtest** — any date in the dataset; predicts the
     following day from the conditions actually observed then and reveals
     the real outcome.

   Both modes show the hero forecast card with probability track, threshold
   marker, confidence/uncertainty/prediction-set pills and RAIN / NO
   SIGNIFICANT RAIN set boxes (an ambiguous forecast highlights both and
   shows a warning strip); class-probability bars with the confidence scale;
   conditions tiles; per-prediction *model feature influence* rows (labelled
   heuristic, not causal); and rainfall/temperature/pressure history plus
   monthly climatology charts.
2. **Dataset explorer** — stat tiles (rows, columns, date range, stations,
   rain/no-rain days), automatic column-mapping table, preprocessing report
   with every step counted, train/validation/test split strip, filtered
   record browser with CSV download, target-distribution charts.
3. **Model performance** — metric tiles (accuracy, precision, recall, F1,
   ROC-AUC, Brier), confusion-matrix tiles, ROC & calibration curves, class
   distribution, conformal diagnostics (target vs empirical coverage, set
   size, ambiguity share), validation-only model comparison, full
   classification report, impurity + permutation importance. All values
   computed from the real test set.
4. **User study** — the controlled experiment (below), with a start screen
   (steps + what-each-condition-shows preview), scenario cards, progress
   bar and outcome feedback strips.
5. **Study results** — condition-comparison cards, accuracy/time charts,
   confidence-calibration chart, decision-change card, statistical-analysis
   cards, demo-data management, raw decision log. Shows a "No study data
   collected yet" empty state until real decisions exist.
6. **Training & config** — thresholds, split fractions with a test-fraction
   visual, RF hyperparameters, confidence-band preview strip, conformal
   coverage, column-mapping override, retrain buttons and the
   trained-artifacts table.
7. **About** — research-question banner, pipeline chips, key definitions,
   limitations, future improvements and quick-reference cards.

A slide deck explaining the whole project (what it uses, how it works,
results, study design) is generated at **`UARPDD_Presentation.pptx`** in the
project root; regenerate it after retraining with
`python scripts/make_ppt.py` (it pulls the real metrics from `models/`).

## User study methodology (`src/study.py`, `views/study_run.py`)

- **Design: within-subject, counterbalanced.** Each participant answers
  10 scenarios (configurable); a random half is rendered as **POINT**
  (prediction + probability only) and half as **UNCERTAINTY** (adds
  confidence, uncertainty level, 90% prediction set). Scenario→condition
  assignment and presentation order are randomised per participant, so
  condition is not confounded with scenario difficulty or order. The two
  renderings are otherwise visually identical.
- **Scenarios** are drawn from the held-out test period, stratified over
  (ambiguous vs. certain predictions) × (rain vs. no-rain actual outcomes).
  Participants never see the outcome before deciding.
- **Decision task:** "Would you plan an outdoor event tomorrow?" (YES/NO) +
  confidence 1–5. Decision time is measured from scenario display to
  submission. The actual outcome is revealed only after logging.
- **Ground truth for decision quality:** planning the event is appropriate
  iff no significant rain actually occurred; declining is appropriate iff
  significant rain occurred. (A simplification — real decisions weigh
  asymmetric costs; see Limitations.)
- **Logging:** SQLite (`data/study.db`) with participant/session UUIDs only —
  no accounts, no PII.
- **No fabricated results:** before any real decisions exist the results tab
  shows "No study data collected yet." A clearly-labelled synthetic **demo
  data generator** exists solely to demonstrate the analysis dashboard; demo
  records are flagged `is_demo=1`, excluded by default, and deletable with
  one click.

## Statistical analysis

Data-driven test selection, reported with the results:

- **Decision accuracy:** 2×2 contingency table — Fisher's exact test when any
  expected cell count < 5, otherwise chi-square with Yates' correction.
  Effect size: Cohen's h; Wilson 95% CIs per condition and a normal-
  approximation CI for the difference.
- **Decision time:** Welch's t-test only if Shapiro–Wilk is consistent with
  normality in both groups; otherwise Mann–Whitney U (decision times are
  typically right-skewed). Effect size: Cohen's d or rank-biserial r.
- **Stated confidence (ordinal 1–5):** Mann–Whitney U.
- Results are labelled **exploratory** when there are fewer than 30 decisions
  per condition or fewer than 5 participants; the app never claims
  significance from insufficient data.

## Installation & running

```bash
pip install -r requirements.txt
streamlit run app.py
```

On Windows you can also just double-click **`run.bat`** in the project
folder, which does the same thing.

Python ≥ 3.10. On first run the app loads the dataset, inspects and cleans
it, builds the target and features, trains and calibrates the models,
computes the conformal quantile and evaluates on the test period (a few
minutes for ~1M rows). All artifacts are cached in `models/`; subsequent
starts load in seconds. If `data/` is empty the app shows an upload prompt.

A headless end-to-end check is available: `python scripts/smoke_test.py`.

## Project structure

```
├── app.py                  # Streamlit entry point (tabs, first-run pipeline)
├── data/
│   ├── dataset.csv         # the weather dataset (auto-detected schema)
│   └── study.db            # SQLite decision log (created on first decision)
├── models/                 # cached artifacts (created on first run)
│   ├── rainfall_bundle.joblib   # model + calibrator + conformal q̂ + metadata
│   ├── processed_features.parquet
│   ├── evaluation.joblib
│   └── config.json
├── src/
│   ├── config.py           # all configurable knobs + physical bounds
│   ├── data_loader.py      # loading, inspection, auto column mapping
│   ├── live_weather.py     # live current conditions for real-time forecasts
│   ├── preprocessing.py    # cleaning, dedup, invalid values, imputation
│   ├── feature_engineering.py  # target + leakage-safe features
│   ├── train.py            # split, model comparison, calibration, persistence
│   ├── uncertainty.py      # split conformal prediction + labels
│   ├── predict.py          # prediction assembly + feature influence
│   ├── evaluation.py       # test-set metrics, curves, importances
│   ├── study.py            # scenarios, design, statistics, demo generator
│   ├── database.py         # SQLite logging (no PII)
│   └── pipeline.py         # first-run orchestration + caching
├── views/                  # Streamlit tab renderers + shared theme
├── notebooks/exploration.ipynb
├── run.bat                 # Windows double-click launcher
├── scripts/smoke_test.py
├── requirements.txt
└── README.md
```

## Limitations

- **Decision ground truth is simplified:** appropriateness ignores asymmetric
  costs of cancelling vs. getting rained out.
- **Statistical independence:** the analysis treats decisions as independent;
  within-participant correlation in the within-subject design would ideally
  be handled with mixed-effects models once enough participants exist.
- **Conformal coverage is marginal**, not conditional per station or season.
- The feature-influence explanation is a heuristic (global importance ×
  value unusualness), not SHAP-style attribution, and is labelled as such.
- Convenience sampling of participants; results from small samples are
  explicitly flagged exploratory.
- Station identity is label-encoded; unseen stations would need retraining.
- The live "tomorrow" mode depends on an external weather feed
  (Open-Meteo) and on the assumption that its variables are comparable to
  the dataset's station observations; the historical evaluation metrics
  were measured on dataset observations, not on the live feed.

## Future improvements

- Mokri/Mondrian (per-station or per-season) conformal prediction for
  approximate conditional coverage.
- Mixed-effects logistic regression for the study analysis.
- Cost-sensitive decision support (expected-utility framing).
- SHAP-based local explanations.
- Probability-of-precipitation regression (rain amount intervals via
  conformalized quantile regression).
