# Heavy Rainfall Early Warning & Inundation Prediction System

> AI/ML-based heavy rainfall early warning with quantified uncertainty and
> inundation risk — for Smart India Hackathon problem statement **26071**.

| | |
|---|---|
| **Problem Statement ID** | 26071 |
| **Title** | AI/ML-Based Integrated heavy rainfall Early Warning and Inundation Prediction System using Satellite, Radar, observational Weather and numerical weather prediction model data |
| **Organization** | Ministry of Earth Sciences (MoES) |
| **Department** | India Meteorological Department |
| **Category / Theme** | Software · Disaster Management |

The system (1) predicts whether **significant rainfall (≥ 2.5 mm)** will
occur **tomorrow** at Indian weather stations from an ML pipeline trained on
historical observations, (2) quantifies the uncertainty of every prediction
with **calibrated probabilities and split conformal prediction**, (3) issues
an **IMD-style colour-coded early warning (Green / Yellow / Orange / Red)**
and an **inundation risk band** per station, and (4) includes a
**controlled human-subject experiment** testing whether communicating
uncertainty improves warning-based decisions.

## Why uncertainty-aware warnings

An early warning is a decision aid: whether a district activates its
response depends not only on the most likely outcome but on how strongly
the model can rule out the alternative. Poorly communicated uncertainty
leads to overconfident or overly conservative decisions — costly in both
directions during heavy-rainfall events. This system therefore reports a
calibrated probability, a conformal prediction set with a coverage
guarantee, and a documented warning matrix, and evaluates the human side
in a controlled study.

## Data source integration (per the problem statement)

| Source | Status | How |
|---|---|---|
| Observational weather | ✅ implemented | Historical station dataset (~1M daily records, 406 stations) the model trains on |
| NWP model data | ✅ implemented | Live current conditions from the Open-Meteo feed, which serves operational NWP model output for each station's coordinates (`src/live_weather.py`) |
| Satellite (INSAT-3D/3DR) | 🔜 roadmap | Ingest channel; the auto-detecting feature pipeline accepts extra columns (e.g. IR brightness temperature) without model-layer changes |
| Doppler weather radar | 🔜 roadmap | Same pluggable ingest path (e.g. reflectivity-derived rain-rate features) |

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
- Missing weather features are **left missing here on purpose**. Imputation
  happens later, in `train.py`, with medians computed on the training split
  alone and reused unchanged for validation, test and live inference. Filling
  them at this stage would compute the medians over the whole file — test
  period included — and leak future information into every training row.
- **Observed rainfall is kept separately (`rainfall_obs`) with missing values
  preserved** — the prediction target is never built from imputed rainfall.
- Rows whose 1–3 day rainfall lags cannot be formed are dropped rather than
  invented. On the shipped dataset (26.4% of rainfall readings missing, in
  station-level clusters rather than scattered) this costs 8.0% of modelling
  rows — 695,275 → 639,803 — and keeps all 406 stations.
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

## Early warning & inundation risk (`src/warning.py`)

The warning layer turns the calibrated model output into actionable,
IMD-aligned guidance per station:

- **IMD 24-h rainfall intensity categories** — light (2.5–15.5 mm),
  moderate (15.6–64.4), heavy (64.5–115.5), very heavy (115.6–204.4),
  extremely heavy (≥ 204.5).
- **Heavy-rain potential** — for the target date, the 90th percentile of
  wet-day rainfall within ±15 days of that day-of-year across all years at
  the station: "if it does rain, how heavy can it plausibly be right now".
- **Warning level** — a documented decision matrix over the calibrated
  probability *p* and the potential *q90*, mirroring IMD's colour code
  (Green *No action* · Yellow *Be updated* · Orange *Be prepared* · Red
  *Take action*):

  | Level | Condition |
  |---|---|
  | RED | p ≥ 0.70 & q90 ≥ 115.6 mm, or p ≥ 0.85 & q90 ≥ 64.5 mm |
  | ORANGE | p ≥ 0.60 & q90 ≥ 64.5 mm, or p ≥ 0.80 & q90 ≥ 15.6 mm |
  | YELLOW | p ≥ classification threshold, or ambiguous conformal set & q90 ≥ 64.5 mm |
  | GREEN | otherwise |

- **Inundation risk index** — 0.45 × saturation percentile (current
  7-day rainfall accumulation ranked against the station's own history, a
  soil-saturation/drainage-load proxy) + 0.35 × p + 0.20 × q90 scaled to
  the extremely-heavy threshold; banded LOW / MODERATE / HIGH / SEVERE.
  A screening indicator for response prioritisation, honestly labelled —
  not a terrain-based hydrological inundation model (that is on the
  roadmap).

## Dashboard — Heavy Rainfall Early Warning System

`streamlit run app.py` opens the app: a dark-navy sidebar shell (brand,
dataset stats, navigation, cached-pipeline footer) with eight pages,
following the approved UI design proposal. Pages can be deep-linked with
`?page=<name>` (e.g. `?page=Model performance`).

1. **Early warning board** (landing page) — sweeps the monitored stations,
   forecasts tomorrow for each from live conditions (climatology fallback,
   always labelled), and shows: summary tiles counting stations per warning
   level, an India map with stations coloured by warning level, ranked
   station cards (warning colour, rain probability, conformal set,
   heavy-rain potential with IMD intensity category, 7-day antecedent
   rainfall percentile, inundation band and advice), and a methodology
   expander with the full warning matrix.
2. **Prediction dashboard** — two forecast modes:
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
3. **Dataset explorer** — stat tiles (rows, columns, date range, stations,
   rain/no-rain days), automatic column-mapping table, preprocessing report
   with every step counted, train/validation/test split strip, filtered
   record browser with CSV download, target-distribution charts.
4. **Model performance** — metric tiles (accuracy, precision, recall, F1,
   ROC-AUC, Brier), confusion-matrix tiles, ROC & calibration curves, class
   distribution, conformal diagnostics (target vs empirical coverage, set
   size, ambiguity share), validation-only model comparison, full
   classification report, impurity + permutation importance. All values
   computed from the real test set.
5. **User study** — the controlled experiment (below), with a start screen
   (steps + what-each-condition-shows preview), scenario cards, progress
   bar and outcome feedback strips.
6. **Study results** — condition-comparison cards, accuracy/time charts,
   confidence-calibration chart, decision-change card, statistical-analysis
   cards, demo-data management, raw decision log. Shows a "No study data
   collected yet" empty state until real decisions exist.
7. **Training & config** — thresholds, split fractions with a test-fraction
   visual, RF hyperparameters, confidence-band preview strip, conformal
   coverage, column-mapping override, retrain buttons and the
   trained-artifacts table.
8. **About** — research-question banner, pipeline chips, key definitions,
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

Or use the launcher for your platform, which creates a private virtual
environment, installs the requirements when they change, starts the server
and opens the browser:

| Platform | Launcher |
| --- | --- |
| macOS / Linux | `./run.sh` |
| Windows | double-click **`run.bat`** |

Python ≥ 3.10. On first run the app loads the dataset, inspects and cleans
it, builds the target and features, trains and calibrates the models,
computes the conformal quantile and evaluates on the test period (a few
minutes for ~1M rows). All artifacts are cached in `models/`; subsequent
starts load in seconds. If `data/` is empty the app shows an upload prompt.

A headless end-to-end check is available: `python scripts/smoke_test.py`.

### Apple Silicon thread sizing

`src/runtime.py` sizes the numeric stack's thread pools for the host. The
two pools want opposite things on Apple Silicon, so it splits them:
HistGradientBoosting is OpenMP code with a barrier at every split and is
~25% faster confined to the performance cores, while a random forest fits
independent trees and loses ~45% when cut to the same four workers. So the
OpenMP/BLAS pools get the performance cores and joblib keeps every logical
CPU. The module documents the measurements behind those numbers. Other
platforms are left on the library defaults.

Set `OMP_NUM_THREADS` yourself to override; the module never replaces a
thread count you chose.

### Dataset read cache

The shipped dataset is a 62 MB `.xlsx`, which pandas parses cell by cell
(~41 s). The first read is mirrored to
`models/raw_cache_<fingerprint>.parquet` and every later read comes from
there (~2 s), so rebuilding after a config change no longer pays the
spreadsheet cost — a full retrain drops from ~68 s to ~27 s.

The sidecar is a cache, not a dataset: it is keyed by the same fingerprint
that invalidates the model cache, it lives outside `data/` so the dataset
auto-detection never sees it, and it is rewritten from the source file if it
is ever missing or unreadable. Delete `models/` at any time to rebuild
everything from `data/dataset.xlsx`.

## Project structure

```
├── app.py                  # Streamlit entry point (tabs, first-run pipeline)
├── data/
│   ├── dataset.xlsx        # the weather dataset, shipped (auto-detected schema)
│   └── study.db            # SQLite decision log (created on first decision)
├── models/                 # cached artifacts (created on first run)
│   ├── rainfall_bundle.joblib   # model + calibrator + conformal q̂ + metadata
│   ├── processed_features.parquet
│   ├── raw_cache_<fingerprint>.parquet  # fast re-read of the source dataset
│   ├── evaluation.joblib
│   └── config.json
├── src/
│   ├── config.py           # all configurable knobs + physical bounds
│   ├── runtime.py          # thread-pool sizing for the host machine
│   ├── data_loader.py      # loading, inspection, auto column mapping
│   ├── live_weather.py     # live current conditions for real-time forecasts
│   ├── preprocessing.py    # cleaning, dedup, invalid values, imputation
│   ├── feature_engineering.py  # target + leakage-safe features
│   ├── train.py            # split, model comparison, calibration, persistence
│   ├── uncertainty.py      # split conformal prediction + labels
│   ├── warning.py          # IMD colour-coded warnings + inundation risk
│   ├── predict.py          # prediction assembly + feature influence
│   ├── evaluation.py       # test-set metrics, curves, importances
│   ├── study.py            # scenarios, design, statistics, demo generator
│   ├── database.py         # SQLite logging (no PII)
│   └── pipeline.py         # first-run orchestration + caching
├── views/                  # Streamlit tab renderers + shared theme
├── notebooks/exploration.ipynb
├── run.sh                  # macOS / Linux launcher
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

- The warning matrix and inundation index are documented heuristics on top
  of the calibrated model output — the colour thresholds are design choices
  aligned to IMD intensity bands, not learned parameters, and the
  inundation band uses no terrain or drainage-network data.

## Roadmap (toward the full PS 26071 vision)

- **Satellite ingestion** — INSAT-3D/3DR IR brightness temperature and
  hydro-estimator rain-rate as additional feature columns.
- **Radar ingestion** — Doppler weather radar reflectivity-derived
  rain-rate features for nowcasting lead times.
- **Direct NWP fields** — GFS/WRF grid values (CAPE, precipitable water,
  850 hPa winds) interpolated to stations, beyond the current NWP-driven
  daily feed.
- **Rainfall-amount prediction** — conformalized quantile regression for
  mm-amount intervals, upgrading the warning matrix from potential-based
  to forecast-amount-based.
- **Hydrological inundation modelling** — DEM/HAND-based flood-spread
  estimation per district on top of the current screening index.
- Mondrian (per-station / per-season) conformal prediction for approximate
  conditional coverage.
- Mixed-effects logistic regression for the study analysis;
  cost-sensitive expected-utility decision support; SHAP-based local
  explanations.
