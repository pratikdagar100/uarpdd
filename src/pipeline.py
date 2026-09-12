"""First-run pipeline orchestration with on-disk caching.

Start app -> find dataset -> inspect -> preprocess -> target + features ->
chronological split -> train + calibrate + conformal -> evaluate -> save.
Subsequent starts load the cached artifacts as long as the dataset file, the
training-relevant config and the pipeline code that produced them are all
unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from .config import AppConfig, DATA_DIR, MODELS_DIR
from .data_loader import (find_dataset, dataset_fingerprint, load_raw,
                          auto_map_columns, inspect_dataset)
from .preprocessing import preprocess
from .feature_engineering import build_features
from .train import train_pipeline, load_bundle, chronological_split
from .evaluation import evaluate_bundle

PROCESSED_PATH = MODELS_DIR / "processed_features.parquet"
META_PATH = MODELS_DIR / "pipeline_meta.json"
EVAL_PATH = MODELS_DIR / "evaluation.joblib"
MAPPING_OVERRIDE_PATH = MODELS_DIR / "column_mapping.json"

# ---------------------------------------------------------- code versioning
# The cached artifacts are only valid for the code that produced them. Keying
# the caches on the dataset and the config alone means an edit to the science
# modules is silently ignored -- the app keeps serving a model built by the
# previous version of the pipeline. Hashing the sources instead of bumping a
# constant by hand is deliberate: the failure mode being fixed here is exactly
# the one a forgotten manual bump reintroduces.
#
# The three stages are layered so an edit invalidates only what it can
# actually change: touching train.py must not force a 64 MB spreadsheet to be
# re-read, but it must retrain, and a retrain must re-evaluate.
_FEATURE_SOURCES = ("preprocessing.py", "feature_engineering.py")
_MODEL_SOURCES = ("train.py", "uncertainty.py")
_EVAL_SOURCES = ("evaluation.py", "predict.py")

_SRC_DIR = Path(__file__).resolve().parent


def _hash_sources(names: tuple[str, ...], *parents: str) -> str:
    h = hashlib.md5()
    for parent in parents:
        h.update(parent.encode())
    for name in names:
        try:
            h.update(hashlib.md5((_SRC_DIR / name).read_bytes()).digest())
        except OSError:                              # source missing: treat
            h.update(b"\0")                          # as its own version
    return h.hexdigest()[:12]


def features_code_version() -> str:
    """Version of the code that produces models/processed_features.parquet."""
    return _hash_sources(_FEATURE_SOURCES)


def model_code_version() -> str:
    """Version of the code that produces the model bundle (features included:
    a changed feature matrix requires a retrain)."""
    return _hash_sources(_MODEL_SOURCES, features_code_version())


def eval_code_version() -> str:
    """Version of the code that produces the cached evaluation."""
    return _hash_sources(_EVAL_SOURCES, model_code_version())


def dataset_cache_key() -> str:
    """Cheap identity of the dataset currently in data/.

    Callers cache pipeline state in memory across reruns; keyed on the config
    alone, that cache survives the dataset being replaced or uploaded and
    keeps serving the previous dataset's model.
    """
    path = find_dataset(DATA_DIR)
    return "none" if path is None else dataset_fingerprint(path)


def load_mapping_override() -> dict:
    if MAPPING_OVERRIDE_PATH.exists():
        try:
            return json.loads(MAPPING_OVERRIDE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_mapping_override(mapping: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MAPPING_OVERRIDE_PATH.write_text(json.dumps(mapping, indent=2),
                                     encoding="utf-8")


def _meta_matches(fingerprint: str, cfg: AppConfig) -> bool:
    if not (META_PATH.exists() and PROCESSED_PATH.exists()):
        return False
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (meta.get("fingerprint") == fingerprint
            and meta.get("config_key") == cfg.model_key()
            and meta.get("code_version") == features_code_version()
            and meta.get("mapping_override") == load_mapping_override())


def run_pipeline(cfg: AppConfig, progress=None, force_retrain: bool = False
                 ) -> dict:
    """Returns a state dict with everything the UI needs, or a dict with
    'error' when no dataset is available."""
    def _report(pct, msg):
        if progress:
            progress(pct, msg)

    dataset_path = find_dataset(DATA_DIR)
    if dataset_path is None:
        return {"error": "no_dataset"}

    fingerprint = dataset_fingerprint(dataset_path)
    state: dict = {"dataset_path": str(dataset_path),
                   "fingerprint": fingerprint}

    # ---------- Processed features (cached) ----------
    if _meta_matches(fingerprint, cfg) and not force_retrain:
        _report(0.05, "Loading cached processed dataset…")
        features = pd.read_parquet(PROCESSED_PATH)
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        inspection = meta["inspection"]
        prep_report = meta["prep_report"]
        feat_report = meta["feat_report"]
        feature_cols = meta["feature_cols"]
    else:
        _report(0.02, f"Loading dataset {dataset_path.name}…")
        raw = load_raw(dataset_path, cache_dir=MODELS_DIR,
                       fingerprint=fingerprint)
        _report(0.08, "Inspecting dataset structure…")
        mapping = auto_map_columns(raw)
        mapping.update({k: v for k, v in load_mapping_override().items()
                        if v in raw.columns or v is None})
        inspection = inspect_dataset(raw, mapping)
        _report(0.12, "Preprocessing (duplicates, invalid values, imputation)…")
        clean, prep_report = preprocess(raw, mapping, cfg)
        del raw
        _report(0.2, "Creating RainTomorrow target and lag features…")
        features, feature_cols, feat_report = build_features(clean, cfg)
        del clean
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        features.to_parquet(PROCESSED_PATH)
        META_PATH.write_text(json.dumps({
            "fingerprint": fingerprint, "config_key": cfg.model_key(),
            "code_version": features_code_version(),
            "mapping_override": load_mapping_override(),
            "inspection": _jsonable(inspection),
            "prep_report": _jsonable(prep_report),
            "feat_report": _jsonable(feat_report),
            "feature_cols": feature_cols,
        }, indent=2, default=str), encoding="utf-8")

    state.update({"inspection": inspection, "prep_report": prep_report,
                  "feat_report": feat_report, "feature_cols": feature_cols,
                  "features": features})

    # ---------- Model bundle (cached) ----------
    bundle = (None if force_retrain else
              load_bundle(fingerprint, cfg, model_code_version()))
    if bundle is None:
        _report(0.25, "Training models (this happens once per dataset/config)…")
        bundle = train_pipeline(features, feature_cols, cfg, fingerprint,
                                progress=lambda p, m: _report(0.25 + 0.6 * p, m),
                                code_version=model_code_version())
        if EVAL_PATH.exists():
            EVAL_PATH.unlink()
    state["bundle"] = bundle

    split = chronological_split(features, cfg)
    state["split"] = split

    # ---------- Evaluation (cached) ----------
    evaluation = None
    if EVAL_PATH.exists() and not force_retrain:
        try:
            cached = joblib.load(EVAL_PATH)
            if cached.get("fingerprint") == fingerprint and \
               cached.get("config_key") == cfg.model_key() and \
               cached.get("code_version") == eval_code_version():
                evaluation = cached["evaluation"]
        except Exception:
            evaluation = None
    if evaluation is None:
        _report(0.9, "Evaluating on the held-out test period…")
        evaluation = evaluate_bundle(bundle, features, split["test_mask"], cfg)
        joblib.dump({"fingerprint": fingerprint,
                     "config_key": cfg.model_key(),
                     "code_version": eval_code_version(),
                     "evaluation": evaluation}, EVAL_PATH, compress=3)
    state["evaluation"] = evaluation

    # Training-set feature statistics for the explanation heuristic
    tr = features.loc[split["train_mask"], feature_cols]
    state["train_stats"] = pd.DataFrame({"mean": tr.mean(), "std": tr.std()})

    _report(1.0, "Ready.")
    return state


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=str))
