"""
Non-interactive synthetic-data experiment runner.

Replaces the exploratory Streamlit app (interactive_make_classification_full.py)
for systematic comparisons: instead of sliders, each experiment is a YAML file
under `tests/` describing a `sklearn.datasets.make_classification` dataset (the
same variables the interactive app's sidebar exposed -- n_samples, n_features,
n_informative, n_redundant, n_clusters_per_class, class_sep, flip_y, n_classes,
weights, random_state), an optional `augmentation` section (skew, outliers,
scale -- see below), a k-fold CV setup, and which registered models to run.
Results are written to `outputs/<experiment_name>/<run_timestamp>/`.

The optional `augmentation` section ports the interactive app's skew/outlier
/scale controls 1:1:
    augmentation:
      skew_factor: 0.0      # 0 = off. X -> exp(X * skew_factor) when > 0.
      outlier_ratio: 0.0    # 0 = off. Fraction of rows scaled by a random
                            # 5x-10x factor when > 0.
      scale_factor: 1.0     # 1.0 = off. Uniform multiplier on X when > 1.0.
      partial_scale: false  # If true, scale_factor only applies to the
                            # first 2 features instead of the whole dataset.
Omit the whole section (as in the plain example config) to skip it entirely.

NOTE (scope): only the "Dataset parameters" section of the interactive app,
plus its skew/outlier/scale controls, were carried over here. The "Regras que
mudam por região" (spatial rule change) control was deliberately left out.

NOTE (metrics): for now this only computes what was asked for -- accuracy
(mean/std over the folds), and wall-clock train/inference time (mean/std over
the folds). AUC and G-mean (used by the old interactive app) are not computed
here; that's a deliberate scope cut, not an oversight.

NOTE (MLflow): each experiment run is also logged to MLflow (parent run per
experiment, one nested child run per model), using the same
sqlite:///mlflow.db backend already set up for debug_experiment_iris.py, under
a separate experiment name ("fuzzy_ensemble_experiments") so sweep runs don't
mix with debug runs. This is additive to the CSV output, not a replacement for
it -- set `mlflow: false` in a config to skip it for that experiment.

Usage:
    uv run python experiments/run_synthetic_experiments.py --config tests/binary_moderate.yaml
        # runs just that one experiment file (the default mode)

    uv run python experiments/run_synthetic_experiments.py --sweep
        # runs every *.yaml / *.yml experiment file found under tests/ instead
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from src.Base_Model import FuzzyEspecialistEnsembleClassifier
from src.model_variants import (
    FastKSelectionEnsembleClassifier,
    FuzzyKMedoidsEspecialistEnsembleClassifier,
    GMMEspecialistEnsembleClassifier,
    PerceptronEspecialistEnsembleClassifier,
    PossibilisticCMeansEspecialistEnsembleClassifier,
    SVMEspecialistEnsembleClassifier,
)
from src.baselines import (
    ForestOfLocalTreesBaseline,
    FuzzyBaggingBaseline,
    RandomForestBaseline,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TESTS_DIR = PROJECT_ROOT / "tests"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Every model available to an experiment config's `models:` list, keyed by the
# name used in the YAML. Each entry is a zero-arg-besides-random_state factory
# so every fold gets a FRESH instance (same reasoning as debug_experiment_iris.py:
# avoids any risk of state leaking between folds).
MODEL_REGISTRY = {
    # Plain sklearn reference points, kept for continuity with the old
    # interactive app's fixed comparison set.
    "DecisionTree": lambda random_state: DecisionTreeClassifier(random_state=random_state),
    "Bagging": lambda random_state: BaggingClassifier(random_state=random_state),
    # Main model.
    "FuzzyEspecialistEnsembleClassifier": lambda random_state: FuzzyEspecialistEnsembleClassifier(random_state=random_state),
    # Clustering-algorithm / k-selection / base-classifier variants.
    "GMMEspecialistEnsembleClassifier": lambda random_state: GMMEspecialistEnsembleClassifier(random_state=random_state),
    "FuzzyKMedoidsEspecialistEnsembleClassifier": lambda random_state: FuzzyKMedoidsEspecialistEnsembleClassifier(random_state=random_state),
    "PossibilisticCMeansEspecialistEnsembleClassifier": lambda random_state: PossibilisticCMeansEspecialistEnsembleClassifier(random_state=random_state),
    "FastKSelectionEnsembleClassifier": lambda random_state: FastKSelectionEnsembleClassifier(random_state=random_state),
    "SVMEspecialistEnsembleClassifier": lambda random_state: SVMEspecialistEnsembleClassifier(random_state=random_state),
    "PerceptronEspecialistEnsembleClassifier": lambda random_state: PerceptronEspecialistEnsembleClassifier(random_state=random_state),
    # Baselines.
    "RandomForestBaseline": lambda random_state: RandomForestBaseline(random_state=random_state),
    "ForestOfLocalTreesBaseline": lambda random_state: ForestOfLocalTreesBaseline(random_state=random_state),
    "FuzzyBaggingBaseline": lambda random_state: FuzzyBaggingBaseline(random_state=random_state),
}


def load_experiment_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "name" not in config:
        raise ValueError(f"{config_path}: experiment config must have a top-level 'name'")
    if "dataset" not in config:
        raise ValueError(f"{config_path}: experiment config must have a top-level 'dataset' section")

    return config


def resolve_model_names(config: dict) -> list:
    models = config.get("models", "all")
    if models == "all" or models is None:
        return list(MODEL_REGISTRY.keys())
    return list(models)


def apply_augmentation(X: np.ndarray, augmentation: dict, random_state: int) -> np.ndarray:
    """
    Ports the skew/outlier/scale transforms from
    interactive_make_classification_full.py (its `apply_skew_and_outliers`
    plus the scale-factor block), applied identically -- same formulas, same
    order (skew+outliers, then scale). The "spatial rule change" control from
    that app is intentionally NOT ported here.
    """
    X = X.copy()
    rng = np.random.RandomState(random_state)

    skew_factor = augmentation.get("skew_factor", 0.0)
    outlier_ratio = augmentation.get("outlier_ratio", 0.0)
    scale_factor = augmentation.get("scale_factor", 1.0)
    partial_scale = augmentation.get("partial_scale", False)

    # --- Skew (exponential transform) ---
    if skew_factor > 0:
        X = np.exp(X * skew_factor)

    # --- Outliers (random rows scaled by a random 5x-10x factor) ---
    if outlier_ratio > 0:
        n_outliers = int(len(X) * outlier_ratio)
        if n_outliers > 0:
            outlier_indices = rng.choice(len(X), n_outliers, replace=False)
            X[outlier_indices] *= rng.uniform(5, 10, size=(n_outliers, X.shape[1]))

    # --- Scale (uniform magnitude multiplier, optionally only 2 features) ---
    if scale_factor > 1.0:
        if partial_scale and X.shape[1] >= 2:
            X[:, :2] *= scale_factor
        else:
            X *= scale_factor

    return X


def build_dataset(config: dict):
    dataset_params = dict(config["dataset"])  # copy -- make_classification mutates nothing, but be safe
    random_state = dataset_params.get("random_state", 42)
    X, y = make_classification(**dataset_params)

    augmentation = config.get("augmentation")
    if augmentation:
        X = apply_augmentation(X, augmentation, random_state)

    return X, y


def run_one_model(model_name: str, X, y, skf: StratifiedKFold, random_state: int) -> dict:
    """Runs k-fold CV for a single model, returning summary stats. Any
    exception during a fold is caught and recorded rather than aborting the
    whole sweep (mirrors debug_experiment_iris.py's per-fold error handling)."""
    fold_acc, fold_train_time, fold_infer_time = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = MODEL_REGISTRY[model_name](random_state)

        try:
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            y_pred = model.predict(X_test)
            infer_time = time.perf_counter() - t0

            acc = accuracy_score(y_test, y_pred)

            print(f"  [{model_name}] fold {fold_idx}: acc={acc:.4f} "
                  f"train={train_time:.3f}s infer={infer_time:.3f}s")

            fold_acc.append(acc)
            fold_train_time.append(train_time)
            fold_infer_time.append(infer_time)

        except Exception as e:
            print(f"  [{model_name}] fold {fold_idx} FAILED: {e}")

    if not fold_acc:
        return {
            "model": model_name,
            "n_folds_ok": 0,
            "accuracy_mean": np.nan,
            "accuracy_std": np.nan,
            "train_time_mean_s": np.nan,
            "train_time_std_s": np.nan,
            "inference_time_mean_s": np.nan,
            "inference_time_std_s": np.nan,
        }

    return {
        "model": model_name,
        "n_folds_ok": len(fold_acc),
        "accuracy_mean": float(np.mean(fold_acc)),
        "accuracy_std": float(np.std(fold_acc)),
        "train_time_mean_s": float(np.mean(fold_train_time)),
        "train_time_std_s": float(np.std(fold_train_time)),
        "inference_time_mean_s": float(np.mean(fold_infer_time)),
        "inference_time_std_s": float(np.std(fold_infer_time)),
    }


def run_experiment(config_path: Path, outputs_dir: Path) -> pd.DataFrame:
    config = load_experiment_config(config_path)
    experiment_name = config["name"]

    print(f"\n=== Experiment: {experiment_name} ({config_path.name}) ===")

    X, y = build_dataset(config)

    cv_config = config.get("cv", {})
    n_splits = cv_config.get("n_splits", 5)
    random_state = cv_config.get("random_state", config["dataset"].get("random_state", 42))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    model_names = resolve_model_names(config)
    unknown = [m for m in model_names if m not in MODEL_REGISTRY]
    if unknown:
        print(f"  Skipping unknown model name(s) not in MODEL_REGISTRY: {unknown}")
        model_names = [m for m in model_names if m in MODEL_REGISTRY]

    use_mlflow = config.get("mlflow", True)
    if use_mlflow:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("fuzzy_ensemble_experiments")

    results = []

    if use_mlflow:
        parent_run_cm = mlflow.start_run(run_name=f"{experiment_name}_{datetime.now():%Y%m%d_%H%M%S}")
    else:
        import contextlib
        parent_run_cm = contextlib.nullcontext()

    with parent_run_cm:
        if use_mlflow:
            mlflow.log_params({
                "experiment_name": experiment_name,
                "n_splits": n_splits,
                "cv_random_state": random_state,
                **{f"dataset.{k}": v for k, v in config["dataset"].items()},
                **{f"augmentation.{k}": v for k, v in config.get("augmentation", {}).items()},
            })

        for model_name in model_names:
            print(f" Running {model_name} ({n_splits}-fold CV)...")

            if use_mlflow:
                with mlflow.start_run(run_name=model_name, nested=True):
                    param_probe = MODEL_REGISTRY[model_name](random_state)
                    if hasattr(param_probe, "get_params"):
                        mlflow.log_params(param_probe.get_params())
                    summary = run_one_model(model_name, X, y, skf, random_state)
                    mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
                    results.append(summary)
            else:
                results.append(run_one_model(model_name, X, y, skf, random_state))

    results_df = pd.DataFrame(results)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = outputs_dir / experiment_name / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.csv"
    results_df.to_csv(results_path, index=False)

    # Copy the resolved config alongside the results for reproducibility.
    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"\n Results written to: {results_path}")
    print(results_df.to_string(index=False))

    return results_df


def find_experiment_configs(tests_dir: Path) -> list:
    return sorted(list(tests_dir.glob("*.yaml")) + list(tests_dir.glob("*.yml")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to a single experiment YAML file to run. This is the "
             "default mode -- required unless --sweep is passed.",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run every *.yaml/*.yml experiment file found under --tests-dir "
             "instead of a single --config. An explicit opt-in.",
    )
    parser.add_argument("--tests-dir", type=Path, default=DEFAULT_TESTS_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    args = parser.parse_args()

    if args.sweep:
        if args.config is not None:
            print("--sweep was passed -- ignoring --config and running every "
                  f"experiment file under {args.tests_dir} instead.")
        config_paths = find_experiment_configs(args.tests_dir)
        if not config_paths:
            print(f"No experiment configs found under {args.tests_dir}")
            return
    else:
        if args.config is None:
            parser.error(
                "Specify --config <path> to run a single experiment "
                "(the default mode), or pass --sweep to run every "
                f"*.yaml/*.yml file under {args.tests_dir} instead."
            )
        config_paths = [args.config]

    for config_path in config_paths:
        run_experiment(config_path, args.outputs_dir)


if __name__ == "__main__":
    main()
