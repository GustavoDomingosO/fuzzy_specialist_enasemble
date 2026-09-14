"""
Debug experiment: FuzzyEspecialistEnsembleClassifier on Iris, with MLflow tracking.

Not a benchmark -- this is a fast, single-dataset sanity check to confirm the
model still fits/predicts correctly and that MLflow logging is wired up
correctly, before pointing the same pattern at the full PMLB sweep.
"""

import time

import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold

# Adjust this import to wherever Base_Model.py actually lives in your project
# (mirroring how the PMLB script imports its models, e.g. Final_Models.<module>).
from src.Base_Model import FuzzyEspecialistEnsembleClassifier


def run_debug_experiment():
    # Recent MLflow versions put the plain "./mlruns" folder store into
    # maintenance mode and refuse to write to it unless you opt in; a local
    # SQLite file is the currently-recommended zero-setup backend instead.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fuzzy_ensemble_debug")

    # 1. Load a small, well-known dataset -- just to sanity-check the pipeline,
    # not to draw any conclusions about model quality.
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")

    dataset_name = "iris"
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=100)

    fold_accuracies = []
    fold_auc = []

    with mlflow.start_run(run_name=f"{dataset_name}_{FuzzyEspecialistEnsembleClassifier.__name__}"):
        # get_params() works out of the box now that the class inherits from
        # BaseEstimator -- logging every hyperparameter is one line instead of
        # listing each one out by hand.
        param_probe = FuzzyEspecialistEnsembleClassifier(random_state=100)
        mlflow.log_params({
            "dataset": dataset_name,
            "k_folds": k_folds,
            **param_probe.get_params(),
        })

        print(f"\n--- Debug run: {FuzzyEspecialistEnsembleClassifier.__name__} on {dataset_name} ---")
        start_time = time.perf_counter()

        for fold_idx, (train_index, val_index) in enumerate(skf.split(X, y), 1):
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

            # Fresh instance per fold -- same reasoning as the PMLB script:
            # avoids any risk of state leaking between folds.
            classifier = FuzzyEspecialistEnsembleClassifier(random_state=100, detailed_logs=False)

            try:
                classifier.fit(X_train_fold, y_train_fold)
                # .evaluate() (not .score()) -- score() now returns a single
                # sklearn-standard accuracy float via ClassifierMixin; evaluate()
                # is the renamed original method that still returns (acc, auc).
                acc, auc = classifier.evaluate(X_val_fold, y_val_fold)

                print(f"Fold {fold_idx}: Accuracy = {acc:.4f} | AUC = {auc:.4f}")
                mlflow.log_metric("fold_accuracy", acc, step=fold_idx)
                mlflow.log_metric("fold_auc", auc, step=fold_idx)

                fold_accuracies.append(acc)
                fold_auc.append(auc)

            except Exception as e:
                print(f"Fold {fold_idx} Failed: {e}")
                mlflow.log_param(f"fold_{fold_idx}_error", str(e))

        processing_time = time.perf_counter() - start_time

        if fold_accuracies:
            summary = {
                "Mean_Accuracy": float(np.mean(fold_accuracies)),
                "STD_Accuracy": float(np.std(fold_accuracies)),
                "Mean_AUC": float(np.mean(fold_auc)),
                "STD_AUC": float(np.std(fold_auc)),
                "Processing_Time_Seconds": processing_time,
            }
            mlflow.log_metrics(summary)

            print("\n--- Debug Run Complete ---")
            for key, value in summary.items():
                print(f"{key}: {value:.4f}")
        else:
            print("\nAll folds failed -- check the errors printed above.")


if __name__ == "__main__":
    run_debug_experiment()