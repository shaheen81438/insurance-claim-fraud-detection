"""Reproducible model comparison and artifact creation for claims."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier

from .data_validation import validate_claims
from .preprocessing import build_preprocessing_pipeline, split_target, get_feature_names

RANDOM_SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
CV_FOLDS = 5


@dataclass
class TrainingResult:
    comparison: pd.DataFrame
    final_test_metrics: dict[str, Any]
    selected_model_name: str
    selected_threshold: float
    calibration_method: str
    artifact_path: Path
    report_dir: Path


def candidate_estimators() -> dict[str, Any]:
    """Return fresh estimators; no estimator is fitted by this function."""

    return {
        "logistic_regression": LogisticRegression(max_iter=3000, random_state=RANDOM_SEED),
        "logistic_regression_balanced": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "decision_tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_SEED),
        "decision_tree_balanced": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250, max_depth=10, n_jobs=-1, random_state=RANDOM_SEED
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=250, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=250, max_depth=10, n_jobs=-1, random_state=RANDOM_SEED
        ),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=250, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=15, random_state=RANDOM_SEED
        ),
    }


def make_pipeline(estimator: Any, excluded_columns: tuple[str, ...] = ()) -> Any:
    """Put preprocessing and estimator in one CV-safe pipeline."""

    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("preprocessing", build_preprocessing_pipeline(excluded_columns=excluded_columns)),
        ("model", estimator),
    ])


def make_smote_pipeline() -> Any:
    """Optional fold-local SMOTE experiment using the same preprocessing path."""

    preprocessing = build_preprocessing_pipeline()
    return ImbPipeline([
        # imbalanced-learn requires intermediate steps to be transformers, not
        # nested sklearn Pipelines, so flatten the shared two-step pipeline.
        ("feature_engineering", preprocessing.named_steps["feature_engineering"]),
        ("preprocessing", preprocessing.named_steps["preprocessing"]),
        ("smote", SMOTE(random_state=RANDOM_SEED)),
        ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_SEED)),
    ])


def metric_dict(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    report = classification_report(y_true, predictions, output_dict=True, zero_division=0)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, predictions)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "false_positive_count": int(fp),
        "false_negative_count": int(fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "classification_report": report,
    }


def select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Select an operating threshold using validation data only.

    F1 is the primary criterion, with recall as the tie-breaker. The returned
    table makes the threshold decision auditable.
    """

    rows = []
    for threshold in np.arange(0.10, 0.91, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": round(float(threshold), 2),
            "precision": precision_score(y_true, predictions, zero_division=0),
            "recall": recall_score(y_true, predictions, zero_division=0),
            "f1": f1_score(y_true, predictions, zero_division=0),
        })
    table = pd.DataFrame(rows)
    selected = table.sort_values(["f1", "recall", "precision", "threshold"], ascending=[False, False, False, True]).iloc[0]
    return float(selected["threshold"]), table


def _model_cv_row(name: str, pipeline: Any, X: pd.DataFrame, y: pd.Series, cv: Any) -> dict[str, Any]:
    started = time.perf_counter()
    scores = cross_validate(
        pipeline, X, y, cv=cv, scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc", "f1": "f1", "recall": "recall", "precision": "precision"},
        n_jobs=1, return_train_score=False, error_score="raise",
    )
    return {
        "model": name,
        "cv_pr_auc_mean": float(scores["test_pr_auc"].mean()),
        "cv_pr_auc_std": float(scores["test_pr_auc"].std()),
        "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
        "cv_f1_mean": float(scores["test_f1"].mean()),
        "cv_recall_mean": float(scores["test_recall"].mean()),
        "cv_precision_mean": float(scores["test_precision"].mean()),
        "fit_time_seconds": float(scores["fit_time"].sum()),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _calibration_choice(base_pipeline: Any, X_fit: pd.DataFrame, y_fit: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> tuple[str, Any, dict[str, float]]:
    candidates = {
        "none": base_pipeline,
        "sigmoid": CalibratedClassifierCV(estimator=clone(base_pipeline), method="sigmoid", cv=5),
        "isotonic": CalibratedClassifierCV(estimator=clone(base_pipeline), method="isotonic", cv=5),
    }
    scores: dict[str, float] = {}
    fitted: dict[str, Any] = {}
    for name, estimator in candidates.items():
        estimator.fit(X_fit, y_fit)
        scores[name] = float(brier_score_loss(y_val, estimator.predict_proba(X_val)[:, 1]))
        fitted[name] = estimator
    selected = min(scores, key=scores.get)
    return selected, fitted[selected], scores


def train_and_evaluate(data_path: str | Path = "data/insurance_claims.csv", report_dir: str | Path = "reports/checkpoint_c_v2", artifact_path: str | Path = "models/claim_risk_pipeline_v2.joblib") -> TrainingResult:
    """Train, evaluate, calibrate, and save one versioned model bundle."""

    data_path = Path(data_path)
    report_dir = Path(report_dir)
    artifact_path = Path(artifact_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {artifact_path}")

    frame = pd.read_csv(data_path)
    validation = validate_claims(frame, require_target=True)
    structural_errors = [e for e in validation.errors if e != "incident_date occurs before policy_bind_date"]
    if structural_errors:
        raise ValueError("Training data failed validation: " + "; ".join(structural_errors))
    X, y = split_target(frame)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_SIZE, stratify=y_train, random_state=RANDOM_SEED)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    rows = [_model_cv_row(name, make_pipeline(estimator), X_train, y_train, cv) for name, estimator in candidate_estimators().items()]
    smote_row = _model_cv_row("logistic_regression_smote", make_smote_pipeline(), X_train, y_train, cv)
    rows.append(smote_row)
    comparison = pd.DataFrame(rows).sort_values(["cv_pr_auc_mean", "cv_pr_auc_std"], ascending=[False, True]).reset_index(drop=True)
    selected_name = str(comparison.iloc[0]["model"])
    if selected_name == "logistic_regression_smote":
        selected_base = make_smote_pipeline()
    else:
        selected_base = make_pipeline(candidate_estimators()[selected_name])

    calibration_method, calibrated_model, calibration_scores = _calibration_choice(selected_base, X_fit, y_fit, X_val, y_val)
    val_probabilities = calibrated_model.predict_proba(X_val)[:, 1]
    threshold, threshold_table = select_threshold(y_val, val_probabilities)
    calibrated_model.fit(X_train, y_train)
    test_probabilities = calibrated_model.predict_proba(X_test)[:, 1]
    final_metrics = metric_dict(y_test, test_probabilities, threshold)
    final_metrics["calibration_validation_brier_scores"] = calibration_scores
    final_metrics["test_rows"] = len(X_test)
    final_metrics["train_rows"] = len(X_train)

    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    threshold_table.to_csv(report_dir / "threshold_validation.csv", index=False)
    pd.DataFrame({"actual": y_test.to_numpy(), "fraud_probability": test_probabilities, "predicted": (test_probabilities >= threshold).astype(int)}).to_csv(report_dir / "test_predictions.csv", index=False)
    pd.DataFrame(final_metrics["classification_report"]).T.reset_index(names="class").to_csv(report_dir / "classification_report.csv", index=False)
    pd.DataFrame(final_metrics["confusion_matrix"], index=["actual_non_fraud", "actual_fraud"], columns=["predicted_non_fraud", "predicted_fraud"]).to_csv(report_dir / "confusion_matrix.csv")
    _save_evaluation_plots(y_test, test_probabilities, threshold, report_dir)
    (report_dir / "final_test_metrics.json").write_text(json.dumps(final_metrics, indent=2, default=float), encoding="utf-8")
    (report_dir / "validation_summary.json").write_text(json.dumps({"validation": validation.__dict__, "selected_model": selected_name, "calibration_method": calibration_method}, indent=2, default=str), encoding="utf-8")

    fitted_base = calibrated_model
    if hasattr(calibrated_model, "calibrated_classifiers_"):
        fitted_base = calibrated_model.calibrated_classifiers_[0].estimator
    bundle = {
        "model": calibrated_model,
        "threshold": threshold,
        "selected_model_name": selected_name,
        "calibration_method": calibration_method,
        "feature_names": get_feature_names(fitted_base),
        "random_seed": RANDOM_SEED,
        "target_column": "fraud_reported",
        "risk_bands": {"High": threshold, "Medium": max(0.30, threshold / 2), "Low": 0.0},
        "metrics": final_metrics,
        "library_versions": {"python": platform.python_version(), "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)
    return TrainingResult(comparison, final_metrics, selected_name, threshold, calibration_method, artifact_path, report_dir)


def _save_evaluation_plots(y_test: pd.Series, probabilities: np.ndarray, threshold: float, report_dir: Path) -> None:
    from sklearn.calibration import CalibrationDisplay
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, probabilities >= threshold, labels=[0, 1])
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-fraud", "Fraud"], yticklabels=["Non-fraud", "Fraud"], xlabel="Predicted", ylabel="Actual", title="Held-out test confusion matrix")
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, int(value), ha="center", va="center")
    fig.tight_layout(); fig.savefig(report_dir / "confusion_matrix.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4)); RocCurveDisplay.from_predictions(y_test, probabilities, ax=ax); ax.set_title("Held-out test ROC curve"); fig.tight_layout(); fig.savefig(report_dir / "roc_curve.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4)); PrecisionRecallDisplay.from_predictions(y_test, probabilities, ax=ax); ax.set_title("Held-out test precision-recall curve"); fig.tight_layout(); fig.savefig(report_dir / "precision_recall_curve.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4)); CalibrationDisplay.from_predictions(y_test, probabilities, n_bins=8, strategy="quantile", ax=ax); ax.set_title("Held-out test calibration"); fig.tight_layout(); fig.savefig(report_dir / "calibration_curve.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    result = train_and_evaluate()
    print(result.comparison.to_string(index=False))
    print(json.dumps({"selected_model": result.selected_model_name, "threshold": result.selected_threshold, "calibration": result.calibration_method, "test": result.final_test_metrics}, indent=2, default=float))

