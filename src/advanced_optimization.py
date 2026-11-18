"""Checkpoint F experiments; never overwrites the current production bundle."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split

from .data_validation import validate_claims
from .modeling import RANDOM_SEED, metric_dict, make_pipeline, select_threshold
from .preprocessing import split_target

OUT = Path("reports/checkpoint_f")
PRODUCTION_ARTIFACT = Path("models/claim_risk_pipeline_v2.joblib")


def threshold_table(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.01):
        predictions = probabilities >= round(float(threshold), 2)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        rows.append({
            "threshold": round(float(threshold), 2),
            "accuracy": accuracy_score(y_true, predictions),
            "precision": precision_score(y_true, predictions, zero_division=0),
            "recall": recall_score(y_true, predictions, zero_division=0),
            "f1": f1_score(y_true, predictions, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })
    return pd.DataFrame(rows)


def cv_summary(name: str, estimator: Any, X: pd.DataFrame, y: pd.Series, cv: Any) -> dict[str, Any]:
    scores = cross_validate(
        estimator, X, y, cv=cv,
        scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc", "accuracy": "accuracy", "f1": "f1", "recall": "recall", "precision": "precision"},
        n_jobs=1, error_score="raise",
    )
    return {
        "model": name,
        "cv_pr_auc_mean": float(scores["test_pr_auc"].mean()),
        "cv_pr_auc_std": float(scores["test_pr_auc"].std()),
        "cv_pr_auc_folds": [float(x) for x in scores["test_pr_auc"]],
        "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
        "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
        "cv_f1_mean": float(scores["test_f1"].mean()),
        "cv_recall_mean": float(scores["test_recall"].mean()),
        "cv_precision_mean": float(scores["test_precision"].mean()),
    }


def test_metrics(estimator: Any, X_fit: pd.DataFrame, y_fit: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict[str, Any], float, pd.DataFrame]:
    validation_model = clone(estimator).fit(X_fit, y_fit)
    validation_probabilities = validation_model.predict_proba(X_val)[:, 1]
    threshold, _ = select_threshold(y_val, validation_probabilities)
    fitted = clone(estimator).fit(pd.concat([X_fit, X_val]), pd.concat([y_fit, y_val]))
    test_probabilities = fitted.predict_proba(X_test)[:, 1]
    return metric_dict(y_test, test_probabilities, threshold), threshold, threshold_table(y_val, validation_probabilities)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv("data/insurance_claims.csv")
    validation = validate_claims(raw, require_target=True)
    X, y = split_target(raw)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED)
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        make_pipeline(HistGradientBoostingClassifier(random_state=RANDOM_SEED)),
        param_distributions={
            "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
            "model__max_iter": [100, 150, 250, 350],
            "model__max_leaf_nodes": [7, 15, 31, 63],
            "model__max_depth": [None, 3, 5, 8],
            "model__min_samples_leaf": [10, 20, 30, 50],
            "model__l2_regularization": [0.0, 0.1, 1.0, 5.0],
        },
        n_iter=12, scoring="average_precision", cv=cv, random_state=RANDOM_SEED, n_jobs=1, refit=True,
    )
    started = time.perf_counter(); search.fit(X_train, y_train); search_seconds = time.perf_counter() - started
    tuned = search.best_estimator_
    tuned_cv = cv_summary("hist_gradient_boosting_tuned", tuned, X_train, y_train, cv)

    candidates: dict[str, Any] = {
        "hist_gradient_boosting_default": make_pipeline(HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, random_state=RANDOM_SEED)),
        "hist_gradient_boosting_tuned": tuned,
        "extra_trees_balanced": make_pipeline(ExtraTreesClassifier(n_estimators=250, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)),
        "random_forest_balanced": make_pipeline(RandomForestClassifier(n_estimators=250, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)),
        "logistic_regression_balanced": make_pipeline(LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_SEED)),
    }
    try:
        from xgboost import XGBClassifier
        candidates["xgboost"] = make_pipeline(XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", tree_method="hist", n_jobs=1, random_state=RANDOM_SEED))
        optional = {"xgboost": "tested", "catboost": "not installed", "lightgbm": "not installed"}
    except ImportError:
        optional = {"xgboost": "unavailable", "catboost": "not installed", "lightgbm": "not installed"}

    rows = []
    threshold_tables = []
    for name, estimator in candidates.items():
        summary = tuned_cv if name == "hist_gradient_boosting_tuned" else cv_summary(name, estimator, X_train, y_train, cv)
        metrics, threshold, table = test_metrics(estimator, X_fit, y_fit, X_val, y_val, X_test, y_test)
        summary.update({f"test_{key}": value for key, value in metrics.items() if key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score", "specificity"]})
        summary["selected_validation_threshold"] = threshold
        rows.append(summary)
        table.insert(0, "model", name); threshold_tables.append(table)
    comparison = pd.DataFrame(rows).sort_values(["cv_pr_auc_mean", "cv_pr_auc_std"], ascending=[False, True])
    comparison.to_csv(OUT / "experiment_comparison.csv", index=False)
    pd.concat(threshold_tables, ignore_index=True).to_csv(OUT / "threshold_analysis.csv", index=False)
    (OUT / "hgb_random_search.json").write_text(json.dumps({"best_params": search.best_params_, "best_cv_pr_auc": search.best_score_, "search_seconds": search_seconds}, indent=2, default=float), encoding="utf-8")

    ablations = {
        "full_features": (),
        "without_insured_hobbies": ("insured_hobbies",),
        "compact_claim_context": ("insured_hobbies", "insured_occupation", "insured_relationship", "auto_model"),
    }
    ablation_rows = []
    for name, excluded in ablations.items():
        estimator = make_pipeline(HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, random_state=RANDOM_SEED), excluded_columns=excluded)
        row = cv_summary(name, estimator, X_train, y_train, cv); row["excluded_columns"] = ",".join(excluded); ablation_rows.append(row)
    pd.DataFrame(ablation_rows).to_csv(OUT / "feature_ablation.csv", index=False)

    production = joblib.load(PRODUCTION_ARTIFACT)
    baseline_metrics = production["metrics"]
    tuned_metrics = next(row for row in rows if row["model"] == "hist_gradient_boosting_tuned")
    (OUT / "audit_summary.json").write_text(json.dumps({
        "validation": validation.__dict__, "optional_packages": optional,
        "production_artifact": str(PRODUCTION_ARTIFACT), "production_threshold": production["threshold"],
        "production_test_metrics": {k: baseline_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score", "specificity"]},
        "tuned_result": tuned_metrics, "test_rows": len(X_test), "validation_rows": len(X_val),
    }, indent=2, default=str), encoding="utf-8")
    print(comparison[["model", "cv_pr_auc_mean", "cv_pr_auc_std", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc", "test_pr_auc", "selected_validation_threshold"]].to_string(index=False))
    print("best_params", search.best_params_)
    print("optional_packages", optional)


if __name__ == "__main__":
    run()

