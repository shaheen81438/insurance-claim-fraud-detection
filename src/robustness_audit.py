"""Read-only robustness and statistical audit for the frozen claim-risk model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from .modeling import RANDOM_SEED, TEST_SIZE, CV_FOLDS, make_pipeline, metric_dict
from .preprocessing import split_target


def _bootstrap_metric_row(name: str, values: np.ndarray) -> dict[str, float | str]:
    return {"metric": name, "lower_2.5": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "upper_97.5": float(np.quantile(values, 0.975))}


def bootstrap_test_intervals(y: pd.Series, probabilities: np.ndarray, threshold: float,
                             n_bootstrap: int = 5000) -> pd.DataFrame:
    """Stratified bootstrap: resample non-fraud and fraud test rows separately."""
    y_np = np.asarray(y, dtype=int)
    negative = np.flatnonzero(y_np == 0)
    positive = np.flatnonzero(y_np == 1)
    rng = np.random.default_rng(RANDOM_SEED)
    values = {key: [] for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score", "specificity")}
    for _ in range(n_bootstrap):
        sample = np.concatenate([
            rng.choice(negative, size=len(negative), replace=True),
            rng.choice(positive, size=len(positive), replace=True),
        ])
        ys = y_np[sample]
        ps = probabilities[sample]
        pred = (ps >= threshold).astype(int)
        tn, fp, _, _ = confusion_matrix(ys, pred, labels=[0, 1]).ravel()
        values["accuracy"].append(accuracy_score(ys, pred))
        values["precision"].append(precision_score(ys, pred, zero_division=0))
        values["recall"].append(recall_score(ys, pred, zero_division=0))
        values["f1"].append(f1_score(ys, pred, zero_division=0))
        values["roc_auc"].append(roc_auc_score(ys, ps))
        values["pr_auc"].append(average_precision_score(ys, ps))
        values["brier_score"].append(brier_score_loss(ys, ps))
        values["specificity"].append(tn / (tn + fp) if (tn + fp) else 0.0)
    return pd.DataFrame([_bootstrap_metric_row(k, np.asarray(v)) for k, v in values.items()])


def fold_metrics(X_train: pd.DataFrame, y_train: pd.Series, threshold: float) -> pd.DataFrame:
    """Base-estimator fold metrics at the frozen operational threshold."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    estimator = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, random_state=RANDOM_SEED)
    rows = []
    for fold, (fit_idx, valid_idx) in enumerate(cv.split(X_train, y_train), start=1):
        pipeline = make_pipeline(clone(estimator))
        pipeline.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        probabilities = pipeline.predict_proba(X_train.iloc[valid_idx])[:, 1]
        metrics = metric_dict(y_train.iloc[valid_idx], probabilities, threshold)
        rows.append({"model": "hist_gradient_boosting", "fold": fold, **{k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "specificity")}})
    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "insurance_claims.csv")
    X, y = split_target(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)
    bundle = joblib.load(root / "models" / "claim_risk_pipeline_v2.joblib")
    threshold = float(bundle["threshold"])
    probabilities = bundle["model"].predict_proba(X_test)[:, 1]
    out = root / "reports" / "checkpoint_g"
    out.mkdir(parents=True, exist_ok=True)
    bootstrap_test_intervals(y_test, probabilities, threshold).to_csv(out / "heldout_bootstrap_ci.csv", index=False)
    fold_metrics(X_train, y_train, threshold).to_csv(out / "hgb_cv_fold_metrics.csv", index=False)
    predictions = (probabilities >= threshold).astype(int)
    summary = metric_dict(y_test, probabilities, threshold)
    summary.update({"test_rows": len(X_test), "test_fraud_rows": int(y_test.sum()), "bootstrap_resamples": 5000, "seed": RANDOM_SEED})
    (out / "recalculated_production_metrics.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    (out / "audit_config.json").write_text(json.dumps({"random_seed": RANDOM_SEED, "test_size": TEST_SIZE, "cv_folds": CV_FOLDS, "threshold": threshold, "artifact": "models/claim_risk_pipeline_v2.joblib"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
