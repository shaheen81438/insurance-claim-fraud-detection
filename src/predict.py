"""Inference helpers for the saved versioned claim-risk bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .data_validation import validate_claims
from .explainability import risk_category


def load_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    required = {"model", "threshold", "feature_names", "selected_model_name"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Artifact is missing metadata: {sorted(missing)}")
    return bundle


def predict_claims(bundle: dict[str, Any], claims: pd.DataFrame) -> pd.DataFrame:
    report = validate_claims(claims, require_target=False)
    if report.errors:
        raise ValueError("Invalid claim input: " + "; ".join(report.errors))
    probabilities = bundle["model"].predict_proba(claims)[:, 1]
    result = claims.copy()
    result["fraud_probability"] = probabilities
    result["risk_category"] = [risk_category(float(p), float(bundle["threshold"])) for p in probabilities]
    return result

