import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.explainability import explain_claims, plain_language_explanation, risk_category
from src.predict import load_bundle, predict_claims


ARTIFACT = Path("models/claim_risk_pipeline_v2.joblib")


def valid_claims(n=2):
    frame = pd.read_csv("data/insurance_claims.csv").drop(columns=["fraud_reported"])
    frame["_bind"] = pd.to_datetime(frame["policy_bind_date"])
    frame["_incident"] = pd.to_datetime(frame["incident_date"])
    return frame.loc[frame["_incident"] >= frame["_bind"]].drop(columns=["_bind", "_incident"]).head(n).reset_index(drop=True)


def test_versioned_artifact_loads_and_prediction_is_consistent():
    bundle = load_bundle(ARTIFACT)
    result = predict_claims(bundle, valid_claims())
    assert len(result) == 2
    assert result["fraud_probability"].between(0, 1).all()
    assert set(result["risk_category"]).issubset({"Low", "Medium", "High"})
    assert bundle["threshold"] == 0.18


def test_invalid_date_input_is_rejected():
    bundle = load_bundle(ARTIFACT)
    claim = valid_claims(1)
    claim.loc[0, "incident_date"] = "1990-01-01"
    try:
        predict_claims(bundle, claim)
    except ValueError as error:
        assert "incident_date occurs before" in str(error)
    else:
        raise AssertionError("Invalid date input was accepted")


def test_shap_explanation_aligns_with_transformed_features():
    bundle = load_bundle(ARTIFACT)
    explanation = explain_claims(bundle, valid_claims(1))
    assert explanation["values"].shape[1] == len(bundle["feature_names"])
    assert explanation["values"].shape[1] == len(explanation["feature_names"])
    assert explanation["global_importance"].iloc[0]["mean_abs_shap"] >= 0
    assert "not proof" in plain_language_explanation(0.8, explanation["local"][0], bundle["threshold"])

