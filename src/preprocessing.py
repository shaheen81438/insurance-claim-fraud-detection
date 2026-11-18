"""Shared leakage-safe preprocessing pipeline for claims."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_engineering import ClaimFeatureEngineer


def split_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate raw claim features and the binary target without fitting anything."""

    if "fraud_reported" not in frame.columns:
        raise ValueError("Training data must contain fraud_reported")
    target = frame["fraud_reported"].map({"N": 0, "Y": 1})
    if target.isna().any():
        raise ValueError("fraud_reported must contain only N and Y")
    return frame.drop(columns=["fraud_reported"]), target.astype("int8")


def build_preprocessing_pipeline(excluded_columns: tuple[str, ...] = ()) -> Pipeline:
    """Build a fresh pipeline; fitting is intentionally deferred to the caller."""

    numeric_columns = [
        "months_as_customer", "age", "policy_deductable", "policy_annual_premium",
        "umbrella_limit", "capital-gains", "capital-loss", "incident_hour_of_the_day",
        "number_of_vehicles_involved", "bodily_injuries", "witnesses", "total_claim_amount",
        "injury_claim", "property_claim", "vehicle_claim", "auto_year",
        "policy_age_days", "customer_tenure_years", "claim_amount_per_tenure_year",
        "is_high_value_claim", "vehicle_claim_ratio", "injury_claim_ratio",
        "property_claim_ratio", "missing_doc_count", "is_odd_hour_incident",
        "vehicle_age_at_incident", "claim_amount_to_annual_premium", "claim_component_gap",
    ]
    categorical_columns = [
        "policy_state", "policy_csl", "insured_sex", "insured_education_level",
        "insured_occupation", "insured_hobbies", "insured_relationship", "incident_type",
        "collision_type", "incident_severity", "authorities_contacted", "incident_state",
        "incident_city", "property_damage", "police_report_available", "auto_make", "auto_model",
    ]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    excluded = set(excluded_columns)
    transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, [c for c in numeric_columns if c not in excluded]),
            ("categorical", categorical_pipeline, [c for c in categorical_columns if c not in excluded]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("feature_engineering", ClaimFeatureEngineer(excluded_columns=excluded_columns)),
        ("preprocessing", transformer),
    ])


def get_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """Return transformed feature names from a fitted preprocessing pipeline."""

    if not hasattr(fitted_pipeline, "named_steps"):
        raise TypeError("Expected a fitted sklearn Pipeline")
    transformer = fitted_pipeline.named_steps["preprocessing"]
    # The training pipeline wraps this shared preprocessing pipeline together
    # with the estimator; the SMOTE branch flattens it into a ColumnTransformer.
    if isinstance(transformer, Pipeline) and "preprocessing" in transformer.named_steps:
        transformer = transformer.named_steps["preprocessing"]
    return list(transformer.get_feature_names_out())

