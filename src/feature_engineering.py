"""Deterministic, fit-aware feature engineering for insurance claims."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .data_validation import DATE_COLUMNS, TARGET_COLUMN


ENGINEERED_FEATURES = (
    "policy_age_days",
    "customer_tenure_years",
    "claim_amount_per_tenure_year",
    "is_high_value_claim",
    "vehicle_claim_ratio",
    "injury_claim_ratio",
    "property_claim_ratio",
    "missing_doc_count",
    "is_odd_hour_incident",
    "vehicle_age_at_incident",
    "claim_amount_to_annual_premium",
    "claim_component_gap",
)


class ClaimFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create claim features and remove unavailable/identifier fields.

    The high-value threshold is fitted on each training fold. All other
    transformations are deterministic formulas based only on the current row.
    """

    def __init__(self, high_value_quantile: float = 0.90, excluded_columns: tuple[str, ...] = ()) -> None:
        self.high_value_quantile = high_value_quantile
        self.excluded_columns = excluded_columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ClaimFeatureEngineer":
        frame = self._validate_frame(X)
        amounts = pd.to_numeric(frame["total_claim_amount"], errors="coerce").dropna()
        if amounts.empty:
            raise ValueError("total_claim_amount must contain at least one numeric value")
        self.high_value_threshold_ = float(amounts.quantile(self.high_value_quantile))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "high_value_threshold_"):
            raise RuntimeError("ClaimFeatureEngineer must be fitted before transform")
        frame = self._validate_frame(X).copy()
        frame = frame.replace("?", np.nan)

        bind = pd.to_datetime(frame["policy_bind_date"], errors="coerce")
        incident = pd.to_datetime(frame["incident_date"], errors="coerce")
        frame["policy_age_days"] = (incident - bind).dt.days
        frame["customer_tenure_years"] = self._number(frame, "months_as_customer") / 12.0

        total = self._number(frame, "total_claim_amount")
        tenure = frame["customer_tenure_years"].replace(0, np.nan)
        frame["claim_amount_per_tenure_year"] = total / tenure
        frame["is_high_value_claim"] = (total >= self.high_value_threshold_).astype(int)

        for part in ("vehicle_claim", "injury_claim", "property_claim"):
            frame[f"{part}_ratio"] = self._number(frame, part) / total.replace(0, np.nan)

        frame["missing_doc_count"] = (
            frame["police_report_available"].isna().astype(int)
            + frame["property_damage"].isna().astype(int)
            + (self._number(frame, "witnesses") == 0).astype(int)
        )
        hour = self._number(frame, "incident_hour_of_the_day")
        frame["is_odd_hour_incident"] = ((hour <= 5) | (hour >= 22)).astype(int)
        frame["vehicle_age_at_incident"] = incident.dt.year - self._number(frame, "auto_year")
        premium = self._number(frame, "policy_annual_premium").replace(0, np.nan)
        frame["claim_amount_to_annual_premium"] = total / premium
        components = sum(self._number(frame, column) for column in ("injury_claim", "property_claim", "vehicle_claim"))
        frame["claim_component_gap"] = total - components

        drop_columns = {
            TARGET_COLUMN,
            "fraud_flag",
            *DATE_COLUMNS,
            "policy_number",
            "insured_zip",
            "incident_location",
        }
        # Older versioned artifacts predate optional feature ablations and do
        # not carry this attribute after unpickling; preserve their behavior.
        drop_columns.update(getattr(self, "excluded_columns", ()))
        return frame.drop(columns=[column for column in drop_columns if column in frame.columns])

    @staticmethod
    def _number(frame: pd.DataFrame, column: str) -> pd.Series:
        return pd.to_numeric(frame[column], errors="coerce")

    @staticmethod
    def _validate_frame(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ClaimFeatureEngineer expects a pandas.DataFrame")
        required = [
            "months_as_customer", "policy_bind_date", "incident_date",
            "total_claim_amount", "vehicle_claim", "injury_claim",
            "property_claim", "police_report_available", "property_damage",
            "witnesses", "incident_hour_of_the_day", "auto_year",
            "policy_annual_premium",
        ]
        missing = [column for column in required if column not in X.columns]
        if missing:
            raise ValueError(f"Missing feature-engineering columns: {', '.join(missing)}")
        return X


def feature_dictionary() -> pd.DataFrame:
    """Return the auditable feature dictionary used by the project."""

    rows = [
        ("policy_age_days", "incident_date - policy_bind_date", "Policy tenure at incident time."),
        ("customer_tenure_years", "months_as_customer / 12", "Normalizes customer tenure."),
        ("claim_amount_per_tenure_year", "total_claim_amount / customer_tenure_years", "Claim size relative to tenure."),
        ("is_high_value_claim", "total_claim_amount >= training 90th percentile", "Flags unusually large claims without test-set fitting."),
        ("vehicle_claim_ratio", "vehicle_claim / total_claim_amount", "Vehicle share of the claim."),
        ("injury_claim_ratio", "injury_claim / total_claim_amount", "Injury share of the claim."),
        ("property_claim_ratio", "property_claim / total_claim_amount", "Property share of the claim."),
        ("missing_doc_count", "missing police report + missing property damage + zero witnesses", "Counts absent supporting evidence signals."),
        ("is_odd_hour_incident", "incident hour <= 5 or >= 22", "Flags incidents at unusual hours."),
        ("vehicle_age_at_incident", "incident year - auto_year", "Approximate insured vehicle age."),
        ("claim_amount_to_annual_premium", "total_claim_amount / policy_annual_premium", "Claim size relative to annual premium."),
        ("claim_component_gap", "total_claim_amount - (injury + property + vehicle)", "Detects amount reconciliation inconsistencies."),
    ]
    return pd.DataFrame(rows, columns=["feature", "formula", "justification"])

