"""Schema and data-quality validation for insurance claims.

The validator is deliberately independent of model training. It accepts raw
claims with or without the target column, so the same checks can be used for
training data and uploaded inference data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


TARGET_COLUMN = "fraud_reported"
RAW_COLUMNS = (
    "months_as_customer",
    "age",
    "policy_number",
    "policy_bind_date",
    "policy_state",
    "policy_csl",
    "policy_deductable",
    "policy_annual_premium",
    "umbrella_limit",
    "insured_zip",
    "insured_sex",
    "insured_education_level",
    "insured_occupation",
    "insured_hobbies",
    "insured_relationship",
    "capital-gains",
    "capital-loss",
    "incident_date",
    "incident_type",
    "collision_type",
    "incident_severity",
    "authorities_contacted",
    "incident_state",
    "incident_city",
    "incident_location",
    "incident_hour_of_the_day",
    "number_of_vehicles_involved",
    "property_damage",
    "bodily_injuries",
    "witnesses",
    "police_report_available",
    "total_claim_amount",
    "injury_claim",
    "property_claim",
    "vehicle_claim",
    "auto_make",
    "auto_model",
    "auto_year",
    TARGET_COLUMN,
)

MISSING_TOKEN_COLUMNS = (
    "collision_type",
    "property_damage",
    "police_report_available",
)
IDENTIFIER_COLUMNS = ("policy_number", "insured_zip", "incident_location")
DATE_COLUMNS = ("policy_bind_date", "incident_date")
EXPECTED_TARGET_VALUES = frozenset({"Y", "N"})


@dataclass(frozen=True)
class ValidationReport:
    """Structured result of validating a claim dataframe."""

    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    missing_columns: tuple[str, ...] = ()
    unexpected_columns: tuple[str, ...] = ()
    missing_value_tokens: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    row_count: int = 0
    column_count: int = 0


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric view, coercing the dataset's missing token to NaN."""

    return pd.to_numeric(frame[column].replace("?", pd.NA), errors="coerce")


def validate_claims(
    frame: pd.DataFrame,
    *,
    require_target: bool = False,
    allow_extra_columns: bool = True,
) -> ValidationReport:
    """Validate raw claim schema and basic domain constraints.

    Parameters
    ----------
    frame:
        Raw claim records. The target is optional for inference.
    require_target:
        Require ``fraud_reported`` and validate its values when true.
    allow_extra_columns:
        Preserve forward compatibility with benign extra input columns while
        reporting them in the result.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas.DataFrame")

    required = list(RAW_COLUMNS if require_target else RAW_COLUMNS[:-1])
    missing = tuple(column for column in required if column not in frame.columns)
    unexpected = tuple(column for column in frame.columns if column not in RAW_COLUMNS)
    errors: list[str] = []
    warnings: list[str] = []

    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
    if unexpected and not allow_extra_columns:
        errors.append(f"Unexpected columns: {', '.join(unexpected)}")
    elif unexpected:
        warnings.append(f"Ignoring extra columns: {', '.join(unexpected)}")

    token_counts: dict[str, int] = {}
    for column in MISSING_TOKEN_COLUMNS:
        if column in frame.columns:
            count = int(frame[column].eq("?").sum())
            if count:
                token_counts[column] = count

    duplicates = int(frame.duplicated().sum())
    if duplicates:
        warnings.append(f"Found {duplicates} duplicate rows")

    if TARGET_COLUMN in frame.columns:
        target_values = set(frame[TARGET_COLUMN].dropna().unique())
        invalid_target = target_values - EXPECTED_TARGET_VALUES
        if invalid_target:
            errors.append(f"Invalid {TARGET_COLUMN} values: {sorted(invalid_target)}")
    elif require_target:
        errors.append(f"Missing target column: {TARGET_COLUMN}")

    if "policy_bind_date" in frame.columns and "incident_date" in frame.columns:
        bind = pd.to_datetime(frame["policy_bind_date"], errors="coerce")
        incident = pd.to_datetime(frame["incident_date"], errors="coerce")
        if int(bind.isna().sum()) or int(incident.isna().sum()):
            errors.append("Date columns contain unparsable values")
        elif bool((incident < bind).any()):
            errors.append("incident_date occurs before policy_bind_date")

    numeric_ranges: dict[str, tuple[float, float]] = {
        "age": (0, 120),
        "months_as_customer": (0, 1200),
        "incident_hour_of_the_day": (0, 23),
        "number_of_vehicles_involved": (1, 100),
        "bodily_injuries": (0, 100),
        "witnesses": (0, 100),
        "auto_year": (1886, 2100),
        "total_claim_amount": (0, float("inf")),
        "injury_claim": (0, float("inf")),
        "property_claim": (0, float("inf")),
        "vehicle_claim": (0, float("inf")),
    }
    for column, (lower, upper) in numeric_ranges.items():
        if column not in frame.columns:
            continue
        values = _numeric_series(frame, column).dropna()
        if bool((values < lower).any()) or bool((values > upper).any()):
            errors.append(f"{column} contains values outside [{lower}, {upper}]")

    components = ["injury_claim", "property_claim", "vehicle_claim"]
    if all(column in frame.columns for column in ["total_claim_amount", *components]):
        total = _numeric_series(frame, "total_claim_amount")
        component_sum = sum((_numeric_series(frame, column) for column in components))
        mismatch = (total - component_sum).abs() > 1e-6
        if int(mismatch.sum()):
            warnings.append(
                f"Claim components do not reconcile to total_claim_amount for {int(mismatch.sum())} rows"
            )

    valid = not errors and not (missing and require_target is False and any(c not in frame.columns for c in required))
    return ValidationReport(
        valid=valid,
        errors=tuple(errors),
        warnings=tuple(warnings),
        missing_columns=missing,
        unexpected_columns=unexpected,
        missing_value_tokens=token_counts,
        duplicate_rows=duplicates,
        row_count=len(frame),
        column_count=len(frame.columns),
    )


def validate_or_raise(frame: pd.DataFrame, *, require_target: bool = False) -> ValidationReport:
    """Validate claims and raise a useful ``ValueError`` when invalid."""

    report = validate_claims(frame, require_target=require_target)
    if not report.valid:
        raise ValueError("Invalid claims data: " + "; ".join(report.errors))
    return report

