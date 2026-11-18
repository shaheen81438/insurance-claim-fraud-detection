import pandas as pd

from src.data_validation import TARGET_COLUMN, validate_claims


def test_raw_dataset_schema_and_target():
    frame = pd.read_csv("data/insurance_claims.csv")
    report = validate_claims(frame, require_target=True)
    # The raw source contains one genuine date-order anomaly; validation must
    # surface it instead of silently changing the source record.
    assert not report.valid
    assert any("incident_date occurs before" in error for error in report.errors)
    assert frame.shape == (1000, 39)
    assert TARGET_COLUMN in frame.columns
    assert frame[TARGET_COLUMN].value_counts().to_dict() == {"N": 753, "Y": 247}


def test_missing_tokens_are_reported_not_silently_dropped():
    frame = pd.read_csv("data/insurance_claims.csv")
    report = validate_claims(frame, require_target=True)
    assert report.missing_value_tokens == {
        "collision_type": 178,
        "property_damage": 360,
        "police_report_available": 343,
    }


def test_invalid_target_and_missing_required_column_are_rejected():
    frame = pd.read_csv("data/insurance_claims.csv").drop(columns=["age"])
    frame[TARGET_COLUMN] = "maybe"
    report = validate_claims(frame, require_target=True)
    assert not report.valid
    assert "age" in report.missing_columns
    assert any("Invalid fraud_reported values" in error for error in report.errors)


def test_invalid_date_order_is_rejected():
    frame = pd.read_csv("data/insurance_claims.csv").iloc[:2].copy()
    frame.loc[frame.index[0], "incident_date"] = "1990-01-01"
    report = validate_claims(frame, require_target=True)
    assert not report.valid
    assert any("incident_date occurs before" in error for error in report.errors)

