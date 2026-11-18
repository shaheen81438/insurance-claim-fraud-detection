import numpy as np
import pandas as pd

from src.feature_engineering import ClaimFeatureEngineer, ENGINEERED_FEATURES
from src.preprocessing import build_preprocessing_pipeline, get_feature_names, split_target


def raw_frame():
    return pd.read_csv("data/insurance_claims.csv")


def test_feature_engineering_creates_documented_features_and_drops_identifiers():
    frame = raw_frame()
    transformed = ClaimFeatureEngineer().fit_transform(frame)
    assert set(ENGINEERED_FEATURES).issubset(transformed.columns)
    assert "fraud_reported" not in transformed.columns
    assert not {"policy_number", "insured_zip", "incident_location"}.intersection(transformed.columns)
    assert transformed["is_high_value_claim"].isin([0, 1]).all()


def test_high_value_threshold_is_fit_on_training_rows_only():
    frame = raw_frame()
    train = frame.iloc[:800]
    engineer = ClaimFeatureEngineer().fit(train)
    expected = pd.to_numeric(train["total_claim_amount"]).quantile(0.90)
    assert engineer.high_value_threshold_ == expected


def test_pipeline_fits_and_transforms_same_way_for_train_and_inference():
    frame = raw_frame()
    X, y = split_target(frame)
    pipeline = build_preprocessing_pipeline()
    train_matrix = pipeline.fit_transform(X.iloc[:800], y.iloc[:800])
    inference_matrix = pipeline.transform(X.iloc[800:])
    assert train_matrix.shape[1] == inference_matrix.shape[1]
    assert train_matrix.shape[0] == 800
    assert inference_matrix.shape[0] == 200
    assert np.isfinite(train_matrix).all()
    assert np.isfinite(inference_matrix).all()
    assert len(get_feature_names(pipeline)) == train_matrix.shape[1]


def test_unknown_category_is_accepted_after_fit():
    frame = raw_frame()
    X, y = split_target(frame)
    pipeline = build_preprocessing_pipeline().fit(X.iloc[:800], y.iloc[:800])
    altered = X.iloc[800:801].copy()
    altered.loc[altered.index[0], "auto_model"] = "Unseen model"
    result = pipeline.transform(altered)
    assert result.shape[0] == 1

