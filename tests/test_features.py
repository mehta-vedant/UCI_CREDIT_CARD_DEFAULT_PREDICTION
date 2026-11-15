import pandas as pd
import pytest

from src.credit_risk.config import RAW_FEATURE_COLUMNS
from src.credit_risk.features import CreditRiskFeatureEngineer, validate_features


def valid_row() -> dict:
    return {
        "LIMIT_BAL": 100000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": -1, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        **{f"BILL_AMT{i}": 10000 for i in range(1, 7)},
        **{f"PAY_AMT{i}": 2000 for i in range(1, 7)},
    }


def test_feature_engineer_creates_expected_values():
    transformed = CreditRiskFeatureEngineer().fit_transform(pd.DataFrame([valid_row()]))
    assert transformed.loc[0, "credit_utilization"] == pytest.approx(0.1)
    assert transformed.loc[0, "payment_ratio_1"] == pytest.approx(0.2)
    assert transformed.loc[0, "months_with_delays"] == 0


def test_validation_rejects_invalid_age():
    row = valid_row()
    row["AGE"] = 12
    with pytest.raises(ValueError, match="AGE"):
        validate_features(pd.DataFrame([row]))


def test_validation_returns_exact_model_input_order():
    validated = validate_features(pd.DataFrame([valid_row()]))
    assert list(validated.columns) == RAW_FEATURE_COLUMNS


def test_validation_rejects_non_finite_and_invalid_source_codes():
    row = valid_row()
    row["LIMIT_BAL"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        validate_features(pd.DataFrame([row]))

    row = valid_row()
    row["PAY_0"] = 99
    with pytest.raises(ValueError, match="PAY_0"):
        validate_features(pd.DataFrame([row]))
