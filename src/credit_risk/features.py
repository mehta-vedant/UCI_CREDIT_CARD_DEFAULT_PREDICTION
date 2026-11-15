from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RAW_FEATURE_COLUMNS


class CreditRiskFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create reproducible, past-only features from the supplied six-month history."""

    payment_status_columns = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill_columns = [f"BILL_AMT{i}" for i in range(1, 7)]
    payment_columns = [f"PAY_AMT{i}" for i in range(1, 7)]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        frame["EDUCATION"] = frame["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
        frame["MARRIAGE"] = frame["MARRIAGE"].replace({0: 3})

        frame["credit_utilization"] = np.where(
            frame["LIMIT_BAL"] > 0, frame["BILL_AMT1"] / frame["LIMIT_BAL"], 0.0
        )
        for month in range(1, 7):
            bill = frame[f"BILL_AMT{month}"]
            payment = frame[f"PAY_AMT{month}"]
            frame[f"payment_ratio_{month}"] = np.where(bill > 0, payment / bill, 0.0)

        statuses = frame[self.payment_status_columns].astype(float)
        frame["avg_payment_status"] = statuses.mean(axis=1)
        frame["max_payment_delay"] = statuses.max(axis=1)
        frame["months_with_delays"] = (statuses > 0).sum(axis=1)
        frame["ever_paid_early"] = (statuses < 0).any(axis=1).astype(int)
        frame["avg_bill_amount"] = frame[self.bill_columns].mean(axis=1)
        frame["avg_payment_amount"] = frame[self.payment_columns].mean(axis=1)
        frame["bill_amount_trend"] = frame[self.bill_columns].apply(self._slope, axis=1)
        frame["payment_amount_trend"] = frame[self.payment_columns].apply(self._slope, axis=1)
        frame["high_utilization"] = (frame["credit_utilization"] > 0.8).astype(int)
        return frame

    @staticmethod
    def _slope(values: pd.Series) -> float:
        return float(np.polyfit(np.arange(len(values)), values.astype(float), 1)[0])


def build_preprocessor() -> Pipeline:
    numerical = [
        "LIMIT_BAL", "AGE", *CreditRiskFeatureEngineer.payment_status_columns,
        *CreditRiskFeatureEngineer.bill_columns, *CreditRiskFeatureEngineer.payment_columns,
        "credit_utilization", "avg_payment_status", "max_payment_delay", "months_with_delays",
        "avg_bill_amount", "avg_payment_amount", "bill_amount_trend", "payment_amount_trend",
        *[f"payment_ratio_{month}" for month in range(1, 7)],
    ]
    categorical = ["SEX", "EDUCATION", "MARRIAGE", "high_utilization", "ever_paid_early"]
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    category_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])
    return Pipeline([
        ("features", CreditRiskFeatureEngineer()),
        ("columns", ColumnTransformer([
            ("numeric", numeric_pipeline, numerical),
            ("categorical", category_pipeline, categorical),
        ], remainder="drop")),
    ])


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(RAW_FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    unexpected = sorted(set(frame.columns) - set(RAW_FEATURE_COLUMNS))
    if unexpected:
        raise ValueError(f"Unexpected fields: {', '.join(unexpected)}")
    clean = frame[RAW_FEATURE_COLUMNS].copy()
    for column in RAW_FEATURE_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="raise")
    if not np.isfinite(clean.to_numpy(dtype=float)).all():
        raise ValueError("All fields must be finite numbers.")
    if (clean["LIMIT_BAL"] <= 0).any():
        raise ValueError("LIMIT_BAL must be greater than zero.")
    if (~clean["AGE"].between(18, 100)).any():
        raise ValueError("AGE must be between 18 and 100.")
    if (~clean["SEX"].isin([1, 2])).any():
        raise ValueError("SEX must use a source-dataset code of 1 or 2.")
    if (~clean["EDUCATION"].isin([0, 1, 2, 3, 4, 5, 6])).any():
        raise ValueError("EDUCATION must use a source-dataset code from 0 to 6.")
    if (~clean["MARRIAGE"].isin([0, 1, 2, 3])).any():
        raise ValueError("MARRIAGE must use a source-dataset code from 0 to 3.")
    for column in CreditRiskFeatureEngineer.payment_status_columns:
        if (~clean[column].between(-2, 8)).any():
            raise ValueError(f"{column} must be between -2 and 8.")
    for column in CreditRiskFeatureEngineer.payment_columns:
        if (clean[column] < 0).any():
            raise ValueError(f"{column} cannot be negative.")
    return clean
