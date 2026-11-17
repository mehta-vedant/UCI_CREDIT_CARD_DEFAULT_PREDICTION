import pandas as pd

from src.credit_risk.benchmark import balance_variants, candidate_registry, standard_pipeline
from src.credit_risk.config import DATA_PATH, RAW_FEATURE_COLUMNS, TARGET_COLUMN


def test_benchmark_registry_is_broad_and_has_a_baseline():
    registry = candidate_registry(42)
    assert registry["dummy_prior"].baseline is True
    assert {"logistic_regression", "linear_svm", "extra_trees", "random_forest", "catboost", "xgboost"} <= set(registry)
    assert len(registry) == 13


def test_sampling_variants_fit_only_inside_their_pipeline():
    data = pd.read_csv(DATA_PATH).groupby(TARGET_COLUMN, group_keys=False).head(80)
    X = data[RAW_FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    estimator = candidate_registry(42)["logistic_regression"].estimator
    variants = balance_variants(estimator, 42)
    assert set(variants) == {"native", "random_oversample", "smotenc"}
    variants["smotenc"].fit(X, y)
    assert len(variants["smotenc"].predict_proba(X)) == len(X)


def test_standard_pipeline_outputs_dense_data_for_dense_estimators():
    data = pd.read_csv(DATA_PATH).head(50)
    estimator = candidate_registry(42)["hist_gradient_boosting"].estimator
    transformed = standard_pipeline(estimator).named_steps["preprocess"].fit_transform(
        data[RAW_FEATURE_COLUMNS], data[TARGET_COLUMN]
    )
    assert getattr(transformed, "ndim", None) == 2
