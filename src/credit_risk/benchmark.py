"""Leakage-safe, broad model benchmark for the credit-risk prototype.

The module deliberately keeps the final test partition untouched.  It first
screens diverse model families with five-fold CV on the training partition,
then tunes the four best non-baseline candidates, compares three class-balance
strategies for the selected candidate, chooses a validation F1 threshold, and
only then reports one final test result.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             classification_report, f1_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import (DATA_PATH, METADATA_PATH, MODEL_DIR, MODEL_PATH,
                     RAW_FEATURE_COLUMNS, REPORT_DIR, TARGET_COLUMN)
from .features import CreditRiskFeatureEngineer, build_preprocessor, validate_features
from .showcase import generate_showcase_assets
from .training import evaluate, save_plots


@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: object
    tuning_space: dict
    baseline: bool = False


def candidate_registry(seed: int) -> dict[str, Candidate]:
    """Return a broad, reproducible set of tabular-classification candidates."""
    positive_weight = (1 - 0.2212) / 0.2212
    return {
        "dummy_prior": Candidate("dummy_prior", DummyClassifier(strategy="prior"), {}, True),
        "logistic_regression": Candidate(
            "logistic_regression", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
            {"model__C": np.logspace(-3, 2, 16)},
        ),
        "gaussian_nb": Candidate("gaussian_nb", GaussianNB(), {"model__var_smoothing": np.logspace(-11, -7, 12)}),
        "linear_svm": Candidate(
            "linear_svm", CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=seed), method="sigmoid", cv=3),
            {"model__estimator__C": np.logspace(-3, 1, 10)},
        ),
        "knn": Candidate(
            "knn", KNeighborsClassifier(n_jobs=2),
            {"model__n_neighbors": list(range(15, 62, 4)), "model__weights": ["uniform", "distance"], "model__p": [1, 2]},
        ),
        "decision_tree": Candidate(
            "decision_tree", DecisionTreeClassifier(class_weight="balanced", random_state=seed),
            {"model__max_depth": [3, 5, 8, 12, None], "model__min_samples_leaf": [1, 5, 15, 30]},
        ),
        "extra_trees": Candidate(
            "extra_trees", ExtraTreesClassifier(n_estimators=250, class_weight="balanced", n_jobs=2, random_state=seed),
            {"model__max_depth": [None, 10, 16], "model__min_samples_leaf": [1, 3, 8], "model__max_features": ["sqrt", 0.7]},
        ),
        "random_forest": Candidate(
            "random_forest", RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample", n_jobs=2, random_state=seed),
            {"model__max_depth": [8, 12, None], "model__min_samples_leaf": [1, 3, 8], "model__max_features": ["sqrt", 0.7]},
        ),
        "gradient_boosting": Candidate(
            "gradient_boosting", GradientBoostingClassifier(random_state=seed),
            {"model__n_estimators": [100, 200, 350], "model__learning_rate": [0.02, 0.05, 0.1], "model__max_depth": [2, 3, 4], "model__min_samples_leaf": [5, 15, 30]},
        ),
        "hist_gradient_boosting": Candidate(
            "hist_gradient_boosting", HistGradientBoostingClassifier(random_state=seed),
            {"model__learning_rate": [0.03, 0.06, 0.1], "model__max_leaf_nodes": [15, 31, 63], "model__l2_regularization": [0.0, 0.5, 2.0]},
        ),
        "adaboost": Candidate(
            "adaboost", AdaBoostClassifier(random_state=seed),
            {"model__n_estimators": [100, 200, 350], "model__learning_rate": [0.02, 0.05, 0.1, 0.3]},
        ),
        "catboost": Candidate(
            "catboost", CatBoostClassifier(verbose=False, random_state=seed, thread_count=2, allow_writing_files=False, auto_class_weights="Balanced"),
            {"model__iterations": [200, 350, 500], "model__depth": [4, 6, 8], "model__learning_rate": [0.02, 0.04, 0.07], "model__l2_leaf_reg": [1, 3, 7, 12]},
        ),
        "xgboost": Candidate(
            "xgboost", XGBClassifier(objective="binary:logistic", eval_metric="aucpr", n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=positive_weight, n_jobs=2, random_state=seed),
            {"model__n_estimators": [200, 350, 500], "model__max_depth": [3, 5, 7], "model__learning_rate": [0.02, 0.04, 0.07], "model__min_child_weight": [1, 5, 10], "model__subsample": [0.7, 0.85, 1.0]},
        ),
    }


def standard_pipeline(estimator: object) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", clone(estimator))])


def scoring() -> dict[str, str]:
    return {"average_precision": "average_precision", "roc_auc": "roc_auc", "neg_log_loss": "neg_log_loss", "f1": "f1"}


def cv_summary(name: str, pipeline, X: pd.DataFrame, y: pd.Series, cv) -> dict:
    started = time.perf_counter()
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring(), n_jobs=4, error_score="raise")
    return {
        "candidate": name,
        "mean_average_precision": float(np.mean(scores["test_average_precision"])),
        "std_average_precision": float(np.std(scores["test_average_precision"])),
        "mean_roc_auc": float(np.mean(scores["test_roc_auc"])),
        "mean_log_loss": float(-np.mean(scores["test_neg_log_loss"])),
        "mean_f1_at_0_5": float(np.mean(scores["test_f1"])),
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }


def build_smotenc_pipeline(estimator: object, seed: int) -> ImbPipeline:
    """Sample mixed data after features are created but before one-hot encoding."""
    preprocessor = build_preprocessor()
    return ImbPipeline([
        ("features", preprocessor.named_steps["features"]),
        ("sampler", SMOTENC(categorical_features=["SEX", "EDUCATION", "MARRIAGE", "high_utilization", "ever_paid_early"], random_state=seed)),
        ("columns", preprocessor.named_steps["columns"]),
        ("model", clone(estimator)),
    ])


def balance_variants(estimator: object, seed: int) -> dict[str, object]:
    return {
        "native": standard_pipeline(estimator),
        # imbalanced-learn deliberately rejects an sklearn Pipeline as an
        # intermediate step.  Keep the two preprocessing transformers as
        # explicit steps so the sampler still lives safely inside each fold.
        "random_oversample": ImbPipeline([
            ("features", CreditRiskFeatureEngineer()),
            ("columns", build_preprocessor().named_steps["columns"]),
            ("sampler", RandomOverSampler(random_state=seed)),
            ("model", clone(estimator)),
        ]),
        "smotenc": build_smotenc_pipeline(estimator, seed),
    }


def save_benchmark_plot(screening: pd.DataFrame, tuning: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ordered = screening.sort_values("mean_average_precision")
    axes[0].barh(ordered["candidate"], ordered["mean_average_precision"], xerr=ordered["std_average_precision"], color="#315ecb")
    axes[0].set(title="Five-fold screening: average precision", xlabel="Mean CV average precision")
    if not tuning.empty:
        ordered_tuned = tuning.sort_values("cv_average_precision")
        axes[1].barh(ordered_tuned["candidate"], ordered_tuned["cv_average_precision"], color="#0f766e")
        axes[1].set(title="Top-four tuned candidates", xlabel="Mean CV average precision")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "benchmark_leaderboard.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_benchmark(data_path: Path = DATA_PATH, seed: int = 42, n_iter: int = 10, resume: bool = False) -> dict:
    data = pd.read_csv(data_path)
    X = validate_features(data[RAW_FEATURE_COLUMNS])
    y = data[TARGET_COLUMN].astype(int)
    X_development, X_test, y_development, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
    X_train, X_validation, y_train, y_validation = train_test_split(X_development, y_development, test_size=0.25, stratify=y_development, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    registry = candidate_registry(seed)

    if resume:
        screening = pd.read_csv(REPORT_DIR / "benchmark_screening.csv")
        tuning = pd.read_json(REPORT_DIR / "benchmark_tuning.json")
        tuning_rows = tuning.to_dict(orient="records")
    else:
        screening_rows = []
        for name, candidate in registry.items():
            print(f"Screening {name}...", flush=True)
            screening_rows.append(cv_summary(name, standard_pipeline(candidate.estimator), X_train, y_train, cv))
        screening = pd.DataFrame(screening_rows).sort_values(["mean_average_precision", "mean_roc_auc"], ascending=False)
        screening.to_csv(REPORT_DIR / "benchmark_screening.csv", index=False)
        finalists = [name for name in screening["candidate"] if not registry[name].baseline][:4]
        tuning_rows = []
        for name in finalists:
            candidate = registry[name]
            print(f"Tuning finalist {name}...", flush=True)
            search = RandomizedSearchCV(standard_pipeline(candidate.estimator), candidate.tuning_space, n_iter=n_iter, scoring=scoring(), refit="average_precision", cv=cv, n_jobs=4, random_state=seed, error_score="raise", return_train_score=False)
            started = time.perf_counter(); search.fit(X_train, y_train)
            tuning_rows.append({"candidate": name, "cv_average_precision": float(search.best_score_), "cv_roc_auc": float(search.cv_results_["mean_test_roc_auc"][search.best_index_]), "cv_log_loss": float(-search.cv_results_["mean_test_neg_log_loss"][search.best_index_]), "best_params": search.best_params_, "runtime_seconds": round(time.perf_counter() - started, 2)})
        tuning = pd.DataFrame(tuning_rows).sort_values(["cv_average_precision", "cv_roc_auc"], ascending=False)
        tuning.to_json(REPORT_DIR / "benchmark_tuning.json", orient="records", indent=2)
    save_benchmark_plot(screening, tuning)

    selected_name = str(tuning.iloc[0]["candidate"])
    selected_params = dict(tuning.iloc[0]["best_params"])
    selected_estimator = clone(registry[selected_name].estimator).set_params(**{key.removeprefix("model__"): value for key, value in selected_params.items()})
    balance_rows, balance_pipelines = [], {}
    for name, pipeline in balance_variants(selected_estimator, seed).items():
        print(f"Testing balance strategy {name}...", flush=True)
        result = cv_summary(name, pipeline, X_train, y_train, cv)
        balance_rows.append(result)
        balance_pipelines[name] = pipeline
    balance = pd.DataFrame(balance_rows).sort_values(["mean_average_precision", "mean_roc_auc"], ascending=False)
    balance.to_csv(REPORT_DIR / "balance_strategy_ablation.csv", index=False)
    balance_name = str(balance.iloc[0]["candidate"])
    champion = balance_pipelines[balance_name].fit(X_train, y_train)

    validation_probabilities = champion.predict_proba(X_validation)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 181)
    threshold_rows = []
    for threshold in thresholds:
        predictions = validation_probabilities >= threshold
        threshold_rows.append({"threshold": float(threshold), "precision": float(precision_score(y_validation, predictions, zero_division=0)), "recall": float(recall_score(y_validation, predictions, zero_division=0)), "f1": float(f1_score(y_validation, predictions, zero_division=0))})
    threshold_table = pd.DataFrame(threshold_rows)
    selected_threshold = float(threshold_table.loc[threshold_table["f1"].idxmax(), "threshold"])
    threshold_table.to_csv(REPORT_DIR / "benchmark_threshold_tradeoffs.csv", index=False)

    test_probabilities = champion.predict_proba(X_test)[:, 1]
    test_metrics = asdict(evaluate(y_test, test_probabilities, selected_threshold))
    test_metrics["mcc"] = float(matthews_corrcoef(y_test, test_probabilities >= selected_threshold))
    calibration = {"brier_score": float(brier_score_loss(y_test, test_probabilities)), "log_loss": float(log_loss(y_test, test_probabilities))}
    joblib.dump(champion, MODEL_PATH)
    save_plots(y_validation, validation_probabilities, selected_threshold)

    metadata = {
        "project": "UCI credit-card default prediction",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "broad_benchmark_v1",
        "random_state": seed,
        "cv_folds": 5,
        "splits": {"train": len(X_train), "validation": len(X_validation), "test": len(X_test)},
        "target": TARGET_COLUMN,
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "selected_model": selected_name,
        "selected_balance_strategy": balance_name,
        "selection_metric": "mean cross-validated average precision",
        "screening": screening.to_dict(orient="records"),
        "tuning": tuning.to_dict(orient="records"),
        "balance_strategy_ablation": balance.to_dict(orient="records"),
        "model_comparison": {row["candidate"]: {"cv_roc_auc": row["cv_roc_auc"], "cv_average_precision": row["cv_average_precision"], "best_params": row["best_params"]} for row in tuning_rows},
        "threshold": selected_threshold,
        "validation_f1_at_threshold": float(threshold_table["f1"].max()),
        "test_metrics": test_metrics,
        "calibration": calibration,
        "classification_report": classification_report(y_test, test_probabilities >= selected_threshold, output_dict=True, zero_division=0),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (REPORT_DIR / "benchmark_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    generate_showcase_assets(data_path, METADATA_PATH)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the broad, leakage-safe credit-risk benchmark.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-iter", type=int, default=10, help="Random search trials per finalist.")
    parser.add_argument("--resume", action="store_true", help="Reuse compatible screening/tuning reports and finish post-tuning evaluation.")
    args = parser.parse_args()
    result = run_benchmark(args.data, args.seed, args.n_iter, args.resume)
    print(json.dumps({"selected_model": result["selected_model"], "threshold": result["threshold"], "test_metrics": result["test_metrics"]}, indent=2))
