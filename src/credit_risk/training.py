from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from .config import DATA_PATH, METADATA_PATH, MODEL_DIR, MODEL_PATH, RAW_FEATURE_COLUMNS, REPORT_DIR, TARGET_COLUMN
from .features import build_preprocessor, validate_features
from .showcase import generate_showcase_assets


@dataclass
class Evaluation:
    roc_auc: float
    average_precision: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    log_loss: float
    confusion_matrix: list[list[int]]


def candidate_searches(random_state: int, full_search: bool = False) -> dict[str, GridSearchCV]:
    """Return comparable, leakage-safe model searches.

    The standard search is intentionally small enough to repeat locally.  The
    expanded search trades a substantially longer runtime for a wider search
    space and five CV folds; it is opt-in so that a newcomer does not
    accidentally start a multi-hour job.
    """
    folds = 5 if full_search else 3
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    scoring = {"roc_auc": "roc_auc", "average_precision": "average_precision", "f1": "f1"}
    if full_search:
        logistic_grid = {"model__C": [0.1, 1.0, 10.0]}
        forest_grid = {
            "model__n_estimators": [300],
            "model__max_depth": [8, 12, None],
            "model__min_samples_leaf": [1, 4],
            "model__max_features": ["sqrt"],
        }
        catboost_grid = {
            "model__iterations": [300, 500],
            "model__depth": [4, 6],
            "model__learning_rate": [0.03, 0.07],
            "model__l2_leaf_reg": [3, 7],
        }
    else:
        logistic_grid = {"model__C": [0.25, 1.0, 4.0]}
        forest_grid = {
            "model__n_estimators": [150],
            "model__max_depth": [8, 12],
            "model__min_samples_leaf": [2, 5],
            "model__max_features": ["sqrt"],
        }
        catboost_grid = {
            "model__iterations": [150],
            "model__depth": [4, 6],
            "model__learning_rate": [0.03, 0.05],
            "model__l2_leaf_reg": [3],
        }
    candidates = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
            logistic_grid,
        ),
        "random_forest": (
            # Reserve most CPU and memory for the rest of a developer's machine.
            RandomForestClassifier(class_weight="balanced_subsample", n_jobs=2, random_state=random_state),
            forest_grid,
        ),
        "catboost": (
            CatBoostClassifier(verbose=False, random_state=random_state, eval_metric="AUC", thread_count=2, allow_writing_files=False),
            catboost_grid,
        ),
    }
    searches = {}
    for name, (estimator, grid) in candidates.items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        searches[name] = GridSearchCV(
            # Keep searches serial by default. This makes training predictable on laptops
            # and avoids spawning a process for every fold/parameter combination.
            pipeline, grid, scoring=scoring, refit="roc_auc", cv=cv, n_jobs=1,
            return_train_score=False, error_score="raise",
        )
    return searches


def select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, probabilities >= threshold) for threshold in thresholds]
    index = int(np.argmax(scores))
    return float(thresholds[index]), float(scores[index])


def evaluate(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> Evaluation:
    predictions = (probabilities >= threshold).astype(int)
    return Evaluation(
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        average_precision=float(average_precision_score(y_true, probabilities)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, predictions)),
        log_loss=float(log_loss(y_true, probabilities)),
        confusion_matrix=confusion_matrix(y_true, predictions).tolist(),
    )


def save_plots(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes[0].plot(fpr, tpr, label=f"ROC-AUC {roc_auc_score(y_true, probabilities):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[0].legend()
    axes[1].plot(recall, precision, label=f"AP {average_precision_score(y_true, probabilities):.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    axes[1].legend()
    CalibrationDisplay.from_predictions(y_true, probabilities, n_bins=10, ax=axes[2], name="CatBoost pipeline")
    axes[2].set_title("Calibration diagnostic")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "evaluation_curves.png", dpi=160)
    plt.close(fig)
    thresholds = np.linspace(0.05, 0.95, 181)
    values = [f1_score(y_true, probabilities >= item) for item in thresholds]
    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, values)
    plt.axvline(threshold, color="crimson", linestyle="--", label=f"Selected {threshold:.2f}")
    plt.xlabel("Classification threshold")
    plt.ylabel("Validation F1")
    plt.title("Threshold analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "threshold_analysis.png", dpi=160)
    plt.close()


def save_model_feature_importance(models: dict[str, Pipeline]) -> dict[str, dict]:
    """Save comparable *descriptive* importance views for every fitted candidate.

    Logistic-regression coefficients are comparable only after the project's
    standardisation step, so their absolute values are shown. Tree models use
    their built-in split-based importance. These methods answer different
    questions and neither one is a causal explanation.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    fig, axes = plt.subplots(1, len(models), figsize=(18, 5.5), sharex=False)
    if len(models) == 1:
        axes = [axes]

    for axis, (name, pipeline) in zip(axes, models.items()):
        transformer = pipeline.named_steps["preprocess"].named_steps["columns"]
        feature_names = [
            feature.removeprefix("numeric__").removeprefix("categorical__")
            for feature in transformer.get_feature_names_out()
        ]
        model = pipeline.named_steps["model"]
        if name == "logistic_regression":
            importance = np.abs(model.coef_.ravel())
            method = "absolute standardized logistic-regression coefficient"
        else:
            importance = np.asarray(model.feature_importances_, dtype=float)
            method = "built-in tree split importance"
        ranking = pd.Series(importance, index=feature_names).sort_values(ascending=False).head(12)
        results[name] = {
            "method": method,
            "top_features": [{"feature": feature, "importance": float(score)} for feature, score in ranking.items()],
        }
        axis.barh(ranking.index[::-1], ranking.values[::-1], color="#315ecb")
        axis.set_title(name.replace("_", " ").title())
        axis.set_xlabel("Absolute coefficient" if name == "logistic_regression" else "Split importance")
        axis.tick_params(axis="y", labelsize=8)
    fig.suptitle("Candidate-model feature importance (training-fitted models)", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "model_feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return results


def train(data_path: Path = DATA_PATH, random_state: int = 42, full_search: bool = False) -> dict:
    data = pd.read_csv(data_path)
    X = validate_features(data[RAW_FEATURE_COLUMNS])
    y = data[TARGET_COLUMN].astype(int)
    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development, y_development, test_size=0.25, stratify=y_development, random_state=random_state,
    )
    searches = candidate_searches(random_state, full_search=full_search)
    comparison = {}
    fitted_models = {}
    for name, search in searches.items():
        print(f"Tuning {name}...", flush=True)
        search.fit(X_train, y_train)
        fitted_models[name] = search.best_estimator_
        comparison[name] = {
            "cv_roc_auc": float(search.best_score_),
            "cv_average_precision": float(search.cv_results_["mean_test_average_precision"][search.best_index_]),
            "best_params": search.best_params_,
        }
    winner_name = max(comparison, key=lambda name: comparison[name]["cv_roc_auc"])
    winner = fitted_models[winner_name]
    validation_probabilities = winner.predict_proba(X_validation)[:, 1]
    threshold, validation_f1 = select_threshold(y_validation, validation_probabilities)
    test_probabilities = winner.predict_proba(X_test)[:, 1]
    final_metrics = evaluate(y_test, test_probabilities, threshold)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner, MODEL_PATH)
    save_plots(y_validation, validation_probabilities, threshold)
    model_feature_importance = save_model_feature_importance(fitted_models)
    metadata = {
        "project": "UCI credit-card default prediction",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "random_state": random_state,
        "search_mode": "expanded" if full_search else "standard",
        "cv_folds": 5 if full_search else 3,
        "splits": {"train": len(X_train), "validation": len(X_validation), "test": len(X_test)},
        "target": TARGET_COLUMN,
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "selected_model": winner_name,
        "model_comparison": comparison,
        "model_feature_importance": model_feature_importance,
        "threshold": threshold,
        "validation_f1_at_threshold": validation_f1,
        "test_metrics": asdict(final_metrics),
        "classification_report": classification_report(y_test, test_probabilities >= threshold, output_dict=True, zero_division=0),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (REPORT_DIR / "training_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    generate_showcase_assets(data_path, METADATA_PATH)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the credit-risk pipeline.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full-search", action="store_true",
        help="Use a wider grid and five CV folds (considerably slower).",
    )
    args = parser.parse_args()
    result = train(args.data, args.seed, full_search=args.full_search)
    print(json.dumps({"selected_model": result["selected_model"], "threshold": result["threshold"], "test_metrics": result["test_metrics"]}, indent=2))
