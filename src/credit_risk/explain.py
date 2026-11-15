from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from .config import DATA_PATH, MODEL_PATH, RAW_FEATURE_COLUMNS, REPORT_DIR, TARGET_COLUMN
from .features import validate_features


def create_shap_report(sample_size: int = 500) -> dict[str, float]:
    """Create a global SHAP summary from a reproducible source-data sample."""
    pipeline = joblib.load(MODEL_PATH)
    model = pipeline.named_steps["model"]
    if model.__class__.__name__ != "CatBoostClassifier":
        raise RuntimeError("SHAP report currently supports the fitted CatBoost pipeline.")

    data = pd.read_csv(DATA_PATH)
    features = validate_features(data[RAW_FEATURE_COLUMNS]).sample(
        n=min(sample_size, len(data)), random_state=42
    )
    preprocessor = pipeline.named_steps["preprocess"]
    engineered = preprocessor.named_steps["features"].transform(features)
    transformer = preprocessor.named_steps["columns"]
    transformed = transformer.transform(engineered)
    feature_names = transformer.get_feature_names_out().tolist()

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(transformed)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(values, transformed, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    importance = pd.Series(abs(values).mean(axis=0), index=feature_names).sort_values(ascending=False)
    result = {name: float(score) for name, score in importance.head(20).items()}
    (REPORT_DIR / "shap_feature_importance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(create_shap_report(), indent=2))
