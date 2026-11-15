from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from .config import METADATA_PATH, MODEL_PATH
from .features import validate_features


@lru_cache(maxsize=1)
def load_assets():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError("Model artifacts are missing. Run `python -m src.credit_risk.training` first.")
    return joblib.load(MODEL_PATH), json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def predict(payload: dict) -> dict:
    pipeline, metadata = load_assets()
    features = validate_features(pd.DataFrame([payload]))
    probability = float(pipeline.predict_proba(features)[0, 1])
    threshold = float(metadata["threshold"])
    prediction = int(probability >= threshold)
    return {
        "prediction": prediction,
        "label": "Higher default risk" if prediction else "Lower default risk",
        "default_probability": round(probability, 4),
        "threshold": threshold,
        "model": metadata["selected_model"],
        "disclaimer": "Educational risk estimate only. It is not an automated credit decision.",
    }
