from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request

from src.credit_risk.config import RAW_FEATURE_COLUMNS
from src.credit_risk.inference import load_assets, predict

app = Flask(__name__)
application = app


def form_payload() -> dict[str, str | None]:
    return {column: request.form.get(column) for column in RAW_FEATURE_COLUMNS}


@app.get("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    try:
        result = predict(form_payload())
        return render_template(
            "home.html", results=result["label"],
            probability=round(result["default_probability"] * 100, 1),
            threshold=round(result["threshold"] * 100, 1),
            risk_level="Review recommended" if result["prediction"] else "No review flag",
            risk_color="high" if result["prediction"] else "low",
            disclaimer=result["disclaimer"],
        )
    except (ValueError, RuntimeError) as error:
        return render_template("home.html", error=str(error)), 400


@app.post("/api/predict")
def predict_api():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object with all required client fields."}), 400
    try:
        return jsonify(predict(payload))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503


@app.get("/health")
def health():
    try:
        _, metadata = load_assets()
        return jsonify({"status": "ok", "model": metadata["selected_model"], "threshold": metadata["threshold"]})
    except RuntimeError as error:
        return jsonify({"status": "unavailable", "error": str(error)}), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
