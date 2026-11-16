from __future__ import annotations

import os
import json
from pathlib import Path

from flask import Flask, Response, jsonify, redirect, render_template, request, send_from_directory

from src.credit_risk.config import RAW_FEATURE_COLUMNS
from src.credit_risk.inference import load_assets, predict

app = Flask(__name__)
application = app
SHOWCASE_SUMMARY_PATH = Path(app.static_folder) / "project-assets" / "summary.json"
COURSE_DIR = Path(app.root_path) / "project_guide" / "course"


def form_payload() -> dict[str, str | None]:
    return {column: request.form.get(column) for column in RAW_FEATURE_COLUMNS}


@app.after_request
def add_security_headers(response):
    """Apply small, deployment-safe defaults to every response."""
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/project")
def project_showcase():
    if not SHOWCASE_SUMMARY_PATH.exists():
        return "Project showcase assets are missing. Run `python -m src.credit_risk.showcase`.", 503
    summary = json.loads(SHOWCASE_SUMMARY_PATH.read_text(encoding="utf-8"))
    return render_template("analysis.html", summary=summary)


@app.get("/guide")
def course_home():
    """Redirect so relative course assets resolve below /guide/."""
    return redirect("/guide/index.html")


@app.get("/guide/<path:filename>")
def course_file(filename: str):
    return send_from_directory(COURSE_DIR, filename)


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Endpoint not found."}), 404
    return render_template("error.html", code=404, title="Page not found", message="That page does not exist or has moved."), 404


@app.errorhandler(405)
def method_not_allowed(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Method not allowed for this endpoint."}), 405
    return render_template("error.html", code=405, title="Method not allowed", message="Use the page controls or return to the dashboard."), 405


@app.get("/favicon.ico")
def favicon():
    """Serve a tiny local icon so the browser has no noisy missing-resource error."""
    icon = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<rect width="64" height="64" rx="14" fill="#2563eb"/>'
        '<path d="M32 10 50 17v13c0 12-7 20-18 25C21 50 14 42 14 30V17z" fill="white"/>'
        '<path d="m24 32 5 5 11-12" fill="none" stroke="#2563eb" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>'
        '</svg>'
    )
    return Response(icon, mimetype="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    try:
        payload = form_payload()
        result = predict(payload)
        return render_template(
            "home.html", results=result["label"],
            probability=round(result["default_probability"] * 100, 1),
            threshold=round(result["threshold"] * 100, 1),
            risk_level="Review recommended" if result["prediction"] else "No review flag",
            risk_color="high" if result["prediction"] else "low",
            disclaimer=result["disclaimer"],
            limit_bal=payload["LIMIT_BAL"], age=payload["AGE"], sex=payload["SEX"],
            education=payload["EDUCATION"], marriage=payload["MARRIAGE"],
            BILL_AMT1=payload["BILL_AMT1"],
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
