from application import app


def test_health_returns_json(monkeypatch):
    monkeypatch.setattr("application.load_assets", lambda: (None, {"selected_model": "catboost", "threshold": 0.42}))
    response = app.test_client().get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"


def test_prediction_api_rejects_invalid_json():
    response = app.test_client().post("/api/predict", json=["not", "an", "object"])
    assert response.status_code == 400


def test_favicon_is_served_locally():
    response = app.test_client().get("/favicon.ico")
    assert response.status_code == 200
    assert response.mimetype == "image/svg+xml"


def test_project_showcase_and_aggregate_assets_are_available():
    response = app.test_client().get("/project")
    assert response.status_code == 200
    assert b"Authentic EDA from the active CSV" in response.data
    assert b"compare feature importance across models" in response.data

    asset = app.test_client().get("/static/project-assets/summary.json")
    assert asset.status_code == 200
    assert asset.json["rows"] == 30000
    assert asset.json["missing_cells"] == 0

    heatmap = app.test_client().get("/static/project-assets/correlation_heatmap.png")
    importance = app.test_client().get("/static/project-assets/model_feature_importance.png")
    assert heatmap.status_code == 200
    assert importance.status_code == 200


def test_beginner_course_is_served_by_flask():
    client = app.test_client()
    response = client.get("/guide")
    assert response.status_code == 302
    page = client.get(response.headers["Location"])
    assert page.status_code == 200
    assert b"Learn this credit risk project from zero" in page.data


def test_api_errors_are_json_and_security_headers_are_set():
    client = app.test_client()
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.json == {"error": "Endpoint not found."}
    assert response.headers["X-Content-Type-Options"] == "nosniff"
