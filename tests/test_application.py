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
    assert b"Data quality and exploratory analysis" in response.data
    assert b"Three models were compared" in response.data

    asset = app.test_client().get("/static/project-assets/summary.json")
    assert asset.status_code == 200
    assert asset.json["rows"] == 30000
    assert asset.json["missing_cells"] == 0
