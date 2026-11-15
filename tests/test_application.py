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
