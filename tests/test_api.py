from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "model_not_loaded")

    def test_predict_without_model(self):
        response = client.post(
            "/predict", files={"file": ("test.jpg", b"not an image", "image/jpeg")}
        )
        assert response.status_code == 503
