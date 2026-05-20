from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Iris Prediction API Running"
    }


def test_predict():
    response = client.post(
        "/predict",
        json={
            "features": [5.1, 3.5, 1.4, 0.2]
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "species" in data