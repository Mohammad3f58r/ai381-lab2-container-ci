import joblib
from pathlib import Path
from fastapi.testclient import TestClient

from src.app.main import app

client = TestClient(app)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"

VALID_CUSTOMER = {
    "tenure_months": 24,
    "monthly_fee": 75.5,
    "contract": "month-to-month",
    "internet_service": "fiber",
    "support_calls": 3,
}


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_loads():
    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_predict_contract():
    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200
    assert "churn_probability" in response.json()


def test_predict_probability_range():
    response = client.post("/predict", json=VALID_CUSTOMER)

    probability = response.json()["churn_probability"]

    assert 0 <= probability <= 1


def test_reject_negative_tenure():
    customer = VALID_CUSTOMER.copy()
    customer["tenure_months"] = -5

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_deterministic_prediction():
    first = client.post("/predict", json=VALID_CUSTOMER)
    second = client.post("/predict", json=VALID_CUSTOMER)

    assert first.json() == second.json()
