from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(title="AI381 Churn Prediction API")


class CustomerInput(BaseModel):
    tenure_months: int = Field(ge=0)
    monthly_fee: float = Field(gt=0)
    contract: str
    internet_service: str
    support_calls: int = Field(ge=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerInput):
    data = pd.DataFrame([customer.model_dump()])

    probability = float(model.predict_proba(data)[0][1])

    return {
        "churn_probability": round(probability, 3)
    }
