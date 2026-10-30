# AI381 Churn Prediction API

[![CI](https://github.com/Mohammad3f58r/ai381-lab2-container-ci/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohammad3f58r/ai381-lab2-container-ci/actions/workflows/ci.yml)

A machine learning API that predicts customer churn probability using a trained scikit-learn pipeline and FastAPI.

## Project Structure

```text
.
├── data/
│   └── churn_v2.csv
├── models/
│   └── model.joblib
├── src/
│   ├── app/
│   │   └── main.py
│   └── train/
│       └── train.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── requirements.txt
└── .github/
    └── workflows/
        └── ci.yml
```

## Run with Docker

Build and start the API with one command:

```bash
docker build -t ai381-churn-api:lab2 . && docker run --rm -p 8000:8000 ai381-churn-api:lab2
```

The API will be available at:

- Health check: http://localhost:8000/health
- API documentation: http://localhost:8000/docs

## API Endpoints

### GET /health

Returns the API health status.

Example response:

```json
{
  "status": "ok"
}
```

### POST /predict

Accepts customer information and returns a churn probability.

Example request:

```json
{
  "tenure_months": 24,
  "monthly_fee": 75.5,
  "contract": "month-to-month",
  "internet_service": "fiber",
  "support_calls": 3
}
```

Example response:

```json
{
  "churn_probability": 0.748
}
```

## Run Tests

Install dependencies and run the test suite:

```bash
pip install -r requirements.txt
python -m pytest -v
```

## Code Quality

Run Ruff:

```bash
ruff check src tests
```

## Continuous Integration

GitHub Actions runs linting and tests on pushes to `main` and pull requests targeting `main`.

## Model Features

The model uses the following customer features:

- Tenure months
- Monthly fee
- Contract type
- Internet service
- Support calls

The prediction pipeline uses preprocessing with `StandardScaler` and `OneHotEncoder`, followed by Logistic Regression.
