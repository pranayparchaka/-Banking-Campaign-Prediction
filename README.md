# Banking Campaign Response Prediction and Contact Prioritization

A college project that predicts bank term-deposit response, ranks customers by pre-call response potential, and analyzes calling-capacity trade-offs.

## Problem statement

Banks often have more customers to call than available calling capacity. This project uses the UCI Bank Marketing `bank-full.csv` dataset to estimate response potential before a call, prioritize customers, and quantify the efficiency-versus-coverage trade-off.

The primary pre-call model excludes `duration` because call duration is only known after a call and would create target leakage. Historical `y` is used only as the training/evaluation target and is never accepted as a future prediction feature.

## Features

- Data validation and exploratory analysis
- Leakage-safe preprocessing
- Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting comparison
- Tuned Gradient Boosting model
- Sigmoid probability calibration
- Customer response prediction
- Raw-score customer prioritization
- Contact-capacity analysis
- Model-versus-random retrospective evaluation
- FastAPI backend with OpenAPI documentation
- Responsive HTML/CSS/JavaScript dashboard
- Chart.js model and capacity visualizations

## Technology stack

- Python, Pandas, NumPy
- Scikit-learn and Joblib
- FastAPI, Pydantic, Uvicorn
- HTML, CSS, Vanilla JavaScript
- Chart.js via CDN
- Pytest

## Architecture

```text
bank-full.csv
    |
    v
Validation and EDA
    |
    v
Leakage-safe preprocessing
    |
    v
Tuned Gradient Boosting pipeline
    |
    v
Sigmoid calibration wrapper
    |
    +--> FastAPI backend
    |       |
    |       +--> Prediction
    |       +--> Prioritization
    |       +--> Capacity analysis
    |
    +--> HTML/CSS/JavaScript dashboard
```

The backend uses the raw positive-class score from the underlying Gradient
Boosting pipeline for ranking. It uses the sigmoid-calibrated probability for
response probability and expected-responder estimates.

## Folder structure

```text
backend/       FastAPI routes, schemas, data service, and ML service
frontend/      Dashboard HTML, CSS, API client, app logic, and charts
models/        Saved raw and calibrated model artifacts
reports/       Phase reports, metrics, predictions, and figures
scripts/       Phase 1–3.5 data and ML workflows
src/           Reusable preprocessing and calibration modules
tests/         Backend API tests
bank-full.csv  Original source dataset
requirements.txt
```

## Installation

From the project root:

```powershell
python -m pip install -r .\requirements.txt
```

## Run the backend

```powershell
python -m uvicorn backend.main:app --reload
```

Backend documentation:

```text
http://127.0.0.1:8000/docs
```

## Run the frontend

In a second terminal:

```powershell
python -m http.server 5500 --directory frontend
```

Open:

```text
http://127.0.0.1:5500
```

The dashboard defaults to:

```text
http://127.0.0.1:8000/api
```

The API base URL can be changed in the dashboard and is saved in browser local storage. It can also be configured before loading the application with `window.BANKING_API_BASE_URL`.

## Single-URL deployment preparation

The FastAPI application serves the frontend from the same process. In a
deployment environment, the same origin provides:

```text
/                 dashboard
/api/...          REST API
/docs             OpenAPI documentation
/api/health       health check
```

For local development with the standalone frontend server, the dashboard uses
`http://127.0.0.1:8000/api`. When served by FastAPI or a cloud web service, it
automatically uses the same-origin `/api` path. The API URL remains manually
configurable in the dashboard.

### Render preparation

The recommended deployment platform is Render Web Service. This repository
contains `render.yaml` with the following deployment commands:

```text
Build: pip install -r requirements.txt
Start: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
Health check: /api/health
```

Before deployment:

1. Create a GitHub repository.
2. Push the project, including `models/calibrated_best_model.joblib` and its metadata.
3. Do not commit virtual environments, Python caches, test caches, or secrets.
4. Create a Render Web Service from the repository.
5. Use the commands defined in `render.yaml`.
6. Verify `/`, `/api/health`, `/api/summary`, `/api/model-info`, and `/docs` after deployment.

The expected Render URL format is:

```text
https://banking-campaign-project.onrender.com
```

The actual public URL must only be reported after a successful deployment.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Application and model status |
| GET | `/api/summary` | Dataset and evaluation summary |
| GET | `/api/model-info` | Features, metrics, and calibration metadata |
| POST | `/api/predict` | Predict one pre-call customer |
| POST | `/api/prioritize` | Rank a customer batch by raw ranking score |
| POST | `/api/capacity-analysis` | Select Top-K customers for a capacity percentage |

Customer inputs use the pre-call schema: `customer_id`, `age`, `job`,
`marital`, `education`, `default`, `balance`, `housing`, `loan`, `contact`,
`day`, `month`, `campaign`, `pdays`, `previous`, and `poutcome`.

`duration` and `y` are intentionally rejected as input features.

## Model summary

The selected model is Tuned Gradient Boosting. On the untouched outer test set:

- Accuracy: 0.8033
- Precision: 0.3268
- Recall: 0.6427
- F1-score: 0.4333
- ROC-AUC: 0.7998
- PR-AUC: 0.4466

The saved calibrated model uses sigmoid/Platt scaling fitted from five-fold
development out-of-fold probabilities. The outer test set was not used to fit
or select calibration.

## Calibration and probability semantics

Two values are intentionally kept separate:

- Raw ranking score: underlying Gradient Boosting positive-class score used to sort customers.
- Calibrated response probability: sigmoid-calibrated probability used for response estimates and expected responders.

The calibrated probability is an estimate, not a guarantee that a customer will subscribe. Expected responders are calculated by summing calibrated probabilities and should not be interpreted as guaranteed outcomes.

## Prioritization and capacity analysis

Prioritization sorts customers in descending raw ranking score. The API and
dashboard show the calibrated probability beside the ranking score so their
different purposes remain visible.

Capacity analysis selects:

```text
floor(N × capacity_percentage / 100)
```

with 100% selecting all available customers. For future customers, actual
historical outcomes are unknown and are never returned by the API.

## Testing

Run backend tests:

```powershell
python -m pytest .\tests\test_api.py
```

The current suite covers health, summary, model information, prediction,
prioritization, capacity validation, feature exclusion, ranking behavior, and
OpenAPI route registration.

JavaScript syntax can be checked with:

```powershell
node --check .\frontend\js\api.js
node --check .\frontend\js\app.js
node --check .\frontend\js\charts.js
```

## Limitations

- The dataset is historical and does not represent live customer behavior.
- The application is a decision-support tool, not an autonomous call or subscription system.
- Probability calibration improves expected-response interpretation but does not guarantee future calibration drift will not occur.
- No authentication, database, user management, deployment configuration, or public URL is included.
- Chart.js is loaded from a CDN for the dashboard visualization.

## Disclaimer

Predictions and expected responders are statistical estimates based on historical
data. They are not guarantees of customer behavior, subscription, approval,
profitability, or suitability. Any operational use should include appropriate
bank policy, compliance, privacy, fairness, and human review controls.

