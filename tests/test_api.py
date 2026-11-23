"""Basic Phase 4 API tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.main import app
from backend.data_service import ProjectDataService
from backend.ml_service import BankingModelService
from backend.routes import capacity_analysis, health, model_info, prioritize, predict, summary
from backend.schemas import CapacityAnalysisRequest, CustomerPredictionRequest, PrioritizationRequest


SAMPLE_CUSTOMER = {
    "customer_id": "demo-001",
    "age": 35,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}


@pytest.fixture(scope="module", autouse=True)
def initialize_test_services():
    """Initialize the same services once without the environment's hanging TestClient."""
    project_root = Path(__file__).resolve().parents[1]
    app.state.model_service = BankingModelService(project_root)
    app.state.data_service = ProjectDataService(project_root)
    yield


class RequestStub:
    app = app


def test_health_and_model_info():
    health_body = health(RequestStub())
    assert health_body["model_loaded"] is True

    info_body = model_info(RequestStub())
    assert info_body["excluded_feature"] == "duration"
    assert "age" in info_body["features_used"]


def test_summary():
    body = summary(RequestStub())
    assert body["total_customers"] == 45211
    assert body["calibration_method"] == "sigmoid"


def test_predict_exposes_separate_scores():
    body = predict(CustomerPredictionRequest(**SAMPLE_CUSTOMER), RequestStub())
    assert 0 <= body["calibrated_response_probability"] <= 1
    assert 0 <= body["raw_ranking_score"] <= 1
    assert body["probability_note"]


def test_prioritize_uses_raw_ranking_score_and_no_actual_outcome():
    customers = [dict(SAMPLE_CUSTOMER, customer_id="a"), dict(SAMPLE_CUSTOMER, customer_id="b", age=70, balance=5000)]
    payload = PrioritizationRequest(customers=customers)
    body = prioritize(payload, RequestStub())
    assert body["customers"][0]["priority_rank"] == 1
    assert "actual_y" not in body["customers"][0]
    assert "ranking_method" in body


def test_capacity_analysis_and_validation():
    customers = [dict(SAMPLE_CUSTOMER, customer_id=f"c-{index}") for index in range(10)]
    payload = CapacityAnalysisRequest(customers=customers, capacity_percentage=50)
    body = capacity_analysis(payload, RequestStub())
    assert body["customers_selected"] == 5
    assert body["capacity_percentage"] == 50
    assert "actual_y" not in body

    with pytest.raises(ValidationError):
        CapacityAnalysisRequest(customers=customers, capacity_percentage=0)


def test_openapi_documents_required_paths():
    paths = app.openapi()["paths"]
    assert "/api/health" in paths
    assert "/api/summary" in paths
    assert "/api/model-info" in paths
    assert "/api/predict" in paths
    assert "/api/prioritize" in paths
    assert "/api/capacity-analysis" in paths

