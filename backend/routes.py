"""FastAPI route definitions."""

from fastapi import APIRouter, HTTPException, Request

from .schemas import (
    CapacityAnalysisRequest,
    CapacityAnalysisResponse,
    CustomerPredictionRequest,
    CustomerPredictionResponse,
    PrioritizationRequest,
    PrioritizationResponse,
)

router = APIRouter(prefix="/api")


def services(request: Request):
    return request.app.state.model_service, request.app.state.data_service


@router.get("/health")
def health(request: Request):
    model_service, _ = services(request)
    return {
        "status": "ok",
        "application": "Banking Campaign Response Prediction and Contact Prioritization",
        "model_loaded": model_service is not None,
        "calibration_method": model_service.calibration_method,
    }


@router.get("/summary")
def summary(request: Request):
    _, data_service = services(request)
    return data_service.summary()


@router.get("/model-info")
def model_info(request: Request):
    model_service, data_service = services(request)
    return data_service.model_info(model_service.FEATURE_COLUMNS)


@router.get("/roc-curve")
def roc_curve(request: Request):
    _, data_service = services(request)
    return data_service.roc_curve()


@router.post("/predict", response_model=CustomerPredictionResponse)
def predict(payload: CustomerPredictionRequest, request: Request):
    model_service, _ = services(request)
    return model_service.predict(payload)


@router.post("/prioritize", response_model=PrioritizationResponse)
def prioritize(payload: PrioritizationRequest, request: Request):
    model_service, _ = services(request)
    return {
        "ranking_method": "Descending raw ranking score from the underlying base Gradient Boosting pipeline",
        "probability_method": "Sigmoid-calibrated probability from calibrated_best_model.joblib",
        "customers": model_service.prioritize(payload.customers),
    }


@router.post("/capacity-analysis", response_model=CapacityAnalysisResponse)
def capacity_analysis(payload: CapacityAnalysisRequest, request: Request):
    model_service, _ = services(request)
    return model_service.capacity_analysis(payload.customers, payload.capacity_percentage)

