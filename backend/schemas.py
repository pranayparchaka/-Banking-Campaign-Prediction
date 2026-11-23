"""Pydantic schemas for the banking campaign API."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomerAttributes(BaseModel):
    """Pre-call customer attributes. Duration and historical y are excluded."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=18, le=100)
    job: str = Field(..., min_length=1)
    marital: str = Field(..., min_length=1)
    education: str = Field(..., min_length=1)
    default: str = Field(..., min_length=1)
    balance: float
    housing: str = Field(..., min_length=1)
    loan: str = Field(..., min_length=1)
    contact: str = Field(..., min_length=1)
    day: int = Field(..., ge=1, le=31)
    month: str = Field(..., min_length=1)
    campaign: int = Field(..., ge=1)
    pdays: int = Field(..., ge=-1)
    previous: int = Field(..., ge=0)
    poutcome: str = Field(..., min_length=1)

    @field_validator("job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome")
    @classmethod
    def strip_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class CustomerPredictionRequest(CustomerAttributes):
    customer_id: str = Field(..., min_length=1)


class CustomerPredictionResponse(BaseModel):
    customer_id: str
    predicted_response_class: str
    calibrated_response_probability: float
    raw_ranking_score: float
    priority_category: str
    probability_note: str


class PrioritizationCustomer(CustomerPredictionRequest):
    pass


class PrioritizedCustomerResponse(BaseModel):
    customer_id: str
    priority_rank: int
    raw_ranking_score: float
    calibrated_response_probability: float
    priority_category: str


class PrioritizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: List[PrioritizationCustomer] = Field(..., min_length=1, max_length=10000)


class PrioritizationResponse(BaseModel):
    ranking_method: str
    probability_method: str
    customers: List[PrioritizedCustomerResponse]


class CapacityAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: List[PrioritizationCustomer] = Field(..., min_length=1, max_length=10000)
    capacity_percentage: float = Field(..., ge=1, le=100)


class CapacityAnalysisResponse(BaseModel):
    capacity_percentage: float
    rounding_method: str
    customers_available: int
    customers_selected: int
    average_calibrated_response_probability: float
    expected_responders_estimate: float
    customers: List[PrioritizedCustomerResponse]
    probability_note: str

