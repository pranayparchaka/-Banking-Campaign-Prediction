"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .data_service import ProjectDataService
from .ml_service import BankingModelService
from .routes import router

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_service = BankingModelService(PROJECT_ROOT)
    app.state.data_service = ProjectDataService(PROJECT_ROOT)
    yield


app = FastAPI(
    title="Banking Campaign Response Prediction API",
    description="Prediction, calibrated probability estimation, customer prioritization, and contact capacity analysis.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Request validation failed.",
                "details": exc.errors(),
            }
        },
    )


app.include_router(router)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

