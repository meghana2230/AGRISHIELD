"""
Main FastAPI Application Entrypoint for AgriShield (SIH 26131).
Exposes REST endpoints for model health and crop disease diagnosis.
"""

from contextlib import asynccontextmanager
from typing import Optional
import logging
import sys

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import config
from .schemas import HealthResponse, PredictionResponse, ErrorResponse
from .model_loader import ModelManager
from .predictor import predict_disease, ImageValidationError, PredictionError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agrishield.backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Loads the trained model once into memory on startup and ensures zero retraining.
    """
    logger.info("Starting AgriShield Backend Service...")
    try:
        ModelManager.load()
        logger.info("ML Model and Class Mapping loaded successfully into memory.")
    except Exception as exc:
        logger.error(f"Failed to load ML model on startup: {exc}", exc_info=True)
    yield
    logger.info("Shutting down AgriShield Backend Service...")


# Initialize FastAPI Application
app = FastAPI(
    title="AgriShield Crop Disease Detection API",
    description=(
        "SIH 26131 — AI/ML Backend Service for real-time crop disease diagnosis "
        "using deep transfer learning (EfficientNet-B0) across 38 crop disease classes."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Configure CORS for Frontend Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS if "*" not in config.CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global Exception Handlers (Prevent Unhandled Crashes & Guarantee JSON Responses)
# ---------------------------------------------------------------------------

@app.exception_handler(ImageValidationError)
async def image_validation_exception_handler(request: Request, exc: ImageValidationError):
    """Handles image validation errors (400 Bad Request)."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            success=False,
            error="Image Validation Error",
            detail=str(exc),
        ).model_dump(),
    )


@app.exception_handler(PredictionError)
async def prediction_exception_handler(request: Request, exc: PredictionError):
    """Handles internal prediction pipeline errors (500 Internal Server Error)."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            success=False,
            error="Prediction Processing Error",
            detail=str(exc),
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Standardizes FastAPI HTTP exceptions into consistent JSON structure."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error=exc.detail if isinstance(exc.detail, str) else "HTTP Error",
            detail=str(exc.detail),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catches any unexpected exceptions and returns safe 500 JSON without crashing."""
    logger.error(f"Unhandled server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            success=False,
            error="Internal Server Error",
            detail="An unexpected error occurred during processing.",
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/",
    summary="Root Service Information",
    tags=["General"],
)
async def root():
    """Provides basic service information and documentation links."""
    return {
        "service": "AgriShield Crop Disease Detection API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_endpoint": "/api/health",
        "prediction_endpoint": "/api/predict",
    }


@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health & Model Status Check",
    tags=["System"],
    responses={
        200: {"description": "Service is healthy and model status reported."},
    },
)
async def check_health():
    """
    Health check endpoint for the backend service and model readiness.
    Used by frontend to verify that the ML backend is active and ready for inference.
    """
    model_loaded = ModelManager.is_loaded()
    model_info = ModelManager.get_info()

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        model_name=model_info.get("model_name", config.MODEL_NAME),
        num_classes=model_info.get("num_classes", config.NUM_CLASSES),
    )


@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    summary="Predict Crop Disease from Leaf Image",
    tags=["Inference"],
    responses={
        200: {
            "description": "Successful crop disease prediction.",
            "model": PredictionResponse,
        },
        400: {
            "description": "Bad Request (missing, corrupt, or invalid image format).",
            "model": ErrorResponse,
        },
        500: {
            "description": "Internal Server Error during inference.",
            "model": ErrorResponse,
        },
    },
)
async def predict_crop_disease(
    file: UploadFile = File(
        ...,
        description="Crop leaf image file (JPEG, PNG, or WebP). Maximum size 15 MB.",
    ),
):
    """
    Receives an uploaded leaf image, executes deep learning inference,
    and returns identified crop species, disease condition, and confidence score.

    - **file**: Uploaded leaf image in multipart/form-data.
    """
    # 1. Check for filename / presence
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided in request. Form field 'file' must contain an image.",
        )

    # 2. Read image content
    try:
        contents = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed reading uploaded file: {str(exc)}",
        )

    if not contents or len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes). Please upload a valid image.",
        )

    # 3. Execute prediction pipeline (handles decoding, RGB conversion, transforms, inference)
    result = predict_disease(contents)

    return PredictionResponse(
        success=result["success"],
        crop=result["crop"],
        disease=result["disease"],
        confidence=result["confidence"],
        is_healthy=result["is_healthy"],
        predicted_class=result["predicted_class"],
        inference_time_ms=result["inference_time_ms"],
        top_predictions=result["top_predictions"],
    )
