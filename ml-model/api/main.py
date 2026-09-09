"""
FastAPI Application Entrypoint for AgriShield Inference Backend (SIH 26131).
Exposes /predict, /health, / documentation endpoints with robust error handling.
"""

from contextlib import asynccontextmanager
from typing import Optional, Dict
from pathlib import Path
from datetime import datetime, timezone
import logging
import sys

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure api directory and ml-model root are importable
API_DIR = Path(__file__).resolve().parent
ML_MODEL_DIR = API_DIR.parent

if str(ML_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(ML_MODEL_DIR))
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from api import config
from api.schemas import (
    RootResponse,
    HealthResponse,
    PredictionResponse,
    PredictionDetail,
    TopPrediction,
    DiseaseInfo,
    HistoryRecord,
    ErrorResponse,
)
from api.model_service import (
    ModelService,
    predict_image,
    ModelNotFoundError,
    ImageValidationError,
    InferenceError,
)
from api.disease_service import DiseaseService, get_disease_info_for_prediction

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agrishield.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Initializes the trained model once into RAM upon server startup.
    """
    logger.info("Initializing AgriShield ML Inference Backend...")
    try:
        ModelService.get_instance()
        logger.info("Model loaded successfully into memory during startup.")
    except Exception as exc:
        logger.warning(f"Model initialization deferred or failed on startup: {exc}")
    yield
    logger.info("AgriShield ML Inference Backend shutting down.")


# Initialize FastAPI App
app = FastAPI(
    title="AgriShield Crop Disease Inference API",
    description=(
        "SIH 26131 — Production-grade AI/ML inference service for detecting "
        "crop diseases across 38 categories using deep transfer learning (EfficientNet-B0)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Configure CORS Middleware for Frontend Development
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Structured Error Handlers (Zero Python Stack Traces Exposed)
# ---------------------------------------------------------------------------

@app.exception_handler(ImageValidationError)
async def handle_image_validation_error(request: Request, exc: ImageValidationError):
    """Handles image validation errors (400 Bad Request)."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            success=False,
            error="Image Validation Error",
            detail=str(exc),
        ).model_dump(),
    )


@app.exception_handler(ModelNotFoundError)
async def handle_model_not_found_error(request: Request, exc: ModelNotFoundError):
    """Handles missing model weights (503 Service Unavailable)."""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            success=False,
            error="Model Unavailable",
            detail=str(exc),
        ).model_dump(),
    )


@app.exception_handler(InferenceError)
async def handle_inference_error(request: Request, exc: InferenceError):
    """Handles forward pass errors (500 Internal Server Error)."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            success=False,
            error="Inference Execution Error",
            detail=str(exc),
        ).model_dump(),
    )


from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    """Handles FastAPI parameter & multipart validation errors (422 Unprocessable Entity)."""
    error_messages = [f"{err['loc'][-1]}: {err['msg']}" for err in exc.errors()]
    detail_str = "; ".join(error_messages) if error_messages else "Request validation failed"
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            success=False,
            error="Validation Error",
            detail=detail_str,
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def handle_http_exception(request: Request, exc: HTTPException):
    """Standardizes FastAPI HTTP exceptions into consistent error schema."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error="Request Error",
            detail=str(exc.detail),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def handle_unhandled_exception(request: Request, exc: Exception):
    """Catches all unexpected exceptions and returns safe JSON without leaking stack traces."""
    logger.error(f"Unhandled server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            success=False,
            error="Internal Server Error",
            detail="An internal server error occurred while processing the request.",
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/",
    response_model=RootResponse,
    summary="API Root Information",
    tags=["General"],
)
async def root():
    """Returns basic service details, project name, version, and status."""
    return RootResponse(
        project="AgriShield",
        api="Crop Disease Detection API",
        version="1.0.0",
        status="running",
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health & Model Status Check",
    tags=["Health"],
)
async def health_check():
    """
    Returns the operational status of the service, model loading state,
    and active computation device (cpu or cuda).
    """
    model_loaded = ModelService.is_loaded()
    device_name = ModelService.get_device_name()

    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        device=device_name,
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
    )


@app.get(
    "/diseases",
    response_model=Dict[str, DiseaseInfo],
    summary="List All Supported Crop Diseases & Guidance",
    tags=["Knowledge Base"],
)
async def list_diseases():
    """Returns actionable pathological descriptions, symptoms, and management for all 38 classes."""
    return DiseaseService.get_all()


@app.get(
    "/diseases/{disease_name}",
    response_model=DiseaseInfo,
    summary="Get Disease Guidance by Name or Taxonomy Key",
    tags=["Knowledge Base"],
    responses={
        200: {"description": "Disease information found."},
        404: {"description": "Disease not found in 38-class taxonomy.", "model": ErrorResponse},
    },
)
async def get_disease(disease_name: str):
    """Retrieves symptoms and management advice for a specific crop disease or healthy state."""
    info = DiseaseService.find_by_key_or_name(disease_name)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Disease '{disease_name}' not found. Supported classes belong to the 38-class PlantVillage taxonomy.",
        )
    return DiseaseInfo(**info)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Crop Disease from Leaf Image",
    tags=["Inference"],
    responses={
        200: {"description": "Successful inference diagnosis."},
        400: {"description": "Corrupted, invalid, or empty image.", "model": ErrorResponse},
        413: {"description": "Image exceeds maximum allowed size.", "model": ErrorResponse},
        415: {"description": "Unsupported media/image format.", "model": ErrorResponse},
        500: {"description": "Inference computation error.", "model": ErrorResponse},
        503: {"description": "Model checkpoint not loaded.", "model": ErrorResponse},
    },
)
async def predict_disease_endpoint(
    file: UploadFile = File(
        ...,
        description="Crop leaf image file (JPG, JPEG, PNG, or WebP). Maximum size 15 MB.",
    ),
):
    """
    Receives an uploaded crop leaf image, validates content securely,
    applies exact training preprocessing, executes EfficientNet-B0 inference,
    attaches actionable disease symptoms and management guidance, and returns JSON.
    """
    # 1. Validate file presence
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded. Request form must include a 'file' field.",
        )

    # 2. File extension check (preliminary guard, not sole source of truth)
    ext = Path(file.filename).suffix.lower()
    if ext and ext not in config.ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file extension '{ext}'. "
                f"Supported image formats are: {', '.join(sorted(config.ALLOWED_FILE_EXTENSIONS))}"
            ),
        )

    # 3. Read image binary content
    try:
        file_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file content: {str(exc)}",
        )

    # 4. Check for empty payload
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes). Please upload a valid image file.",
        )

    # 5. Check size limit
    if len(file_bytes) > config.MAX_UPLOAD_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        max_mb = config.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded image ({size_mb:.1f} MB) exceeds maximum allowed size of {max_mb:.0f} MB.",
        )

    # 6. Execute validated inference
    prediction_raw = predict_image(file_bytes)

    # 7. Compile structured Pydantic response
    top_candidates = [
        TopPrediction(
            disease=c["disease"],
            confidence=c["confidence"],
            confidence_percentage=c.get("confidence_percentage"),
            crop=c.get("crop"),
            class_name=c.get("class_name"),
            is_healthy=c.get("is_healthy"),
        )
        for c in prediction_raw["top_predictions"]
    ]

    detail = PredictionDetail(
        disease=prediction_raw["disease"],
        crop=prediction_raw["crop"],
        confidence=prediction_raw["confidence"],
        confidence_percentage=prediction_raw["confidence_percentage"],
        predicted_class=prediction_raw["predicted_class"],
        is_healthy=prediction_raw["is_healthy"],
        low_confidence=prediction_raw["low_confidence"],
        inference_time_ms=prediction_raw["inference_time_ms"],
        top_predictions=top_candidates,
    )

    # 8. Retrieve actionable disease information
    disease_info_dict = get_disease_info_for_prediction(
        prediction_raw["predicted_class"],
        prediction_raw["disease"],
        prediction_raw["crop"],
    )
    disease_info_obj = DiseaseInfo(**disease_info_dict) if disease_info_dict else None

    # 9. Prepare future prediction history record structure
    now_iso = datetime.now(timezone.utc).isoformat()
    history_obj = HistoryRecord(
        timestamp=now_iso,
        crop=prediction_raw["crop"],
        disease=prediction_raw["disease"],
        confidence=prediction_raw["confidence"],
        confidence_percentage=prediction_raw["confidence_percentage"],
        top_predictions=top_candidates,
        image_reference=file.filename,
    )

    return PredictionResponse(
        success=True,
        prediction=detail,
        disease_info=disease_info_obj,
        history_record=history_obj,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=config.API_HOST, port=config.API_PORT, reload=False)

