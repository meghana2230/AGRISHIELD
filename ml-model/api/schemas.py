"""
Pydantic schemas for AgriShield Inference API (SIH 26131).
Enforces structured, validated request and response contracts for frontend integration.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    """Root service information schema."""
    project: str = Field(default="AgriShield", example="AgriShield")
    api: str = Field(default="Crop Disease Detection API", example="Crop Disease Detection API")
    version: str = Field(default="1.0.0", example="1.0.0")
    status: str = Field(default="running", example="running")


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(default="healthy", example="healthy")
    model_loaded: bool = Field(..., description="Whether model weights are loaded in memory", example=True)
    device: str = Field(..., description="Compute device running inference (cpu or cuda)", example="cpu")
    model_name: Optional[str] = Field(default="efficientnet_b0", example="efficientnet_b0")
    num_classes: Optional[int] = Field(default=38, example=38)


class TopPrediction(BaseModel):
    """Individual candidate class prediction."""
    disease: str = Field(..., description="Identified disease pathology", example="Early Blight")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0", example=0.5393)
    confidence_percentage: Optional[float] = Field(default=None, ge=0.0, le=100.0, example=53.93)
    crop: Optional[str] = Field(default=None, description="Crop species", example="Potato")
    class_name: Optional[str] = Field(default=None, description="Raw taxonomic identifier", example="Potato___Early_blight")
    is_healthy: Optional[bool] = Field(default=None, description="Healthy leaf indicator", example=False)


class PredictionDetails(BaseModel):
    """Comprehensive diagnostic detail for the top predicted disease."""
    disease: str = Field(..., description="Primary identified disease pathology", example="Early Blight")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0 to 1.0)", example=0.5393)
    confidence_percentage: float = Field(..., ge=0.0, le=100.0, description="Confidence percentage (0.0 to 100.0)", example=53.93)
    low_confidence: bool = Field(..., description="Flag indicating confidence below threshold", example=False)
    top_predictions: List[TopPrediction] = Field(
        default_factory=list,
        description="Top-3 alternative candidate predictions",
    )
    crop: Optional[str] = Field(default=None, description="Identified crop species", example="Potato")
    predicted_class: Optional[str] = Field(default=None, description="Raw dataset folder class identifier", example="Potato___Early_blight")
    is_healthy: Optional[bool] = Field(default=None, description="Whether the leaf shows no pathology", example=False)
    inference_time_ms: Optional[float] = Field(default=None, description="Inference latency in milliseconds", example=35.2)


# Backward-compatible alias
PredictionDetail = PredictionDetails


class DiseaseInfo(BaseModel):
    """Actionable agronomic information for a specific crop disease or healthy state."""
    disease: str = Field(..., description="Disease or condition name", example="Early Blight")
    crop: str = Field(..., description="Affected crop species", example="Potato")
    is_healthy: Optional[bool] = Field(default=False, description="Whether the class represents healthy foliage", example=False)
    description: str = Field(..., description="Concise pathological description of the condition")
    symptoms: List[str] = Field(default_factory=list, description="Common visible leaf/foliage symptoms")
    management: List[str] = Field(default_factory=list, description="Actionable, safe agronomic management and prevention guidance")


class HistoryRecord(BaseModel):
    """Prediction history record structure prepared for future persistence."""
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of inference", example="2026-09-09T15:45:00Z")
    crop: str = Field(..., example="Potato")
    disease: str = Field(..., example="Early Blight")
    confidence: float = Field(..., example=0.5393)
    confidence_percentage: float = Field(..., example=53.93)
    top_predictions: List[TopPrediction] = Field(default_factory=list)
    image_reference: Optional[str] = Field(default=None, description="Future image file or URL identifier")


class PredictionResponse(BaseModel):
    """
    Standard Prediction Response returned by POST /predict.
    Enriched with actionable disease_info, decision-support disclaimer, and history structure.
    """
    success: bool = Field(default=True, example=True)
    prediction: PredictionDetails
    disease_info: Optional[DiseaseInfo] = Field(
        default=None,
        description="Actionable crop disease description, symptoms, and management guidance",
    )
    disclaimer: str = Field(
        default="This AI prediction is intended as a decision-support aid. Confirm important crop-treatment decisions with a qualified agricultural expert.",
        description="Advisory disclaimer for AI-assisted crop diagnosis",
    )
    history_record: Optional[HistoryRecord] = Field(
        default=None,
        description="Prediction record structure ready for future history storage",
    )


class ErrorResponse(BaseModel):
    """Standardized error response preventing stack trace leakage."""
    success: bool = Field(default=False, example=False)
    error: str = Field(..., description="Category of the error", example="Invalid Image")
    detail: Optional[str] = Field(default=None, description="Specific error description", example="Unsupported file format")
