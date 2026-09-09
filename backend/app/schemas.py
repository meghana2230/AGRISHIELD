"""
Pydantic Schemas for Request/Response Models (SIH 26131 AgriShield).
Adheres strictly to the expected API contract for frontend integration.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(default="ok", description="Service health status", example="ok")
    model_loaded: bool = Field(..., description="Whether the ML model is successfully loaded in memory", example=True)
    model_name: Optional[str] = Field(default="efficientnet_b0", description="Loaded model architecture", example="efficientnet_b0")
    num_classes: Optional[int] = Field(default=38, description="Number of recognized disease classes", example=38)


class PredictionCandidate(BaseModel):
    """Top-K candidate prediction detail."""
    crop: str = Field(..., description="Crop name", example="Tomato")
    disease: str = Field(..., description="Pathology or condition", example="Early Blight")
    confidence: float = Field(..., description="Confidence score percentage (0-100)", example=94.6)
    is_healthy: bool = Field(..., description="Whether the crop leaf is healthy", example=False)
    class_name: str = Field(..., description="Raw dataset class identifier", example="Tomato___Early_blight")


class PredictionResponse(BaseModel):
    """
    Standard prediction response schema.
    Guarantees top-level 'success', 'crop', 'disease', and 'confidence'
    while providing extended diagnostic details for UI richness.
    """
    success: bool = Field(default=True, description="Indicates inference success", example=True)
    crop: str = Field(..., description="Predicted crop species", example="Tomato")
    disease: str = Field(..., description="Identified disease diagnosis", example="Early Blight")
    confidence: float = Field(..., description="Classification confidence percentage (0.0 - 100.0)", example=94.6)
    is_healthy: Optional[bool] = Field(default=None, description="Healthy leaf indicator", example=False)
    predicted_class: Optional[str] = Field(default=None, description="Raw taxonomic class label", example="Tomato___Early_blight")
    inference_time_ms: Optional[float] = Field(default=None, description="Latency in milliseconds", example=32.4)
    top_predictions: Optional[List[PredictionCandidate]] = Field(default=None, description="Top candidate predictions")


class ErrorResponse(BaseModel):
    """Standard error response structure."""
    success: bool = Field(default=False, description="Always False on failure", example=False)
    error: str = Field(..., description="High-level error description", example="Invalid image file")
    detail: Optional[str] = Field(default=None, description="Detailed error explanation", example="Uploaded file is not a valid JPEG/PNG image")
