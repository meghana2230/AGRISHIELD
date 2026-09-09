"""
Configuration module for AgriShield FastAPI Backend (SIH 26131).
Centralizes model paths, preprocessing parameters matching training,
CORS origins, and API server settings.
"""

from pathlib import Path
from typing import List
import os

# Base directory paths
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

# ML Model Paths (reusing trained assets without duplication)
ML_MODEL_DIR = PROJECT_ROOT / "ml-model"
DEFAULT_MODEL_PATH = ML_MODEL_DIR / "models" / "best_model.pth"
DEFAULT_CLASS_INDICES_PATH = ML_MODEL_DIR / "dataset" / "class_indices.json"

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
CLASS_INDICES_PATH = Path(os.getenv("CLASS_INDICES_PATH", str(DEFAULT_CLASS_INDICES_PATH)))

# Exact Preprocessing Parameters (Strictly matching training pipeline)
IMAGE_SIZE: int = 224
IMAGE_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGE_STD: List[float] = [0.229, 0.224, 0.225]
NUM_CLASSES: int = 38
MODEL_NAME: str = "efficientnet_b0"

# Server & Network settings
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

# CORS Configuration
# Allows configurable comma-separated origins, defaulting to standard dev ports and open dev access
_CORS_RAW = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,*")
CORS_ORIGINS: List[str] = [origin.strip() for origin in _CORS_RAW.split(",") if origin.strip()]

# Upload & Validation Constraints
MAX_IMAGE_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "application/octet-stream",  # Fallback for some frontend clients
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
