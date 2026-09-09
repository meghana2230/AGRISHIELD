"""
Configuration module for AgriShield Inference API (SIH 26131).
Inherits base ML settings from src.config and provides API-specific parameters.
"""

from pathlib import Path
from typing import List, Set
import os
import torch

# Base directories
API_DIR = Path(__file__).resolve().parent
ML_MODEL_DIR = API_DIR.parent
PROJECT_ROOT = ML_MODEL_DIR.parent

# Import base ML configuration
import sys
if str(ML_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(ML_MODEL_DIR))

from src import config as ml_config

# Model and Dataset Paths
MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", str(ml_config.BEST_MODEL_PATH)))
CLASS_INDICES_PATH: Path = Path(os.getenv("CLASS_INDICES_PATH", str(ml_config.CLASS_INDICES_PATH)))

# Hardware Execution Device (Automatic CUDA / CPU selection)
DEVICE_OVERRIDE = os.getenv("DEVICE", "").lower()
if DEVICE_OVERRIDE in ("cuda", "cpu"):
    DEVICE = torch.device(DEVICE_OVERRIDE)
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inference Hyperparameters
IMAGE_SIZE: int = ml_config.IMAGE_SIZE
NUM_CLASSES: int = ml_config.NUM_CLASSES
MODEL_NAME: str = ml_config.MODEL_NAME
IMAGE_MEAN: List[float] = ml_config.IMAGE_MEAN
IMAGE_STD: List[float] = ml_config.IMAGE_STD

# Decision Thresholds & Top-K Ranking
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
TOP_K: int = int(os.getenv("TOP_K", "3"))

# Upload & File Constraints
MAX_UPLOAD_SIZE_BYTES: int = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(15 * 1024 * 1024)))  # 15 MB
ALLOWED_MIME_TYPES: Set[str] = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "application/octet-stream",
}
ALLOWED_FILE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}

# Server Host & Port (Supports standard cloud PORT variable)
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("PORT") or os.getenv("API_PORT", "8000"))

# CORS Configuration (Restricted to development origins, configurable via env)
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:4173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:4173",
]
_ENV_CORS = os.getenv("CORS_ORIGINS")
if _ENV_CORS:
    _custom_origins = [origin.strip() for origin in _ENV_CORS.split(",") if origin.strip()]
    # Combine custom production origins with local dev origins without duplicates
    CORS_ORIGINS: List[str] = list(dict.fromkeys(DEFAULT_CORS_ORIGINS + _custom_origins))
else:
    CORS_ORIGINS: List[str] = DEFAULT_CORS_ORIGINS
