"""
Model Loader Module for AgriShield FastAPI Backend (SIH 26131).
Ensures the trained EfficientNet-B0 model checkpoint and 38-class taxonomy
are loaded into memory exactly ONCE on application startup, without retraining.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
import sys

from . import config

# Ensure ml-model is importable to reuse exact model architecture & preprocessing pipeline
if str(config.ML_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(config.ML_MODEL_DIR))

from src.predict import DiseasePredictor

logger = logging.getLogger("agrishield.backend.model_loader")


class ModelManager:
    """
    Manages the lifecycle of the trained ML model.
    Guarantees single in-memory instantiation on startup and fast inference access.
    """
    _predictor: Optional[DiseasePredictor] = None

    @classmethod
    def load(cls) -> DiseasePredictor:
        """
        Loads the trained model once during FastAPI startup.
        Uses the exact weights, class mappings, and preprocessing transforms from training.
        """
        if cls._predictor is None:
            logger.info("Initializing trained model weights from disk...")
            logger.info(f"Model Checkpoint Path : {config.MODEL_PATH}")
            logger.info(f"Class Mapping Path    : {config.CLASS_INDICES_PATH}")

            if not config.MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model checkpoint not found at: {config.MODEL_PATH}. "
                    "Ensure Part 5 training/checkpoint is completed."
                )
            if not config.CLASS_INDICES_PATH.exists():
                raise FileNotFoundError(
                    f"Class indices file not found at: {config.CLASS_INDICES_PATH}."
                )

            cls._predictor = DiseasePredictor.get_instance(
                model_path=config.MODEL_PATH,
                class_indices_path=config.CLASS_INDICES_PATH,
            )
            logger.info(
                f"Model successfully loaded: {cls._predictor.model_name} "
                f"({cls._predictor.num_classes} classes on device: {cls._predictor.device})"
            )
        return cls._predictor

    @classmethod
    def get_predictor(cls) -> DiseasePredictor:
        """Returns the loaded predictor instance. Loads if not already loaded."""
        if cls._predictor is None:
            return cls.load()
        return cls._predictor

    @classmethod
    def is_loaded(cls) -> bool:
        """Checks if model is currently loaded in memory."""
        return cls._predictor is not None

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Provides diagnostic metadata about the loaded model."""
        if cls._predictor is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "model_name": cls._predictor.model_name,
            "num_classes": cls._predictor.num_classes,
            "image_size": cls._predictor.image_size,
            "device": str(cls._predictor.device),
            "val_acc": cls._predictor.val_acc,
        }


def get_model() -> DiseasePredictor:
    """FastAPI Dependency for obtaining the loaded model instance."""
    return ModelManager.get_predictor()
