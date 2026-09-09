"""
Inference and Model Loading Module for AgriShield (SIH 26131).
Provides singleton/cached model loading, exact preprocessing replication,
and production-ready crop disease prediction for backend integration.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import io
import json
import logging
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# Ensure ml-model is importable
BASE_DIR = Path(__file__).resolve().parent.parent
import sys
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import config
from src.model import create_model
from src.dataset import get_val_test_transforms

logger = logging.getLogger(__name__)


class DiseasePredictor:
    """
    Production-ready Model Loader & Predictor.
    Loads the trained model checkpoint once into memory, caches class metadata,
    and applies deterministic preprocessing matching the training pipeline.
    """

    _instance: Optional["DiseasePredictor"] = None

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        class_indices_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or config.DEVICE
        self.model_path = Path(model_path or config.BEST_MODEL_PATH)
        self.class_indices_path = Path(class_indices_path or config.CLASS_INDICES_PATH)

        logger.info(f"Initializing DiseasePredictor on device: {self.device}")
        self._load_class_mapping()
        self._load_model()
        self._setup_transforms()

    @classmethod
    def get_instance(
        cls,
        model_path: Optional[Union[str, Path]] = None,
        class_indices_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ) -> "DiseasePredictor":
        """Singleton accessor ensuring model weights are loaded only once in memory."""
        if cls._instance is None:
            cls._instance = cls(
                model_path=model_path,
                class_indices_path=class_indices_path,
                device=device,
            )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Resets the singleton instance if reloading weights is needed."""
        cls._instance = None

    def _load_class_mapping(self) -> None:
        """Loads and parses the 38-class taxonomic mapping."""
        if not self.class_indices_path.exists():
            raise FileNotFoundError(f"Class mapping file not found at: {self.class_indices_path}")

        with open(self.class_indices_path, "r", encoding="utf-8") as f:
            self.class_indices = json.load(f)

        self.num_classes = len(self.class_indices)
        self.idx_to_meta: Dict[int, Dict[str, Any]] = {
            int(k): v for k, v in self.class_indices.items()
        }
        logger.info(f"Loaded {self.num_classes} classes from {self.class_indices_path.name}")

    def _load_model(self) -> None:
        """Loads the model architecture and restored weights from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {self.model_path}. "
                f"Please ensure a model has been saved before running predictions."
            )

        logger.info(f"Loading trained weights from checkpoint: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Extract checkpoint metadata
        model_name = checkpoint.get("model_name", config.MODEL_NAME)
        num_classes = checkpoint.get("num_classes", config.NUM_CLASSES)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Build architecture without downloading fresh ImageNet weights
        self.model = create_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
            dropout_rate=config.DROPOUT_RATE,
        )

        # Load weights
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.model_name = model_name
        self.val_acc = checkpoint.get("val_acc", None)
        logger.info(
            f"Successfully loaded {model_name} (Classes={num_classes}, "
            f"Checkpoint Val Acc={self.val_acc if self.val_acc is not None else 'N/A'}%)"
        )

    def _setup_transforms(self) -> None:
        """Configures exact deterministic preprocessing matching validation/testing."""
        self.image_size = config.IMAGE_SIZE
        self.image_mean = config.IMAGE_MEAN
        self.image_std = config.IMAGE_STD
        self.transform = get_val_test_transforms(image_size=self.image_size)

    def preprocess(self, image_input: Union[str, Path, bytes, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Standardizes input image to tensor:
        1. Opens image from file path, raw bytes, numpy array, or PIL Image.
        2. Converts to 3-channel RGB.
        3. Resizes to 224x224.
        4. Normalizes with ImageNet channel statistics.
        5. Returns batch tensor of shape [1, 3, 224, 224].
        """
        # 1. Convert to RGB PIL Image
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Image file does not exist: {img_path}")
            pil_img = Image.open(img_path).convert("RGB")
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                pil_img = Image.fromarray(image_input).convert("RGB")
            elif image_input.shape[2] == 4:
                pil_img = Image.fromarray(image_input).convert("RGB")
            else:
                pil_img = Image.fromarray(image_input)
        else:
            raise TypeError(
                f"Unsupported image input type: {type(image_input)}. "
                "Expected file path, bytes, PIL.Image, or numpy.ndarray."
            )

        # 2. Convert to numpy array for Albumentations
        np_img = np.array(pil_img)

        # 3. Apply exact preprocessing pipeline
        augmented = self.transform(image=np_img)
        tensor_img = augmented["image"]  # shape: [3, 224, 224]

        # 4. Add batch dimension: [1, 3, 224, 224]
        batch_tensor = tensor_img.unsqueeze(0).to(self.device)
        return batch_tensor

    def predict(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference on an image and returns structured predictions.

        Returns:
            Dict containing:
                - crop: Identified crop species (e.g., 'Tomato')
                - disease: Specific pathological condition (e.g., 'Early Blight')
                - confidence: Confidence percentage (e.g., 94.6)
                - predicted_class: Raw class identifier
                - is_healthy: Boolean indicating if leaf is healthy
                - top_predictions: Top-K alternative predictions with confidence
                - inference_time_ms: Latency in milliseconds
        """
        t0 = time.time()
        input_tensor = self.preprocess(image_input)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)

        inference_time_ms = round((time.time() - t0) * 1000.0, 2)

        # Extract top-K predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, self.num_classes))
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        top_pred_idx = int(top_indices[0])
        top_meta = self.idx_to_meta[top_pred_idx]
        confidence_percent = round(float(top_probs[0]) * 100.0, 2)

        # Compile top-K list
        top_list = []
        for prob, idx in zip(top_probs, top_indices):
            meta = self.idx_to_meta[int(idx)]
            top_list.append({
                "crop": meta["crop"],
                "disease": meta["disease"],
                "confidence": round(float(prob) * 100.0, 2),
                "is_healthy": meta["is_healthy"],
                "class_name": meta["raw_name"],
            })

        # Structured result adhering to backend integration contract
        result = {
            "crop": top_meta["crop"],
            "disease": top_meta["disease"],
            "confidence": confidence_percent,
            "is_healthy": top_meta["is_healthy"],
            "predicted_class": top_meta["raw_name"],
            "class_index": top_pred_idx,
            "top_predictions": top_list,
            "inference_time_ms": inference_time_ms,
        }
        return result


# ---------------------------------------------------------------------------
# Global Helper Function for Direct API & Backend Calls
# ---------------------------------------------------------------------------

def predict_disease(
    image: Union[str, Path, bytes, Image.Image, np.ndarray],
    model_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Standard entry-point function for backend integration.
    Loads model once via singleton and executes prediction.

    Example output:
    {
        "crop": "Tomato",
        "disease": "Early Blight",
        "confidence": 94.6
    }
    """
    predictor = DiseasePredictor.get_instance(model_path=model_path)
    full_pred = predictor.predict(image)

    # Simplified contract requested by user, with rich metadata preserved
    return {
        "crop": full_pred["crop"],
        "disease": full_pred["disease"],
        "confidence": full_pred["confidence"],
        "is_healthy": full_pred["is_healthy"],
        "predicted_class": full_pred["predicted_class"],
        "inference_time_ms": full_pred["inference_time_ms"],
        "top_predictions": full_pred["top_predictions"],
    }
