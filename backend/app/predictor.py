"""
Prediction Service Module for AgriShield Backend (SIH 26131).
Validates incoming image data, applies exact training preprocessing,
executes inference via the loaded model, and compiles structured predictions.
"""

from typing import Dict, Any, Union, BinaryIO, Optional
from pathlib import Path
import io
import time
import logging

from PIL import Image, UnidentifiedImageError
import numpy as np
import torch

from .model_loader import ModelManager
from . import config

logger = logging.getLogger("agrishield.backend.predictor")


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails validation (corrupted, empty, or invalid format)."""
    pass


class PredictionError(RuntimeError):
    """Raised when model inference fails."""
    pass


def validate_and_open_image(image_input: Union[bytes, BinaryIO, Image.Image, Path, str]) -> Image.Image:
    """
    Validates and decodes input image into a standardized 3-channel RGB PIL Image.

    Validation criteria:
    - Non-empty byte payload
    - Legitimate image format recognized by Pillow (JPEG, PNG, WebP, BMP)
    - Valid non-zero spatial dimensions (at least 10x10 pixels)
    - Successfully converted to 3-channel RGB color space
    """
    if image_input is None:
        raise ImageValidationError("No image data provided. An image file is required.")

    pil_img: Optional[Image.Image] = None

    if isinstance(image_input, Image.Image):
        pil_img = image_input
    elif isinstance(image_input, bytes):
        if len(image_input) == 0:
            raise ImageValidationError("Uploaded image file is empty (0 bytes).")
        if len(image_input) > config.MAX_IMAGE_SIZE_BYTES:
            size_mb = len(image_input) / (1024 * 1024)
            max_mb = config.MAX_IMAGE_SIZE_BYTES / (1024 * 1024)
            raise ImageValidationError(f"Image size ({size_mb:.1f} MB) exceeds limit of {max_mb:.0f} MB.")
        try:
            pil_img = Image.open(io.BytesIO(image_input))
            # Verify file integrity by loading pixel buffer
            pil_img.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ImageValidationError(
                f"Corrupted or invalid image file. Please provide a valid JPEG, PNG, or WebP image. Error: {str(exc)}"
            ) from exc
    elif isinstance(image_input, (str, Path)):
        file_path = Path(image_input)
        if not file_path.exists():
            raise ImageValidationError(f"Image file not found on disk: {file_path}")
        if file_path.stat().st_size == 0:
            raise ImageValidationError("Specified image file on disk is empty (0 bytes).")
        try:
            pil_img = Image.open(file_path)
            pil_img.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ImageValidationError(f"Failed to open image file: {str(exc)}") from exc
    elif hasattr(image_input, "read"):
        content = image_input.read()
        return validate_and_open_image(content)
    else:
        raise ImageValidationError(
            f"Unsupported image input type: {type(image_input)}. Expected bytes, PIL.Image, or file path."
        )

    # Validate image dimensions
    width, height = pil_img.size
    if width < 10 or height < 10:
        raise ImageValidationError(
            f"Image dimensions ({width}x{height}) are too small. Leaf image must be at least 10x10 pixels."
        )

    # Ensure standardization to 3-channel RGB (handles RGBA, Palette, Grayscale, CMYK)
    if pil_img.mode != "RGB":
        try:
            pil_img = pil_img.convert("RGB")
        except Exception as exc:
            raise ImageValidationError(f"Failed to convert image mode {pil_img.mode} to RGB: {str(exc)}") from exc

    return pil_img


def predict_disease(
    image: Union[bytes, BinaryIO, Image.Image, Path, str],
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Executes end-to-end disease prediction on an uploaded image.

    Workflow:
    1. Validates and opens image.
    2. Applies exact training preprocessing (Resize 224x224, ImageNet normalize).
    3. Runs PyTorch model inference using cached singleton.
    4. Computes softmax probabilities.
    5. Returns structured prediction dictionary with top diagnosis and alternatives.

    Returns:
        Dict:
            - success (bool): True
            - crop (str): e.g. "Tomato"
            - disease (str): e.g. "Early Blight"
            - confidence (float): e.g. 94.6
            - is_healthy (bool): True/False
            - predicted_class (str): e.g. "Tomato___Early_blight"
            - inference_time_ms (float): e.g. 34.2
            - top_predictions (list): top-K alternative candidates
    """
    t0 = time.time()

    # 1. Validate and convert image to RGB
    pil_img = validate_and_open_image(image)

    # 2. Retrieve cached model instance
    try:
        predictor = ModelManager.get_predictor()
    except Exception as exc:
        logger.error(f"Model retrieval failure: {exc}", exc_info=True)
        raise PredictionError(f"Model is not loaded or available: {str(exc)}") from exc

    # 3. Apply exact preprocessing pipeline
    try:
        input_tensor = predictor.preprocess(pil_img)
    except Exception as exc:
        logger.error(f"Preprocessing error: {exc}", exc_info=True)
        raise PredictionError(f"Failed applying preprocessing transform: {str(exc)}") from exc

    # 4. Model inference
    try:
        with torch.no_grad():
            logits = predictor.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)
    except Exception as exc:
        logger.error(f"Inference execution failure: {exc}", exc_info=True)
        raise PredictionError(f"Neural network forward pass failed: {str(exc)}") from exc

    inference_time_ms = round((time.time() - t0) * 1000.0, 2)

    # 5. Extract top-K classes
    num_classes = predictor.num_classes
    k = min(top_k, num_classes)
    top_probs, top_indices = torch.topk(probabilities, k=k)
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()

    top_idx = int(top_indices[0])
    top_meta = predictor.idx_to_meta[top_idx]
    confidence = round(float(top_probs[0]) * 100.0, 2)

    top_candidates = []
    for prob, idx in zip(top_probs, top_indices):
        meta = predictor.idx_to_meta[int(idx)]
        top_candidates.append({
            "crop": meta["crop"],
            "disease": meta["disease"],
            "confidence": round(float(prob) * 100.0, 2),
            "is_healthy": meta["is_healthy"],
            "class_name": meta["raw_name"],
        })

    # 6. Build contract adhering result
    return {
        "success": True,
        "crop": top_meta["crop"],
        "disease": top_meta["disease"],
        "confidence": confidence,
        "is_healthy": top_meta["is_healthy"],
        "predicted_class": top_meta["raw_name"],
        "inference_time_ms": inference_time_ms,
        "top_predictions": top_candidates,
    }
