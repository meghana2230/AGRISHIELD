"""
Model Service for AgriShield Inference API (SIH 26131).
Loads the trained EfficientNet-B0 checkpoint once in evaluation mode,
configures automatic CUDA/CPU execution, reuses existing preprocessing,
and provides robust, validated image inference.
"""

from typing import Dict, Any, Union, BinaryIO, Optional
from pathlib import Path
import io
import time
import logging

from PIL import Image, UnidentifiedImageError
import torch

from . import config

# Ensure ml-model root is in sys.path
import sys
if str(config.ML_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(config.ML_MODEL_DIR))

from src.predict import DiseasePredictor

logger = logging.getLogger("agrishield.api.model_service")


class ModelNotFoundError(FileNotFoundError):
    """Raised when the requested model checkpoint file does not exist on disk."""
    pass


class ImageValidationError(ValueError):
    """Raised when input image data is corrupted, empty, or an unsupported type."""
    pass


class InferenceError(RuntimeError):
    """Raised when forward pass inference fails."""
    pass


class ModelService:
    """
    Singleton service managing the trained model lifecycle.
    Guarantees that model weights and taxonomy are loaded into memory exactly ONCE.
    """
    _instance: Optional["ModelService"] = None
    _predictor: Optional[DiseasePredictor] = None

    def __init__(self):
        self.device = config.DEVICE
        self.model_path = config.MODEL_PATH
        self.class_indices_path = config.CLASS_INDICES_PATH
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        self.top_k = config.TOP_K

        logger.info(f"Initializing ModelService on compute device: {self.device}")
        self._load_predictor()

    def _load_predictor(self) -> None:
        """Loads the trained weights and sets evaluation mode."""
        if not self.model_path.exists():
            raise ModelNotFoundError(
                f"Trained model checkpoint not found at: {self.model_path}. "
                "Ensure Part 5 model training has produced best_model.pth."
            )
        if not self.class_indices_path.exists():
            raise FileNotFoundError(
                f"Class taxonomy indices not found at: {self.class_indices_path}."
            )

        # Reuses exact DiseasePredictor from src.predict
        self._predictor = DiseasePredictor.get_instance(
            model_path=self.model_path,
            class_indices_path=self.class_indices_path,
            device=self.device,
        )
        self._predictor.model.eval()
        logger.info(
            f"ModelService ready: {self._predictor.model_name} "
            f"({self._predictor.num_classes} classes, eval mode on {self.device})"
        )

    @classmethod
    def get_instance(cls) -> "ModelService":
        """Singleton accessor ensuring only one instance exists across requests."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def is_loaded(cls) -> bool:
        """Checks if the service and predictor are actively loaded in memory."""
        return cls._instance is not None and cls._instance._predictor is not None

    @classmethod
    def get_device_name(cls) -> str:
        """Returns the device identifier string (CPU or CUDA)."""
        if cls._instance and cls._instance.device:
            return "cuda" if cls._instance.device.type == "cuda" else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def validate_and_decode_image(
        self,
        image_input: Union[bytes, BinaryIO, Image.Image, Path, str],
    ) -> Image.Image:
        """
        Validates the incoming image content and decodes it to a 3-channel RGB PIL Image.
        Rejects corrupted payloads, zero-length files, and oversized images.
        """
        if image_input is None:
            raise ImageValidationError("No image data provided. An image file is required.")

        pil_img: Optional[Image.Image] = None

        if isinstance(image_input, Image.Image):
            pil_img = image_input
        elif isinstance(image_input, bytes):
            if len(image_input) == 0:
                raise ImageValidationError("Uploaded image file is empty (0 bytes).")
            if len(image_input) > config.MAX_UPLOAD_SIZE_BYTES:
                size_mb = len(image_input) / (1024 * 1024)
                max_mb = config.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
                raise ImageValidationError(
                    f"File size ({size_mb:.1f} MB) exceeds maximum allowed size ({max_mb:.0f} MB)."
                )
            try:
                pil_img = Image.open(io.BytesIO(image_input))
                pil_img.load()  # Force reading pixel buffer to detect corruption
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise ImageValidationError(
                    f"Corrupted or invalid image data. Supported formats: JPEG, PNG, WebP. Error: {str(exc)}"
                ) from exc
        elif isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise ImageValidationError(f"Image path does not exist on disk: {img_path}")
            if img_path.stat().st_size == 0:
                raise ImageValidationError("Image file on disk is empty (0 bytes).")
            try:
                pil_img = Image.open(img_path)
                pil_img.load()
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise ImageValidationError(f"Could not open image file: {str(exc)}") from exc
        elif hasattr(image_input, "read"):
            content = image_input.read()
            return self.validate_and_decode_image(content)
        else:
            raise ImageValidationError(f"Unsupported image input type: {type(image_input)}")

        # Validate minimum spatial dimensions
        width, height = pil_img.size
        if width < 10 or height < 10:
            raise ImageValidationError(
                f"Image dimensions ({width}x{height}) are too small. Minimum required is 10x10 pixels."
            )

        # Ensure standard 3-channel RGB mode
        if pil_img.mode != "RGB":
            try:
                pil_img = pil_img.convert("RGB")
            except Exception as exc:
                raise ImageValidationError(f"Failed converting image mode {pil_img.mode} to RGB: {exc}")

        return pil_img

    def predict_image(
        self,
        image: Union[bytes, BinaryIO, Image.Image, Path, str],
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference on an uploaded image.

        1. Validates and decodes image.
        2. Applies exact Albumentations resize (224x224) and ImageNet normalization.
        3. Runs EfficientNet-B0 inference with torch.no_grad().
        4. Calculates softmax probabilities.
        5. Evaluates confidence against confidence threshold.
        6. Returns structured diagnosis and top-K candidate predictions.
        """
        t0 = time.time()
        k = top_k or self.top_k

        # 1. Image Validation
        pil_img = self.validate_and_decode_image(image)

        # 2. Preprocessing via established pipeline
        try:
            input_tensor = self._predictor.preprocess(pil_img)
            # Ensure input tensor is placed on the active compute device
            input_tensor = input_tensor.to(self.device)
        except Exception as exc:
            logger.error(f"Preprocessing error: {exc}", exc_info=True)
            raise InferenceError(f"Image preprocessing transform failed: {str(exc)}") from exc

        # 3. Model Inference (evaluation mode, no gradients)
        try:
            with torch.no_grad():
                logits = self._predictor.model(input_tensor)
                probabilities = torch.softmax(logits, dim=1).squeeze(0)
        except Exception as exc:
            logger.error(f"Model forward pass error: {exc}", exc_info=True)
            raise InferenceError(f"Neural network inference failed: {str(exc)}") from exc

        inference_time_ms = round((time.time() - t0) * 1000.0, 2)

        # 4. Extract Top-K
        num_classes = self._predictor.num_classes
        k_val = min(k, num_classes)
        top_probs, top_indices = torch.topk(probabilities, k=k_val)
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        top_pred_idx = int(top_indices[0])
        top_meta = self._predictor.idx_to_meta[top_pred_idx]

        # Confidence value between 0 and 1
        confidence_val = round(float(top_probs[0]), 4)
        confidence_percentage = round(confidence_val * 100.0, 2)

        # Evaluate low-confidence threshold without fabricating disease
        is_low_confidence = bool(confidence_val < self.confidence_threshold)

        # Top candidates compilation
        top_predictions_list = []
        for prob, idx in zip(top_probs, top_indices):
            meta = self._predictor.idx_to_meta[int(idx)]
            prob_float = round(float(prob), 4)
            top_predictions_list.append({
                "disease": meta["disease"],
                "crop": meta["crop"],
                "confidence": prob_float,
                "confidence_percentage": round(prob_float * 100.0, 2),
                "class_name": meta["raw_name"],
                "is_healthy": meta["is_healthy"],
            })

        return {
            "disease": top_meta["disease"],
            "crop": top_meta["crop"],
            "confidence": confidence_val,
            "confidence_percentage": confidence_percentage,
            "predicted_class": top_meta["raw_name"],
            "is_healthy": top_meta["is_healthy"],
            "low_confidence": is_low_confidence,
            "inference_time_ms": inference_time_ms,
            "top_predictions": top_predictions_list,
        }


def predict_image(
    image: Union[bytes, BinaryIO, Image.Image, Path, str],
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Module-level entry-point for predicting crop disease from an image."""
    service = ModelService.get_instance()
    return service.predict_image(image, top_k=top_k)
