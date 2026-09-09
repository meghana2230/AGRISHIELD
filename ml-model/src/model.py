"""
Model Architecture Definition and Factory Module for AgriShield (SIH 26131).
Implements primary EfficientNet-B0 and alternative MobileNetV3-Large vision backbones
with custom 38-class classification heads, ImageNet transfer learning,
and parameter management utilities.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Primary Architecture: EfficientNet-B0
# ---------------------------------------------------------------------------

def build_efficientnet_b0(
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = config.PRETRAINED,
    dropout_rate: float = config.DROPOUT_RATE,
) -> nn.Module:
    """
    Constructs the primary EfficientNet-B0 architecture with a custom classification head.

    Args:
        num_classes: Number of output disease classes (default: 38).
        pretrained: If True, loads official ImageNet-1K pretrained weights.
        dropout_rate: Dropout probability in the classification head (default: 0.3).

    Returns:
        torch.nn.Module: Configured EfficientNet-B0 model.
    """
    weights = None
    if pretrained:
        try:
            weights = EfficientNet_B0_Weights.DEFAULT
            logger.info("Loading ImageNet-1K pretrained weights for EfficientNet-B0...")
        except Exception as e:
            logger.warning(
                f"Could not load pretrained weights due to network error: {e}. "
                "Falling back to randomly initialized weights (pretrained=False)."
            )
            weights = None

    try:
        model = efficientnet_b0(weights=weights)
    except Exception as e:
        logger.warning(
            f"Pretrained weights download failed ({e}). Instantiating EfficientNet-B0 without pretraining."
        )
        model = efficientnet_b0(weights=None)

    # Standard EfficientNet-B0 classifier has:
    # classifier[0] = Dropout(p=0.2, inplace=True)
    # classifier[1] = Linear(in_features=1280, out_features=1000)
    in_features = model.classifier[1].in_features  # 1280

    # Replace classifier head cleanly without arbitrary redesign
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes, bias=True),
    )

    model.model_name = "efficientnet_b0"
    model.num_classes = num_classes
    return model


# ---------------------------------------------------------------------------
# 2. Alternative Architecture: MobileNetV3-Large
# ---------------------------------------------------------------------------

def build_mobilenet_v3_large(
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = config.PRETRAINED,
    dropout_rate: float = config.DROPOUT_RATE,
) -> nn.Module:
    """
    Constructs the alternative MobileNetV3-Large architecture for low-latency edge deployment.

    Args:
        num_classes: Number of output disease classes (default: 38).
        pretrained: If True, loads official ImageNet-1K pretrained weights.
        dropout_rate: Dropout probability in the classification head (default: 0.3).

    Returns:
        torch.nn.Module: Configured MobileNetV3-Large model.
    """
    weights = None
    if pretrained:
        try:
            weights = MobileNet_V3_Large_Weights.DEFAULT
            logger.info("Loading ImageNet-1K pretrained weights for MobileNetV3-Large...")
        except Exception as e:
            logger.warning(
                f"Could not load pretrained weights due to network error: {e}. "
                "Falling back to randomly initialized weights (pretrained=False)."
            )
            weights = None

    try:
        model = mobilenet_v3_large(weights=weights)
    except Exception as e:
        logger.warning(
            f"Pretrained weights download failed ({e}). Instantiating MobileNetV3-Large without pretraining."
        )
        model = mobilenet_v3_large(weights=None)

    # Standard MobileNetV3-Large classifier has:
    # classifier[0] = Linear(in_features=960, out_features=1280)
    # classifier[1] = Hardswish()
    # classifier[2] = Dropout(p=0.2, inplace=True)
    # classifier[3] = Linear(in_features=1280, out_features=1000)
    in_features = model.classifier[3].in_features  # 1280

    # Replace classifier head cleanly
    model.classifier[2] = nn.Dropout(p=dropout_rate, inplace=True)
    model.classifier[3] = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    model.model_name = "mobilenet_v3_large"
    model.num_classes = num_classes
    return model


# ---------------------------------------------------------------------------
# 3. Clean Model Factory
# ---------------------------------------------------------------------------

def create_model(
    model_name: str = config.MODEL_NAME,
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = config.PRETRAINED,
    dropout_rate: float = config.DROPOUT_RATE,
) -> nn.Module:
    """
    Model factory function instantiating the requested architecture.

    Supported model_name values:
        - 'efficientnet_b0' (default / primary)
        - 'mobilenet_v3_large' (alternative)
    """
    name = model_name.lower().strip()
    if name == "efficientnet_b0":
        return build_efficientnet_b0(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )
    elif name == "mobilenet_v3_large":
        return build_mobilenet_v3_large(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )
    else:
        raise ValueError(
            f"Unsupported model architecture: '{model_name}'. "
            f"Supported options: ['efficientnet_b0', 'mobilenet_v3_large']"
        )


# ---------------------------------------------------------------------------
# 4. Parameter Management & Transfer Learning Helpers
# ---------------------------------------------------------------------------

def freeze_backbone(model: nn.Module) -> None:
    """
    Freezes all convolutional backbone feature extractor layers.
    Only the classification head remains trainable for Phase 1 warmup training.
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the classifier head
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
        logger.info(f"Backbone frozen. Classifier parameters remain trainable.")
    else:
        raise AttributeError(f"Model {type(model).__name__} does not have a 'classifier' attribute.")


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreezes all parameters across the entire network for Phase 2 fine-tuning.
    """
    for param in model.parameters():
        param.requires_grad = True
    logger.info("All model parameters unfrozen for fine-tuning.")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Returns parameter counts: total, trainable, and frozen.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
    }


def get_trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Returns a list of parameters that have requires_grad=True.
    """
    return [p for p in model.parameters() if p.requires_grad]
