"""
Model Architecture Verification Script for AgriShield (SIH 26131).
Verifies:
1. Python / PyTorch / torchvision imports
2. Primary EfficientNet-B0 instantiation with custom 38-class head
3. Alternative MobileNetV3-Large instantiation with custom 38-class head
4. Forward-pass validation with dummy tensor [1, 3, 224, 224] -> expected [1, 38]
5. Parameter counts (total, trainable, frozen)
6. Backbone freeze / unfreeze functionality for two-phase transfer learning
7. Hardware device assignment (CPU / CUDA)
"""

import sys
from pathlib import Path

# Add ml-model to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
import torchvision
from src import config
from src.model import (
    create_model,
    build_efficientnet_b0,
    build_mobilenet_v3_large,
    freeze_backbone,
    unfreeze_backbone,
    count_parameters,
)


def verify_architecture(model_name: str, pretrained: bool, device: torch.device):
    print("-" * 70)
    print(f"Testing Architecture: {model_name.upper()} (pretrained={pretrained})")
    print("-" * 70)

    # 1. Model Instantiation
    print(f"1. Instantiating {model_name} (num_classes={config.NUM_CLASSES}, pretrained={pretrained})...")
    model = create_model(
        model_name=model_name,
        num_classes=config.NUM_CLASSES,
        pretrained=pretrained,
        dropout_rate=config.DROPOUT_RATE,
    )
    model.to(device)
    model.eval()

    # 2. Parameter Inspection (Initial State)
    params_initial = count_parameters(model)
    print(f"   Total Parameters     : {params_initial['total_parameters']:,}")
    print(f"   Trainable Parameters : {params_initial['trainable_parameters']:,}")
    print(f"   Frozen Parameters    : {params_initial['frozen_parameters']:,}")

    # 3. Test Phase 1 Warmup (Frozen Backbone)
    freeze_backbone(model)
    params_phase1 = count_parameters(model)
    print(f"   [Phase 1 Check] Trainable (Classifier Only): {params_phase1['trainable_parameters']:,}")
    print(f"   [Phase 1 Check] Frozen (Feature Extractor) : {params_phase1['frozen_parameters']:,}")
    assert params_phase1["trainable_parameters"] > 0, "Phase 1 requires trainable classifier!"
    assert params_phase1["frozen_parameters"] > 0, "Phase 1 requires frozen backbone!"

    # 4. Test Phase 2 Fine-Tuning (Unfrozen Backbone)
    unfreeze_backbone(model)
    params_phase2 = count_parameters(model)
    print(f"   [Phase 2 Check] All Parameters Trainable  : {params_phase2['trainable_parameters']:,}")
    assert params_phase2["trainable_parameters"] == params_phase2["total_parameters"], "Phase 2 requires all parameters unfrozen!"

    # 5. Forward Pass Verification with Dummy Input
    dummy_input = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    print(f"2. Forward Pass Test:")
    print(f"   Input Tensor Shape   : {list(dummy_input.shape)} (Expected: [1, 3, 224, 224])")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"   Output Tensor Shape  : {list(output.shape)} (Expected: [1, 38])")
    print(f"   Output Tensor Dtype  : {output.dtype}")
    print(f"   Output Logits Range  : [{output.min().item():.3f} .. {output.max().item():.3f}]")

    # Assertions
    assert output.shape == (1, config.NUM_CLASSES), (
        f"Output shape mismatch! Expected (1, {config.NUM_CLASSES}), got {output.shape}"
    )
    print(f"   Forward Pass Status  : PASSED [Output Dimension strictly matches {config.NUM_CLASSES}]")

    return {
        "model_name": model_name,
        "pretrained": pretrained,
        "total_params": params_initial["total_parameters"],
        "classifier_params": params_phase1["trainable_parameters"],
        "output_shape": list(output.shape),
    }


def run_model_verification():
    print("=" * 75)
    print("AgriShield Part 4: Model Architecture & Factory Verification")
    print("=" * 75)

    device = config.DEVICE
    print(f"Detected Computation Device : {device}")
    print(f"PyTorch Version             : {torch.__version__}")
    print(f"Torchvision Version         : {torchvision.__version__}")
    print(f"Configured Number of Classes: {config.NUM_CLASSES}")
    print(f"Configured Input Resolution : {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")

    results = []

    # 1. Primary Model: EfficientNet-B0 (Pretrained ImageNet weights loaded from cache)
    res_eff = verify_architecture("efficientnet_b0", pretrained=True, device=device)
    results.append(res_eff)

    # 2. Alternative Model: MobileNetV3-Large (Structural verification)
    res_mob = verify_architecture("mobilenet_v3_large", pretrained=False, device=device)
    results.append(res_mob)

    # Summary Table
    print("\n" + "=" * 75)
    print("SUMMARY OF VERIFIED ARCHITECTURES")
    print("=" * 75)
    print(f"{'Architecture':<20} | {'Role':<12} | {'Pretrained':<10} | {'Total Params':<13} | {'Classifier Params':<17} | {'Output Shape':<12}")
    print("-" * 75)
    for r in results:
        role = "Primary" if r["model_name"] == "efficientnet_b0" else "Alternative"
        print(
            f"{r['model_name']:<20} | "
            f"{role:<12} | "
            f"{str(r['pretrained']):<10} | "
            f"{r['total_params']:<13,d} | "
            f"{r['classifier_params']:<17,d} | "
            f"{str(r['output_shape']):<12}"
        )

    print("=" * 75)
    print("ALL ARCHITECTURAL VERIFICATIONS COMPLETED SUCCESSFULLY.")
    print("NO MODEL WAS TRAINED. NO MODEL WEIGHTS WERE SAVED TO DISK.")
    print("=" * 75)


if __name__ == "__main__":
    run_model_verification()
