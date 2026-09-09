"""
Environment Verification Script for AgriShield ML Module (SIH 26131).
Verifies installed packages, versions, hardware acceleration, and PyTorch device.
"""

import sys
import platform

def verify_environment():
    print("=" * 60)
    print("AgriShield AI/ML Environment Verification")
    print("=" * 60)

    # 1. Python Information
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {platform.python_version()} ({platform.architecture()[0]})")
    print(f"Platform          : {platform.system()} {platform.release()}")
    print("-" * 60)

    # 2. Deep Learning Frameworks
    try:
        import torch
        print(f"[OK] PyTorch Version      : {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"[OK] CUDA Available       : {cuda_available}")
        if cuda_available:
            print(f"[OK] CUDA Device Name     : {torch.cuda.get_device_name(0)}")
            print(f"[OK] CUDA Device Count    : {torch.cuda.device_count()}")
            selected_device = torch.device("cuda:0")
        else:
            print("[INFO] CUDA not detected. Using CPU for execution.")
            selected_device = torch.device("cpu")
        print(f"[OK] Selected Computation Device : {selected_device}")
    except ImportError as e:
        print(f"[FAIL] PyTorch is NOT installed: {e}")

    try:
        import torchvision
        print(f"[OK] torchvision Version  : {torchvision.__version__}")
    except ImportError as e:
        print(f"[FAIL] torchvision is NOT installed: {e}")

    # 3. Scientific & Data Libraries
    try:
        import numpy as np
        print(f"[OK] NumPy Version        : {np.__version__}")
    except ImportError as e:
        print(f"[FAIL] NumPy is NOT installed: {e}")

    try:
        import PIL
        print(f"[OK] Pillow (PIL) Version : {PIL.__version__}")
    except ImportError as e:
        print(f"[FAIL] Pillow is NOT installed: {e}")

    try:
        import sklearn
        print(f"[OK] scikit-learn Version : {sklearn.__version__}")
    except ImportError as e:
        print(f"[FAIL] scikit-learn is NOT installed: {e}")

    try:
        import pandas as pd
        print(f"[OK] Pandas Version       : {pd.__version__}")
    except ImportError as e:
        print(f"[FAIL] Pandas is NOT installed: {e}")

    # 4. Computer Vision & Augmentation
    try:
        import albumentations as A
        print(f"[OK] Albumentations Ver   : {A.__version__}")
    except ImportError as e:
        print(f"[FAIL] Albumentations is NOT installed: {e}")

    # 5. Production & Export Runtimes
    try:
        import onnx
        print(f"[OK] ONNX Version         : {onnx.__version__}")
    except ImportError as e:
        print(f"[FAIL] ONNX is NOT installed: {e}")

    try:
        import onnxruntime as ort
        print(f"[OK] ONNX Runtime Ver     : {ort.__version__}")
    except ImportError as e:
        print(f"[FAIL] ONNX Runtime is NOT installed: {e}")

    try:
        import fastapi
        print(f"[OK] FastAPI Version      : {fastapi.__version__}")
    except ImportError as e:
        print(f"[FAIL] FastAPI is NOT installed: {e}")

    print("=" * 60)
    print("Verification Completed Successfully.")
    print("=" * 60)

if __name__ == "__main__":
    verify_environment()
