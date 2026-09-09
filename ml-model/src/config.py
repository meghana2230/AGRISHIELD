"""
Configuration module for AgriShield Crop Disease Detection (SIH 26131).
Contains path definitions, model architecture settings, hyperparameters,
and hardware execution configuration.
"""

from pathlib import Path
import torch

# Base directory for ML project
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset Paths
DATASET_DIR = BASE_DIR / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"
CLASS_INDICES_PATH = DATASET_DIR / "class_indices.json"

# Train / Val / Test Manifest Paths (Part 3 generation)
TRAIN_MANIFEST_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_MANIFEST_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_MANIFEST_PATH = PROCESSED_DATA_DIR / "test.csv"

# Model Checkpoints & Artifact Paths
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
ONNX_EXPORT_PATH = MODELS_DIR / "crop_disease_model.onnx"

# Output Paths (Plots, Metrics, Reports)
OUTPUTS_DIR = BASE_DIR / "outputs"
METRICS_REPORT_PATH = OUTPUTS_DIR / "test_metrics_report.json"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"
TRAINING_CURVES_PATH = OUTPUTS_DIR / "training_curves.png"

# Core Model Hyperparameters
IMAGE_SIZE = 224
NUM_CLASSES = 38
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 2

# Model Architecture Configuration
# Primary: efficientnet_b0 | Alternative: mobilenet_v3_large
MODEL_NAME = "efficientnet_b0"
PRETRAINED = True
DROPOUT_RATE = 0.3

# Normalization Constants (ImageNet standards)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Two-Phase Learning Rates & Optimization
PHASE1_EPOCHS = 5
PHASE1_LR = 1e-3

PHASE2_EPOCHS = 20
PHASE2_LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 4

# Hardware Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if torch.cuda.is_available() else False
