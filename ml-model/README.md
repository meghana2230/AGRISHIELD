# AgriShield ML Module — SIH 26131

This repository contains the machine learning components for **SIH 26131: Crop Disease Detection**. The system classifies 38 distinct crop-disease conditions across 14 agricultural crops using deep transfer learning.

---

## Directory Structure

```
ml-model/
├── dataset/
│   ├── raw/                  # Downloaded raw PlantVillage image files (ignored by Git)
│   ├── processed/            # Stratified train/val/test split manifests (.csv) (ignored by Git)
│   └── class_indices.json    # Standardized 38-class metadata mapping (alphabetical order)
├── notebooks/                # Prototyping & Exploratory Data Analysis (EDA)
├── src/                      # Modular production source code
│   ├── __init__.py           # Package init
│   ├── config.py             # Hyperparameters, paths, and hardware device setup
│   ├── dataset.py            # Dataset loader, transforms, stratified splitting & DataLoaders
│   ├── verify_dataset.py     # Dataset pipeline & preprocessing verification script
│   ├── model.py              # Architecture definitions (EfficientNet-B0 & MobileNetV3-Large)
│   ├── verify_model.py       # Model architecture & forward-pass verification script
│   ├── train.py              # Training pipeline with class-weighted loss & checkpointing
│   ├── predict.py            # Singleton model loader and inference engine
│   ├── evaluate.py           # Test set evaluation & confusion matrix placeholder (Part 6)
│   └── export.py             # ONNX model serialization placeholder (Part 7)
├── models/
│   ├── checkpoints/          # Training epoch checkpoints
│   ├── best_model.pth        # Best trained PyTorch model checkpoint (Part 5)
│   └── crop_disease_model.onnx # Exported ONNX artifact for backend (Part 7)
├── outputs/                  # Evaluation plots, confusion matrices, and metrics JSON
├── venv/                     # Local Python virtual environment (ignored by Git)
├── requirements.txt          # Pinned project dependencies
├── verify_env.py             # Hardware & library diagnostic verification script
├── test_inference.py         # Local inference verification script
└── README.md                 # Documentation and instructions
```

---

## Environment Setup & Activation (Windows)

### 1. Activating the Environment on Windows
From the `ml-model/` directory in PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```
*If PowerShell restricts script execution, run first:*
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Or from Command Prompt (`cmd.exe`):
```cmd
venv\Scripts\activate.bat
```

### 2. Verifying the Setup
Run the environment verification script:
```powershell
python verify_env.py
```

---

## Dataset Pipeline & Preprocessing (Part 3)

* **Dataset**: PlantVillage (Color Leaf Dataset), 10,878 cataloged RGB images across 38 classes.
* **Input Resolution**: $224 \times 224$ pixels, 3-channel RGB.
* **Alphabetical Class Index Mapping**: `ml-model/dataset/class_indices.json`.
* **Splits**: Stratified 70/15/15 (7,614 train / 1,632 val / 1,632 test).
* **Data Leakage Guarantee**: Strict assertion confirms zero intersection between image paths.
* **Verification Command**:
  ```powershell
  python src/verify_dataset.py
  ```

---

## Model Architecture & Factory (Part 4)

* **Primary Model**: **EfficientNet-B0** (4,056,226 parameters; 48,678 classifier parameters).
* **Alternative Model**: **MobileNetV3-Large** (4,250,710 parameters; 1,278,758 classifier parameters).
* **Transfer Learning**: Custom 38-class classification head with dropout ($p=0.3$) and ImageNet-1K pretrained feature extractor.
* **Verification Command**:
  ```powershell
  python src/verify_model.py
  ```

---

## Model Loading & Inference Engine (Part 5)

### 1. Model Loading Architecture (`DiseasePredictor`)
Implemented in `ml-model/src/predict.py`:
* **Singleton Design Pattern**: Loads the model weights into memory **once**. The model is never re-instantiated or retrained during HTTP request handling.
* **Checkpoint File**: `ml-model/models/best_model.pth` (16.13 MB).
* **Deterministic Preprocessing**: Automatically replicates the exact validation/testing pipeline:
  1. Input conversion to 3-channel RGB (supports file path, raw bytes, PIL Image, or numpy array).
  2. Bilinear resize to $224 \times 224$.
  3. ImageNet channel normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
  4. Batch dimension expansion to `[1, 3, 224, 224]`.

### 2. Standardized Prediction Function
```python
from src.predict import predict_disease

# Predict from file path, PIL Image, or byte stream
result = predict_disease("path/to/leaf_image.jpg")
print(result)
```

**Output Schema**:
```json
{
  "crop": "Tomato",
  "disease": "Early Blight",
  "confidence": 94.6,
  "is_healthy": false,
  "predicted_class": "Tomato___Early_blight",
  "inference_time_ms": 18.5,
  "top_predictions": [
    {"crop": "Tomato", "disease": "Early Blight", "confidence": 94.6},
    {"crop": "Tomato", "disease": "Late Blight", "confidence": 3.8},
    {"crop": "Tomato", "disease": "Target Spot", "confidence": 0.9}
  ]
}
```

### 3. Local Inference Verification
Run the local inference test:
```powershell
python test_inference.py
```
This script validates:
* Loading the saved `best_model.pth` checkpoint.
* Executing inference on real test dataset images.
* Validating response keys (`crop`, `disease`, `confidence`).
* Verifying singleton in-memory reuse.

---

## Current Status & Next Steps
* **Phase 1 (Planning & Research)**: Completed & Approved.
* **Phase 2 (ML Environment Setup)**: Completed & Approved.
* **Phase 3 (Dataset Setup & Preprocessing)**: Completed & Approved.
* **Phase 4 (Model Development)**: Completed & Approved.
* **Phase 5 (Model Loading & Inference Engine)**: Completed.
* **Phase 6 / Next**: Backend Integration (FastAPI REST service exposing `POST /predict`) and frontend integration.
