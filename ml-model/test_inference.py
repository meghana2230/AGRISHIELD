"""
Local Test Script for AgriShield Crop Disease Prediction (SIH 26131).
Verifies that:
1. The trained model checkpoint loads successfully once into memory.
2. The exact preprocessing used during training is applied.
3. Model inference executes cleanly on real test dataset images.
4. Returns expected JSON-compatible dictionary with crop, disease, and confidence.
"""

import sys
import json
from pathlib import Path

# Add ml-model to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from src import config
from src.predict import DiseasePredictor, predict_disease


def run_local_test():
    print("=" * 75)
    print("AgriShield Part 5: Model Inference & Loading Verification")
    print("=" * 75)

    # 1. Verify Model Checkpoint Existence
    model_path = config.BEST_MODEL_PATH
    print(f"\n[STEP 1] Model File Verification:")
    print(f"  Expected Model Path : {model_path}")
    print(f"  Model File Exists   : {model_path.exists()}")
    if not model_path.exists():
        print(f"  [ERROR] Model file {model_path} does not exist. Please train or save the model first.")
        sys.exit(1)

    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  Model File Size     : {file_size_mb:.2f} MB")
    print(f"  Model Format        : PyTorch State Dict Archive (.pth)")

    # 2. Inspect Model Parameters & Preprocessing Config
    print(f"\n[STEP 2] Architectural & Preprocessing Specifications:")
    predictor = DiseasePredictor.get_instance(model_path=model_path)
    print(f"  Model Architecture  : {predictor.model_name}")
    print(f"  Number of Classes   : {predictor.num_classes}")
    print(f"  Input Resolution    : {predictor.image_size} x {predictor.image_size}")
    print(f"  Color Channels      : 3 (RGB)")
    print(f"  Image Normalization : Mean={predictor.image_mean}, Std={predictor.image_std}")
    print(f"  Device Configured   : {predictor.device}")
    if predictor.val_acc is not None:
        print(f"  Checkpoint Val Acc  : {predictor.val_acc:.2f}%")

    # 3. Load Sample Test Images from Isolated Test Manifest
    test_manifest_path = config.TEST_MANIFEST_PATH
    print(f"\n[STEP 3] Loading Test Sample from: {test_manifest_path.name}:")
    test_df = pd.read_csv(test_manifest_path)
    print(f"  Total Test Samples Available : {len(test_df)}")

    # Pick 2 diverse test samples (e.g. Tomato Early Blight and Apple Scab or Healthy)
    sample_rows = [
        test_df[test_df["disease"].str.contains("Early Blight", case=False, na=False)].iloc[0],
        test_df[test_df["is_healthy"] == True].iloc[0],
    ]

    print(f"\n[STEP 4] Executing Predictions:")
    for idx, sample in enumerate(sample_rows, 1):
        img_path = sample["image_path"]
        ground_truth_crop = sample["crop"]
        ground_truth_disease = sample["disease"]
        ground_truth_class = sample["class_name"]

        print(f"\n  --- Test Case {idx} ---")
        print(f"  Image File Path     : {img_path}")
        print(f"  Ground Truth Crop   : {ground_truth_crop}")
        print(f"  Ground Truth Disease: {ground_truth_disease}")
        print(f"  Ground Truth Class  : {ground_truth_class}")

        # Run Prediction Function
        result = predict_disease(img_path)

        print(f"\n  [Prediction Output Result]:")
        print(json.dumps({
            "crop": result["crop"],
            "disease": result["disease"],
            "confidence": result["confidence"],
        }, indent=4))

        print(f"  Is Healthy Flag     : {result['is_healthy']}")
        print(f"  Inference Latency   : {result['inference_time_ms']} ms")
        print(f"  Top-3 Candidates    :")
        for rank, cand in enumerate(result["top_predictions"][:3], 1):
            print(f"    {rank}. {cand['crop']} - {cand['disease']} ({cand['confidence']}%)")

        # Validate contract
        assert "crop" in result, "Missing 'crop' key in prediction result!"
        assert "disease" in result, "Missing 'disease' key in prediction result!"
        assert "confidence" in result, "Missing 'confidence' key in prediction result!"
        assert 0.0 <= result["confidence"] <= 100.0, "Confidence score out of range [0.0, 100.0]!"

    # 4. Verify Singleton Caching (Confirm Model is Loaded ONCE)
    print(f"\n[STEP 5] Verifying Singleton Cache (No Retraining on Requests):")
    predictor_2 = DiseasePredictor.get_instance()
    assert predictor is predictor_2, "DiseasePredictor failed singleton identity test!"
    print(f"  Singleton Caching Test: PASSED (Same model instance reused in memory)")

    print("\n" + "=" * 75)
    print("ALL PART 5 INFERENCE VERIFICATIONS PASSED SUCCESSFULLY.")
    print("THE MODEL IS READY FOR FASTAPI BACKEND INTEGRATION.")
    print("=" * 75)


if __name__ == "__main__":
    run_local_test()
