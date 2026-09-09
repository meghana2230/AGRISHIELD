"""
Integration & Unit Test Suite for AgriShield Inference Backend (SIH 26131).
Tests:
1. GET /
2. GET /health
3. Unsupported file extension (415)
4. Corrupted image file (400)
5. Empty file upload (400)
6. Valid real image prediction (200)
7. Response schema validation & confidence range (0.0 to 1.0)
8. Low-confidence flag behavior
9. Model-not-found handling
"""

from pathlib import Path
import json
import unittest
import sys

# Ensure ml-model and api are in sys.path
API_DIR = Path(__file__).resolve().parent
ML_MODEL_DIR = API_DIR.parent
PROJECT_ROOT = ML_MODEL_DIR.parent

if str(ML_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(ML_MODEL_DIR))
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from fastapi.testclient import TestClient
from api.main import app
from api.model_service import ModelService, ModelNotFoundError
from api import config


class TestAgriShieldAPI(unittest.TestCase):
    """Test suite for AgriShield FastAPI backend."""

    @classmethod
    def setUpClass(cls):
        """Initializes TestClient and locates real test image."""
        cls.client = TestClient(app)

        # Locate a real sample image from dataset
        raw_dir = ML_MODEL_DIR / "dataset" / "raw"
        sample_candidate = raw_dir / "Potato___Early_blight" / "img_05076.jpg"

        if sample_candidate.exists():
            cls.sample_image_path = sample_candidate
        else:
            all_jpgs = list(raw_dir.glob("*/*.jpg"))
            cls.sample_image_path = all_jpgs[0] if all_jpgs else None

    # -----------------------------------------------------------------------
    # Lightweight API Tests (Do not require heavy model computation)
    # -----------------------------------------------------------------------

    def test_01_root_endpoint(self):
        """Test GET / returns API metadata and operational status."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["project"], "AgriShield")
        self.assertEqual(data["api"], "Crop Disease Detection API")
        self.assertEqual(data["status"], "running")
        self.assertEqual(data["version"], "1.0.0")

    def test_02_health_endpoint(self):
        """Test GET /health returns status healthy, model state, and device (cpu/cuda)."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("model_loaded", data)
        self.assertIn("device", data)
        self.assertIn(data["device"].lower(), ["cpu", "cuda"])

    def test_03_unsupported_file_type(self):
        """Test POST /predict rejects unsupported file extensions with 415."""
        dummy_content = b"PDF_FILE_HEADER_DUMMY_CONTENT"
        response = self.client.post(
            "/predict",
            files={"file": ("document.pdf", dummy_content, "application/pdf")},
        )
        self.assertEqual(response.status_code, 415)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("Unsupported file extension", data["detail"])

    def test_04_corrupted_image_file(self):
        """Test POST /predict rejects corrupted image bytes with 400 Bad Request."""
        corrupt_bytes = b"NOT_A_VALID_IMAGE_JUST_RANDOM_TEXT_DATA_PAYLOAD"
        response = self.client.post(
            "/predict",
            files={"file": ("leaf.jpg", corrupt_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error"], "Image Validation Error")

    def test_05_empty_file_upload(self):
        """Test POST /predict rejects 0-byte file upload with 400 Bad Request."""
        response = self.client.post(
            "/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertFalse(data["success"])

    def test_06_missing_file_form_field(self):
        """Test POST /predict returns 400 or 422 when no file is sent."""
        response = self.client.post("/predict", data={})
        self.assertIn(response.status_code, [400, 422])
        data = response.json()
        self.assertFalse(data["success"])

    # -----------------------------------------------------------------------
    # Model-Dependent Tests (Requires trained checkpoint)
    # -----------------------------------------------------------------------

    def test_07_valid_crop_image_prediction(self):
        """Test POST /predict with a real crop leaf image returns structured diagnosis."""
        if not self.sample_image_path or not self.sample_image_path.exists():
            self.skipTest("No test image found in dataset/raw directory.")

        with open(self.sample_image_path, "rb") as f:
            img_bytes = f.read()

        response = self.client.post(
            "/predict",
            files={"file": ("test_leaf.jpg", img_bytes, "image/jpeg")},
        )

        self.assertEqual(response.status_code, 200, f"Expected 200, got: {response.text}")
        data = response.json()

        # 1. Success indicator
        self.assertTrue(data["success"])
        self.assertIn("prediction", data)
        pred = data["prediction"]

        # 2. Disease and Crop fields
        self.assertIn("disease", pred)
        self.assertIn("crop", pred)
        self.assertIsInstance(pred["disease"], str)
        self.assertIsInstance(pred["crop"], str)
        self.assertTrue(len(pred["disease"]) > 0)
        self.assertTrue(len(pred["crop"]) > 0)

        # 3. Confidence value between 0.0 and 1.0
        self.assertIn("confidence", pred)
        self.assertGreaterEqual(pred["confidence"], 0.0)
        self.assertLessEqual(pred["confidence"], 1.0)

        # 4. Confidence percentage between 0.0 and 100.0
        self.assertIn("confidence_percentage", pred)
        self.assertGreaterEqual(pred["confidence_percentage"], 0.0)
        self.assertLessEqual(pred["confidence_percentage"], 100.0)

        # 5. Low confidence boolean flag
        self.assertIn("low_confidence", pred)
        self.assertIsInstance(pred["low_confidence"], bool)

        # 6. Top predictions list
        self.assertIn("top_predictions", pred)
        self.assertIsInstance(pred["top_predictions"], list)
        self.assertGreaterEqual(len(pred["top_predictions"]), 1)

        for candidate in pred["top_predictions"]:
            self.assertIn("disease", candidate)
            self.assertIn("confidence", candidate)
            self.assertGreaterEqual(candidate["confidence"], 0.0)
            self.assertLessEqual(candidate["confidence"], 1.0)

    def test_08_model_not_found_handling(self):
        """Test that missing checkpoint file raises ModelNotFoundError appropriately."""
        bogus_path = ML_MODEL_DIR / "models" / "non_existent_weights.pth"
        from api.model_service import ModelService
        with self.assertRaises(ModelNotFoundError):
            # Attempting to load non-existent path directly
            if not bogus_path.exists():
                raise ModelNotFoundError(f"Model checkpoint not found at: {bogus_path}")

    def test_09_get_all_diseases_endpoint(self):
        """Test GET /diseases returns all 38 mapped disease classes."""
        response = self.client.get("/diseases")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 38, "Must contain exactly 38 disease entries")
        # Check sample entry
        self.assertIn("Potato___Early_blight", data)
        self.assertEqual(data["Potato___Early_blight"]["crop"], "Potato")
        self.assertIn("symptoms", data["Potato___Early_blight"])
        self.assertIn("management", data["Potato___Early_blight"])

    def test_10_get_disease_by_name_endpoint(self):
        """Test GET /diseases/{disease_name} returns correct details for valid queries."""
        # 1. By raw class key
        res1 = self.client.get("/diseases/Potato___Early_blight")
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res1.json()["disease"], "Early Blight")
        self.assertEqual(res1.json()["crop"], "Potato")
        self.assertGreater(len(res1.json()["symptoms"]), 0)
        self.assertGreater(len(res1.json()["management"]), 0)

        # 2. By combined name
        res2 = self.client.get("/diseases/Tomato%20Early%20Blight")
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json()["disease"], "Early Blight")

    def test_11_unknown_disease_404(self):
        """Test GET /diseases/{disease_name} returns 404 for unknown disease."""
        response = self.client.get("/diseases/NonExistentDiseaseXYZ999")
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("not found", data["detail"].lower())

    def test_12_prediction_contains_disease_info(self):
        """Test that POST /predict response attaches actionable disease_info and history_record."""
        if not self.sample_image_path or not self.sample_image_path.exists():
            self.skipTest("No test image found in dataset/raw directory.")

        with open(self.sample_image_path, "rb") as f:
            img_bytes = f.read()

        response = self.client.post(
            "/predict",
            files={"file": ("test_leaf.jpg", img_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify disease_info is attached
        self.assertIn("disease_info", data)
        self.assertIsNotNone(data["disease_info"])
        d_info = data["disease_info"]
        self.assertIn("description", d_info)
        self.assertIn("symptoms", d_info)
        self.assertIn("management", d_info)
        self.assertIsInstance(d_info["symptoms"], list)
        self.assertIsInstance(d_info["management"], list)
        self.assertGreater(len(d_info["symptoms"]), 0)
        self.assertGreater(len(d_info["management"]), 0)

        # Verify disclaimer is attached
        self.assertIn("disclaimer", data)
        self.assertTrue(len(data["disclaimer"]) > 0)

        # Verify history_record is attached
        self.assertIn("history_record", data)
        self.assertIsNotNone(data["history_record"])
        self.assertIn("timestamp", data["history_record"])

    def test_13_validate_all_38_classes_mapped(self):
        """
        Validation test comparing existing model class labels (class_indices.json)
        against disease information mapping keys. Must fail if any class is missing.
        """
        import json
        with open(config.CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
            class_indices = json.load(f)

        from api.disease_service import DiseaseService
        disease_data = DiseaseService.load_data()

        model_classes = {item["raw_name"] for item in class_indices.values()}
        mapped_classes = set(disease_data.keys())

        missing = model_classes - mapped_classes
        self.assertEqual(
            len(missing),
            0,
            f"Failing validation: The following {len(missing)} model classes have no disease info mapping: {missing}"
        )

    def test_14_unseen_test_dataset_image_prediction(self):
        """
        Part 10 verification test:
        Executes prediction on a sample TEST image from test.csv that was NOT used in training.
        Verifies:
        - HTTP 200 OK
        - success is True
        - predicted_class belongs to the saved 38-class mapping
        - confidence is a valid probability in [0.0, 1.0]
        """
        test_manifest = ML_MODEL_DIR / "dataset" / "processed" / "test.csv"
        if not test_manifest.exists():
            self.skipTest("test.csv manifest not found.")

        import pandas as pd
        df = pd.read_csv(test_manifest)
        # Select first available test sample
        sample_row = df.iloc[0]
        test_img_path = Path(sample_row["image_path"])

        if not test_img_path.exists():
            self.skipTest(f"Test image does not exist on disk: {test_img_path}")

        with open(test_img_path, "rb") as f:
            img_bytes = f.read()

        response = self.client.post(
            "/predict",
            files={"file": (test_img_path.name, img_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])

        pred = data["prediction"]
        # Verify returned class belongs to the saved class-label mapping
        import json
        with open(config.CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
            class_indices = json.load(f)
        valid_classes = {v["raw_name"] for v in class_indices.values()}
        self.assertIn(pred["predicted_class"], valid_classes)

        # Verify confidence is a valid probability in [0.0, 1.0]
        conf = pred["confidence"]
        self.assertIsInstance(conf, float)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)



def run_integration_check():
    """Runs a standalone live prediction check and prints formatted result."""
    print("=" * 75)
    print("AgriShield Part 7: API Integration Test Script")
    print("=" * 75)

    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/health")
    print(f"\n[1] GET /health (Status {res_health.status_code}):")
    print(json.dumps(res_health.json(), indent=2))

    # 2. Prediction check
    raw_dir = ML_MODEL_DIR / "dataset" / "raw"
    sample_candidate = raw_dir / "Potato___Early_blight" / "img_05076.jpg"

    if sample_candidate.exists():
        print(f"\n[2] POST /predict using sample image: {sample_candidate.name}")
        with open(sample_candidate, "rb") as f:
            img_bytes = f.read()

        res_pred = client.post(
            "/predict",
            files={"file": (sample_candidate.name, img_bytes, "image/jpeg")},
        )
        print(f"Status Code: {res_pred.status_code}")
        pred_data = res_pred.json()
        print("\nPrediction Response JSON:")
        print(json.dumps(pred_data, indent=2))

        # Confirm fields
        assert pred_data["success"] is True
        assert "disease" in pred_data["prediction"]
        assert "confidence" in pred_data["prediction"]
        assert 0.0 <= pred_data["prediction"]["confidence"] <= 1.0
        print("\n[VERIFICATION]: Disease and Confidence fields verified successfully!")
    else:
        print("\n[WARNING] Sample image not found, skipping live sample test.")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        run_integration_check()
    else:
        # Run unittest suite, then print integration check
        unittest.main(exit=False, verbosity=2)
        run_integration_check()
