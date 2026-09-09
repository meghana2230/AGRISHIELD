"""
Automated Integration & Unit Tests for AgriShield FastAPI Backend (SIH 26131).
Tests health endpoint, model loading, valid image predictions using real dataset samples,
and error handling on invalid/corrupted/missing inputs.
"""

from pathlib import Path
import json
import unittest
import sys

# Ensure project root and backend are in sys.path
TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app
from app.model_loader import ModelManager
from app import config


class TestAgriShieldBackend(unittest.TestCase):
    """Test suite covering the complete AgriShield API inference service."""

    @classmethod
    def setUpClass(cls):
        """Initializes the TestClient and loads the model into memory."""
        cls.client = TestClient(app)
        # Ensure model is initialized via lifespan / manager
        ModelManager.load()

        # Locate a verified real dataset image
        raw_dataset_dir = PROJECT_ROOT / "ml-model" / "dataset" / "raw"
        cls.sample_image_path = raw_dataset_dir / "Potato___Early_blight" / "img_05076.jpg"

        # Fallback to any valid JPEG in dataset if img_05076.jpg is not directly found
        if not cls.sample_image_path.exists():
            all_jpgs = list(raw_dataset_dir.glob("*/*.jpg"))
            if all_jpgs:
                cls.sample_image_path = all_jpgs[0]

        # Load valid class taxonomy
        with open(config.CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
            cls.class_indices = json.load(f)

        cls.valid_diseases = {meta["disease"] for meta in cls.class_indices.values()}
        cls.valid_crops = {meta["crop"] for meta in cls.class_indices.values()}
        cls.valid_raw_names = {meta["raw_name"] for meta in cls.class_indices.values()}

    def test_01_model_loading_and_singleton(self):
        """1. Verify that the trained model loads once and preserves singleton identity."""
        self.assertTrue(ModelManager.is_loaded(), "ModelManager should report model as loaded")
        predictor1 = ModelManager.get_predictor()
        predictor2 = ModelManager.get_predictor()
        self.assertIs(predictor1, predictor2, "ModelManager must reuse the same singleton in-memory instance")

        info = ModelManager.get_info()
        self.assertEqual(info["num_classes"], 38, "Model must be configured with 38 disease classes")
        self.assertEqual(info["image_size"], 224, "Model input resolution must be 224x224")

    def test_02_health_endpoint(self):
        """2. Verify GET /api/health returns status ok and model_loaded true."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["num_classes"], 38)
        self.assertEqual(data["model_name"], "efficientnet_b0")

    def test_03_predict_valid_crop_image(self):
        """3. Verify POST /api/predict with a real crop leaf image returns valid diagnosis."""
        self.assertTrue(self.sample_image_path.exists(), f"Test image not found at: {self.sample_image_path}")

        with open(self.sample_image_path, "rb") as f:
            img_bytes = f.read()

        response = self.client.post(
            "/api/predict",
            files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
        )

        self.assertEqual(response.status_code, 200, f"Expected 200, got: {response.text}")
        data = response.json()

        # Verify contract keys
        self.assertTrue(data["success"])
        self.assertIn("crop", data)
        self.assertIn("disease", data)
        self.assertIn("confidence", data)

        # Verify values adhere to actual trained taxonomy
        self.assertIn(data["crop"], self.valid_crops, f"Unknown crop: {data['crop']}")
        self.assertIn(data["disease"], self.valid_diseases, f"Unknown disease: {data['disease']}")
        self.assertIn(data["predicted_class"], self.valid_raw_names, f"Unknown class: {data['predicted_class']}")

        # Verify confidence range
        self.assertGreaterEqual(data["confidence"], 0.0)
        self.assertLessEqual(data["confidence"], 100.0)

        # Verify top predictions list
        self.assertIn("top_predictions", data)
        self.assertIsInstance(data["top_predictions"], list)
        self.assertGreaterEqual(len(data["top_predictions"]), 1)

    def test_04_predict_corrupt_or_invalid_image(self):
        """4. Verify POST /api/predict returns HTTP 400 on corrupted or non-image payload."""
        corrupt_bytes = b"NOT_A_VALID_IMAGE_CONTENT_PLAIN_TEXT_PAYLOAD_1234567890"

        response = self.client.post(
            "/api/predict",
            files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")},
        )

        self.assertEqual(response.status_code, 400, "Corrupted image must return 400 Bad Request")
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("error", data)

    def test_05_predict_empty_image_file(self):
        """5. Verify POST /api/predict returns HTTP 400 on empty 0-byte file."""
        empty_bytes = b""

        response = self.client.post(
            "/api/predict",
            files={"file": ("empty.jpg", empty_bytes, "image/jpeg")},
        )

        self.assertEqual(response.status_code, 400, "Empty image must return 400 Bad Request")
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("error", data)

    def test_06_predict_missing_file(self):
        """6. Verify POST /api/predict returns HTTP 400 or 422 when no file is uploaded."""
        response = self.client.post("/api/predict", data={})
        self.assertIn(response.status_code, [400, 422], "Missing file must return 400 or 422")

    def test_07_documentation_endpoints(self):
        """7. Verify OpenAPI documentation endpoints /docs and /openapi.json are accessible."""
        docs_res = self.client.get("/docs")
        self.assertEqual(docs_res.status_code, 200, "Swagger UI /docs must be accessible")

        openapi_res = self.client.get("/openapi.json")
        self.assertEqual(openapi_res.status_code, 200, "OpenAPI spec /openapi.json must be accessible")
        schema = openapi_res.json()
        self.assertIn("paths", schema)
        self.assertIn("/api/predict", schema["paths"])
        self.assertIn("/api/health", schema["paths"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
