# AgriShield — Frontend API Integration Contract

**Project**: SIH 26131 — Crop Disease Detection  
**Module**: AI/ML Inference & Knowledge Base Backend API  
**Target Client**: Web / Mobile Frontend (React / Vite / Next.js)

---

## 1. Base URL & Endpoints

| Environment | Base URL |
| :--- | :--- |
| **Local Development** | `http://localhost:8000` |
| **Alternative Loopback** | `http://127.0.0.1:8000` |
| **Network Interface** | `http://0.0.0.0:8000` |

### Endpoint Summary

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | API Root & Service Metadata | No |
| `GET` | `/health` | Service & ML Model Health Status | No |
| `GET` | `/diseases` | Retrieve Knowledge Base for all 38 Supported Disease Classes | No |
| `GET` | `/diseases/{disease_name}` | Retrieve Actionable Info for a Specific Crop Disease | No |
| `POST` | `/predict` | Crop Disease Classification with Actionable Disease Guidance | No |
| `GET` | `/docs` | Interactive Swagger UI Documentation | No |
| `GET` | `/redoc` | Redoc Alternative Documentation | No |
| `GET` | `/openapi.json` | OpenAPI 3.1.0 Specification Schema | No |

---

## 2. Root Endpoint

### Request
```http
GET / HTTP/1.1
Host: localhost:8000
Accept: application/json
```

### Response (`200 OK`)
```json
{
  "project": "AgriShield",
  "api": "Crop Disease Detection API",
  "version": "1.0.0",
  "status": "running"
}
```

---

## 3. Health Check Endpoint

### Request
```http
GET /health HTTP/1.1
Host: localhost:8000
Accept: application/json
```

### Response (`200 OK`)
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "model_name": "efficientnet_b0",
  "num_classes": 38
}
```

---

## 4. Disease Information Endpoints (Part 9)

### 4.1. Get All Supported Diseases (`GET /diseases`)
Returns complete actionable information for all 38 supported classes (26 pathological diseases + 12 healthy controls).

#### Request
```http
GET /diseases HTTP/1.1
Host: localhost:8000
Accept: application/json
```

#### Response (`200 OK`)
```json
{
  "count": 38,
  "diseases": [
    {
      "disease": "Apple Scab",
      "crop": "Apple",
      "is_healthy": false,
      "description": "A serious fungal disease caused by Venturia inaequalis affecting apple leaves and fruit in cool, humid spring weather.",
      "symptoms": [
        "Olive-green to brown velvety spots on the upper leaf surface",
        "Leaves may become distorted, turn yellow, and drop prematurely",
        "Dark, scabby lesions develop on fruit skin, leading to cracking"
      ],
      "management": [
        "Rake and destroy fallen leaves in autumn to eliminate overwintering fungal ascocarps",
        "Prune tree canopy during dormancy to promote air circulation and rapid leaf drying",
        "Apply preventive protective fungicides (e.g. captan, sulfur) starting at green tip bud stage",
        "Plant scab-resistant apple cultivars (e.g., Enterprise, GoldRush, Liberty, Prima)"
      ]
    }
  ]
}
```

---

### 4.2. Get Specific Disease Information (`GET /diseases/{disease_name}`)
Resolves by exact class name (e.g. `Potato___Early_blight`), formatted name (`Early Blight`), crop combined name (`Potato Early Blight`), or slug (`potato-early-blight`).

#### Request
```http
GET /diseases/Potato___Early_blight HTTP/1.1
Host: localhost:8000
Accept: application/json
```

#### Response (`200 OK`)
```json
{
  "disease": "Early Blight",
  "crop": "Potato",
  "is_healthy": false,
  "description": "A widespread fungal disease caused by Alternaria solani affecting potato and tomato foliage, stems, and tubers during alternating dry and wet periods.",
  "symptoms": [
    "Small, dark brown to black spots developing on older lower leaves first",
    "Lesions expand into characteristic circular patterns with concentric rings ('target board' pattern)",
    "Yellow chlorotic halos surround advancing necrotic lesions",
    "Severe infections cause leaves to dry up, wither, and hang down on the stem"
  ],
  "management": [
    "Rotate crops with non-solanaceous plants such as grains or corn for at least 3 years",
    "Avoid overhead sprinkler irrigation and water early in the day to allow leaves to dry",
    "Ensure adequate nitrogen and potassium fertility; stressed plants are far more susceptible",
    "Apply preventive fungicides (e.g., chlorothalonil or copper) when lower leaves begin to senesce"
  ]
}
```

#### Unknown Disease Response (`404 Not Found`)
```json
{
  "success": false,
  "error": "Disease Not Found",
  "detail": "Disease 'NonExistentDisease' is not supported in the 38-class knowledge base."
}
```

---

## 5. Crop Disease Prediction Endpoint (`POST /predict`)

### Request Specifications
* **URL**: `POST http://localhost:8000/predict`
* **Content-Type**: `multipart/form-data`
* **Form Field Name**: `file`
* **Supported Formats**: `JPEG` (`.jpg`, `.jpeg`), `PNG` (`.png`), `WebP` (`.webp`)
* **Maximum File Size**: 15 MB
* **Minimum Image Resolution**: $10 \times 10$ pixels

### Frontend JavaScript Integration (FormData)
```javascript
const formData = new FormData();
formData.append("file", imageFile);

const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData,
});

const result = await response.json();
console.log(result);
```

### Successful Response (`200 OK`)
```json
{
  "success": true,
  "prediction": {
    "disease": "Early Blight",
    "confidence": 0.5393,
    "confidence_percentage": 53.93,
    "low_confidence": true,
    "top_predictions": [
      {
        "disease": "Early Blight",
        "confidence": 0.5393,
        "confidence_percentage": 53.93,
        "crop": "Potato",
        "class_name": "Potato___Early_blight",
        "is_healthy": false
      },
      {
        "disease": "Late Blight",
        "confidence": 0.1066,
        "confidence_percentage": 10.66,
        "crop": "Potato",
        "class_name": "Potato___Late_blight",
        "is_healthy": false
      },
      {
        "disease": "Leaf Scorch",
        "confidence": 0.0649,
        "confidence_percentage": 6.49,
        "crop": "Strawberry",
        "class_name": "Strawberry___Leaf_scorch",
        "is_healthy": false
      }
    ],
    "crop": "Potato",
    "predicted_class": "Potato___Early_blight",
    "is_healthy": false,
    "inference_time_ms": 32.37
  },
  "disease_info": {
    "disease": "Early Blight",
    "crop": "Potato",
    "is_healthy": false,
    "description": "A widespread fungal disease caused by Alternaria solani affecting potato and tomato foliage, stems, and tubers during alternating dry and wet periods.",
    "symptoms": [
      "Small, dark brown to black spots developing on older lower leaves first",
      "Lesions expand into characteristic circular patterns with concentric rings ('target board' pattern)",
      "Yellow chlorotic halos surround advancing necrotic lesions",
      "Severe infections cause leaves to dry up, wither, and hang down on the stem"
    ],
    "management": [
      "Rotate crops with non-solanaceous plants such as grains or corn for at least 3 years",
      "Avoid overhead sprinkler irrigation and water early in the day to allow leaves to dry",
      "Ensure adequate nitrogen and potassium fertility; stressed plants are far more susceptible",
      "Apply preventive fungicides (e.g., chlorothalonil or copper) when lower leaves begin to senesce"
    ]
  },
  "disclaimer": "This AI prediction is intended as a decision-support aid. Confirm important crop-treatment decisions with a qualified agricultural expert.",
  "history_record": {
    "timestamp": "2026-09-09T15:56:01.684869+00:00",
    "crop": "Potato",
    "disease": "Early Blight",
    "confidence": 0.5393,
    "confidence_percentage": 53.93,
    "top_predictions": [ ... ],
    "image_reference": "potato_early_blight.jpg"
  }
}
```

---

## 6. Confidence Behavior & Safety Guidelines

### 6.1. Visual States
1. **High Confidence (`low_confidence === false`)**:
   - Classification confidence exceeds threshold (60%).
   - Diagnosis, symptoms, and prevention guidance are displayed normally.

2. **Low Confidence (`low_confidence === true`)**:
   - Confidence is below the threshold.
   - The UI presents an advisory warning alerting the farmer that the result is uncertain.
   - Clear advice is shown: *"Please capture or upload a clearer, well-lit, close-up image of the leaf for an accurate diagnosis."*

3. **Inference Unavailable**:
   - If the backend is unreachable or model initialization failed (HTTP 503 / NetworkError), the UI displays a clean technical error banner.
   - An unavailable or erroneous response is **never presented as a valid diagnosis**.

### 6.2. Agricultural & Safety Disclaimer
Near the prediction result, AgriShield displays:
> *"This AI prediction is intended as a decision-support aid. Confirm important crop-treatment decisions with a qualified agricultural expert."*

* **Safety Restrictions**:
  - No medical claims.
  - No chemical or pesticide dosage calculations.
  - No dangerous handling procedures.
  - Only vetted agronomic cultural controls, sanitation, crop rotation, and standard organic/preventative management.

---

## 7. Prediction History-Ready Data Structure

The `history_record` object in the prediction response prepares the schema for future persistence (Part 10+):

| Field | Type | Description |
| :--- | :--- | :--- |
| `timestamp` | `string` (ISO 8601 UTC) | Exact UTC timestamp of inference request |
| `crop` | `string` | Predicted or affected crop |
| `disease` | `string` | Predicted disease pathology |
| `confidence` | `float` | Raw confidence probability `[0.0, 1.0]` |
| `confidence_percentage` | `float` | Human-readable confidence percentage `[0.0, 100.0]` |
| `top_predictions` | `array` | Top-3 ranked candidate classes with scores |
| `image_reference` | `string` (optional) | Filename or storage reference for uploaded leaf image |

---

## 8. Error Responses

| HTTP Status | Error Type | Condition |
| :--- | :--- | :--- |
| `400 Bad Request` | `Image Validation Error` | File is corrupted, 0 bytes, or not a recognizable image. |
| `404 Not Found` | `Disease Not Found` | Requested disease in `/diseases/{disease_name}` does not exist in the 38-class catalog. |
| `413 Payload Too Large` | `Request Entity Too Large` | Image file size exceeds 15 MB. |
| `415 Unsupported Media Type` | `Unsupported File Type` | Uploaded file extension is not `.jpg`, `.jpeg`, `.png`, or `.webp`. |
| `422 Unprocessable Entity` | `Validation Error` | Form field `file` is missing from multipart payload. |
| `503 Service Unavailable` | `Model Unavailable` | Model checkpoint (`best_model.pth`) is missing or failed to initialize. |
| `500 Internal Server Error` | `Internal Server Error` | Unhandled error during inference. |
