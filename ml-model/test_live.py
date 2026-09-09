import urllib.request
import urllib.error
import json
import uuid

BASE = "http://127.0.0.1:8000"

print("--- Step 1: Health ---")
with urllib.request.urlopen(f"{BASE}/health") as res:
    h = json.loads(res.read().decode())
print("Health:", h["status"], "| Model:", h["model_name"])

print("\n--- Step 2: GET /diseases ---")
with urllib.request.urlopen(f"{BASE}/diseases") as res:
    d = json.loads(res.read().decode())
cnt = len(d)
print("Diseases count:", cnt)
assert cnt == 38, f"Expected 38 classes, got {cnt}"

print("\n--- Step 3: GET /diseases/{disease_name} ---")
with urllib.request.urlopen(f"{BASE}/diseases/Potato___Early_blight") as res:
    p = json.loads(res.read().decode())
print("Found disease:", p["disease"])
print("Crop:", p["crop"])
print("Symptoms count:", len(p["symptoms"]))
print("Management tips count:", len(p["management"]))

print("\n--- Step 4: Unknown disease 404 ---")
try:
    urllib.request.urlopen(f"{BASE}/diseases/NonExistentDisease123")
    raise AssertionError("Expected 404 error but request succeeded")
except urllib.error.HTTPError as e:
    print("Status code:", e.code)
    err_data = json.loads(e.read().decode())
    print("Error response:", err_data)
    assert e.code == 404

print("\n--- Step 5: POST /predict with sample leaf ---")
sample_path = "../frontend/public/samples/potato_early_blight.jpg"
boundary = "----WebKitFormBoundary" + uuid.uuid4().hex
with open(sample_path, "rb") as f:
    img_bytes = f.read()

part1 = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="potato_early_blight.jpg"\r\n'
    f"Content-Type: image/jpeg\r\n\r\n"
).encode("utf-8")
part2 = f"\r\n--{boundary}--\r\n".encode("utf-8")
payload = part1 + img_bytes + part2

req = urllib.request.Request(
    f"{BASE}/predict",
    data=payload,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)

with urllib.request.urlopen(req) as res:
    pred = json.loads(res.read().decode())

print("Predict HTTP status: 200")
print("Disease:", pred["prediction"]["disease"])
print("Crop:", pred["prediction"]["crop"])
print("Confidence %:", pred["prediction"]["confidence_percentage"])
print("Low confidence:", pred["prediction"]["low_confidence"])
print("Has disease_info:", "disease_info" in pred and pred["disease_info"] is not None)
print("Symptoms:", pred["disease_info"]["symptoms"])
print("Management:", pred["disease_info"]["management"])
print("Disclaimer:", pred.get("disclaimer"))
print("History record:", pred.get("history_record"))
print("\n>>> ALL LIVE TESTS VERIFIED SUCCESSFULLY! <<<")
