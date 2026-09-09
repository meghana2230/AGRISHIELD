import fs from 'fs';

const API_BASE_URL = process.env.VITE_API_BASE_URL || 'https://agrishield-backend.onrender.com';

async function sendPredict(filename, buffer) {
  const boundary = '----WebKitFormBoundary' + Math.random().toString(16).substring(2);
  const header = Buffer.from(
    `--${boundary}\r\n` +
    `Content-Disposition: form-data; name="file"; filename="${filename}"\r\n` +
    `Content-Type: image/jpeg\r\n\r\n`
  );
  const footer = Buffer.from(`\r\n--${boundary}--\r\n`);
  const payload = Buffer.concat([header, buffer, footer]);

  return await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': `multipart/form-data; boundary=${boundary}`,
    },
    body: payload,
  });
}

async function runTests() {
  console.log('=== AgriShield Frontend <-> Backend Full Integration Test ===\n');

  // Test 1: Health
  console.log('[Test 1] GET /health');
  const healthRes = await fetch(`${API_BASE_URL}/health`);
  const healthData = await healthRes.json();
  console.log('Status:', healthRes.status, '| Health:', healthData.status, '| Model:', healthData.model_name);
  if (healthRes.status !== 200 || !healthData.model_loaded) throw new Error('Health check failed');

  // Test 2: Pathological sample
  console.log('\n[Test 2] POST /predict with potato_early_blight.jpg');
  const potatoBytes = fs.readFileSync('public/samples/potato_early_blight.jpg');
  const pred1Res = await sendPredict('potato_early_blight.jpg', potatoBytes);
  const pred1 = await pred1Res.json();
  console.log('Status:', pred1Res.status);
  console.log('Predicted Crop:', pred1.prediction.crop);
  console.log('Predicted Disease:', pred1.prediction.disease);
  console.log('Confidence %:', pred1.prediction.confidence_percentage);
  console.log('Is Healthy:', pred1.prediction.is_healthy);
  console.log('Symptoms:', pred1.disease_info.symptoms.length, 'items');
  console.log('Management:', pred1.disease_info.management.length, 'items');
  if (pred1Res.status !== 200 || !pred1.success) throw new Error('Potato early blight prediction failed');

  // Test 3: Healthy sample
  console.log('\n[Test 3] POST /predict with tomato_healthy.jpg');
  const tomatoBytes = fs.readFileSync('public/samples/tomato_healthy.jpg');
  const pred2Res = await sendPredict('tomato_healthy.jpg', tomatoBytes);
  const pred2 = await pred2Res.json();
  console.log('Status:', pred2Res.status);
  console.log('Predicted Crop:', pred2.prediction.crop);
  console.log('Predicted Disease:', pred2.prediction.disease);
  console.log('Confidence %:', pred2.prediction.confidence_percentage);
  console.log('Is Healthy:', pred2.prediction.is_healthy);
  if (pred2Res.status !== 200 || !pred2.success) throw new Error('Tomato healthy prediction failed');

  // Test 4: Error handling - corrupted/unsupported file
  console.log('\n[Test 4] Error Handling: POST /predict with invalid extension test.txt');
  const invalidRes = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'multipart/form-data; boundary=----WebKitFormBoundary123',
    },
    body: Buffer.from(
      '------WebKitFormBoundary123\r\n' +
      'Content-Disposition: form-data; name="file"; filename="test.txt"\r\n' +
      'Content-Type: text/plain\r\n\r\n' +
      'hello world\r\n' +
      '------WebKitFormBoundary123--\r\n'
    ),
  });
  console.log('Status:', invalidRes.status, '(Expected 415)');
  if (invalidRes.status !== 415) throw new Error('Expected 415 for invalid extension');

  console.log('\n=============================================================');
  console.log('>>> ALL FRONTEND <-> BACKEND INTEGRATION TESTS PASSED! <<<');
  console.log('=============================================================');
}

runTests().catch((err) => {
  console.error('Integration test failed:', err);
  process.exit(1);
});
