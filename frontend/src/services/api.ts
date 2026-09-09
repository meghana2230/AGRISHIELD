/**
 * AgriShield Backend API Client Service (SIH 26131).
 * Communicates with FastAPI ML inference endpoints for disease classification and health status.
 */

import type { PredictionResponse, HealthResponse } from '../types/api';

// Configurable API base URL defaulting to standard local dev backend
export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024; // 15 MB
const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp'];

/**
 * Checks backend API health and model availability.
 */
export async function checkBackendHealth(): Promise<HealthResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });

    if (!response.ok) {
      throw new Error(`Health check failed with HTTP ${response.status}`);
    }

    return (await response.json()) as HealthResponse;
  } catch (error: unknown) {
    if (error instanceof TypeError && (error.message.includes('fetch') || error.message.includes('NetworkError'))) {
      throw new Error(
        `Unable to connect to the disease detection server at ${API_BASE_URL}. Please ensure the FastAPI backend is running.`
      );
    }
    throw error;
  }
}

/**
 * Sends a crop leaf image to the FastAPI backend for ML disease prediction.
 *
 * @param file Uploaded image File object
 * @returns PredictionResponse with disease diagnosis, confidence, and top-3 ranking
 */
export async function predictDisease(file: File): Promise<PredictionResponse> {
  // 1. Client-side presence validation
  if (!file) {
    throw new Error('No image file selected. Please select or capture a crop leaf image.');
  }

  // 2. Client-side file extension & MIME validation
  const fileNameLower = file.name.toLowerCase();
  const hasValidExt = ALLOWED_EXTENSIONS.some((ext) => fileNameLower.endsWith(ext));
  const isImageMime = file.type.startsWith('image/') || file.type === 'application/octet-stream';

  if (!hasValidExt && !isImageMime) {
    throw new Error(
      `Unsupported file format (${file.type || 'unknown'}). Supported formats: JPG, JPEG, PNG, WebP.`
    );
  }

  // 3. Client-side file size validation
  if (file.size === 0) {
    throw new Error('Uploaded image file is empty (0 bytes). Please select a valid photo.');
  }

  if (file.size > MAX_IMAGE_SIZE_BYTES) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    throw new Error(`Image size (${sizeMb} MB) exceeds maximum allowed limit of 15 MB.`);
  }

  // 4. Construct multipart payload with form field name 'file'
  const formData = new FormData();
  formData.append('file', file, file.name);

  // 5. Execute HTTP POST to /predict with comprehensive network & HTTP status handling
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
    });
  } catch (networkError: unknown) {
    throw new Error(
      `Unable to connect to the disease detection server at ${API_BASE_URL}. Please ensure the FastAPI backend is running.`
    );
  }

  let responseData: any = null;
  try {
    responseData = await response.json();
  } catch {
    responseData = null;
  }

  if (!response.ok) {
    // 1. Check for structured backend error details
    if (responseData && (responseData.detail || responseData.error)) {
      const msg = typeof responseData.detail === 'string' ? responseData.detail : responseData.error;
      throw new Error(msg);
    }

    // 2. Fallback to friendly status-code-specific messages
    if (response.status === 400) {
      throw new Error('Corrupted or invalid image file. Please provide a clear crop leaf photo.');
    } else if (response.status === 413) {
      throw new Error('The uploaded image exceeds the maximum payload size of 15 MB.');
    } else if (response.status === 415) {
      throw new Error('Unsupported image format. Supported formats: JPG, JPEG, PNG, WebP.');
    } else if (response.status === 422) {
      throw new Error('Missing image file in request payload (Unprocessable Content).');
    } else if (response.status === 503) {
      throw new Error('The crop disease detection model service is temporarily unavailable on the server.');
    } else if (response.status >= 500) {
      throw new Error(`Internal server error (${response.status}) occurred during disease prediction.`);
    }

    throw new Error(`Server responded with HTTP status code ${response.status}.`);
  }

  if (!responseData || !responseData.success || !responseData.prediction) {
    throw new Error('Malformed or incomplete diagnosis response received from disease detection server.');
  }

  return responseData as PredictionResponse;
}
