/**
 * TypeScript interfaces for AgriShield Inference API (SIH 26131).
 * Enforces strict typing for backend request and response contracts.
 */

export interface TopPrediction {
  disease: string;
  confidence: number;
  confidence_percentage?: number;
  crop?: string;
  class_name?: string;
  is_healthy?: boolean;
}

export interface PredictionDetails {
  disease: string;
  confidence: number;
  confidence_percentage: number;
  low_confidence: boolean;
  top_predictions: TopPrediction[];
  crop?: string;
  predicted_class?: string;
  is_healthy?: boolean;
  inference_time_ms?: number;
}

export interface DiseaseInfo {
  disease: string;
  crop: string;
  is_healthy: boolean;
  description: string;
  symptoms: string[];
  management: string[];
}

export interface HistoryRecord {
  timestamp: string;
  crop: string;
  disease: string;
  confidence: number;
  confidence_percentage: number;
  top_predictions: TopPrediction[];
  image_reference?: string;
}

export interface PredictionResponse {
  success: boolean;
  prediction: PredictionDetails;
  disease_info?: DiseaseInfo;
  disclaimer?: string;
  history_record?: HistoryRecord;
}

export interface ErrorResponse {
  success: boolean;
  error: string;
  detail?: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
  model_name?: string;
  num_classes?: number;
}

export interface RootResponse {
  project: string;
  api: string;
  version: string;
  status: string;
}
