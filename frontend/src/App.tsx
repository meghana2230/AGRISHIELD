import React, { useState, useEffect, useRef } from 'react';
import { predictDisease, checkBackendHealth, API_BASE_URL } from './services/api';
import type { PredictionResponse, HealthResponse } from './types/api';

export default function App() {
  // Application State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isDragActive, setIsDragActive] = useState<boolean>(false);

  // Backend Health State
  const [backendHealth, setBackendHealth] = useState<HealthResponse | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Probe backend health upon component mount and periodically
  useEffect(() => {
    let isMounted = true;

    async function verifyHealth() {
      try {
        const health = await checkBackendHealth();
        if (isMounted) {
          setBackendHealth(health);
          setBackendStatus(health.model_loaded ? 'online' : 'offline');
        }
      } catch {
        if (isMounted) {
          setBackendHealth(null);
          setBackendStatus('offline');
        }
      }
    }

    verifyHealth();
    const interval = setInterval(verifyHealth, 10000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // Cleanup object URLs on unmount or file change
  useEffect(() => {
    return () => {
      if (previewUrl && previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  // Handle file selection
  const handleFileChange = (file: File) => {
    setErrorMessage(null);
    setPredictionResult(null);

    // Validate type
    const validExtensions = ['.jpg', '.jpeg', '.png', '.webp'];
    const lowerName = file.name.toLowerCase();
    const hasValidExt = validExtensions.some((ext) => lowerName.endsWith(ext));

    if (!hasValidExt && !file.type.startsWith('image/')) {
      setErrorMessage('Unsupported file format. Please upload a valid JPG, JPEG, PNG, or WebP image.');
      return;
    }

    // Validate size (15 MB)
    if (file.size > 15 * 1024 * 1024) {
      setErrorMessage('The selected image exceeds 15 MB. Please select a smaller photo.');
      return;
    }

    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileChange(e.target.files[0]);
    }
  };

  // Drag & Drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  // Clear current image
  const handleClearImage = () => {
    setSelectedFile(null);
    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setPredictionResult(null);
    setErrorMessage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Load sample dataset image for zero-friction evaluation
  const handleLoadSample = async (samplePath: string, fileName: string) => {
    try {
      setLoading(true);
      setErrorMessage(null);
      const response = await fetch(samplePath);
      if (!response.ok) throw new Error('Failed to load sample image');
      const blob = await response.blob();
      const file = new File([blob], fileName, { type: 'image/jpeg' });
      handleFileChange(file);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Error loading sample image';
      setErrorMessage(msg);
    } finally {
      setLoading(false);
    }
  };

  // Execute inference request
  const handleDetectDisease = async () => {
    if (!selectedFile || loading) return;

    setLoading(true);
    setErrorMessage(null);

    try {
      const response = await predictDisease(selectedFile);
      setPredictionResult(response);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unexpected error during disease prediction.';
      setErrorMessage(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Navigation Header */}
      <header className="navbar">
        <div className="brand-container">
          <div className="brand-logo-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2a10 10 0 0 1 10 10c0 5.523-4.477 10-10 10S2 17.523 2 12a10 10 0 0 1 10-10z" />
              <path d="M12 2a7 7 0 0 1 7 7c0 3.866-3.134 7-7 7a7 7 0 0 1-7-7c0-3.866 3.134-7 7-7z" />
              <path d="M12 6v12" />
            </svg>
          </div>
          <div>
            <span className="brand-name">AgriShield</span>
          </div>
          <span className="brand-badge">SIH 26131</span>
        </div>

        {/* Live Backend Connection Indicator */}
        <div className="nav-status">
          <span className={`status-dot ${backendStatus === 'online' ? 'online' : 'offline'}`}></span>
          <span>
            {backendStatus === 'online'
              ? `Backend Online (${backendHealth?.model_name || 'EfficientNet-B0'} • ${backendHealth?.device?.toUpperCase() || 'CPU'})`
              : backendStatus === 'checking'
              ? 'Checking server...'
              : 'Backend Offline'}
          </span>
        </div>
      </header>

      {/* Main Container */}
      <main className="main-content">
        {/* Hero Section */}
        <section className="hero-section">
          <div className="hero-pill">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
              <polyline points="22 4 12 14.01 9 11.01" />
            </svg>
            <span>Deep Learning Plant Pathology System</span>
          </div>
          <h1 className="hero-title">
            Crop Disease Detection <span>Powered by AI</span>
          </h1>
          <p className="hero-subtitle">
            Instant leaf diagnostics across 38 crop disease classes using deep transfer learning.
            Upload or snap a photo of any crop leaf to detect pathologies in milliseconds.
          </p>
        </section>

        {/* Two-Column Workspace */}
        <div className="detector-grid">
          {/* Left Column: Image Input & Upload */}
          <div className="glass-card">
            <div className="card-header">
              <h2 className="card-title">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                Leaf Image Input
              </h2>
              {selectedFile && (
                <span className="badge badge-emerald">
                  {(selectedFile.size / 1024).toFixed(0)} KB
                </span>
              )}
            </div>

            {/* Error Banner */}
            {errorMessage && (
              <div className="error-banner">
                <div className="error-banner-content">
                  <svg className="error-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="8" x2="12" y2="12" />
                    <line x1="12" y1="16" x2="12.01" y2="16" />
                  </svg>
                  <div>
                    <strong>Action Required:</strong>
                    <div>{errorMessage}</div>
                  </div>
                </div>
                <button
                  onClick={() => setErrorMessage(null)}
                  style={{ background: 'transparent', border: 'none', color: '#fecdd3', cursor: 'pointer' }}
                >
                  ✕
                </button>
              </div>
            )}

            {/* Upload Area / Dropzone */}
            {!previewUrl ? (
              <div
                className={`dropzone ${isDragActive ? 'active' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="dropzone-icon-box">
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                    <circle cx="8.5" cy="8.5" r="1.5" />
                    <polyline points="21 15 16 10 5 21" />
                  </svg>
                </div>
                <h3 className="dropzone-title">Click to upload or drag & drop</h3>
                <p className="dropzone-desc">Supports JPEG, PNG, or WebP (up to 15 MB)</p>
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  accept="image/jpeg,image/png,image/webp"
                  onChange={handleInputChange}
                />
              </div>
            ) : (
              /* Selected Image Preview */
              <div>
                <div className="preview-container">
                  <img src={previewUrl} alt="Selected Crop Leaf" className="preview-img" />
                  {loading && <div className="scanning-line"></div>}
                  <div className="preview-overlay">
                    <div className="preview-details">
                      <div>File: <b>{selectedFile?.name || 'sample_leaf.jpg'}</b></div>
                      <div>Format: <b>{selectedFile?.type || 'image/jpeg'}</b></div>
                    </div>
                    {!loading && (
                      <button className="btn-remove" onClick={handleClearImage}>
                        Remove
                      </button>
                    )}
                  </div>
                </div>

                {/* Primary Action Button */}
                <button
                  className="btn-detect"
                  disabled={loading || !selectedFile}
                  onClick={handleDetectDisease}
                >
                  {loading ? (
                    <>
                      <div className="spinner"></div>
                      <span>Analyzing leaf pathology...</span>
                    </>
                  ) : (
                    <>
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                        <circle cx="11" cy="11" r="8" />
                        <line x1="21" y1="21" x2="16.65" y2="16.65" />
                      </svg>
                      <span>Detect Crop Disease</span>
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Quick Demo Sample Selector */}
            <div className="samples-wrapper">
              <div className="samples-label">Quick Test Samples from Dataset</div>
              <div className="samples-grid">
                <button
                  className="sample-chip"
                  disabled={loading}
                  onClick={() => handleLoadSample('/samples/potato_early_blight.jpg', 'potato_early_blight.jpg')}
                >
                  <span style={{ fontSize: '1.1rem' }}>🥔</span>
                  <div>
                    <div style={{ fontWeight: 600, color: '#f8fafc' }}>Potato Early Blight</div>
                    <div style={{ fontSize: '0.72rem', color: '#94a3b8' }}>Pathological test sample</div>
                  </div>
                </button>
                <button
                  className="sample-chip"
                  disabled={loading}
                  onClick={() => handleLoadSample('/samples/tomato_healthy.jpg', 'tomato_healthy.jpg')}
                >
                  <span style={{ fontSize: '1.1rem' }}>🍅</span>
                  <div>
                    <div style={{ fontWeight: 600, color: '#f8fafc' }}>Tomato Healthy</div>
                    <div style={{ fontSize: '0.72rem', color: '#94a3b8' }}>Healthy control sample</div>
                  </div>
                </button>
              </div>
            </div>
          </div>

          {/* Right Column: Disease Prediction Results */}
          <div className="glass-card">
            <div className="card-header">
              <h2 className="card-title">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" strokeWidth="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="16" y1="13" x2="8" y2="13" />
                  <line x1="16" y1="17" x2="8" y2="17" />
                  <polyline points="10 9 9 9 8 9" />
                </svg>
                Diagnostic Report
              </h2>
              {predictionResult && (
                <span className={`badge ${predictionResult.prediction.is_healthy ? 'badge-emerald' : 'badge-amber'}`}>
                  {predictionResult.prediction.is_healthy ? 'Healthy Leaf' : 'Pathology Detected'}
                </span>
              )}
            </div>

            {/* Diagnostic Report Body: Prediction Result vs Error State vs Empty State */}
            {predictionResult ? (
              <div>
                {/* Primary Diagnosis Header */}
                <div className="result-header">
                  <div className="result-crop-tag">
                    {predictionResult.prediction.crop || predictionResult.disease_info?.crop || 'Diagnosed Crop'}
                  </div>
                  <h3 className="result-disease-name">
                    {predictionResult.prediction.crop && !predictionResult.prediction.disease.toLowerCase().includes(predictionResult.prediction.crop.toLowerCase())
                      ? `${predictionResult.prediction.crop} ${predictionResult.prediction.disease}`
                      : predictionResult.prediction.disease}
                  </h3>
                  <div className="result-meta-row">
                    <div>
                      Crop: <b>{predictionResult.prediction.crop || predictionResult.disease_info?.crop || 'N/A'}</b>
                    </div>
                    <div>
                      Taxonomy: <code>{predictionResult.prediction.predicted_class || 'Standard'}</code>
                    </div>
                    {predictionResult.prediction.inference_time_ms && (
                      <div>
                        Latency: <b>{predictionResult.prediction.inference_time_ms} ms</b>
                      </div>
                    )}
                  </div>
                </div>

                {/* State 2: Low-Confidence Warning Alert */}
                {predictionResult.prediction.low_confidence && (
                  <div className="alert-low-confidence">
                    <div className="alert-low-confidence-icon">⚠️</div>
                    <div>
                      <strong>Prediction Uncertain (Low Confidence):</strong> Classification confidence is{' '}
                      {predictionResult.prediction.confidence_percentage.toFixed(1)}% (below recommended threshold).
                      Please capture or upload a clearer, well-lit, close-up image of the leaf for an accurate diagnosis.
                    </div>
                  </div>
                )}

                {/* Primary Confidence Gauge */}
                <div className="confidence-box">
                  <div className="confidence-header">
                    <span className="confidence-title">Classification Confidence</span>
                    <span
                      className="confidence-value"
                      style={{
                        color:
                          predictionResult.prediction.confidence_percentage >= 70
                            ? '#34d399'
                            : predictionResult.prediction.confidence_percentage >= 40
                            ? '#fbbf24'
                            : '#fb7185',
                      }}
                    >
                      {predictionResult.prediction.confidence_percentage.toFixed(1)}%
                    </span>
                  </div>
                  <div className="confidence-bar-bg">
                    <div
                      className="confidence-bar-fill"
                      style={{
                        width: `${Math.min(100, Math.max(0, predictionResult.prediction.confidence_percentage))}%`,
                        backgroundColor:
                          predictionResult.prediction.confidence_percentage >= 70
                            ? '#10b981'
                            : predictionResult.prediction.confidence_percentage >= 40
                            ? '#f59e0b'
                            : '#f43f5e',
                      }}
                    ></div>
                  </div>
                </div>

                {/* Actionable Disease Information: Symptoms & Description */}
                {predictionResult.disease_info && (
                  <>
                    <div className="info-section">
                      <div className="info-section-header">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                          <circle cx="12" cy="12" r="10" />
                          <line x1="12" y1="8" x2="12" y2="12" />
                          <line x1="12" y1="16" x2="12.01" y2="16" />
                        </svg>
                        <h4 className="info-section-title">Common Visible Symptoms</h4>
                      </div>
                      {predictionResult.disease_info.description && (
                        <p className="info-section-desc">{predictionResult.disease_info.description}</p>
                      )}
                      <ul className="info-list">
                        {predictionResult.disease_info.symptoms.map((symptom, idx) => (
                          <li key={idx} className="info-list-item">
                            <span className="info-bullet-icon info-bullet-symptom">●</span>
                            <span>{symptom}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* Actionable Disease Information: Prevention & Management */}
                    <div className="info-section">
                      <div className="info-section-header">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#34d399" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                          <path d="M9 12l2 2 4-4" />
                        </svg>
                        <h4 className="info-section-title">Crop Management &amp; Prevention Guidance</h4>
                      </div>
                      <ul className="info-list">
                        {predictionResult.disease_info.management.map((tip, idx) => (
                          <li key={idx} className="info-list-item">
                            <span className="info-bullet-icon info-bullet-management">✔</span>
                            <span>{tip}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </>
                )}

                {/* Top-3 Candidates Breakdown */}
                {predictionResult.prediction.top_predictions && predictionResult.prediction.top_predictions.length > 0 && (
                  <div style={{ marginBottom: '1.25rem' }}>
                    <h4 className="top-candidates-title">Top 3 Candidates:</h4>
                    {predictionResult.prediction.top_predictions.slice(0, 3).map((item, index) => {
                      const percentage =
                        item.confidence_percentage !== undefined
                          ? item.confidence_percentage
                          : (item.confidence * 100);

                      const formattedName =
                        item.crop && !item.disease.toLowerCase().includes(item.crop.toLowerCase())
                          ? `${item.crop} ${item.disease}`
                          : item.disease;

                      return (
                        <div key={index} className="candidate-card">
                          <div className="candidate-info">
                            <span className={`candidate-rank ${index === 0 ? 'rank-1' : ''}`}>
                              {index + 1}
                            </span>
                            <div>
                              <span className="candidate-name">{formattedName}</span>
                              {item.crop && <span className="candidate-crop">({item.crop})</span>}
                            </div>
                          </div>
                          <span
                            className="candidate-score"
                            style={{
                              color: index === 0 ? '#34d399' : '#94a3b8',
                            }}
                          >
                            {percentage.toFixed(1)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Advisory Safety Disclaimer */}
                <div className="disclaimer-box">
                  <svg className="disclaimer-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="16" x2="12" y2="12" />
                    <line x1="12" y1="8" x2="12.01" y2="8" />
                  </svg>
                  <div>
                    {predictionResult.disclaimer ||
                      'This AI prediction is intended as a decision-support aid. Confirm important crop-treatment decisions with a qualified agricultural expert.'}
                  </div>
                </div>

                {/* Repeat Analysis Flow: Analyze Another Leaf */}
                <button
                  className="btn-new-analysis"
                  onClick={handleClearImage}
                  title="Upload or select another crop leaf photo to analyze"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                  <span>Analyze Another Leaf Image</span>
                </button>
              </div>
            ) : errorMessage ? (
              /* State 3: Technical Error / Model Unavailable State */
              <div className="server-error-state">
                <div className="server-error-icon">
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="8" x2="12" y2="12" />
                    <line x1="12" y1="16" x2="12.01" y2="16" />
                  </svg>
                </div>
                <h3 className="server-error-title">Inference Service Unavailable</h3>
                <p className="server-error-desc">{errorMessage}</p>
                <div style={{ marginTop: '1.25rem', fontSize: '0.8rem', color: '#94a3b8' }}>
                  Please verify API server connectivity: <code>{API_BASE_URL}</code>
                </div>
              </div>
            ) : (
              /* State 0: Awaiting Image Placeholder */
              <div className="empty-state">
                <div className="empty-state-icon">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                    <path d="M9 12l2 2 4-4" />
                  </svg>
                </div>
                <h3 className="empty-state-title">Awaiting Crop Leaf Image</h3>
                <p style={{ maxWidth: '320px', margin: '0 auto', fontSize: '0.9rem' }}>
                  Select or drop an infected crop leaf photo and click &ldquo;Detect Crop Disease&rdquo; to view real-time AI pathology analysis.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="footer">
        <div>
          AgriShield • Smart India Hackathon (SIH 26131) • Powered by PyTorch, EfficientNet-B0 &amp; FastAPI
        </div>
        <div style={{ marginTop: '0.35rem', color: '#64748b' }}>
          Backend API Service endpoint: <code>{API_BASE_URL}/predict</code>
        </div>
      </footer>
    </div>
  );
}
