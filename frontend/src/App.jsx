import React, { useState } from 'react';
import { BrainCircuit } from 'lucide-react';
import './index.css';

// We will create these component files next
import UploadPanel from './components/UploadPanel';
import ImageViewer from './components/ImageViewer';
import ResultPanel from './components/ResultPanel';

function App() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  // Handle uploading and calling Flask API
  const handleAnalyze = async (formData) => {
    try {
      setLoading(true);
      setResults(null); // Clear previous results

      // POST request to our Flask backend via Vite Proxy
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
        // Notice: We don't set Content-Type header.
        // The browser automatically sets it to multipart/form-data with a boundary.
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Something went wrong on the server.');
      }

      const data = await response.json();
      setResults(data);
      
    } catch (error) {
      console.error('Error during analysis:', error);
      alert(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* HEADER */}
      <header className="header">
        <BrainCircuit size={32} color="#6366f1" />
        <h1>IPEM - eXplainable AI Model</h1>
      </header>

      {/* MAIN LAYOUT (3 Columns via CSS Grid) */}
      <main className="main-content">
        
        {/* Left Column: Configuration & Upload */}
        <div className="col-left">
          <UploadPanel onAnalyze={handleAnalyze} disabled={loading} />
        </div>

        {/* Center Column: Image Viewer */}
        <div className="col-center">
          <ImageViewer resultData={results} />
        </div>

        {/* Right Column: Metrics & Predictions */}
        <div className="col-right">
          <ResultPanel resultData={results} />
        </div>

      </main>

      {/* LOADING OVERLAY */}
      {loading && (
        <div className="loader-overlay">
          <div className="spinner"></div>
          <div className="loader-text">Analyzing Image & Generating Explanation...</div>
        </div>
      )}
    </div>
  );
}

export default App;
