import React, { useState } from 'react';
import { Image as ImageIcon } from 'lucide-react';

const ImageViewer = ({ resultData }) => {
  const [activeTab, setActiveTab] = useState('heatmap'); // 'original', 'heatmap', 'perturbed'

  // If no data, show empty state
  if (!resultData) {
    return (
      <div className="glass-panel viewer-card" style={{ justifyContent: 'center', alignItems: 'center' }}>
        <ImageIcon size={64} style={{ color: 'var(--text-muted)', marginBottom: '16px', opacity: 0.5 }} />
        <h3 style={{ color: 'var(--text-muted)' }}>No Image to Display</h3>
        <p style={{ color: 'var(--border-highlight)', fontSize: '0.9rem', textAlign: 'center', marginTop: '8px' }}>
          Upload an image and run analysis from the left panel.
        </p>
      </div>
    );
  }

  const { images } = resultData;

  return (
    <div className="glass-panel viewer-card">
      {/* View Selector Tabs */}
      <div className="viewer-header">
        <button 
          className={`view-tab ${activeTab === 'original' ? 'active' : ''}`}
          onClick={() => setActiveTab('original')}
        >
          Original Image
        </button>
        <button 
          className={`view-tab ${activeTab === 'heatmap' ? 'active' : ''}`}
          onClick={() => setActiveTab('heatmap')}
        >
          XAI Heatmap
        </button>
      </div>

      {/* Image Display */}
      <div className="image-display-area">
        {/* Render base64 string directly in img src */}
        <img 
          key={activeTab} /* Key forces remount for CSS animation to trigger */
          src={images[activeTab]} 
          alt={`${activeTab} view`} 
          className="main-image"
        />
      </div>
    </div>
  );
};

export default ImageViewer;
