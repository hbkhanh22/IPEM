import React, { useState, useRef } from 'react';
import { UploadCloud, X, Play } from 'lucide-react';

const UploadPanel = ({ onAnalyze, disabled }) => {
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  
  const [dataset, setDataset] = useState('brain-tumor');
  const [model, setModel] = useState('efficientnet_b3');
  const [xaiMethod, setXaiMethod] = useState('gradcam');
  
  const fileInputRef = useRef(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      // Automatically switch off GradCAM if Transformer is selected later
    }
  };

  const handleRemoveImage = (e) => {
    e.stopPropagation();
    setImage(null);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!image) {
      alert("Please upload an image first!");
      return;
    }

    const formData = new FormData();
    formData.append('image', image);
    formData.append('dataset', dataset);
    formData.append('model_name', model);
    formData.append('xai_method', xaiMethod);

    onAnalyze(formData);
  };

  // If Transformer is selected, GradCAM must be disabled
  const isTransformer = model === 'transformer';
  
  // Auto-switch away from GradCAM if user picks Transformer
  if (isTransformer && xaiMethod === 'gradcam') {
    setXaiMethod('lime');
  }

  return (
    <form className="glass-panel" onSubmit={handleSubmit}>
      
      {/* 1. Image Upload Section */}
      <h2 style={{ marginBottom: '20px' }}>Configuration</h2>
      
      <div className="input-group">
        <label className="label">Upload Image</label>
        <div 
          className={`dropzone ${previewUrl ? 'active' : ''}`}
          onClick={() => !previewUrl && fileInputRef.current.click()}
        >
          <input 
            type="file" 
            accept="image/*" 
            style={{ display: 'none' }} 
            ref={fileInputRef}
            onChange={handleImageChange}
            disabled={disabled}
          />
          
          {previewUrl ? (
            <div className="preview-container">
              <img src={previewUrl} alt="Preview" />
              <button 
                type="button" 
                className="remove-btn" 
                onClick={handleRemoveImage}
                disabled={disabled}
              >
                <X size={16} />
              </button>
            </div>
          ) : (
            <>
              <UploadCloud size={40} className="dropzone-icon" />
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                Click to browse or drag image here
              </p>
            </>
          )}
        </div>
      </div>

      {/* 2. Dataset Selection */}
      <div className="input-group">
        <label className="label">Dataset</label>
        <select 
          className="btn" 
          style={{ width: '100%', justifyContent: 'space-between', border: '1px solid var(--border-color)' }}
          value={dataset}
          onChange={(e) => setDataset(e.target.value)}
          disabled={disabled}
        >
          <option value="brain-tumor">Brain Tumor</option>
          <option value="caltech-101">Caltech-101</option>
        </select>
      </div>

      {/* 3. Model Selection */}
      <div className="input-group">
        <label className="label">Model Architecture</label>
        <select 
          className="btn" 
          style={{ width: '100%', justifyContent: 'space-between', border: '1px solid var(--border-color)' }}
          value={model}
          onChange={(e) => setModel(e.target.value)}
          disabled={disabled}
        >
          <option value="efficientnet_b3">EfficientNet-B3</option>
          <option value="resnet50">ResNet-50</option>
          <option value="transformer">Vision Transformer (ViT)</option>
        </select>
      </div>

      {/* 4. XAI Method Selection (Radio Buttons for better UI than select) */}
      <div className="input-group">
        <label className="label">Explanation Method</label>
        <div className="radio-options">
          {['lime', 'gradcam', 'rise', 'ipem'].map((method) => {
            const labelMap = {
              lime: 'LIME (Local Interpretable)',
              gradcam: 'Grad-CAM',
              rise: 'RISE',
              ipem: 'IPEM (Proposed)'
            };
            
            const isDisabled = method === 'gradcam' && isTransformer;

            return (
              <label key={method} style={{ opacity: isDisabled ? 0.5 : 1 }}>
                <input 
                  type="radio" 
                  name="xai_method" 
                  value={method}
                  checked={xaiMethod === method}
                  onChange={(e) => setXaiMethod(e.target.value)}
                  disabled={disabled || isDisabled}
                />
                <div className="radio-label">
                  <div className="radio-indicator"></div>
                  {labelMap[method]}
                  {isDisabled && <span style={{fontSize: '0.75rem', marginLeft: 'auto', color: 'var(--danger)'}}>Not supported</span>}
                </div>
              </label>
            );
          })}
        </div>
      </div>

      {/* Submit Button */}
      <button 
        type="submit" 
        className="btn btn-primary" 
        style={{ width: '100%', marginTop: '10px' }}
        disabled={disabled || !image}
      >
        <Play size={18} />
        {disabled ? 'Processing...' : 'Run Analysis'}
      </button>

    </form>
  );
};

export default UploadPanel;
