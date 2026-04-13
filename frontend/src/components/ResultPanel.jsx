import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Activity, Target, Timer } from 'lucide-react';

const ResultPanel = ({ resultData }) => {
  if (!resultData) {
    return (
      <div className="glass-panel" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <p style={{ color: 'var(--text-muted)' }}>Results will appear here</p>
      </div>
    );
  }

  const { prediction, metrics } = resultData;
  const confidencePercent = (prediction.confidence * 100).toFixed(2);

  // Format data for Recharts [ {name: 'cat', value: 95.5}, ... ]
  const chartData = prediction.top5.map(item => ({
    name: item[0],
    value: parseFloat((item[1] * 100).toFixed(2))
  }));

  return (
    <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <h2 style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Activity size={24} color="var(--primary)" /> Analysis Results
      </h2>

      {/* Top Prediction Highlight */}
      <div className="prediction-box">
        <p style={{ color: 'rgba(255,255,255,0.7)', textTransform: 'uppercase', fontSize: '0.8rem', letterSpacing: '2px' }}>
          Top Prediction
        </p>
        <h2>{prediction.class_name}</h2>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', color: 'var(--success)' }}>
          <Target size={18} />
          <span style={{ fontWeight: 600, fontSize: '1.2rem' }}>{confidencePercent}% Confidence</span>
        </div>
      </div>

      {/* Basic Metrics Display */}
      <h3 style={{ marginBottom: '12px', fontSize: '1rem', color: 'var(--text-muted)' }}>XAI Performance Metrics</h3>
      <div className="metric-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
        <div className="metric-item">
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            <Activity size={14} /> AOPC-MORF
          </div>
          <div className="value">{metrics.aopc_morf !== undefined ? metrics.aopc_morf.toFixed(4) : 'N/A'}</div>
        </div>
        
        <div className="metric-item">
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
             <Target size={14} /> Insertion
          </div>
          <div className="value">{metrics.insertion !== undefined ? metrics.insertion.toFixed(4) : 'N/A'}</div>
        </div>

        <div className="metric-item">
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
             <Target size={14} /> Deletion
          </div>
          <div className="value">{metrics.deletion !== undefined ? metrics.deletion.toFixed(4) : 'N/A'}</div>
        </div>

        <div className="metric-item">
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            <Timer size={14} /> Time
          </div>
          <div className="value">{metrics.explanation_time !== undefined ? `${metrics.explanation_time}s` : 'N/A'}</div>
        </div>
      </div>

      {/* Top 5 Chart */}
      <h3 style={{ marginBottom: '16px', marginTop: 'auto', fontSize: '1rem', color: 'var(--text-muted)' }}>Top-5 Probabilities</h3>
      <div style={{ width: '100%', height: '220px', marginBottom: '10px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 30, left: 40, bottom: 0 }}>
            <XAxis type="number" hide domain={[0, 100]} />
            <YAxis 
              dataKey="name" 
              type="category" 
              axisLine={false} 
              tickLine={false} 
              tick={{ fill: 'var(--text-main)', fontSize: 12 }} 
              width={100}
            />
            {/* Tooltip Styling modified: added contentStyle and itemStyle to ensure white text */}
            <Tooltip 
              cursor={{ fill: 'rgba(255,255,255,0.05)' }} 
              formatter={(val) => `${val}%`} 
              contentStyle={{ backgroundColor: 'var(--bg-panel-solid)', borderColor: 'var(--border-color)', color: '#fff' }}
              itemStyle={{ color: '#fff' }}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20} animationDuration={1000}>
              {
                chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={index === 0 ? 'var(--primary)' : 'rgba(99, 102, 241, 0.4)'} />
                ))
              }
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      
    </div>
  );
};

export default ResultPanel;
