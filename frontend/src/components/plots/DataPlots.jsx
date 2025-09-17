import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';

const DataPlots = ({ data }) => {
  const [plotData, setPlotData] = useState(null);
  const [activeParameter, setActiveParameter] = useState('temperature');

  useEffect(() => {
    if (data && data.profiles) {
      setPlotData(data);
    }
  }, [data]);

  // Sample data for demonstration
  const sampleData = {
    temperature: {
      depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
      values: [28.5, 28.2, 27.8, 25.1, 22.3, 18.7, 12.4, 8.2, 5.1, 3.8],
      title: 'Temperature Profile',
      yLabel: 'Temperature (°C)',
      color: '#ff6b6b'
    },
    salinity: {
      depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
      values: [34.2, 34.3, 34.4, 34.6, 34.8, 35.1, 35.3, 35.0, 34.8, 34.7],
      title: 'Salinity Profile',
      yLabel: 'Salinity (PSU)',
      color: '#4ecdc4'
    },
    oxygen: {
      depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
      values: [220, 218, 215, 200, 180, 150, 120, 180, 200, 220],
      title: 'Dissolved Oxygen Profile',
      yLabel: 'Oxygen (μmol/kg)',
      color: '#45b7d1'
    }
  };

  const getCurrentData = () => {
    if (plotData && plotData.profiles && plotData.profiles[activeParameter]) {
      return plotData.profiles[activeParameter];
    }
    return sampleData[activeParameter];
  };

  const currentData = getCurrentData();

  const plotConfig = {
    data: [
      {
        x: currentData.values,
        y: currentData.depths,
        type: 'scatter',
        mode: 'lines+markers',
        marker: { color: currentData.color, size: 6 },
        line: { color: currentData.color, width: 2 },
        name: currentData.title
      }
    ],
    layout: {
      title: {
        text: currentData.title,
        font: { size: 16 }
      },
      xaxis: {
        title: currentData.yLabel,
        showgrid: true,
        gridcolor: '#e0e0e0'
      },
      yaxis: {
        title: 'Depth (m)',
        autorange: 'reversed',
        showgrid: true,
        gridcolor: '#e0e0e0'
      },
      margin: { l: 60, r: 40, t: 60, b: 60 },
      plot_bgcolor: 'white',
      paper_bgcolor: 'white',
      font: { size: 12 }
    },
    config: {
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
    }
  };

  return (
    <div className="h-full flex flex-col">
      <div className="bg-gray-50 px-4 py-2 border-b border-gray-200">
        <h3 className="text-lg font-medium text-gray-900">Ocean Profile Data</h3>
        <p className="text-sm text-gray-600">
          Depth profiles of ocean parameters from ARGO floats
        </p>
      </div>
      
      {/* Parameter Selection */}
      <div className="bg-white border-b border-gray-200 p-3">
        <div className="flex space-x-2">
          {Object.keys(sampleData).map((param) => (
            <button
              key={param}
              onClick={() => setActiveParameter(param)}
              className={`px-3 py-1 rounded-md text-sm font-medium ${
                activeParameter === param
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {param.charAt(0).toUpperCase() + param.slice(1)}
            </button>
          ))}
        </div>
      </div>
      
      {/* Plot Area */}
      <div className="flex-1 bg-white">
        <Plot
          data={plotConfig.data}
          layout={plotConfig.layout}
          config={plotConfig.config}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler={true}
        />
      </div>
      
      {/* Plot Info */}
      <div className="bg-gray-50 border-t border-gray-200 p-2">
        <div className="text-xs text-gray-600">
          <span className="font-medium">Data Range:</span> 
          {` ${Math.min(...currentData.values).toFixed(1)} - ${Math.max(...currentData.values).toFixed(1)} ${currentData.yLabel.split('(')[1]?.replace(')', '') || ''}`}
          <span className="ml-4 font-medium">Max Depth:</span> 
          {` ${Math.max(...currentData.depths)}m`}
        </div>
      </div>
    </div>
  );
};

export default DataPlots;