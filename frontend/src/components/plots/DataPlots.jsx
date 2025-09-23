// import React, { useState, useEffect } from 'react';
// import Plot from 'react-plotly.js';


// const DataPlots = ({ data }) => {
//   const [plotData, setPlotData] = useState(null);
//   const [activeParameter, setActiveParameter] = useState('temperature');

//   useEffect(() => {
//     // Accepts both old and new backend formats
//     if (data && typeof data === 'object') {
//       // New backend: { temperature_profile: {...}, salinity_profile: {...} }
//       if (data.temperature_profile || data.salinity_profile) {
//         setPlotData({
//           temperature: data.temperature_profile || null,
//           salinity: data.salinity_profile || null
//         });
//       } else if (data.profiles) {
//         setPlotData(data);
//       } else {
//         setPlotData(null);
//       }
//     } else {
//       setPlotData(null);
//     }
//   }, [data]);

//   // Sample data for demonstration
//   const sampleData = {
//     temperature: {
//       depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
//       values: [28.5, 28.2, 27.8, 25.1, 22.3, 18.7, 12.4, 8.2, 5.1, 3.8],
//       title: 'Temperature Profile',
//       yLabel: 'Temperature (°C)',
//       color: '#ff6b6b'
//     },
//     salinity: {
//       depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
//       values: [34.2, 34.3, 34.4, 34.6, 34.8, 35.1, 35.3, 35.0, 34.8, 34.7],
//       title: 'Salinity Profile',
//       yLabel: 'Salinity (PSU)',
//       color: '#4ecdc4'
//     },
//     oxygen: {
//       depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
//       values: [220, 218, 215, 200, 180, 150, 120, 180, 200, 220],
//       title: 'Dissolved Oxygen Profile',
//       yLabel: 'Oxygen (μmol/kg)',
//       color: '#45b7d1'
//     }
//   };


//   const getCurrentData = () => {
//     // New backend: plotData.temperature or plotData.salinity
//     if (plotData && plotData[activeParameter]) {
//       return {
//         ...sampleData[activeParameter],
//         ...plotData[activeParameter]
//       };
//     }
//     // Old backend: plotData.profiles[activeParameter]
//     if (plotData && plotData.profiles && plotData.profiles[activeParameter]) {
//       return plotData.profiles[activeParameter];
//     }
//     return sampleData[activeParameter];
//   };

//   const currentData = getCurrentData();

//   const plotConfig = {
//     data: [
//       {
//         x: currentData.values,
//         y: currentData.depths,
//         type: 'scatter',
//         mode: 'lines+markers',
//         marker: { color: currentData.color, size: 6 },
//         line: { color: currentData.color, width: 2 },
//         name: currentData.title
//       }
//     ],
//     layout: {
//       title: {
//         text: currentData.title,
//         font: { size: 16 }
//       },
//       xaxis: {
//         title: currentData.yLabel,
//         showgrid: true,
//         gridcolor: '#e0e0e0'
//       },
//       yaxis: {
//         title: 'Depth (m)',
//         autorange: 'reversed',
//         showgrid: true,
//         gridcolor: '#e0e0e0'
//       },
//       margin: { l: 60, r: 40, t: 60, b: 60 },
//       plot_bgcolor: 'white',
//       paper_bgcolor: 'white',
//       font: { size: 12 }
//     },
//     config: {
//       displayModeBar: true,
//       displaylogo: false,
//       modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
//     }
//   };

//   return (
//     <div className="h-full flex flex-col">
//       <div className="bg-gray-50 px-4 py-2 border-b border-gray-200">
//         <h3 className="text-lg font-medium text-gray-900">Ocean Profile Data</h3>
//         <p className="text-sm text-gray-600">
//           Depth profiles of ocean parameters from ARGO floats
//         </p>
//       </div>
      
//       {/* Parameter Selection */}
//       <div className="bg-white border-b border-gray-200 p-3">
//         <div className="flex space-x-2">
//           {Object.keys(sampleData).map((param) => (
//             <button
//               key={param}
//               onClick={() => setActiveParameter(param)}
//               className={`px-3 py-1 rounded-md text-sm font-medium ${
//                 activeParameter === param
//                   ? 'bg-blue-500 text-white'
//                   : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
//               }`}
//             >
//               {param.charAt(0).toUpperCase() + param.slice(1)}
//             </button>
//           ))}
//         </div>
//       </div>
      
//       {/* Plot Area */}
//       <div className="flex-1 bg-white">
//         <Plot
//           data={plotConfig.data}
//           layout={plotConfig.layout}
//           config={plotConfig.config}
//           style={{ width: '100%', height: '100%' }}
//           useResizeHandler={true}
//         />
//       </div>
      
//       {/* Plot Info */}
//       <div className="bg-gray-50 border-t border-gray-200 p-2">
//         <div className="text-xs text-gray-600">
//           <span className="font-medium">Data Range:</span> 
//           {` ${Math.min(...currentData.values).toFixed(1)} - ${Math.max(...currentData.values).toFixed(1)} ${currentData.yLabel.split('(')[1]?.replace(')', '') || ''}`}
//           <span className="ml-4 font-medium">Max Depth:</span> 
//           {` ${Math.max(...currentData.depths)}m`}
//         </div>
//       </div>
//     </div>
//   );
// };

// export default DataPlots;

// Replace the entire DataPlots.jsx with this updated version


import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';

// Add this function to handle parameter detection
const detectParameterFromQuery = (query) => {
  if (!query) return 'temperature';
  const lowerQuery = query.toLowerCase();
  if (lowerQuery.includes('salinity')) return 'salinity';
  if (lowerQuery.includes('temperature') || lowerQuery.includes('temp')) return 'temperature';
  if (lowerQuery.includes('oxygen')) return 'oxygen';
  return 'temperature'; // default
};

const DataPlots = ({ data }) => {
  const [plotData, setPlotData] = useState(null);
  const [activeParameter, setActiveParameter] = useState('temperature');
  const [availableParameters, setAvailableParameters] = useState(['temperature']);

  useEffect(() => {
    if (data && typeof data === 'object') {
      // Detect parameter from metadata if available
      let detectedParam = 'temperature';
      if (data.metadata && data.metadata.query_type) {
        detectedParam = detectParameterFromQuery(data.metadata.query_type);
      } else if (data.metadata && data.metadata.parameter) {
        detectedParam = data.metadata.parameter;
      }

      // Handle new backend format with type and available_params
      if (data.type && data.available_params) {
        setAvailableParameters(data.available_params);

        // Set active parameter based on detection
        if (data.available_params.includes(detectedParam)) {
          setActiveParameter(detectedParam);
        } else if (data.available_params.length > 0) {
          setActiveParameter(data.available_params[0]);
        }

        // Extract plot data
        if (data.all_data) {
          setPlotData(data.all_data);
        } else if (data.data) {
          const paramFromType = data.type.replace('_profile', '');
          const plotDataObj = {};
          plotDataObj[paramFromType] = data.data;
          setPlotData(plotDataObj);
        }
      } 
      // Handle old format for backward compatibility
      else if (data.temperature_profile || data.salinity_profile) {
        const params = [];
        const plotDataObj = {};

        if (data.temperature_profile) {
          params.push('temperature');
          plotDataObj.temperature = data.temperature_profile;
        }

        if (data.salinity_profile) {
          params.push('salinity');
          plotDataObj.salinity = data.salinity_profile;
        }

        setAvailableParameters(params);
        setPlotData(plotDataObj);

        if (params.length > 0) {
          setActiveParameter(params[0]);
        }
      } else if (data.profiles) {
        setPlotData(data.profiles);
        setAvailableParameters(Object.keys(data.profiles));
      } else {
        setPlotData(null);
        setAvailableParameters(['temperature']);
      }
    } else {
      setPlotData(null);
      setAvailableParameters(['temperature']);
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
    }
  };

  const getCurrentData = () => {
    if (plotData && plotData[activeParameter]) {
      return {
        ...sampleData[activeParameter],
        ...plotData[activeParameter]
      };
    }
    return sampleData[activeParameter];
  };

  const currentData = getCurrentData();
  
  const getParameterUnit = (param) => {
    const units = {
      temperature: '°C',
      salinity: 'PSU',
      oxygen: 'μmol/kg'
    };
    return units[param] || '';
  };

  const plotConfig = {
    data: [
      {
        x: currentData.values,
        y: currentData.depths,
        type: 'scatter',
        mode: 'lines+markers',
        marker: { color: currentData.color, size: 6 },
        line: { color: currentData.color, width: 2 },
        name: currentData.title,
        hovertemplate: 
          '<b>Depth</b>: %{y}m<br>' +
          `<b>${currentData.yLabel.split('(')[0]}</b>: %{x} ${getParameterUnit(activeParameter)}<br>` +
          '<extra></extra>'
      }
    ],
    layout: {
      title: {
        text: currentData.title,
        font: { size: 18, family: 'Arial', color: '#2c3e50' }
      },
      xaxis: {
        title: {
          text: currentData.yLabel,
          font: { size: 14, family: 'Arial' }
        },
        showgrid: true,
        gridcolor: '#e0e0e0',
        zeroline: true,
        zerolinecolor: '#bdc3c7'
      },
      yaxis: {
        title: {
          text: 'Depth (m)',
          font: { size: 14, family: 'Arial' }
        },
        autorange: 'reversed',
        showgrid: true,
        gridcolor: '#e0e0e0',
        zeroline: true,
        zerolinecolor: '#bdc3c7'
      },
      margin: { l: 70, r: 40, t: 80, b: 70 },
      plot_bgcolor: '#f8f9fa',
      paper_bgcolor: '#ffffff',
      font: { family: 'Arial', size: 12, color: '#2c3e50' },
      hoverlabel: {
        bgcolor: '#fff',
        bordercolor: '#ddd',
        font: { family: 'Arial', size: 12 }
      }
    },
    config: {
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
      toImageButtonOptions: {
        format: 'png',
        filename: `${activeParameter}_profile`,
        height: 600,
        width: 800,
        scale: 2
      }
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
          {availableParameters.map((param) => (
            <button
              key={param}
              onClick={() => setActiveParameter(param)}
              className={`px-3 py-1 rounded-md text-sm font-medium ${
                activeParameter === param
                  ? 'bg-blue-500 text-white shadow-md'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {param.charAt(0).toUpperCase() + param.slice(1)}
            </button>
          ))}
        </div>
      </div>
      
      {/* Plot Area */}
      <div className="flex-1 bg-white p-4">
        <Plot
          data={plotConfig.data}
          layout={plotConfig.layout}
          config={plotConfig.config}
          style={{ width: '100%', height: '100%', minHeight: '400px' }}
          useResizeHandler={true}
        />
      </div>
      
      {/* Data Summary */}
      <div className="bg-gray-50 border-t border-gray-200 p-3">
        <div className="text-sm text-gray-600">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <span className="font-medium">Parameter: </span>
              {activeParameter}
            </div>
            <div>
              <span className="font-medium">Data Range: </span>
              {currentData.values && currentData.values.length > 0 
                ? `${Math.min(...currentData.values).toFixed(1)} - ${Math.max(...currentData.values).toFixed(1)} ${getParameterUnit(activeParameter)}`
                : 'N/A'
              }
            </div>
            <div>
              <span className="font-medium">Depth Range: </span>
              {currentData.depths && currentData.depths.length > 0 
                ? `${Math.min(...currentData.depths)} - ${Math.max(...currentData.depths)}m`
                : 'N/A'
              }
            </div>
            <div>
              <span className="font-medium">Data Points: </span>
              {currentData.values ? currentData.values.length : 0}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataPlots;