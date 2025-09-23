import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { validatePlotData } from '../../utils/dataTransformers';

const DataPlots = ({ data }) => {
    const [activeParameter, setActiveParameter] = useState('temperature');
    const [availableParameters, setAvailableParameters] = useState([]);
    const [plotReadyData, setPlotReadyData] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        console.log('=== DATAPLOTS COMPONENT ===');
        console.log('DataPlots received data:', data);
        
        setError(null);
        
        if (!data) {
            console.log('No data provided to DataPlots');
            setAvailableParameters(['temperature']);
            setPlotReadyData(null);
            return;
        }
        
        // Validate the data structure
        if (!validatePlotData(data)) {
            console.log('Invalid plot data structure');
            setError('Invalid data structure received');
            setAvailableParameters(['temperature']);
            setPlotReadyData(null);
            return;
        }
        
        try {
            const params = Object.keys(data.profiles).filter(param => {
                const profile = data.profiles[param];
                return profile && 
                       Array.isArray(profile.depths) && 
                       Array.isArray(profile.values) &&
                       profile.depths.length > 0 &&
                       profile.values.length > 0;
            });
            
            console.log('Valid parameters found:', params);
            
            if (params.length === 0) {
                setError('No valid parameters found in data');
                setAvailableParameters(['temperature']);
                setPlotReadyData(null);
                return;
            }
            
            setAvailableParameters(params);
            
            // Set initial parameter
            if (params.includes('salinity')) {
                setActiveParameter('salinity');
            } else if (params.includes('temperature')) {
                setActiveParameter('temperature');
            } else {
                setActiveParameter(params[0]);
            }
            
            setPlotReadyData(data.profiles);
            
        } catch (err) {
            console.error('Error processing data:', err);
            setError('Error processing data: ' + err.message);
            setAvailableParameters(['temperature']);
            setPlotReadyData(null);
        }
    }, [data]);

    // Sample data for fallback
    const sampleData = {
        temperature: {
            depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
            values: [28.5, 28.2, 27.8, 25.1, 22.3, 18.7, 12.4, 8.2, 5.1, 3.8],
            title: 'Temperature Profile (Sample)',
            yLabel: 'Temperature (°C)',
            color: '#ff6b6b'
        },
        salinity: {
            depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
            values: [34.2, 34.3, 34.4, 34.6, 34.8, 35.1, 35.3, 35.0, 34.8, 34.7],
            title: 'Salinity Profile (Sample)',
            yLabel: 'Salinity (PSU)',
            color: '#4ecdc4'
        }
    };

    const getCurrentData = () => {
        if (plotReadyData && plotReadyData[activeParameter]) {
            return plotReadyData[activeParameter];
        }
        return sampleData[activeParameter] || sampleData.temperature;
    };

    const getParameterUnit = (param) => {
        const units = {
            temperature: '°C',
            salinity: 'PSU',
            oxygen: 'μmol/kg',
            chlorophyll: 'mg/m³'
        };
        return units[param] || '';
    };

    const currentData = getCurrentData();
    const isRealData = plotReadyData && plotReadyData[activeParameter];
    
    // Check if we have valid data points
    const hasValidData = currentData.values && 
                        currentData.depths && 
                        currentData.values.length > 0 && 
                        currentData.depths.length > 0;

    const plotConfig = {
        data: hasValidData ? [
            {
                x: currentData.values,
                y: currentData.depths,
                type: 'scatter',
                mode: 'lines+markers',
                marker: { 
                    color: currentData.color, 
                    size: 6,
                    symbol: isRealData ? 'circle' : 'diamond-dot'
                },
                line: { 
                    color: currentData.color, 
                    width: 2,
                    dash: isRealData ? 'solid' : 'dot'
                },
                name: currentData.title,
                hovertemplate: 
                    '<b>Depth</b>: %{y}m<br>' +
                    `<b>${currentData.yLabel}</b>: %{x} ${getParameterUnit(activeParameter)}<br>` +
                    '<extra></extra>'
            }
        ] : [],
        layout: {
            title: {
                text: hasValidData ? 
                    (isRealData ? `${currentData.title} - Real ARGO Data` : `${currentData.title} - Sample Data`) 
                    : 'No Data Available',
                font: { size: 16, family: 'Arial', color: '#2c3e50' }
            },
            xaxis: {
                title: {
                    text: hasValidData ? currentData.yLabel : 'Values',
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
            margin: { l: 70, r: 40, t: 60, b: 60 },
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

    if (error) {
        return (
            <div className="h-full flex flex-col items-center justify-center bg-white rounded-lg border border-gray-200 p-8">
                <div className="text-center">
                    <div className="text-red-500 text-6xl mb-4">⚠️</div>
                    <h3 className="text-lg font-medium text-gray-900 mb-2">Data Error</h3>
                    <p className="text-gray-600 mb-4">{error}</p>
                    <p className="text-sm text-gray-500">
                        Please check the browser console for detailed error information.
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col bg-white rounded-lg border border-gray-200">
            <div className="bg-gray-50 px-4 py-3 border-b border-gray-200">
                <h3 className="text-lg font-medium text-gray-900">Ocean Profile Data</h3>
                <p className="text-sm text-gray-600">
                    {isRealData ? 'Real ARGO float data' : 'Sample data demonstration'}
                </p>
            </div>
            
            {/* Parameter Selection */}
            <div className="bg-white border-b border-gray-200 p-3">
                <div className="flex flex-wrap gap-2">
                    {availableParameters.map((param) => (
                        <button
                            key={param}
                            onClick={() => setActiveParameter(param)}
                            className={`px-3 py-2 rounded-md text-sm font-medium transition-all ${
                                activeParameter === param
                                    ? 'bg-blue-500 text-white shadow-md'
                                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                            }`}
                        >
                            {param.charAt(0).toUpperCase() + param.slice(1)}
                            {plotReadyData && plotReadyData[param] && (
                                <span className="ml-1 text-xs opacity-75">
                                    ({plotReadyData[param].values.length} pts)
                                </span>
                            )}
                        </button>
                    ))}
                </div>
            </div>
            
            {/* Plot Area */}
            <div className="flex-1 p-4">
                {hasValidData ? (
                    <Plot
                        data={plotConfig.data}
                        layout={plotConfig.layout}
                        config={plotConfig.config}
                        style={{ width: '100%', height: '100%', minHeight: '400px' }}
                        useResizeHandler={true}
                    />
                ) : (
                    <div className="flex flex-col items-center justify-center h-full text-gray-500">
                        <div className="text-6xl mb-4">📊</div>
                        <p className="text-lg mb-2">No data available for plotting</p>
                        <p className="text-sm text-center">
                            The data received doesn't contain valid depth-value pairs for {activeParameter}.
                        </p>
                    </div>
                )}
            </div>
            
            {/* Data Summary */}
            <div className="bg-gray-50 border-t border-gray-200 p-3">
                <div className="text-sm text-gray-600">
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                        <div>
                            <span className="font-medium">Status: </span>
                            <span className={isRealData ? "text-green-600" : "text-yellow-600"}>
                                {isRealData ? "✅ Real Data" : "📋 Sample Data"}
                            </span>
                        </div>
                        <div>
                            <span className="font-medium">Parameter: </span>
                            {activeParameter}
                        </div>
                        <div>
                            <span className="font-medium">Data Points: </span>
                            {hasValidData ? currentData.values.length : 0}
                        </div>
                        {hasValidData && (
                            <>
                                <div>
                                    <span className="font-medium">Value Range: </span>
                                    {Math.min(...currentData.values).toFixed(2)} - {Math.max(...currentData.values).toFixed(2)} {getParameterUnit(activeParameter)}
                                </div>
                                <div>
                                    <span className="font-medium">Depth Range: </span>
                                    {Math.min(...currentData.depths)} - {Math.max(...currentData.depths)}m
                                </div>
                            </>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default DataPlots;