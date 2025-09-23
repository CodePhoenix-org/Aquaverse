import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { validatePlotData, has3DData, extract3DParameterData, extract3DOceanData, transformApiData } from '../../utils/dataTransformers';
import ThreeDScatterPlot from './ThreeDScatterPlot';
import OceanScatter3DPlot from './OceanScatter3DPlot';
import { BarChart3, Globe, Database, Loader } from 'lucide-react';

const DataPlots = ({ data }) => {
    const [activeTab, setActiveTab] = useState('2d'); // '2d' or '3d'
    const [active3DView, setActive3DView] = useState('parameter'); // 'parameter' or 'ocean'
    const [threeDData, setThreeDData] = useState(null);
    const [loading3D, setLoading3D] = useState(false);
    const [availableParameters, setAvailableParameters] = useState([]);
    const [activeParameter, setActiveParameter] = useState('temperature');
    const [plotReadyData, setPlotReadyData] = useState(null);
    const [error, setError] = useState(null);

    // Process incoming data
    useEffect(() => {
        console.log('=== DATAPLOTS COMPONENT ===');
        console.log('DataPlots received data:', data);
        
        setError(null);
        
        if (!data) {
            console.log('No data provided to DataPlots');
            setAvailableParameters(['temperature']);
            setPlotReadyData(null);
            
            // Check if we should auto-switch to 3D based on data type
            if (data && has3DData(data)) {
                setActiveTab('3d');
                setThreeDData(data);
            }
            return;
        }
        
        // Check for 3D data
        if (has3DData(data)) {
            console.log('3D data detected in incoming data');
            setThreeDData(data);
            setActiveTab('3d');
            
            // Determine which 3D view to show
            if (data.type === '3d_parameter_plot') {
                setActive3DView('parameter');
            } else if (data.type === '3d_ocean_plot') {
                setActive3DView('ocean');
            }
        }
        
        // Process 2D profile data
        if (!validatePlotData(data)) {
            console.log('Invalid plot data structure');
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


    // Fetch 3D data from backend
    const fetch3DData = async (type = 'parameter') => {
        setLoading3D(true);
        setError(null);
        try {
            const endpoint = `/api/3d/${type === 'parameter' ? 'parameter-plot' : 'ocean-plot'}?region=indian&limit=${type === 'parameter' ? 1000 : 500}`;
            const response = await fetch(endpoint);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const apiData = await response.json();
            console.log('API Response:', apiData);  // Debug the raw response
            const transformedData = transformApiData(apiData);
      
            if (!transformedData || !transformedData.threeDData) {
                throw new Error('Invalid or empty 3D data received');
            }

            setThreeDData(transformedData);
        } catch (err) {
            console.error('Error fetching 3D data:', err);
            setError(`Failed to fetch 3D data: ${err.message}. Check backend logs.`);
            // Remove mock data fallback in production
            if (type === 'parameter') {
                setThreeDData(transformApiData({ type: '3d_parameter_plot', normalData: [], anomalyData: [] }));
            } else {
                setThreeDData(transformApiData({ type: '3d_ocean_plot', oceanData: [] }));
            }
        } finally {
            setLoading3D(false);
        }
    };

    // Handle 3D view change
    const handle3DViewChange = (newView) => {
        setActive3DView(newView);
        fetch3DData(newView);  // Fetch data on view change
    };

    // Effect to fetch initial data and handle tab switches
    useEffect(() => {
        if (activeTab === '3d') {
            fetch3DData(active3DView);
            const interval = setInterval(() => fetch3DData(active3DView), 60000);  // Refresh every 60s
            return () => clearInterval(interval);
        }
    }, [activeTab, active3DView]);

    // (Removed duplicate handle3DViewChange definition)

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

    const getCurrent2DData = () => {
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

    const current2DData = getCurrent2DData();
    const isReal2DData = plotReadyData && plotReadyData[activeParameter];
    const hasValid2DData = current2DData.values && 
                          current2DData.depths && 
                          current2DData.values.length > 0 && 
                          current2DData.depths.length > 0;

    // 2D Plot configuration
    const plotConfig = {
        data: hasValid2DData ? [
            {
                x: current2DData.values,
                y: current2DData.depths,
                type: 'scatter',
                mode: 'lines+markers',
                marker: { 
                    color: current2DData.color, 
                    size: 6,
                    symbol: isReal2DData ? 'circle' : 'diamond-dot'
                },
                line: { 
                    color: current2DData.color, 
                    width: 2,
                    dash: isReal2DData ? 'solid' : 'dot'
                },
                name: current2DData.title,
                hovertemplate: 
                    '<b>Depth</b>: %{y}m<br>' +
                    `<b>${current2DData.yLabel}</b>: %{x} ${getParameterUnit(activeParameter)}<br>` +
                    '<extra></extra>'
            }
        ] : [],
        layout: {
            title: {
                text: hasValid2DData ? 
                    (isReal2DData ? `${current2DData.title} - Real ARGO Data` : `${current2DData.title} - Sample Data`) 
                    : 'No Data Available',
                font: { size: 16, family: 'Arial', color: '#2c3e50' }
            },
            xaxis: {
                title: {
                    text: hasValid2DData ? current2DData.yLabel : 'Values',
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
            {/* Header */}
            <div className="bg-gray-50 px-4 py-3 border-b border-gray-200">
                <div className="flex items-center justify-between">
                    <div>
                        <h3 className="text-lg font-medium text-gray-900">Ocean Data Visualizations</h3>
                        <p className="text-sm text-gray-600">
                            {activeTab === '2d' ? '2D Depth Profiles' : '3D Interactive Visualizations'}
                        </p>
                    </div>
                    
                    {/* Tab Navigation */}
                    <div className="flex gap-2 bg-white rounded-lg p-1 border">
                        <button
                            onClick={() => setActiveTab('2d')}
                            className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                                activeTab === '2d' 
                                    ? 'bg-blue-500 text-white shadow-md' 
                                    : 'text-gray-700 hover:bg-gray-100'
                            }`}
                        >
                            <BarChart3 className="w-4 h-4 inline mr-2" />
                            2D Profiles
                        </button>
                        <button
                            onClick={() => {
                                setActiveTab('3d');
                                if (!threeDData) fetch3DData('parameter');
                            }}
                            className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                                activeTab === '3d' 
                                    ? 'bg-green-500 text-white shadow-md' 
                                    : 'text-gray-700 hover:bg-gray-100'
                            }`}
                        >
                            <Globe className="w-4 h-4 inline mr-2" />
                            3D Visualizations
                        </button>
                    </div>
                </div>
            </div>
            
            {/* Content Area */}
            <div className="flex-1 p-4">
                {activeTab === '2d' ? (
                    /* 2D Profile Content */
                    <div className="h-full flex flex-col">
                        {/* Parameter Selection */}
                        <div className="bg-white border-b border-gray-200 p-3 mb-4">
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
                        
                        {/* 2D Plot */}
                        <div className="flex-1">
                            {hasValid2DData ? (
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
                    </div>
                ) : (
                    /* 3D Visualization Content */
                    <div className="h-full flex flex-col">
                        {/* 3D View Selection */}
                        <div className="bg-white border-b border-gray-200 p-3 mb-4">
                            <div className="flex gap-2">
                                <button
                                    onClick={() => handle3DViewChange('parameter')}
                                    className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                                        active3DView === 'parameter' 
                                            ? 'bg-purple-500 text-white shadow-md' 
                                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                                    }`}
                                >
                                    Parameter 3D
                                </button>
                                <button
                                    onClick={() => handle3DViewChange('ocean')}
                                    className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                                        active3DView === 'ocean' 
                                            ? 'bg-teal-500 text-white shadow-md' 
                                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                                    }`}
                                >
                                    Ocean 3D
                                </button>
                            </div>
                        </div>
                        
                        {/* 3D Plot Area */}
                        <div className="flex-1">
                            {loading3D ? (
                                <div className="flex flex-col items-center justify-center h-full">
                                    <Loader className="w-8 h-8 animate-spin text-blue-500 mb-4" />
                                    <p className="text-gray-600">Loading 3D visualization...</p>
                                </div>
                            ) : threeDData ? (
                                <div className="h-full">
                                    {active3DView === 'parameter' && threeDData.type === '3d_parameter_plot' && (
                                        <ThreeDScatterPlot 
                                            normalData={extract3DParameterData(threeDData).normalData}
                                            anomalyData={extract3DParameterData(threeDData).anomalyData}
                                        />
                                    )}
                                    {active3DView === 'ocean' && threeDData.type === '3d_ocean_plot' && (
                                        <OceanScatter3DPlot 
                                            data={extract3DOceanData(threeDData)}
                                        />
                                    )}
                                    
                                    {/* Metadata Display */}
                                    {threeDData.metadata && (
                                        <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                                            <div className="text-sm text-gray-600 grid grid-cols-2 md:grid-cols-4 gap-2">
                                                <div><span className="font-medium">Title:</span> {threeDData.metadata.title}</div>
                                                <div><span className="font-medium">Region:</span> {threeDData.metadata.region}</div>
                                                <div><span className="font-medium">Total Points:</span> {threeDData.metadata.total_points}</div>
                                                {threeDData.metadata.normal_count && (
                                                    <div><span className="font-medium">Normal Points:</span> {threeDData.metadata.normal_count}</div>
                                                )}
                                                {threeDData.metadata.anomaly_count && (
                                                    <div><span className="font-medium">Anomaly Points:</span> {threeDData.metadata.anomaly_count}</div>
                                                )}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div className="flex flex-col items-center justify-center h-full text-gray-500">
                                    <Database className="w-12 h-12 mb-4 opacity-50" />
                                    <p className="text-lg mb-2">No 3D data available</p>
                                    <p className="text-sm text-center">
                                        Click on the buttons above to load 3D visualizations.
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
            
            {/* Footer Summary */}
            <div className="bg-gray-50 border-t border-gray-200 p-3">
                <div className="text-sm text-gray-600">
                    {activeTab === '2d' ? (
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                            <div>
                                <span className="font-medium">Status: </span>
                                <span className={isReal2DData ? "text-green-600" : "text-yellow-600"}>
                                    {isReal2DData ? "✅ Real Data" : "📋 Sample Data"}
                                </span>
                            </div>
                            <div>
                                <span className="font-medium">Parameter: </span>
                                {activeParameter}
                            </div>
                            <div>
                                <span className="font-medium">Data Points: </span>
                                {hasValid2DData ? current2DData.values.length : 0}
                            </div>
                            {hasValid2DData && (
                                <>
                                    <div>
                                        <span className="font-medium">Value Range: </span>
                                        {Math.min(...current2DData.values).toFixed(2)} - {Math.max(...current2DData.values).toFixed(2)} {getParameterUnit(activeParameter)}
                                    </div>
                                    <div>
                                        <span className="font-medium">Depth Range: </span>
                                        {Math.min(...current2DData.depths)} - {Math.max(...current2DData.depths)}m
                                    </div>
                                </>
                            )}
                        </div>
                    ) : (
                        <div className="text-center">
                            <span className="font-medium">3D Visualization Mode: </span>
                            {active3DView === 'parameter' ? 'Parameter Space (T-S-O)' : 'Geographic Space (Lon-Lat-Depth)'}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default DataPlots;