import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';

const ProfileComparison = ({ data }) => {
  const [selectedProfiles, setSelectedProfiles] = useState(['region1_temp', 'region2_temp']);
  const [activeParameter, setActiveParameter] = useState('temperature');
  const [comparisonMode, setComparisonMode] = useState('region'); // 'region' or 'time'

  // Sample comparison data for demonstration
  const sampleComparisonData = {
    temperature: {
      region1_temp: {
        depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
        values: [28.5, 28.2, 27.8, 25.1, 22.3, 18.7, 12.4, 8.2, 5.1, 3.8],
        name: 'Arabian Sea',
        color: '#ff6b6b'
      },
      region2_temp: {
        depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
        values: [29.1, 28.8, 28.3, 26.2, 23.5, 19.8, 13.1, 8.8, 5.5, 4.1],
        name: 'Bay of Bengal',
        color: '#4ecdc4'
      },
      time1_temp: {
        depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
        values: [28.5, 28.2, 27.8, 25.1, 22.3, 18.7, 12.4, 8.2, 5.1, 3.8],
        name: 'March 2023',
        color: '#45b7d1'
      },
      time2_temp: {
        depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
        values: [27.8, 27.5, 27.1, 24.3, 21.6, 17.9, 11.8, 7.9, 4.8, 3.5],
        name: 'September 2023',
        color: '#f39c12'
      }
    },
    salinity: {
      region1_sal: {
        depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
        values: [34.2, 34.3, 34.4, 34.6, 34.8, 35.1, 35.3, 35.0, 34.8, 34.7],
        name: 'Arabian Sea',
        color: '#ff6b6b'
      },
      region2_sal: {
        depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
        values: [33.8, 33.9, 34.0, 34.2, 34.5, 34.9, 35.1, 34.8, 34.6, 34.5],
        name: 'Bay of Bengal',
        color: '#4ecdc4'
      }
    },
    oxygen: {
      region1_oxy: {
        depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
        values: [220, 218, 215, 200, 180, 150, 120, 180, 200, 220],
        name: 'Arabian Sea',
        color: '#ff6b6b'
      },
      region2_oxy: {
        depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
        values: [215, 212, 208, 190, 170, 140, 110, 170, 190, 210],
        name: 'Bay of Bengal',
        color: '#4ecdc4'
      }
    }
  };

  const getParameterUnit = (param) => {
    const units = {
      temperature: '°C',
      salinity: 'PSU',
      oxygen: 'μmol/kg'
    };
    return units[param] || '';
  };

  // Schema validation functions
  const isValidComparisonProfile = (profile) => {
    return profile && 
           Array.isArray(profile.depths) && 
           Array.isArray(profile.values) && 
           typeof profile.name === 'string' && 
           typeof profile.color === 'string' &&
           profile.depths.length === profile.values.length;
  };

  const isValidComparisonData = (dataStructure) => {
    if (!dataStructure || typeof dataStructure !== 'object') return false;
    
    // Check if it has parameter-level organization
    for (const param of ['temperature', 'salinity', 'oxygen']) {
      if (dataStructure[param]) {
        const paramData = dataStructure[param];
        
        // Check if it has comparison profiles (region*/time* keys)
        const profileKeys = Object.keys(paramData);
        const hasComparisonKeys = profileKeys.some(key => 
          key.includes('region') || key.includes('time')
        );
        
        if (hasComparisonKeys) {
          // Validate that profiles have the right structure
          const validProfiles = profileKeys.filter(key => 
            isValidComparisonProfile(paramData[key])
          );
          
          if (validProfiles.length >= 2) {
            return true; // At least one parameter has valid comparison data
          }
        }
      }
    }
    
    return false;
  };

  const isSingleProfileData = (dataStructure) => {
    if (!dataStructure || typeof dataStructure !== 'object') return false;
    
    // Check if it's single-profile format (DataPlots format)
    for (const param of ['temperature', 'salinity', 'oxygen']) {
      if (dataStructure[param]) {
        const paramData = dataStructure[param];
        // Single profile has depths, values, title, yLabel directly
        if (Array.isArray(paramData.depths) && 
            Array.isArray(paramData.values) && 
            typeof paramData.title === 'string') {
          return true;
        }
      }
    }
    
    return false;
  };

  // Function to get comparison data with proper validation
  const getComparisonData = () => {
    // First check for explicit comparison data
    if (data && data.profileComparisons && isValidComparisonData(data.profileComparisons)) {
      console.log('ProfileComparison: Using data.profileComparisons');
      return data.profileComparisons;
    }
    
    // Check if data.profiles is structured for comparisons
    if (data && data.profiles && isValidComparisonData(data.profiles)) {
      console.log('ProfileComparison: Using data.profiles (comparison format)');
      return data.profiles;
    }
    
    // Check if it's single profile data that we could potentially use
    if (data && data.profiles && isSingleProfileData(data.profiles)) {
      console.log('ProfileComparison: Received single-profile data, using sample comparison data');
    } else if (data && data.profiles) {
      console.log('ProfileComparison: Received data.profiles but structure is not recognized, using sample data');
    }
    
    // Fall back to sample data
    return sampleComparisonData;
  };

  const getAvailableProfiles = () => {
    const comparisonData = getComparisonData();
    const profiles = comparisonData[activeParameter] || {};
    
    // Safely get profile keys
    const profileKeys = Object.keys(profiles);
    
    if (comparisonMode === 'region') {
      return profileKeys.filter(key => key.includes('region'));
    } else {
      return profileKeys.filter(key => key.includes('time'));
    }
  };

  const getPlotData = () => {
    const comparisonData = getComparisonData();
    const profiles = comparisonData[activeParameter] || {};
    
    return selectedProfiles
      .filter(profileId => {
        const profile = profiles[profileId];
        return profile && isValidComparisonProfile(profile);
      })
      .map(profileId => {
        const profile = profiles[profileId];
        return {
          x: profile.values,
          y: profile.depths,
          type: 'scatter',
          mode: 'lines+markers',
          marker: { color: profile.color, size: 6 },
          line: { color: profile.color, width: 2 },
          name: profile.name
        };
      });
  };

  const plotConfig = {
    data: getPlotData(),
    layout: {
      title: {
        text: `${activeParameter.charAt(0).toUpperCase() + activeParameter.slice(1)} Profile Comparison`,
        font: { size: 16 }
      },
      xaxis: {
        title: `${activeParameter.charAt(0).toUpperCase() + activeParameter.slice(1)} (${getParameterUnit(activeParameter)})`,
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
      font: { size: 12 },
      showlegend: true,
      legend: {
        x: 1,
        xanchor: 'left',
        y: 1
      }
    },
    config: {
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
    }
  };

  useEffect(() => {
    const availableProfiles = getAvailableProfiles();
    if (availableProfiles.length >= 2) {
      setSelectedProfiles(availableProfiles.slice(0, 2));
    }
  }, [activeParameter, comparisonMode]);

  return (
    <div className="h-full flex flex-col">
      <div className="bg-gray-50 px-4 py-2 border-b border-gray-200">
        <h3 className="text-lg font-medium text-gray-900">Profile Comparison</h3>
        <p className="text-sm text-gray-600">
          Compare profiles across different regions or time periods
        </p>
      </div>
      
      {/* Controls */}
      <div className="bg-white border-b border-gray-200 p-3 space-y-3">
        {/* Parameter Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Parameter</label>
          <div className="flex space-x-2">
            {Object.keys(getComparisonData()).map((param) => (
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

        {/* Comparison Mode */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Comparison Type</label>
          <div className="flex space-x-2">
            <button
              onClick={() => setComparisonMode('region')}
              className={`px-3 py-1 rounded-md text-sm font-medium ${
                comparisonMode === 'region'
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              By Region
            </button>
            <button
              onClick={() => setComparisonMode('time')}
              className={`px-3 py-1 rounded-md text-sm font-medium ${
                comparisonMode === 'time'
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              By Time Period
            </button>
          </div>
        </div>

        {/* Profile Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Select Profiles to Compare ({comparisonMode === 'region' ? 'Regions' : 'Time Periods'})
          </label>
          <div className="flex flex-wrap gap-2">
            {getAvailableProfiles().map((profileId) => {
              const comparisonData = getComparisonData();
              const profile = comparisonData[activeParameter]?.[profileId];
              
              if (!profile || !isValidComparisonProfile(profile)) {
                return null; // Skip invalid profiles
              }
              
              return (
                <label key={profileId} className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={selectedProfiles.includes(profileId)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedProfiles([...selectedProfiles, profileId]);
                      } else {
                        setSelectedProfiles(selectedProfiles.filter(id => id !== profileId));
                      }
                    }}
                    className="rounded"
                  />
                  <span 
                    className="w-3 h-3 rounded-full" 
                    style={{ backgroundColor: profile.color }}
                  ></span>
                  <span className="text-sm text-gray-700">{profile.name}</span>
                </label>
              );
            })}
          </div>
        </div>
      </div>
      
      {/* Plot Area */}
      <div className="flex-1 bg-white">
        {(() => {
          const availableProfiles = getAvailableProfiles();
          const plotData = getPlotData();
          
          if (availableProfiles.length === 0) {
            return (
              <div className="flex items-center justify-center h-full text-gray-500">
                <div className="text-center">
                  <p className="text-lg mb-2">No comparison profiles available</p>
                  <p className="text-sm mb-2">
                    Switch to "{comparisonMode === 'region' ? 'time' : 'region'}" mode or provide data with comparison profiles
                  </p>
                  <p className="text-xs text-gray-400">
                    {data && data.profiles && isSingleProfileData(data.profiles) 
                      ? 'Single-profile data detected - use Data Plots view instead'
                      : 'Using sample comparison data for demonstration'
                    }
                  </p>
                </div>
              </div>
            );
          }
          
          if (selectedProfiles.length === 0) {
            return (
              <div className="flex items-center justify-center h-full text-gray-500">
                <div className="text-center">
                  <p className="text-lg mb-2">No profiles selected for comparison</p>
                  <p className="text-sm">Please select at least one profile to compare</p>
                </div>
              </div>
            );
          }
          
          if (plotData.length === 0) {
            return (
              <div className="flex items-center justify-center h-full text-gray-500">
                <div className="text-center">
                  <p className="text-lg mb-2">Selected profiles are invalid</p>
                  <p className="text-sm">The selected profiles don't have valid comparison data</p>
                </div>
              </div>
            );
          }
          
          return (
            <Plot
              data={plotData}
              layout={plotConfig.layout}
              config={plotConfig.config}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler={true}
            />
          );
        })()}
      </div>
      
      {/* Comparison Info */}
      <div className="bg-gray-50 border-t border-gray-200 p-2">
        <div className="text-xs text-gray-600">
          <span className="font-medium">Profiles being compared:</span> 
          {selectedProfiles.length > 0 ? (
            selectedProfiles.map(profileId => {
              const comparisonData = getComparisonData();
              const profile = comparisonData[activeParameter]?.[profileId];
              return profile && isValidComparisonProfile(profile) ? profile.name : profileId;
            }).join(', ')
          ) : (
            'None selected'
          )}
        </div>
      </div>
    </div>
  );
};

export default ProfileComparison;