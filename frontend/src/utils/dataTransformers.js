// Unified transformer for API data to frontend format
export const transformApiData = (apiData) => {
    if (!apiData) return null;
    // Handle different data types from the API
    switch(apiData.type) {
        case 'temperature_profile':
            return {
                profiles: {
                    temperature: {
                        depths: apiData.data.depths,
                        values: apiData.data.values,
                        title: 'Temperature Profile',
                        yLabel: 'Temperature (°C)',
                        color: '#ff6b6b'
                    }
                }
            };
        case 'salinity_profile':
            return {
                profiles: {
                    salinity: {
                        depths: apiData.data.depths,
                        values: apiData.data.values,
                        title: 'Salinity Profile',
                        yLabel: 'Salinity (PSU)',
                        color: '#4ecdc4'
                    }
                }
            };
        default:
            return null;
    }
};
// utils/dataTransformers.js

export const transformToPlotFormat = (apiData) => {
    if (!apiData) return null;
    
    switch(apiData.type) {
        case 'temperature_profile':
            return {
                depths: apiData.data.depths,
                values: apiData.data.values,
                title: `Temperature Profile - ${apiData.metadata.location}`,
                yLabel: 'Temperature (°C)',
                color: '#ff6b6b'
            };
        
        case 'salinity_profile':
            return {
                depths: apiData.data.depths,
                values: apiData.data.values,
                title: `Salinity Profile - ${apiData.metadata.location}`,
                yLabel: 'Salinity (PSU)',
                color: '#4ecdc4'
            };
        
        default:
            return null;
    }
};

export const transformToMapData = (apiData) => {
    // Transform to map-friendly format
    if (apiData.type === 'float_locations') {
        return {
            floats: apiData.data.map(float => ({
                id: float.id,
                latitude: float.lat,
                longitude: float.lng,
                status: float.status,
                lastUpdate: float.date
            })),
            trajectories: apiData.trajectories || []
        };
    }
    return null;
};