// utils/dataTransformers.js

// Enhanced transformer for API data to frontend format
export const transformApiData = (apiData) => {
    if (!apiData) {
        console.log('transformApiData: No API data provided');
        return null;
    }
    
    console.log('=== TRANSFORMER INPUT ===');
    console.log('Raw API data:', apiData);
    console.log('API data type:', typeof apiData);
    console.log('API data keys:', Object.keys(apiData));

    try {
            // Handle 3D parameter plot format
            if (apiData.type === '3d_parameter_plot') {
                console.log('3D Parameter plot format detected');
                return {
                    type: apiData.type,
                    threeDData: {
                        parameterPlot: {
                            normalData: apiData.normalData || [],
                            anomalyData: apiData.anomalyData || []
                        }
                    },
                    metadata: apiData.metadata || {},
                    available_params: ['temperature', 'salinity', 'oxygen']
                };
            }

            // Handle 3D ocean plot format
            if (apiData.type === '3d_ocean_plot') {
                console.log('3D Ocean plot format detected');
                return {
                    type: apiData.type,
                    threeDData: {
                        oceanPlot: {
                            data: apiData.oceanData || []
                        }
                    },
                    metadata: apiData.metadata || {},
                    available_params: ['temperature', 'salinity', 'oxygen']
                };
            }
        
        if (apiData.all_data && typeof apiData.all_data === 'object') {
            console.log('Multiple profiles format detected');
            const result = {
                type: apiData.type || 'multiple_profiles',
                available_params: apiData.available_params || Object.keys(apiData.all_data),
                metadata: apiData.metadata || {},
                profiles: apiData.all_data
            };
            
            console.log('Transformed result:', result);
            return result;
        }
        
        // Handle single profile format
        if (apiData.data && apiData.type) {
            console.log('Single profile format detected');
            const paramName = apiData.type.replace('_profile', '');
            return {
                type: apiData.type,
                available_params: [paramName],
                metadata: apiData.metadata || {},
                profiles: { [paramName]: apiData.data }
            };
        }
        
        // Handle direct profiles format
        if (apiData.profiles) {
            console.log('Direct profiles format detected');
            return apiData;
        }
        
        console.log('Unknown data format');
        return null;
        
    } catch (error) {
        console.error('Error transforming API data:', error);
        return null;
    }
};

// Helper function to extract plot-ready data
export const extractPlotData = (transformedData, parameter) => {
    if (!transformedData || !transformedData.profiles) return null;
    
    const profileData = transformedData.profiles[parameter];
    if (!profileData) return null;
    
    // Ensure we have the required structure
    return {
        depths: Array.isArray(profileData.depths) ? profileData.depths : [],
        values: Array.isArray(profileData.values) ? profileData.values : [],
        title: profileData.title || `${parameter} Profile`,
        yLabel: profileData.yLabel || parameter,
        color: profileData.color || '#ff6b6b'
    };
};

// Helper function to validate plot data
export const validatePlotData = (data) => {
    if (!data || !data.profiles) {
        console.log('validatePlotData: No profiles data');
        return false;
    }
    
    const validParams = Object.keys(data.profiles).filter(param => {
        const profile = data.profiles[param];
        return profile && 
               Array.isArray(profile.depths) && 
               Array.isArray(profile.values) &&
               profile.depths.length > 0 &&
               profile.values.length > 0 &&
               profile.depths.length === profile.values.length;
    });
    
    console.log('validatePlotData: Valid parameters found:', validParams);
    return validParams.length > 0;
};
    // New function to check if data contains 3D visualizations
    export const has3DData = (data) => {
        return data && data.threeDData && (
            data.threeDData.parameterPlot || 
            data.threeDData.oceanPlot
        );
    };

    // Extract 3D parameter data
    export const extract3DParameterData = (data) => {
        if (!has3DData(data)) return { normalData: [], anomalyData: [] };
        return data.threeDData.parameterPlot || { normalData: [], anomalyData: [] };
    };

    // Extract 3D ocean data
    export const extract3DOceanData = (data) => {
        if (!has3DData(data)) return [];
        return data.threeDData.oceanPlot?.data || [];
    };