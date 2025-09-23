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
        // Handle multiple profiles format
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