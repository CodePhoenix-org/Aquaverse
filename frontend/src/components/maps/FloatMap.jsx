import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, CircleMarker, Polyline } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix for default markers in react-leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const FloatMap = ({ data }) => {
  const [mapData, setMapData] = useState(null);
  
  // Default center on Indian Ocean as specified in requirements
  const defaultCenter = [-20, 80];
  const defaultZoom = 4;

  useEffect(() => {
    if (data && (data.floats || data.trajectories)) {
      setMapData(data);
    }
  }, [data]);

  // Generate sample data for demo purposes
  const sampleFloats = [
    { id: 'ARGO_001', lat: -10.5, lng: 75.2, status: 'active', lastUpdate: '2023-03-15' },
    { id: 'ARGO_002', lat: -15.3, lng: 82.1, status: 'active', lastUpdate: '2023-03-14' },
    { id: 'ARGO_003', lat: -8.7, lng: 70.8, status: 'inactive', lastUpdate: '2023-02-28' },
    { id: 'ARGO_004', lat: -25.1, lng: 88.5, status: 'active', lastUpdate: '2023-03-16' },
  ];

  const sampleTrajectories = [
    [
      [-10.5, 75.2], [-11.2, 75.8], [-12.1, 76.3], [-13.0, 76.9]
    ],
    [
      [-15.3, 82.1], [-16.0, 82.7], [-16.8, 83.2], [-17.5, 83.8]
    ]
  ];

  const getMarkerColor = (status) => {
    return status === 'active' ? 'green' : 'red';
  };

  return (
    <div className="h-full">
      <div className="bg-gray-50 px-4 py-2 border-b border-gray-200">
        <h3 className="text-lg font-medium text-gray-900">ARGO Float Locations</h3>
        <p className="text-sm text-gray-600">
          Interactive map showing float trajectories and current positions
        </p>
      </div>
      
      <div className="h-full" style={{ minHeight: '400px' }}>
        <MapContainer
          center={defaultCenter}
          zoom={defaultZoom}
          style={{ height: '100%', width: '100%' }}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          
          {/* Render sample float positions only if no real data */}
          {!mapData && sampleFloats.map((float) => (
            <CircleMarker
              key={float.id}
              center={[float.lat, float.lng]}
              radius={8}
              fillColor={getMarkerColor(float.status)}
              color={getMarkerColor(float.status)}
              weight={2}
              opacity={0.8}
              fillOpacity={0.6}
            >
              <Popup>
                <div className="text-sm">
                  <strong>Float ID:</strong> {float.id}<br />
                  <strong>Status:</strong> {float.status}<br />
                  <strong>Position:</strong> {float.lat.toFixed(2)}°, {float.lng.toFixed(2)}°<br />
                  <strong>Last Update:</strong> {float.lastUpdate}
                </div>
              </Popup>
            </CircleMarker>
          ))}
          
          {/* Render sample trajectories only if no real data */}
          {!mapData && sampleTrajectories.map((trajectory, index) => (
            <Polyline
              key={`sample-trajectory-${index}`}
              positions={trajectory}
              color="blue"
              weight={2}
              opacity={0.6}
            />
          ))}
          
          {/* Render actual trajectories if available */}
          {mapData && mapData.trajectories && mapData.trajectories.map((trajectory, index) => (
            <Polyline
              key={`real-trajectory-${index}`}
              positions={trajectory.points || trajectory}
              color="blue"
              weight={2}
              opacity={0.8}
            />
          ))}
          
          {/* Render actual data if available */}
          {mapData && mapData.floats && mapData.floats.map((float) => (
            <CircleMarker
              key={float.id}
              center={[float.latitude, float.longitude]}
              radius={8}
              fillColor={getMarkerColor(float.status)}
              color={getMarkerColor(float.status)}
              weight={2}
              opacity={0.8}
              fillOpacity={0.6}
            >
              <Popup>
                <div className="text-sm">
                  <strong>Float ID:</strong> {float.id}<br />
                  <strong>Status:</strong> {float.status}<br />
                  <strong>Position:</strong> {float.latitude?.toFixed(2)}°, {float.longitude?.toFixed(2)}°<br />
                  <strong>Last Update:</strong> {float.lastUpdate}
                </div>
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>
      
      {/* Legend */}
      <div className="bg-white border-t border-gray-200 p-2">
        <div className="flex items-center space-x-4 text-xs">
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span>Active Floats</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <span>Inactive Floats</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-8 h-0.5 bg-blue-500"></div>
            <span>Float Trajectories</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FloatMap;