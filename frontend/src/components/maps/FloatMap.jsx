import React, { useEffect, useRef, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

const activeIcon = new L.Icon({
  iconUrl: 'https://cdn-icons-png.flaticon.com/512/190/190411.png', // green marker
  iconSize: [30, 30],
  iconAnchor: [15, 30],
  popupAnchor: [0, -30],
});

const inactiveIcon = new L.Icon({
  iconUrl: 'https://cdn-icons-png.flaticon.com/512/190/190406.png', // red marker
  iconSize: [30, 30],
  iconAnchor: [15, 30],
  popupAnchor: [0, -30],
});

const FloatMap = ({ data }) => {
  const [mapData, setMapData] = useState(null);
  const mapRef = useRef(null);

  const defaultCenter = [-20, 80];
  const defaultZoom = 4;


  useEffect(() => {
    // Accepts both old and new backend formats
    if (data && typeof data === 'object') {
      // New backend: { float_locations: [...] }
      if (data.float_locations) {
        setMapData({ floats: data.float_locations });
      } else if (data.floats || data.trajectories) {
        setMapData(data);
      } else {
        setMapData(null);
      }
    } else {
      setMapData(null);
    }
  }, [data]);

  useEffect(() => {
    const timer = setTimeout(() => {
      if (mapRef.current) mapRef.current.invalidateSize(false);
    }, 150);
    return () => clearTimeout(timer);
  });

  const sampleFloats = [
    { id: 'ARGO_001', lat: -10.5, lng: 75.2, status: 'active', lastUpdate: '2023-03-15' },
    { id: 'ARGO_002', lat: -15.3, lng: 82.1, status: 'active', lastUpdate: '2023-03-14' },
    { id: 'ARGO_003', lat: -8.7, lng: 70.8, status: 'inactive', lastUpdate: '2023-02-28' },
    { id: 'ARGO_004', lat: -25.1, lng: 88.5, status: 'active', lastUpdate: '2023-03-16' },
  ];

  const sampleTrajectories = [
    [[-10.5, 75.2], [-11.2, 75.8], [-12.1, 76.3], [-13.0, 76.9]],
    [[-15.3, 82.1], [-16.0, 82.7], [-16.8, 83.2], [-17.5, 83.8]],
  ];

  const getIcon = (status) => (status === 'active' ? activeIcon : inactiveIcon);

  return (
    <div className="h-full">
      <div className="bg-gray-50 px-4 py-2 border-b border-gray-200">
        <h3 className="text-lg font-medium text-gray-900">ARGO Float Locations</h3>
        <p className="text-sm text-gray-600">
          Interactive map showing float trajectories and current positions
        </p>
      </div>

      <div className="h-full" style={{ minHeight: '400px', height: '400px' }}>
        <MapContainer
          center={defaultCenter}
          zoom={defaultZoom}
          whenCreated={(map) => {
            mapRef.current = map;
            setTimeout(() => map.invalidateSize(false), 50);
          }}
          style={{ height: '100%', width: '100%' }}
        >
          {/* MapTiler TileLayer */}
          <TileLayer
            attribution='&copy; <a href="https://www.maptiler.com/">MapTiler</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://api.maptiler.com/maps/streets/{z}/{x}/{y}.png?key=guH98SuP1qEfBsrk2TPM"
          />

          {/* Sample floats */}
          {!mapData &&
            sampleFloats.map((float) => (
              <Marker key={float.id} position={[float.lat, float.lng]} icon={getIcon(float.status)}>
                <Popup>
                  <div className="text-sm">
                    <strong>Float ID:</strong> {float.id}
                    <br />
                    <strong>Status:</strong> {float.status}
                    <br />
                    <strong>Position:</strong> {float.lat.toFixed(2)}°, {float.lng.toFixed(2)}°
                    <br />
                    <strong>Last Update:</strong> {float.lastUpdate}
                  </div>
                </Popup>
              </Marker>
            ))}

          {/* Sample trajectories */}
          {!mapData &&
            sampleTrajectories.map((trajectory, index) => (
              <Polyline
                key={`sample-trajectory-${index}`}
                positions={trajectory}
                color="blue"
                weight={2}
                opacity={0.6}
              />
            ))}

          {/* Actual trajectories */}
          {mapData &&
            mapData.trajectories &&
            mapData.trajectories.map((trajectory, index) => (
              <Polyline
                key={`real-trajectory-${index}`}
                positions={trajectory.points || trajectory}
                color="blue"
                weight={2}
                opacity={0.8}
              />
            ))}

          {/* Actual floats */}
          {mapData &&
            mapData.floats &&
            mapData.floats.map((float) => (
              <Marker
                key={float.id}
                position={[float.latitude, float.longitude]}
                icon={getIcon(float.status)}
              >
                <Popup>
                  <div className="text-sm">
                    <strong>Float ID:</strong> {float.id}
                    <br />
                    <strong>Status:</strong> {float.status}
                    <br />
                    <strong>Position:</strong> {float.latitude?.toFixed(2)}°, {float.longitude?.toFixed(2)}°
                    <br />
                    <strong>Last Update:</strong> {float.lastUpdate}
                  </div>
                </Popup>
              </Marker>
            ))}
        </MapContainer>
      </div>

      {/* Legend */}
      <div className="bg-white border-t border-gray-200 p-2">
        <div className="flex items-center space-x-4 text-xs">
          <div className="flex items-center space-x-1">
            <img src="https://cdn-icons-png.flaticon.com/512/190/190411.png" className="w-4 h-4" />
            <span>Active Floats</span>
          </div>
          <div className="flex items-center space-x-1">
            <img src="https://cdn-icons-png.flaticon.com/512/190/190406.png" className="w-4 h-4" />
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
