import React, { useEffect, useRef, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup, Polyline } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

// Marker icons
const activeIcon = new L.Icon({
  iconUrl: "https://cdn-icons-png.flaticon.com/512/190/190411.png",
  iconSize: [30, 30],
  iconAnchor: [15, 30],
  popupAnchor: [0, -30],
});

const inactiveIcon = new L.Icon({
  iconUrl: "https://cdn-icons-png.flaticon.com/512/190/190406.png",
  iconSize: [30, 30],
  iconAnchor: [15, 30],
  popupAnchor: [0, -30],
});

const FloatMap = ({ data }) => {
  const [mapData, setMapData] = useState(null);
  const [floatPositions, setFloatPositions] = useState([]);
  const mapRef = useRef(null);

  const defaultCenter = [-20, 80];
  const defaultZoom = 4;

  // Sample mock floats
  const sampleFloats = [
    { id: "ARGO_001", lat: -10.5, lng: 75.2, status: "active", lastUpdate: "2023-03-15", depth: 200 },
    { id: "ARGO_002", lat: -15.3, lng: 82.1, status: "active", lastUpdate: "2023-03-14", depth: 150 },
    { id: "ARGO_003", lat: -8.7, lng: 70.8, status: "inactive", lastUpdate: "2023-02-28", depth: 300 },
    { id: "ARGO_004", lat: -25.1, lng: 88.5, status: "active", lastUpdate: "2023-03-16", depth: 100 },
    { id: "ARGO_005", lat: -18.2, lng: 78.4, status: "inactive", lastUpdate: "2023-03-10", depth: 250 },
    { id: "ARGO_006", lat: -12.9, lng: 80.7, status: "active", lastUpdate: "2023-03-12", depth: 180 },
  ];

  // Sample trajectories
  const sampleTrajectories = [
    [[-10.5, 75.2], [-11.2, 75.8], [-12.1, 76.3], [-13.0, 76.9]],
    [[-15.3, 82.1], [-16.0, 82.7], [-16.8, 83.2], [-17.5, 83.8]],
    [[-8.7, 70.8], [-9.3, 71.1], [-10.0, 71.5]],
    [[-25.1, 88.5], [-24.5, 88.0], [-23.9, 87.5]],
    [[-18.2, 78.4], [-18.7, 79.0], [-19.3, 79.5]],
    [[-12.9, 80.7], [-13.5, 81.1], [-14.0, 81.6]],
  ];

  // Determine icon
  const getIcon = (status) => (status === "active" ? activeIcon : inactiveIcon);

  // Handle backend data
  useEffect(() => {
    if (data && typeof data === "object") {
      if (data.float_locations) {
        setMapData({ floats: data.float_locations, trajectories: data.float_trajectories || [] });
      } else if (data.floats || data.trajectories) {
        setMapData(data);
      } else {
        setMapData(null);
      }
    } else {
      setMapData(null);
    }
  }, [data]);

  // Initialize float positions for animation
  useEffect(() => {
    if (!mapData) setFloatPositions(sampleFloats);
    else setFloatPositions(mapData.floats || []);
  }, [mapData]);

  // Animate floats randomly (mock)
  useEffect(() => {
    const interval = setInterval(() => {
      setFloatPositions((prev) =>
        prev.map((f) => ({
          ...f,
          lat: f.lat + (Math.random() - 0.5) * 0.05, // small random drift
          lng: f.lng + (Math.random() - 0.5) * 0.05,
        }))
      );
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  // Fix map resize
  useEffect(() => {
    const timer = setTimeout(() => {
      if (mapRef.current) mapRef.current.invalidateSize(false);
    }, 150);
    return () => clearTimeout(timer);
  });

  return (
    <div className="h-full">
      {/* Header */}
      <div className="bg-gray-50 px-4 py-2 border-b border-gray-200">
        <h3 className="text-lg font-medium text-gray-900">ARGO Float Locations</h3>
        <p className="text-sm text-gray-600">Interactive map showing float trajectories and current positions</p>
      </div>

      {/* Map */}
      <div className="h-full" style={{ minHeight: "400px", height: "400px" }}>
        <MapContainer
          center={defaultCenter}
          zoom={defaultZoom}
          whenCreated={(map) => {
            mapRef.current = map;
            setTimeout(() => map.invalidateSize(false), 50);
          }}
          style={{ height: "100%", width: "100%" }}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.maptiler.com/">MapTiler</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://api.maptiler.com/maps/streets/{z}/{x}/{y}.png?key=guH98SuP1qEfBsrk2TPM"
          />

          {/* Trajectories */}
          {!mapData &&
            sampleTrajectories.map((trajectory, idx) => (
              <Polyline key={idx} positions={trajectory} color="blue" weight={2} opacity={0.6} />
            ))}

          {mapData &&
            mapData.trajectories?.map((trajectory, idx) => (
              <Polyline
                key={idx}
                positions={trajectory.points || trajectory}
                color="blue"
                weight={2}
                opacity={0.8}
              />
            ))}

          {/* Markers */}
          {floatPositions.map((float) => (
            <Marker
              key={float.id}
              position={[float.lat ?? float.latitude, float.lng ?? float.longitude]}
              icon={getIcon(float.status)}
            >
              <Popup>
                <div className="text-sm">
                  <strong>Float ID:</strong> {float.id}
                  <br />
                  <strong>Status:</strong> {float.status}
                  <br />
                  <strong>Position:</strong> {(float.lat ?? float.latitude)?.toFixed(2)}°,{" "}
                  {(float.lng ?? float.longitude)?.toFixed(2)}°
                  <br />
                  <strong>Depth:</strong> {float.depth ?? "N/A"} m
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
