import { useEffect, useRef, useState } from "react";
import { MapContainer, Marker, Popup, Polyline, TileLayer } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { Compass, Waves } from "lucide-react";

const activeIcon = new L.Icon({
  iconUrl: "https://cdn-icons-png.flaticon.com/512/190/190411.png",
  iconSize: [28, 28],
  iconAnchor: [14, 28],
  popupAnchor: [0, -28],
});

const inactiveIcon = new L.Icon({
  iconUrl: "https://cdn-icons-png.flaticon.com/512/190/190406.png",
  iconSize: [28, 28],
  iconAnchor: [14, 28],
  popupAnchor: [0, -28],
});

const sampleFloats = [
  { id: "ARGO_001", lat: -10.5, lng: 75.2, status: "active", lastUpdate: "2023-03-15", depth: 200 },
  { id: "ARGO_002", lat: -15.3, lng: 82.1, status: "active", lastUpdate: "2023-03-14", depth: 150 },
  { id: "ARGO_003", lat: -8.7, lng: 70.8, status: "inactive", lastUpdate: "2023-02-28", depth: 300 },
  { id: "ARGO_004", lat: -25.1, lng: 88.5, status: "active", lastUpdate: "2023-03-16", depth: 100 },
  { id: "ARGO_005", lat: -18.2, lng: 78.4, status: "inactive", lastUpdate: "2023-03-10", depth: 250 },
  { id: "ARGO_006", lat: -12.9, lng: 80.7, status: "active", lastUpdate: "2023-03-12", depth: 180 },
];

const sampleTrajectories = [
  [[-10.5, 75.2], [-11.2, 75.8], [-12.1, 76.3], [-13.0, 76.9]],
  [[-15.3, 82.1], [-16.0, 82.7], [-16.8, 83.2], [-17.5, 83.8]],
  [[-8.7, 70.8], [-9.3, 71.1], [-10.0, 71.5]],
  [[-25.1, 88.5], [-24.5, 88.0], [-23.9, 87.5]],
  [[-18.2, 78.4], [-18.7, 79.0], [-19.3, 79.5]],
  [[-12.9, 80.7], [-13.5, 81.1], [-14.0, 81.6]],
];

export default function FloatMap({ data }) {
  const [mapData, setMapData] = useState(null);
  const [floatPositions, setFloatPositions] = useState([]);
  const mapRef = useRef(null);

  useEffect(() => {
    if (data && typeof data === "object") {
      if (data.float_locations) {
        setMapData({
          floats: data.float_locations,
          trajectories: data.float_trajectories || [],
        });
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
    setFloatPositions(mapData?.floats || sampleFloats);
  }, [mapData]);

  useEffect(() => {
    const interval = setInterval(() => {
      setFloatPositions((previous) =>
        previous.map((float) => ({
          ...float,
          lat: (float.lat ?? float.latitude) + (Math.random() - 0.5) * 0.035,
          lng: (float.lng ?? float.longitude) + (Math.random() - 0.5) * 0.035,
        }))
      );
    }, 2200);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      mapRef.current?.invalidateSize(false);
    }, 200);

    return () => clearTimeout(timer);
  });

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="premium-kicker">Geographic Layer</p>
          <h3 className="mt-2 font-display text-2xl font-semibold text-white">
            Float fleet positions and trajectories
          </h3>
        </div>
        <div className="flex flex-wrap gap-3">
          <span className="premium-chip">
            <Waves className="h-4 w-4 text-cyan-100" />
            {floatPositions.length} tracked floats
          </span>
          <span className="premium-chip">
            <Compass className="h-4 w-4 text-cyan-100" />
            Premium dark basemap
          </span>
        </div>
      </div>

      <div className="overflow-hidden rounded-[24px] border border-white/10">
        <div style={{ minHeight: "28rem", height: "28rem" }}>
          <MapContainer
            center={[-20, 80]}
            zoom={4}
            whenReady={(event) => {
              mapRef.current = event.target;
              setTimeout(() => event.target.invalidateSize(false), 50);
            }}
            style={{ height: "100%", width: "100%" }}
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; CARTO'
              url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            />

            {!mapData
              ? sampleTrajectories.map((trajectory, index) => (
                  <Polyline
                    key={index}
                    positions={trajectory}
                    color="#38bdf8"
                    weight={2}
                    opacity={0.55}
                  />
                ))
              : mapData.trajectories?.map((trajectory, index) => (
                  <Polyline
                    key={index}
                    positions={trajectory.points || trajectory}
                    color="#38bdf8"
                    weight={2}
                    opacity={0.75}
                  />
                ))}

            {floatPositions.map((float) => {
              const lat = float.lat ?? float.latitude;
              const lng = float.lng ?? float.longitude;
              const status = float.status || "active";

              return (
                <Marker
                  key={float.id}
                  position={[lat, lng]}
                  icon={status === "active" ? activeIcon : inactiveIcon}
                >
                  <Popup>
                    <div className="space-y-1 text-sm">
                      <p className="font-semibold text-cyan-100">{float.id}</p>
                      <p>Status: {status}</p>
                      <p>
                        Position: {lat?.toFixed(2)}, {lng?.toFixed(2)}
                      </p>
                      <p>Depth: {float.depth ?? "N/A"} m</p>
                      <p>Last update: {float.lastUpdate || "Unknown"}</p>
                    </div>
                  </Popup>
                </Marker>
              );
            })}
          </MapContainer>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="premium-card p-4">
          <p className="text-sm text-slate-300">Trajectory mode</p>
          <p className="mt-2 font-display text-2xl font-semibold text-white">Enabled</p>
        </div>
        <div className="premium-card p-4">
          <p className="text-sm text-slate-300">Active marker style</p>
          <p className="mt-2 font-display text-2xl font-semibold text-white">Neon pinpoint</p>
        </div>
        <div className="premium-card p-4">
          <p className="text-sm text-slate-300">Coverage source</p>
          <p className="mt-2 font-display text-2xl font-semibold text-white">
            {mapData ? "Live query" : "Demo fleet"}
          </p>
        </div>
      </div>
    </div>
  );
}
