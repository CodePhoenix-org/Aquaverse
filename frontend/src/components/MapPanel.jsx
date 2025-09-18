import { MapContainer, TileLayer, CircleMarker, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";

function MapPanel({ points }) {
  const center = points?.length
    ? [points[0].latitude, points[0].longitude]
    : [20, 0];

  return (
    <div className="h-[320px] rounded-lg overflow-hidden">
      <MapContainer center={center} zoom={3} className="h-full w-full">
        <TileLayer
          attribution='&copy; <a href="https://carto.com/">CARTO</a>, <a href="https://www.openstreetmap.org/">OSM</a>'
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        />
        {points?.map((p, idx) => (
          <CircleMarker
            key={idx}
            center={[p.latitude, p.longitude]}
            pathOptions={{ color: p.color || "#06b6d4" }}
            radius={4}
          >
            <Tooltip>
              <div className="text-xs">
                <div><b>Float</b>: {p.float_id ?? "N/A"}</div>
                <div><b>Lat</b>: {p.latitude?.toFixed?.(2)}</div>
                <div><b>Lon</b>: {p.longitude?.toFixed?.(2)}</div>
                {p.variable && <div><b>{p.variable}</b>: {p.value}</div>}
              </div>
            </Tooltip>
          </CircleMarker>
        ))}
      </MapContainer>
    </div>
  );
}

export default MapPanel;


