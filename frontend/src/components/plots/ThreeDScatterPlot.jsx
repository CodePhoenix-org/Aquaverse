import Plot from "react-plotly.js";

export default function ThreeDScatterPlot({ normalData = [], anomalyData = [] }) {
  return (
    <Plot
      data={[
        {
          x: normalData.map((item) => item.Temperature),
          y: normalData.map((item) => item.Salinity),
          z: normalData.map((item) => item.Oxygen),
          mode: "markers",
          type: "scatter3d",
          marker: {
            size: normalData.map((item) =>
              Math.max(4, ((item.Temperature - 2.728) / (31.018 - 2.728)) * 10)
            ),
            color: "#7fe7ff",
            symbol: "circle",
            opacity: 0.85,
            line: { color: "#0f172a", width: 1 },
          },
          name: "Normal",
          hovertemplate:
            "Temp: %{x} C<br>Salinity: %{y} PSU<br>Oxygen: %{z} umol/kg<br>Status: Normal<extra></extra>",
        },
        {
          x: anomalyData.map((item) => item.Temperature),
          y: anomalyData.map((item) => item.Salinity),
          z: anomalyData.map((item) => item.Oxygen),
          mode: "markers",
          type: "scatter3d",
          marker: {
            size: anomalyData.map((item) =>
              Math.max(6, ((item.Temperature - 2.728) / (31.018 - 2.728)) * 12)
            ),
            color: "#f59e7c",
            symbol: "diamond",
            opacity: 0.95,
            line: { color: "#0f172a", width: 1.5 },
          },
          name: "Anomaly",
          hovertemplate:
            "Temp: %{x} C<br>Salinity: %{y} PSU<br>Oxygen: %{z} umol/kg<br>Status: Anomaly<extra></extra>",
        },
      ]}
      layout={{
        title: {
          text: "3D Parameter Space",
          font: { family: "Space Grotesk, sans-serif", color: "#f8fdff", size: 18 },
        },
        paper_bgcolor: "rgba(7,24,42,0)",
        scene: {
          bgcolor: "rgba(7,24,42,0)",
          xaxis: {
            title: "Temperature (C)",
            color: "#d9ecff",
            gridcolor: "rgba(184,214,236,0.16)",
            backgroundcolor: "rgba(8, 25, 44, 0.45)",
          },
          yaxis: {
            title: "Salinity (PSU)",
            color: "#d9ecff",
            gridcolor: "rgba(184,214,236,0.16)",
            backgroundcolor: "rgba(8, 25, 44, 0.45)",
          },
          zaxis: {
            title: "Oxygen (umol/kg)",
            color: "#d9ecff",
            gridcolor: "rgba(184,214,236,0.16)",
            backgroundcolor: "rgba(8, 25, 44, 0.45)",
          },
          camera: { eye: { x: 1.5, y: 1.5, z: 1.25 } },
          dragmode: "orbit",
        },
        legend: {
          bgcolor: "rgba(7,24,42,0.55)",
          bordercolor: "rgba(125,211,252,0.16)",
          font: { color: "#d9ecff" },
        },
        margin: { l: 0, r: 0, b: 0, t: 40 },
        height: 700,
      }}
      config={{ displayModeBar: true, responsive: true, scrollZoom: true }}
      style={{ width: "100%", height: "700px" }}
    />
  );
}
