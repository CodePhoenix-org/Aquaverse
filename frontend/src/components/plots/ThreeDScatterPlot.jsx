// import React from "react";
// import Plot from "react-plotly.js";

// // Expects props: normalData, anomalyData each array of objects { Temperature, Salinity, Oxygen }
// const ThreeDScatterPlot = ({ normalData, anomalyData }) => {
//   return (
//     <Plot
//       data={[
//         {
//           x: normalData.map(d => d.Temperature),
//           y: normalData.map(d => d.Salinity),
//           z: normalData.map(d => d.Oxygen),
//           mode: "markers",
//           type: "scatter3d",
//           marker: {
//             size: 5,
//             color: "blue",
//             symbol: "circle",
//             opacity: 0.8
//           },
//           name: "Normal"
//         },
//         {
//           x: anomalyData.map(d => d.Temperature),
//           y: anomalyData.map(d => d.Salinity),
//           z: anomalyData.map(d => d.Oxygen),
//           mode: "markers",
//           type: "scatter3d",
//           marker: {
//             size: 7,
//             color: "red",
//             symbol: "diamond",
//             opacity: 0.8
//           },
//           name: "Anomaly"
//         }
//       ]}
//       layout={{
//         title: "3D: Temperature (x), Salinity (y), Oxygen (z), Anomaly (Color/Symbol)",
//         scene: {
//           xaxis: { title: "Temperature (°C)" },
//           yaxis: { title: "Salinity" },
//           zaxis: { title: "Oxygen" }
//         },
//         height: 700
//       }}
//       config={{ displayModeBar: true }}
//       style={{ width: "100%", height: "700px" }}
//     />
//   );
// };
// export default ThreeDScatterPlot;






import React from "react";
import Plot from "react-plotly.js";

const ThreeDScatterPlot = ({ normalData, anomalyData }) => {
  return (
    <Plot
      data={[
        {
          x: normalData.map(d => d.Temperature),
          y: normalData.map(d => d.Salinity),
          z: normalData.map(d => d.Oxygen),
          mode: "markers",
          type: "scatter3d",
          marker: {
            size: normalData.map(d => Math.max(4, (d.Temperature - 2.728) / (31.018 - 2.728) * 10)), // Scale size by temp
            color: "#1E90FF",
            symbol: "circle",
            opacity: 0.85,
            line: { color: "#4169E1", width: 1 },
          },
          name: "Normal",
          hovertemplate: "Temp: %{x}°C<br>Sal: %{y} PSU<br>Oxy: %{z} μmol/kg<br>Status: Normal"
        },
        {
          x: anomalyData.map(d => d.Temperature),
          y: anomalyData.map(d => d.Salinity),
          z: anomalyData.map(d => d.Oxygen),
          mode: "markers",
          type: "scatter3d",
          marker: {
            size: anomalyData.map(d => Math.max(6, (d.Temperature - 2.728) / (31.018 - 2.728) * 12)),
            color: "#FF4500",
            symbol: "diamond",
            opacity: 0.9,
            line: { color: "#B22222", width: 2 },
          },
          name: "Anomaly",
          hovertemplate: "Temp: %{x}°C<br>Sal: %{y} PSU<br>Oxy: %{z} μmol/kg<br>Status: Anomaly"
        }
      ]}
      layout={{
        title: "3D Parameter Space: Temp vs Sal vs Oxy",
        scene: {
          xaxis: { title: "Temperature (°C)", range: [2, 32], backgroundcolor: "#F0F8FF", gridcolor: "#A9CDEF" },
          yaxis: { title: "Salinity (PSU)", range: [33.5, 36.6], backgroundcolor: "#F0F8FF", gridcolor: "#A9CDEF" },
          zaxis: { title: "Oxygen (μmol/kg)", backgroundcolor: "#F0F8FF", gridcolor: "#A9CDEF" },
          bgcolor: "#E0FFFF",
          aspectmode: "cube",
          camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } },
          dragmode: "orbit"
        },
        legend: { x: 0, y: 1, bgcolor: "rgba(255,255,255,0.8)" },
        margin: { l: 0, r: 0, b: 0, t: 40 },
        height: 700,
        annotations: [{ text: "Anomalies in red", xref: "paper", yref: "paper", x: 0.05, y: 0.05, showarrow: false, font: { color: "#666" } }]
      }}
      config={{ displayModeBar: true, responsive: true, scrollZoom: true }}
      style={{ width: "100%", height: "700px" }}
    />
  );
};

export default ThreeDScatterPlot;