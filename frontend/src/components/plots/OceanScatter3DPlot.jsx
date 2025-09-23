// // import React from "react";
// // import Plot from "react-plotly.js";

// // const OceanScatter3DPlot = ({ data }) => {
// //   const symbolArr = data.map(d => (d.Anomaly_Status === "Normal" ? "circle" : "diamond"));
  
// //   // Add stem lines: For each point, create a line from (x,y,0) to (x,y,z)
// //   const stemTraces = data.map((d, i) => ({
// //     type: "scatter3d",
// //     mode: "lines",
// //     x: [d.Longitude, d.Longitude],
// //     y: [d.Latitude, d.Latitude],
// //     z: [0, d.Depth],  // From surface (z=0) to point
// //     line: {
// //       color: "#4682B4",  // Steel blue
// //       width: 2,
// //       dash: d.Anomaly_Status === "Anomaly" ? "dash" : "solid"  // Dash for anomalies
// //     },
// //     hoverinfo: "none",
// //     showlegend: false
// //   }));

// //   return (
// //     <Plot
// //       data={[
// //         ...stemTraces,  // Add stems first (behind markers)
// //         {
// //           x: data.map(d => d.Longitude),
// //           y: data.map(d => d.Latitude),
// //           z: data.map(d => d.Depth),
// //           mode: "markers",
// //           type: "scatter3d",
// //           marker: {
// //             size: 6,  // Dynamic size? Use: data.map(d => Math.max(3, d.Temperature / 5)),
// //             color: data.map(d => d.Temperature),
// //             colorscale: "Portland",  // Better scale: Blues to reds
// //             symbol: symbolArr,
// //             opacity: 0.85,
// //             colorbar: { 
// //               title: "Temperature (°C)", 
// //               thickness: 20,
// //               len: 0.5
// //             },
// //             line: { color: "#333", width: 1 }  // Subtle border
// //           },
// //           text: data.map(d => `Status: ${d.Anomaly_Status}<br>Temp: ${d.Temperature}°C`),
// //           hovertemplate: "%{text}<br>Lon: %{x}<br>Lat: %{y}<br>Depth: %{z}m",
// //           name: "Data Points"
// //         }
// //       ]}
// //       layout={{
// //         title: {
// //           text: "3D Ocean Geography: Lon/Lat/Depth with Temperature",
// //           font: { size: 18, color: "#333" }
// //         },
// //         scene: {
// //           xaxis: { title: "Longitude (°)", backgroundcolor: "#F0F8FF", gridcolor: "#A9CDEF" },
// //           yaxis: { title: "Latitude (°)", backgroundcolor: "#F0F8FF", gridcolor: "#A9CDEF" },
// //           zaxis: { title: "Depth (m)", autorange: "reversed", backgroundcolor: "#F0F8FF", gridcolor: "#A9CDEF" },  // Reverse z for depth down
// //           bgcolor: "#B0E0E6",  // Powder blue background
// //           aspectratio: { x: 1, y: 1, z: 0.5 },  // Compress z for ocean feel
// //           camera: { eye: { x: 2, y: 2, z: 0.5 } },  // Overhead angled view
// //           dragmode: "orbit"
// //         },
// //         legend: { orientation: "h", y: -0.1 },
// //         margin: { l: 0, r: 0, b: 0, t: 40 },
// //         height: 700
// //       }}
// //       config={{ 
// //         displayModeBar: true,
// //         responsive: true,
// //         scrollZoom: true
// //       }}
// //       style={{ width: "100%", height: "700px" }}
// //     />
// //   );
// // };

// // export default OceanScatter3DPlot;


// import React from "react";
// import Plot from "react-plotly.js";

// const OceanScatter3DPlot = ({ data }) => {
//   const symbolArr = data.map(d => (d.Anomaly_Status === "Normal" ? "circle" : "diamond"));
//   const stemTraces = data.map((d, i) => ({
//     type: "scatter3d",
//     mode: "lines",
//     x: [d.Longitude, d.Longitude],
//     y: [d.Latitude, d.Latitude],
//     z: [0, d.Depth || d.pressure], // Use pressure as depth proxy if Depth missing
//     line: { color: "#4682B4", width: 2, dash: d.Anomaly_Status === "Anomaly" ? "dash" : "solid" },
//     hoverinfo: "none",
//     showlegend: false
//   }));

//   return (
//     <Plot
//       data={[
//         ...stemTraces,
//         {
//           x: data.map(d => d.Longitude),
//           y: data.map(d => d.Latitude),
//           z: data.map(d => d.Depth || d.pressure),
//           mode: "markers",
//           type: "scatter3d",
//           marker: {
//             size: data.map(d => Math.max(4, (d.Temperature - 2.728) / (31.018 - 2.728) * 8)),
//             color: data.map(d => d.Temperature),
//             colorscale: "Portland",
//             symbol: symbolArr,
//             opacity: 0.85,
//             colorbar: { title: "Temp (°C)", thickness: 20, len: 0.5 },
//             line: { color: "#333", width: 1 }
//           },
//           text: data.map(d => `Status: ${d.Anomaly_Status}<br>Temp: ${d.Temperature}°C`),
//           hovertemplate: "%{text}<br>Lon: %{x}<br>Lat: %{y}<br>Depth: %{z}m",
//           name: "Data Points"
//         }
//       ]}
//       layout={{
//         title: "3D Ocean: Lon/Lat/Depth with Temp",
//         scene: {
//           xaxis: { title: "Longitude (°)", backgroundcolor: "#F0F8FF", gridcolor: "#A9CDEF" },
//           yaxis: { title: "Latitude (°)", backgroundcolor: "#F0F8FF", gridcolor: "#A9CDEF" },
//           zaxis: { title: "Depth (m)", autorange: "reversed", backgroundcolor: "#F0F8FF", gridcolor: "#A9CDEF" },
//           bgcolor: "#B0E0E6",
//           aspectratio: { x: 1, y: 1, z: 0.5 },
//           camera: { eye: { x: 2, y: 2, z: 0.5 } },
//           dragmode: "orbit"
//         },
//         legend: { orientation: "h", y: -0.1 },
//         margin: { l: 0, r: 0, b: 0, t: 40 },
//         height: 700
//       }}
//       config={{ displayModeBar: true, responsive: true, scrollZoom: true }}
//       style={{ width: "100%", height: "700px" }}
//     />
//   );
// };

// export default OceanScatter3DPlot;





import React from "react";
import Plot from "react-plotly.js";

const OceanScatter3DPlot = ({ data }) => {
  const symbolArr = data.map(d => (d.Anomaly_Status === "Normal" ? "circle" : "diamond"));
  const stemTraces = data.map((d, i) => ({
    type: "scatter3d",
    mode: "lines",
    x: [d.Longitude, d.Longitude],
    y: [d.Latitude, d.Latitude],
    z: [0, d.Depth || d.pressure], // Use pressure as depth proxy if Depth missing
    line: { color: "#4682B4", width: 2, dash: d.Anomaly_Status === "Anomaly" ? "dash" : "solid" },
    hoverinfo: "none",
    showlegend: false
  }));

  return (
    <Plot
      data={[
        ...stemTraces,
        {
          x: data.map(d => d.Longitude),
          y: data.map(d => d.Latitude),
          z: data.map(d => d.Depth || d.pressure),
          mode: "markers",
          type: "scatter3d",
          marker: {
            size: data.map(d => Math.max(4, (d.Temperature - 2.728) / (31.018 - 2.728) * 8)),
            color: data.map(d => d.Temperature),
            colorscale: "Portland",
            symbol: symbolArr,
            opacity: 0.85,
            colorbar: { title: "Temp (°C)", thickness: 20, len: 0.5 },
            line: { color: "#333", width: 1 }
          },
          text: data.map(d => `Status: ${d.Anomaly_Status}<br>Temp: ${d.Temperature}°C`),
          hovertemplate: "%{text}<br>Lon: %{x}<br>Lat: %{y}<br>Depth: %{z}m",
          name: "Data Points"
        }
      ]}
      layout={{
        title: "3D Ocean: Lon/Lat/Depth with Temp",
        scene: {
          xaxis: { title: "Longitude (°)", backgroundcolor: "#F0F8FF", gridcolor: "#A9CDEF" },
          yaxis: { title: "Latitude (°)", backgroundcolor: "#F0F8FF", gridcolor: "#A9CDEF" },
          zaxis: { title: "Depth (m)", autorange: "reversed", backgroundcolor: "#F0F8FF", gridcolor: "#A9CDEF" },
          bgcolor: "#B0E0E6",
          aspectratio: { x: 1, y: 1, z: 0.5 },
          camera: { eye: { x: 2, y: 2, z: 0.5 } },
          dragmode: "orbit"
        },
        legend: { orientation: "h", y: -0.1 },
        margin: { l: 0, r: 0, b: 0, t: 40 },
        height: 700
      }}
      config={{ displayModeBar: true, responsive: true, scrollZoom: true }}
      style={{ width: "100%", height: "700px" }}
    />
  );
};

export default OceanScatter3DPlot;