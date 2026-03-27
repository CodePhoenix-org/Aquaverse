import Plot from "react-plotly.js";

export default function OceanScatter3DPlot({ data = [] }) {
  const symbolArr = data.map((item) =>
    item.Anomaly_Status === "Normal" ? "circle" : "diamond"
  );

  const stemTraces = data.map((item) => ({
    type: "scatter3d",
    mode: "lines",
    x: [item.Longitude, item.Longitude],
    y: [item.Latitude, item.Latitude],
    z: [0, item.Depth || item.pressure],
    line: {
      color: "rgba(125, 211, 252, 0.45)",
      width: 2,
      dash: item.Anomaly_Status === "Anomaly" ? "dash" : "solid",
    },
    hoverinfo: "none",
    showlegend: false,
  }));

  return (
    <Plot
      data={[
        ...stemTraces,
        {
          x: data.map((item) => item.Longitude),
          y: data.map((item) => item.Latitude),
          z: data.map((item) => item.Depth || item.pressure),
          mode: "markers",
          type: "scatter3d",
          marker: {
            size: data.map((item) =>
              Math.max(4, ((item.Temperature - 2.728) / (31.018 - 2.728)) * 8)
            ),
            color: data.map((item) => item.Temperature),
            colorscale: [
              [0, "#0ea5e9"],
              [0.5, "#7fe7ff"],
              [1, "#f59e7c"],
            ],
            symbol: symbolArr,
            opacity: 0.9,
            colorbar: {
              title: "Temp (C)",
              thickness: 18,
              len: 0.55,
              tickfont: { color: "#d9ecff" },
              titlefont: { color: "#d9ecff" },
            },
            line: { color: "#04101d", width: 1 },
          },
          text: data.map(
            (item) => `Status: ${item.Anomaly_Status}<br>Temp: ${item.Temperature} C`
          ),
          hovertemplate:
            "%{text}<br>Lon: %{x}<br>Lat: %{y}<br>Depth: %{z} m<extra></extra>",
          name: "Data Points",
        },
      ]}
      layout={{
        title: {
          text: "3D Ocean Geography",
          font: { family: "Space Grotesk, sans-serif", color: "#f8fdff", size: 18 },
        },
        paper_bgcolor: "rgba(7,24,42,0)",
        scene: {
          bgcolor: "rgba(7,24,42,0)",
          xaxis: {
            title: "Longitude (deg)",
            color: "#d9ecff",
            gridcolor: "rgba(184,214,236,0.16)",
            backgroundcolor: "rgba(8, 25, 44, 0.45)",
          },
          yaxis: {
            title: "Latitude (deg)",
            color: "#d9ecff",
            gridcolor: "rgba(184,214,236,0.16)",
            backgroundcolor: "rgba(8, 25, 44, 0.45)",
          },
          zaxis: {
            title: "Depth (m)",
            autorange: "reversed",
            color: "#d9ecff",
            gridcolor: "rgba(184,214,236,0.16)",
            backgroundcolor: "rgba(8, 25, 44, 0.45)",
          },
          aspectratio: { x: 1, y: 1, z: 0.55 },
          camera: { eye: { x: 2, y: 2, z: 0.6 } },
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
