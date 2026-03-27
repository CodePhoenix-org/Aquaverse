import { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";
import { BarChart3, Database, Globe, Loader2 } from "lucide-react";
import {
  extract3DOceanData,
  extract3DParameterData,
  has3DData,
  transformApiData,
  validatePlotData,
} from "../../utils/dataTransformers";
import OceanScatter3DPlot from "./OceanScatter3DPlot";
import ThreeDScatterPlot from "./ThreeDScatterPlot";

const sampleData = {
  temperature: {
    depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
    values: [28.5, 28.2, 27.8, 25.1, 22.3, 18.7, 12.4, 8.2, 5.1, 3.8],
    title: "Temperature Profile",
    yLabel: "Temperature (C)",
    color: "#6fe7ff",
  },
  salinity: {
    depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
    values: [34.2, 34.3, 34.4, 34.6, 34.8, 35.1, 35.3, 35.0, 34.8, 34.7],
    title: "Salinity Profile",
    yLabel: "Salinity (PSU)",
    color: "#2dd4bf",
  },
};

const plotTheme = {
  paper_bgcolor: "rgba(7, 24, 42, 0)",
  plot_bgcolor: "rgba(7, 24, 42, 0)",
  font: { family: "Manrope, sans-serif", color: "#d9ecff", size: 12 },
  margin: { l: 70, r: 24, t: 56, b: 60 },
};

function TabButton({ active, onClick, icon: Icon, children }) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition-all ${
        active
          ? "bg-gradient-to-r from-cyan-300 to-sky-400 text-slate-950 shadow-lg"
          : "border border-white/10 bg-white/[0.04] text-slate-200 hover:bg-white/[0.08]"
      }`}
    >
      <Icon className="h-4 w-4" />
      {children}
    </button>
  );
}

export default function DataPlots({ data }) {
  const [activeTab, setActiveTab] = useState("2d");
  const [active3DView, setActive3DView] = useState("parameter");
  const [threeDData, setThreeDData] = useState(null);
  const [loading3D, setLoading3D] = useState(false);
  const [availableParameters, setAvailableParameters] = useState(["temperature"]);
  const [activeParameter, setActiveParameter] = useState("temperature");
  const [plotReadyData, setPlotReadyData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setError(null);

    if (!data) {
      setAvailableParameters(["temperature"]);
      setPlotReadyData(null);
      return;
    }

    if (has3DData(data)) {
      setThreeDData(data);
      setActiveTab("3d");
      setActive3DView(data.type === "3d_ocean_plot" ? "ocean" : "parameter");
    }

    if (!validatePlotData(data)) {
      setAvailableParameters(["temperature"]);
      setPlotReadyData(null);
      return;
    }

    try {
      const params = Object.keys(data.profiles).filter((param) => {
        const profile = data.profiles[param];
        return (
          profile &&
          Array.isArray(profile.depths) &&
          Array.isArray(profile.values) &&
          profile.depths.length > 0 &&
          profile.values.length > 0
        );
      });

      if (!params.length) {
        setError("No valid parameters were found in the current dataset.");
        setAvailableParameters(["temperature"]);
        setPlotReadyData(null);
        return;
      }

      setAvailableParameters(params);
      setActiveParameter(
        params.includes("salinity")
          ? "salinity"
          : params.includes("temperature")
            ? "temperature"
            : params[0]
      );
      setPlotReadyData(data.profiles);
    } catch (processingError) {
      setError(`Unable to prepare plots: ${processingError.message}`);
      setAvailableParameters(["temperature"]);
      setPlotReadyData(null);
    }
  }, [data]);

  const fetch3DData = async (type = "parameter") => {
    setLoading3D(true);
    setError(null);

    try {
      const endpoint = `/api/3d/${type === "parameter" ? "parameter-plot" : "ocean-plot"}?region=indian&limit=${type === "parameter" ? 1000 : 500}`;
      const response = await fetch(endpoint);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const apiData = await response.json();
      const transformed = transformApiData(apiData);

      if (!transformed || !transformed.threeDData) {
        throw new Error("The backend returned empty 3D data.");
      }

      setThreeDData(transformed);
    } catch (fetchError) {
      setError(`Failed to load 3D visualization data: ${fetchError.message}`);
      setThreeDData(
        type === "parameter"
          ? transformApiData({ type: "3d_parameter_plot", normalData: [], anomalyData: [] })
          : transformApiData({ type: "3d_ocean_plot", oceanData: [] })
      );
    } finally {
      setLoading3D(false);
    }
  };

  useEffect(() => {
    if (activeTab !== "3d") return undefined;

    fetch3DData(active3DView);
    const interval = setInterval(() => fetch3DData(active3DView), 60000);
    return () => clearInterval(interval);
  }, [activeTab, active3DView]);

  const current2DData = useMemo(() => {
    if (plotReadyData?.[activeParameter]) {
      return plotReadyData[activeParameter];
    }

    return sampleData[activeParameter] || sampleData.temperature;
  }, [activeParameter, plotReadyData]);

  const isReal2DData = Boolean(plotReadyData?.[activeParameter]);
  const hasValid2DData =
    Array.isArray(current2DData?.values) &&
    Array.isArray(current2DData?.depths) &&
    current2DData.values.length > 0 &&
    current2DData.depths.length > 0;

  const getParameterUnit = (parameter) => {
    const units = {
      temperature: "C",
      salinity: "PSU",
      oxygen: "umol/kg",
      chlorophyll: "mg/m3",
    };
    return units[parameter] || "";
  };

  const plotConfig = useMemo(
    () => ({
      data: hasValid2DData
        ? [
            {
              x: current2DData.values,
              y: current2DData.depths,
              type: "scatter",
              mode: "lines+markers",
              marker: {
                color: current2DData.color,
                size: 7,
                symbol: isReal2DData ? "circle" : "diamond",
                line: { color: "#04101d", width: 1 },
              },
              line: {
                color: current2DData.color,
                width: 3,
                dash: isReal2DData ? "solid" : "dot",
              },
              hovertemplate:
                "<b>Depth</b>: %{y} m<br>" +
                `<b>${current2DData.yLabel}</b>: %{x} ${getParameterUnit(activeParameter)}<extra></extra>`,
              name: current2DData.title,
            },
          ]
        : [],
      layout: {
        ...plotTheme,
        title: {
          text: hasValid2DData
            ? `${current2DData.title}${isReal2DData ? " - Live Data" : " - Sample Data"}`
            : "No data available",
          font: { family: "Space Grotesk, sans-serif", size: 18, color: "#f8fdff" },
        },
        xaxis: {
          title: { text: current2DData.yLabel, font: { color: "#cce6ff" } },
          gridcolor: "rgba(184, 214, 236, 0.16)",
          zerolinecolor: "rgba(184, 214, 236, 0.16)",
          color: "#cce6ff",
        },
        yaxis: {
          title: { text: "Depth (m)", font: { color: "#cce6ff" } },
          autorange: "reversed",
          gridcolor: "rgba(184, 214, 236, 0.16)",
          zerolinecolor: "rgba(184, 214, 236, 0.16)",
          color: "#cce6ff",
        },
        hoverlabel: {
          bgcolor: "#07192c",
          bordercolor: "rgba(125,211,252,0.24)",
          font: { family: "Manrope, sans-serif", color: "#f8fdff" },
        },
      },
      config: {
        displayModeBar: true,
        displaylogo: false,
        responsive: true,
        modeBarButtonsToRemove: ["pan2d", "lasso2d", "select2d"],
      },
    }),
    [activeParameter, current2DData, hasValid2DData, isReal2DData]
  );

  if (error && activeTab === "2d") {
    return (
      <div className="flex min-h-[28rem] items-center justify-center rounded-[24px] border border-rose-300/20 bg-rose-300/8 p-8 text-center">
        <div>
          <p className="font-display text-2xl font-semibold text-white">Plot preparation issue</p>
          <p className="mt-3 max-w-lg text-sm leading-7 text-slate-300">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="premium-kicker">Visualization Layer</p>
          <h3 className="mt-2 font-display text-2xl font-semibold text-white">
            {activeTab === "2d" ? "Depth profile analysis" : "Interactive 3D exploration"}
          </h3>
          <p className="mt-2 text-sm leading-6 text-slate-300">
            Move between precise 2D parameter profiles and immersive 3D structures.
          </p>
        </div>

        <div className="flex flex-wrap gap-3">
          <TabButton active={activeTab === "2d"} onClick={() => setActiveTab("2d")} icon={BarChart3}>
            2D Profiles
          </TabButton>
          <TabButton
            active={activeTab === "3d"}
            onClick={() => {
              setActiveTab("3d");
              if (!threeDData) fetch3DData("parameter");
            }}
            icon={Globe}
          >
            3D Views
          </TabButton>
        </div>
      </div>

      {activeTab === "2d" ? (
        <div className="space-y-5">
          <div className="flex flex-wrap gap-3">
            {availableParameters.map((parameter) => (
              <button
                key={parameter}
                onClick={() => setActiveParameter(parameter)}
                className={`rounded-full px-4 py-2 text-sm font-semibold transition-all ${
                  activeParameter === parameter
                    ? "bg-gradient-to-r from-cyan-300 to-sky-400 text-slate-950 shadow-lg"
                    : "border border-white/10 bg-white/[0.04] text-slate-200 hover:bg-white/[0.08]"
                }`}
              >
                {parameter}
              </button>
            ))}
          </div>

          <div className="premium-card p-2">
            {hasValid2DData ? (
              <Plot
                data={plotConfig.data}
                layout={plotConfig.layout}
                config={plotConfig.config}
                style={{ width: "100%", minHeight: "26rem", height: "100%" }}
                useResizeHandler
              />
            ) : (
              <div className="flex min-h-[26rem] items-center justify-center text-center">
                <div>
                  <p className="font-display text-2xl font-semibold text-white">
                    No profile data available
                  </p>
                  <p className="mt-3 max-w-md text-sm leading-7 text-slate-300">
                    The current payload does not contain matching depth and value arrays
                    for this parameter.
                  </p>
                </div>
              </div>
            )}
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            <div className="premium-card p-4">
              <p className="text-sm text-slate-300">Dataset mode</p>
              <p className="mt-2 font-display text-2xl font-semibold text-white">
                {isReal2DData ? "Live" : "Sample"}
              </p>
            </div>
            <div className="premium-card p-4">
              <p className="text-sm text-slate-300">Selected parameter</p>
              <p className="mt-2 font-display text-2xl font-semibold text-white">
                {activeParameter}
              </p>
            </div>
            <div className="premium-card p-4">
              <p className="text-sm text-slate-300">Data points</p>
              <p className="mt-2 font-display text-2xl font-semibold text-white">
                {hasValid2DData ? current2DData.values.length : 0}
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-5">
          <div className="flex flex-wrap gap-3">
            <TabButton
              active={active3DView === "parameter"}
              onClick={() => setActive3DView("parameter")}
              icon={BarChart3}
            >
              Parameter Space
            </TabButton>
            <TabButton
              active={active3DView === "ocean"}
              onClick={() => setActive3DView("ocean")}
              icon={Globe}
            >
              Ocean Geography
            </TabButton>
          </div>

          <div className="premium-card p-2">
            {loading3D ? (
              <div className="flex min-h-[28rem] items-center justify-center gap-3 text-slate-200">
                <Loader2 className="h-5 w-5 animate-spin" />
                Loading 3D visualization...
              </div>
            ) : threeDData ? (
              <>
                {active3DView === "parameter" && threeDData.type === "3d_parameter_plot" ? (
                  <ThreeDScatterPlot {...extract3DParameterData(threeDData)} />
                ) : null}

                {active3DView === "ocean" && threeDData.type === "3d_ocean_plot" ? (
                  <OceanScatter3DPlot data={extract3DOceanData(threeDData)} />
                ) : null}
              </>
            ) : (
              <div className="flex min-h-[28rem] items-center justify-center text-center">
                <div>
                  <Database className="mx-auto h-10 w-10 text-slate-400" />
                  <p className="mt-4 font-display text-2xl font-semibold text-white">
                    No 3D data loaded
                  </p>
                  <p className="mt-3 max-w-md text-sm leading-7 text-slate-300">
                    Load a 3D mode above to inspect parameter clusters or ocean geography.
                  </p>
                </div>
              </div>
            )}
          </div>

          {error ? (
            <div className="rounded-[24px] border border-amber-300/20 bg-amber-300/8 px-5 py-4 text-sm text-slate-200">
              {error}
            </div>
          ) : null}

          {threeDData?.metadata ? (
            <div className="grid gap-4 md:grid-cols-4">
              {Object.entries({
                Title: threeDData.metadata.title,
                Region: threeDData.metadata.region,
                "Total Points": threeDData.metadata.total_points,
                "Normal Count":
                  threeDData.metadata.normal_count ?? threeDData.metadata.anomaly_count,
              }).map(([label, value]) => (
                <div key={label} className="premium-card p-4">
                  <p className="text-sm text-slate-300">{label}</p>
                  <p className="mt-2 font-display text-xl font-semibold text-white">
                    {value ?? "-"}
                  </p>
                </div>
              ))}
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}
