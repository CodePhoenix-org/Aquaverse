import { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";

const sampleComparisonData = {
  temperature: {
    region1_temp: {
      depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
      values: [28.5, 28.2, 27.8, 25.1, 22.3, 18.7, 12.4, 8.2, 5.1, 3.8],
      name: "Arabian Sea",
      color: "#7fe7ff",
    },
    region2_temp: {
      depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
      values: [29.1, 28.8, 28.3, 26.2, 23.5, 19.8, 13.1, 8.8, 5.5, 4.1],
      name: "Bay of Bengal",
      color: "#2dd4bf",
    },
    time1_temp: {
      depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
      values: [28.5, 28.2, 27.8, 25.1, 22.3, 18.7, 12.4, 8.2, 5.1, 3.8],
      name: "March 2023",
      color: "#f59e7c",
    },
    time2_temp: {
      depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
      values: [27.8, 27.5, 27.1, 24.3, 21.6, 17.9, 11.8, 7.9, 4.8, 3.5],
      name: "September 2023",
      color: "#c084fc",
    },
  },
  salinity: {
    region1_sal: {
      depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
      values: [34.2, 34.3, 34.4, 34.6, 34.8, 35.1, 35.3, 35.0, 34.8, 34.7],
      name: "Arabian Sea",
      color: "#7fe7ff",
    },
    region2_sal: {
      depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
      values: [33.8, 33.9, 34.0, 34.2, 34.5, 34.9, 35.1, 34.8, 34.6, 34.5],
      name: "Bay of Bengal",
      color: "#2dd4bf",
    },
  },
  oxygen: {
    region1_oxy: {
      depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
      values: [220, 218, 215, 200, 180, 150, 120, 180, 200, 220],
      name: "Arabian Sea",
      color: "#7fe7ff",
    },
    region2_oxy: {
      depths: [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
      values: [215, 212, 208, 190, 170, 140, 110, 170, 190, 210],
      name: "Bay of Bengal",
      color: "#2dd4bf",
    },
  },
};

const plotTheme = {
  paper_bgcolor: "rgba(7, 24, 42, 0)",
  plot_bgcolor: "rgba(7, 24, 42, 0)",
  font: { family: "Manrope, sans-serif", color: "#d9ecff", size: 12 },
  margin: { l: 60, r: 24, t: 56, b: 60 },
};

function isValidComparisonProfile(profile) {
  return (
    profile &&
    Array.isArray(profile.depths) &&
    Array.isArray(profile.values) &&
    typeof profile.name === "string" &&
    typeof profile.color === "string" &&
    profile.depths.length === profile.values.length
  );
}

function isValidComparisonData(dataStructure) {
  if (!dataStructure || typeof dataStructure !== "object") return false;

  return ["temperature", "salinity", "oxygen"].some((parameter) => {
    const parameterData = dataStructure[parameter];
    if (!parameterData) return false;
    return Object.values(parameterData).filter(isValidComparisonProfile).length >= 2;
  });
}

function isSingleProfileData(dataStructure) {
  if (!dataStructure || typeof dataStructure !== "object") return false;

  return ["temperature", "salinity", "oxygen"].some((parameter) => {
    const parameterData = dataStructure[parameter];
    return (
      parameterData &&
      Array.isArray(parameterData.depths) &&
      Array.isArray(parameterData.values) &&
      typeof parameterData.title === "string"
    );
  });
}

function getComparisonData(data) {
  if (data && typeof data === "object" && (data.temperature_profile || data.salinity_profile)) {
    const comparisonData = {};
    if (data.temperature_profile) {
      comparisonData.temperature = {
        temp1: { ...data.temperature_profile, name: "Profile 1", color: "#7fe7ff" },
      };
    }
    if (data.salinity_profile) {
      comparisonData.salinity = {
        sal1: { ...data.salinity_profile, name: "Profile 1", color: "#2dd4bf" },
      };
    }
    if (Object.keys(comparisonData).length > 0) return comparisonData;
  }

  if (data?.profileComparisons && isValidComparisonData(data.profileComparisons)) {
    return data.profileComparisons;
  }

  if (data?.profiles && isValidComparisonData(data.profiles)) {
    return data.profiles;
  }

  if (data?.profiles && isSingleProfileData(data.profiles)) {
    return sampleComparisonData;
  }

  return sampleComparisonData;
}

export default function ProfileComparison({ data }) {
  const [selectedProfiles, setSelectedProfiles] = useState(["region1_temp", "region2_temp"]);
  const [activeParameter, setActiveParameter] = useState("temperature");
  const [comparisonMode, setComparisonMode] = useState("region");

  const comparisonData = useMemo(() => getComparisonData(data), [data]);

  const getAvailableProfiles = () => {
    const profiles = comparisonData[activeParameter] || {};
    return Object.keys(profiles).filter((key) =>
      comparisonMode === "region" ? key.includes("region") : key.includes("time")
    );
  };

  useEffect(() => {
    const availableProfiles = getAvailableProfiles();
    if (availableProfiles.length >= 2) {
      setSelectedProfiles(availableProfiles.slice(0, 2));
    } else if (availableProfiles.length === 1) {
      setSelectedProfiles(availableProfiles);
    } else {
      setSelectedProfiles([]);
    }
  }, [activeParameter, comparisonMode]); // eslint-disable-line react-hooks/exhaustive-deps

  const plotData = selectedProfiles
    .filter((profileId) => isValidComparisonProfile(comparisonData[activeParameter]?.[profileId]))
    .map((profileId) => {
      const profile = comparisonData[activeParameter][profileId];
      return {
        x: profile.values,
        y: profile.depths,
        type: "scatter",
        mode: "lines+markers",
        marker: { color: profile.color, size: 7, line: { color: "#04101d", width: 1 } },
        line: { color: profile.color, width: 3 },
        name: profile.name,
      };
    });

  const getParameterUnit = (parameter) => {
    const units = {
      temperature: "C",
      salinity: "PSU",
      oxygen: "umol/kg",
    };
    return units[parameter] || "";
  };

  return (
    <div className="space-y-5">
      <div>
        <p className="premium-kicker">Comparison Layer</p>
        <h3 className="mt-2 font-display text-2xl font-semibold text-white">
          Compare profiles across region or time
        </h3>
        <p className="mt-2 text-sm leading-6 text-slate-300">
          Overlay multiple ocean profiles inside a cleaner, easier-to-read premium plot surface.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-[0.42fr_0.58fr]">
        <div className="space-y-4">
          <div className="premium-card p-4">
            <p className="text-sm font-medium text-slate-200">Parameter</p>
            <div className="mt-3 flex flex-wrap gap-2">
              {Object.keys(comparisonData).map((parameter) => (
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
          </div>

          <div className="premium-card p-4">
            <p className="text-sm font-medium text-slate-200">Comparison type</p>
            <div className="mt-3 flex flex-wrap gap-2">
              {[
                { id: "region", label: "By Region" },
                { id: "time", label: "By Time" },
              ].map((mode) => (
                <button
                  key={mode.id}
                  onClick={() => setComparisonMode(mode.id)}
                  className={`rounded-full px-4 py-2 text-sm font-semibold transition-all ${
                    comparisonMode === mode.id
                      ? "bg-gradient-to-r from-cyan-300 to-sky-400 text-slate-950 shadow-lg"
                      : "border border-white/10 bg-white/[0.04] text-slate-200 hover:bg-white/[0.08]"
                  }`}
                >
                  {mode.label}
                </button>
              ))}
            </div>
          </div>

          <div className="premium-card p-4">
            <p className="text-sm font-medium text-slate-200">
              Active profiles ({comparisonMode === "region" ? "regions" : "time periods"})
            </p>
            <div className="mt-3 space-y-3">
              {getAvailableProfiles().map((profileId) => {
                const profile = comparisonData[activeParameter]?.[profileId];
                if (!profile || !isValidComparisonProfile(profile)) return null;

                const checked = selectedProfiles.includes(profileId);

                return (
                  <label
                    key={profileId}
                    className="flex items-center gap-3 rounded-2xl border border-white/[0.08] bg-white/[0.03] px-4 py-3"
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={(event) => {
                        if (event.target.checked) {
                          setSelectedProfiles((current) => [...current, profileId]);
                        } else {
                          setSelectedProfiles((current) => current.filter((id) => id !== profileId));
                        }
                      }}
                      className="h-4 w-4 rounded border-white/20 bg-slate-900 text-cyan-300"
                    />
                    <span
                      className="h-3 w-3 rounded-full"
                      style={{ backgroundColor: profile.color }}
                    />
                    <span className="text-sm text-slate-200">{profile.name}</span>
                  </label>
                );
              })}
            </div>
          </div>
        </div>

        <div className="premium-card p-2">
          {plotData.length ? (
            <Plot
              data={plotData}
              layout={{
                ...plotTheme,
                title: {
                  text: `${activeParameter} comparison`,
                  font: { family: "Space Grotesk, sans-serif", size: 18, color: "#f8fdff" },
                },
                xaxis: {
                  title: `${activeParameter} (${getParameterUnit(activeParameter)})`,
                  gridcolor: "rgba(184, 214, 236, 0.16)",
                  color: "#cce6ff",
                },
                yaxis: {
                  title: "Depth (m)",
                  autorange: "reversed",
                  gridcolor: "rgba(184, 214, 236, 0.16)",
                  color: "#cce6ff",
                },
                legend: {
                  bgcolor: "rgba(7,24,42,0.4)",
                  bordercolor: "rgba(125,211,252,0.16)",
                  font: { color: "#d9ecff" },
                },
              }}
              config={{
                displayModeBar: true,
                displaylogo: false,
                responsive: true,
                modeBarButtonsToRemove: ["pan2d", "lasso2d", "select2d"],
              }}
              style={{ width: "100%", minHeight: "32rem", height: "100%" }}
              useResizeHandler
            />
          ) : (
            <div className="flex min-h-[32rem] items-center justify-center text-center">
              <div>
                <p className="font-display text-2xl font-semibold text-white">
                  No profiles selected
                </p>
                <p className="mt-3 max-w-md text-sm leading-7 text-slate-300">
                  Choose at least one valid profile from the left panel to start comparing.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="premium-card p-4">
          <p className="text-sm text-slate-300">Parameter</p>
          <p className="mt-2 font-display text-2xl font-semibold text-white">
            {activeParameter}
          </p>
        </div>
        <div className="premium-card p-4">
          <p className="text-sm text-slate-300">Mode</p>
          <p className="mt-2 font-display text-2xl font-semibold text-white">
            {comparisonMode}
          </p>
        </div>
        <div className="premium-card p-4">
          <p className="text-sm text-slate-300">Profiles selected</p>
          <p className="mt-2 font-display text-2xl font-semibold text-white">
            {selectedProfiles.length}
          </p>
        </div>
      </div>
    </div>
  );
}
