import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/Authcontext";
import {
  AlertCircle,
  AlertTriangle,
  BarChart3,
  Compass,
  Droplets,
  HeartPulse,
  LifeBuoy,
  MapPin,
  Menu,
  MessageCircle,
  ShieldAlert,
  Sparkles,
  ThermometerSun,
  Waves,
  X,
  Zap,
} from "lucide-react";
import PageShell from "./ui/PageShell";
import BrandMark from "./ui/BrandMark";

function ResultCard({ title, badge, children, icon: Icon }) {
  return (
    <div className="premium-panel p-5 sm:p-6">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.06]">
            <Icon className="h-5 w-5 text-cyan-100" />
          </div>
          <h2 className="font-display text-2xl font-semibold text-white">{title}</h2>
        </div>
        {badge ? (
          <span className="rounded-full border border-white/10 bg-white/[0.06] px-4 py-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-200">
            {badge}
          </span>
        ) : null}
      </div>
      {children}
    </div>
  );
}

export default function Predictor() {
  const navigate = useNavigate();
  const { user } = useAuth();

  const [lat, setLat] = useState(9);
  const [lon, setLon] = useState(68);
  const [depth, setDepth] = useState(980);
  const [temperature, setTemperature] = useState(29.5);
  const [salinity, setSalinity] = useState(34.3);
  const [oxygen, setOxygen] = useState(58.9);
  const [chlorophyll, setChlorophyll] = useState(0.1);
  const [isLoading, setIsLoading] = useState(false);
  const [data, setData] = useState(null);
  const [risk, setRisk] = useState(null);
  const [survival, setSurvival] = useState(null);
  const [error, setError] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const fields = useMemo(
    () => [
      {
        label: "Latitude (deg)",
        value: lat,
        setValue: setLat,
        step: "0.1",
        min: -90,
        max: 90,
      },
      {
        label: "Longitude (deg)",
        value: lon,
        setValue: setLon,
        step: "0.1",
        min: -180,
        max: 180,
      },
      {
        label: "Depth (m)",
        value: depth,
        setValue: setDepth,
        step: "1",
        min: 0,
      },
      {
        label: "Temperature (C)",
        value: temperature,
        setValue: setTemperature,
        step: "0.1",
      },
      {
        label: "Salinity (PSU)",
        value: salinity,
        setValue: setSalinity,
        step: "0.1",
      },
      {
        label: "Oxygen (umol/kg)",
        value: oxygen,
        setValue: setOxygen,
        step: "0.1",
      },
      {
        label: "Chlorophyll (mg/m3)",
        value: chlorophyll,
        setValue: setChlorophyll,
        step: "0.01",
      },
    ],
    [lat, lon, depth, temperature, salinity, oxygen, chlorophyll]
  );

  const handlePredict = async () => {
    setIsLoading(true);
    setError(null);
    setData(null);
    setRisk(null);
    setSurvival(null);

    const payload = {
      latitude: parseFloat(lat),
      longitude: parseFloat(lon),
      depth: parseFloat(depth),
      temperature: parseFloat(temperature),
      salinity: parseFloat(salinity),
      oxygen: parseFloat(oxygen),
      chlorophyll: parseFloat(chlorophyll),
    };

    try {
      const response = await fetch("http://localhost:8000/predict/disaster", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const result = await response.json();
      setData(result);
      setRisk({
        level: result.disaster_prediction === "Anomaly" ? "High Risk" : "Low Risk",
        probability: Math.round(result.prediction_confidence * 100),
      });

      let estimatedSurvival = 100;
      const temp = parseFloat(temperature);
      const oxy = parseFloat(oxygen);
      const chl = parseFloat(chlorophyll);

      if (temp < 20 || temp > 25) {
        estimatedSurvival -= Math.abs(temp - 22.5) * 2;
      }
      if (oxy < 50) {
        estimatedSurvival -= (50 - oxy) * 2;
      } else if (oxy < 100) {
        estimatedSurvival -= (100 - oxy) * 0.5;
      }
      if (chl < 0.5) {
        estimatedSurvival -= (0.5 - chl) * 100;
      } else if (chl > 1.0) {
        estimatedSurvival -= (chl - 1.0) * 50;
      }

      estimatedSurvival = Math.max(0, Math.min(100, Math.round(estimatedSurvival)));

      setSurvival({
        level:
          estimatedSurvival > 70
            ? "High Survival"
            : estimatedSurvival > 40
              ? "Medium Survival"
              : "Low Survival",
        probability: estimatedSurvival,
      });
    } catch (predictionError) {
      setError(predictionError.message || "Failed to fetch prediction");
    } finally {
      setIsLoading(false);
    }
  };

  if (!user) {
    return (
      <PageShell>
        <div className="flex min-h-screen items-center justify-center px-4 py-10">
          <div className="premium-panel premium-panel-strong max-w-xl p-8 text-center">
            <AlertTriangle className="mx-auto h-12 w-12 text-rose-300" />
            <h1 className="mt-5 font-display text-4xl font-semibold text-white">
              Sign in required
            </h1>
            <p className="mt-4 text-base leading-7 text-slate-300">
              The prediction suite is part of the authenticated premium workspace.
              Sign in to test environmental conditions and anomaly risk scenarios.
            </p>
            <div className="mt-8 flex flex-col gap-3 sm:flex-row sm:justify-center">
              <button onClick={() => navigate("/auth")} className="premium-button">
                Go to Auth
              </button>
              <button onClick={() => navigate("/home")} className="premium-button-secondary">
                Back to Home
              </button>
            </div>
          </div>
        </div>
      </PageShell>
    );
  }

  return (
    <PageShell>
      <div className="mx-auto flex min-h-screen w-full max-w-[1400px] gap-6 px-4 py-6 sm:px-6 lg:px-8">
        <aside
          className={`premium-panel premium-panel-strong fixed inset-y-4 left-4 z-40 w-72 p-5 transition-transform lg:static lg:translate-x-0 ${
            sidebarOpen ? "translate-x-0" : "-translate-x-[120%]"
          }`}
        >
          <div className="flex items-center justify-between">
            <BrandMark compact />
            <button
              onClick={() => setSidebarOpen(false)}
              className="inline-flex h-10 w-10 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-white lg:hidden"
              aria-label="Close navigation"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          <div className="mt-8 space-y-3">
            {[
              { label: "FloatChat", icon: MessageCircle, path: "/floatchat" },
              { label: "Dashboard", icon: BarChart3, path: "/dashboard" },
              { label: "3D Globe", icon: Waves, path: "/visuals" },
            ].map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.path}
                  onClick={() => navigate(item.path)}
                  className="flex w-full items-center gap-3 rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 py-3 text-left text-sm font-medium text-slate-200 transition-colors hover:bg-white/[0.08]"
                >
                  <Icon className="h-4 w-4 text-cyan-100" />
                  {item.label}
                </button>
              );
            })}
          </div>

          <div className="mt-8 premium-divider" />

          <div className="mt-8 space-y-4">
            <div className="premium-card p-4">
              <p className="text-sm text-slate-300">Model scope</p>
              <p className="mt-2 font-display text-2xl font-semibold text-white">
                Anomaly + survival
              </p>
            </div>
            <div className="premium-card p-4">
              <p className="text-sm text-slate-300">Signal inputs</p>
              <p className="mt-2 font-display text-2xl font-semibold text-white">
                7 variables
              </p>
            </div>
          </div>
        </aside>

        {sidebarOpen ? (
          <button
            className="fixed inset-0 z-30 bg-slate-950/70 backdrop-blur-sm lg:hidden"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close sidebar overlay"
          />
        ) : null}

        <main className="min-w-0 flex-1 space-y-6">
          <section className="premium-panel premium-panel-strong p-6 sm:p-8">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <button
                  onClick={() => setSidebarOpen(true)}
                  className="mb-4 inline-flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-white lg:hidden"
                  aria-label="Open navigation"
                >
                  <Menu className="h-5 w-5" />
                </button>
                <p className="premium-kicker">Prediction Suite</p>
                <h1 className="mt-2 font-display text-4xl font-bold tracking-[-0.05em] text-white sm:text-5xl">
                  Premium environmental risk assessment.
                </h1>
                <p className="mt-4 max-w-3xl text-base leading-8 text-slate-300">
                  Feed in ocean conditions and test how the model reads anomaly risk
                  and aquatic-life survival. The UI is tuned for clarity so you can
                  focus on the signal, not the screen.
                </p>
              </div>

              <div className="flex flex-wrap gap-3">
                <span className="premium-chip">
                  <ShieldAlert className="h-4 w-4 text-cyan-100" />
                  Risk monitoring
                </span>
                <span className="premium-chip">
                  <HeartPulse className="h-4 w-4 text-cyan-100" />
                  Survival estimate
                </span>
              </div>
            </div>
          </section>

          <section className="premium-panel p-6 sm:p-8">
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.06]">
                <Compass className="h-5 w-5 text-cyan-100" />
              </div>
              <div>
                <p className="premium-kicker">Input Console</p>
                <h2 className="mt-1 font-display text-2xl font-semibold text-white">
                  Ocean condition inputs
                </h2>
              </div>
            </div>

            <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {fields.map((field) => (
                <label key={field.label} className="space-y-2">
                  <span className="text-sm font-medium text-slate-200">{field.label}</span>
                  <input
                    type="number"
                    value={field.value}
                    onChange={(event) =>
                      field.setValue(event.target.value ? parseFloat(event.target.value) : "")
                    }
                    step={field.step}
                    min={field.min}
                    max={field.max}
                    className="premium-input"
                  />
                </label>
              ))}
            </div>

            <button
              onClick={handlePredict}
              disabled={
                isLoading ||
                [lat, lon, depth, temperature, salinity, oxygen, chlorophyll].some(
                  (value) => value === ""
                )
              }
              className="mt-6 premium-button"
            >
              {isLoading ? (
                <>
                  <Sparkles className="h-4 w-4 animate-pulse" />
                  Analyzing conditions...
                </>
              ) : (
                <>
                  <Zap className="h-4 w-4" />
                  Predict Risk
                </>
              )}
            </button>
          </section>

          <div className="grid gap-6 xl:grid-cols-2">
            <ResultCard
              title="Disaster Risk Assessment"
              badge={risk?.level}
              icon={ShieldAlert}
            >
              {error ? (
                <div className="rounded-[22px] border border-rose-300/20 bg-rose-300/8 px-4 py-4 text-sm text-slate-100">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="h-4 w-4 text-rose-200" />
                    {error}
                  </div>
                </div>
              ) : data && risk ? (
                <div className="space-y-5">
                  <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                    <div className="premium-card p-4 sm:col-span-2 xl:col-span-1">
                      <p className="text-sm text-slate-300">Confidence</p>
                      <p className="mt-2 font-display text-3xl font-semibold text-white">
                        {risk.probability}%
                      </p>
                    </div>
                    <div className="premium-card p-4">
                      <ThermometerSun className="h-5 w-5 text-amber-200" />
                      <p className="mt-3 text-sm text-slate-300">
                        Temperature {data.temperature.toFixed(1)} C
                      </p>
                    </div>
                    <div className="premium-card p-4">
                      <Droplets className="h-5 w-5 text-sky-200" />
                      <p className="mt-3 text-sm text-slate-300">
                        Salinity {data.salinity.toFixed(1)} PSU
                      </p>
                    </div>
                    <div className="premium-card p-4">
                      <Waves className="h-5 w-5 text-emerald-200" />
                      <p className="mt-3 text-sm text-slate-300">
                        Chl-a {data.chlorophyll.toFixed(2)} mg/m3
                      </p>
                    </div>
                  </div>
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div className="premium-card p-4">
                      <div className="flex items-center gap-2 text-sm text-slate-200">
                        <MapPin className="h-4 w-4 text-cyan-100" />
                        Location
                      </div>
                      <p className="mt-3 text-sm text-slate-300">
                        Lat {data.latitude.toFixed(1)}, Lon {data.longitude.toFixed(1)}
                      </p>
                    </div>
                    <div className="premium-card p-4">
                      <div className="flex items-center gap-2 text-sm text-slate-200">
                        <Zap className="h-4 w-4 text-cyan-100" />
                        Oxygen and Depth
                      </div>
                      <p className="mt-3 text-sm text-slate-300">
                        Oxygen {data.oxygen.toFixed(1)} umol/kg, Depth {data.depth.toFixed(0)} m
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-sm leading-7 text-slate-300">
                  Enter ocean inputs above to generate a premium anomaly risk readout.
                </p>
              )}
            </ResultCard>

            <ResultCard
              title="Aquatic Life Survival"
              badge={survival?.level}
              icon={LifeBuoy}
            >
              {error ? (
                <div className="rounded-[22px] border border-rose-300/20 bg-rose-300/8 px-4 py-4 text-sm text-slate-100">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="h-4 w-4 text-rose-200" />
                    {error}
                  </div>
                </div>
              ) : data && survival ? (
                <div className="space-y-5">
                  <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                    <div className="premium-card p-4 sm:col-span-2 xl:col-span-1">
                      <p className="text-sm text-slate-300">Estimated survival</p>
                      <p className="mt-2 font-display text-3xl font-semibold text-white">
                        {survival.probability}%
                      </p>
                    </div>
                    <div className="premium-card p-4">
                      <ThermometerSun className="h-5 w-5 text-amber-200" />
                      <p className="mt-3 text-sm text-slate-300">
                        Temperature {data.temperature.toFixed(1)} C
                      </p>
                    </div>
                    <div className="premium-card p-4">
                      <Zap className="h-5 w-5 text-violet-200" />
                      <p className="mt-3 text-sm text-slate-300">
                        Oxygen {data.oxygen.toFixed(1)} umol/kg
                      </p>
                    </div>
                    <div className="premium-card p-4">
                      <Waves className="h-5 w-5 text-emerald-200" />
                      <p className="mt-3 text-sm text-slate-300">
                        Chl-a {data.chlorophyll.toFixed(2)} mg/m3
                      </p>
                    </div>
                  </div>

                  <div className="grid gap-4 sm:grid-cols-2">
                    <div className="premium-card p-4">
                      <div className="flex items-center gap-2 text-sm text-slate-200">
                        <MapPin className="h-4 w-4 text-cyan-100" />
                        Location
                      </div>
                      <p className="mt-3 text-sm text-slate-300">
                        Lat {data.latitude.toFixed(1)}, Lon {data.longitude.toFixed(1)}
                      </p>
                    </div>
                    <div className="premium-card p-4">
                      <div className="flex items-center gap-2 text-sm text-slate-200">
                        <Droplets className="h-4 w-4 text-cyan-100" />
                        Other metrics
                      </div>
                      <p className="mt-3 text-sm text-slate-300">
                        Salinity {data.salinity.toFixed(1)} PSU, Depth {data.depth.toFixed(0)} m
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-sm leading-7 text-slate-300">
                  Survival estimates appear here once a prediction request completes.
                </p>
              )}
            </ResultCard>
          </div>

          <section className="premium-panel p-6 sm:p-8">
            <p className="premium-kicker">Method</p>
            <h2 className="mt-2 font-display text-2xl font-semibold text-white">
              How this predictor reads the ocean
            </h2>
            <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              {[
                "Uses ARGO-derived temperature, salinity, oxygen, and chlorophyll signals.",
                "Runs a machine learning anomaly read for disaster-oriented risk assessment.",
                "Estimates marine-life survival from environmental tolerances and oxygen availability.",
                "Pairs model output with a calmer premium UI so interpretation is faster.",
              ].map((item) => (
                <div key={item} className="premium-card p-4">
                  <p className="text-sm leading-7 text-slate-300">{item}</p>
                </div>
              ))}
            </div>
          </section>
        </main>
      </div>
    </PageShell>
  );
}
