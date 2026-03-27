import { useState } from "react";
import {
  CalendarDaysIcon,
  HashtagIcon,
  MapPinIcon,
} from "@heroicons/react/24/outline";
import {
  Activity,
  Filter,
  Globe,
  RotateCcw,
  Sparkles,
  Thermometer,
  Waves,
} from "lucide-react";

const variableOptions = [
  { value: "", label: "All variables" },
  { value: "temperature", label: "Temperature" },
  { value: "salinity", label: "Salinity" },
  { value: "oxygen", label: "Oxygen" },
  { value: "bgc", label: "BGC" },
];

const regionOptions = [
  { value: "", label: "All regions" },
  { value: "north-atlantic", label: "North Atlantic" },
  { value: "pacific", label: "Pacific" },
  { value: "indian", label: "Indian Ocean" },
  { value: "southern", label: "Southern Ocean" },
];

export default function Sidebar({ onApply }) {
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [variable, setVariable] = useState("");
  const [region, setRegion] = useState("");
  const [floatId, setFloatId] = useState("");

  const apply = () => {
    onApply?.({ dateFrom, dateTo, variable, region, floatId });
  };

  const clearFilters = () => {
    setDateFrom("");
    setDateTo("");
    setVariable("");
    setRegion("");
    setFloatId("");
  };

  return (
    <aside className="premium-panel premium-panel-strong h-fit overflow-hidden lg:sticky lg:top-28">
      <div className="border-b border-white/10 bg-white/[0.03] px-5 py-5">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-cyan-300/16 bg-cyan-300/10">
            <Filter className="h-5 w-5 text-cyan-100" />
          </div>
          <div>
            <p className="premium-kicker">Control Panel</p>
            <h2 className="mt-1 font-display text-xl font-semibold text-white">
              Refine the signal
            </h2>
          </div>
        </div>
        <p className="mt-4 text-sm leading-6 text-slate-300">
          Tighten the ocean lens before you map, chart, compare, or export.
        </p>
      </div>

      <div className="space-y-5 px-5 py-5">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1">
          <label className="space-y-2">
            <span className="flex items-center gap-2 text-sm font-medium text-slate-200">
              <CalendarDaysIcon className="h-4 w-4" />
              Start date
            </span>
            <input
              type="date"
              value={dateFrom}
              onChange={(event) => setDateFrom(event.target.value)}
              className="premium-input"
            />
          </label>

          <label className="space-y-2">
            <span className="flex items-center gap-2 text-sm font-medium text-slate-200">
              <CalendarDaysIcon className="h-4 w-4" />
              End date
            </span>
            <input
              type="date"
              value={dateTo}
              onChange={(event) => setDateTo(event.target.value)}
              className="premium-input"
            />
          </label>
        </div>

        <label className="space-y-2">
          <span className="flex items-center gap-2 text-sm font-medium text-slate-200">
            <Thermometer className="h-4 w-4" />
            Variable
          </span>
          <select
            value={variable}
            onChange={(event) => setVariable(event.target.value)}
            className="premium-select"
          >
            {variableOptions.map((option) => (
              <option key={option.value || "all"} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        <label className="space-y-2">
          <span className="flex items-center gap-2 text-sm font-medium text-slate-200">
            <MapPinIcon className="h-4 w-4" />
            Region
          </span>
          <select
            value={region}
            onChange={(event) => setRegion(event.target.value)}
            className="premium-select"
          >
            {regionOptions.map((option) => (
              <option key={option.value || "all"} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        <label className="space-y-2">
          <span className="flex items-center gap-2 text-sm font-medium text-slate-200">
            <HashtagIcon className="h-4 w-4" />
            Float ID
          </span>
          <input
            value={floatId}
            onChange={(event) => setFloatId(event.target.value)}
            placeholder="Search by float identifier"
            className="premium-input"
          />
        </label>

        <div className="flex gap-3">
          <button onClick={apply} className="premium-button flex-1">
            <Sparkles className="h-4 w-4" />
            Apply
          </button>
          <button onClick={clearFilters} className="premium-button-secondary">
            <RotateCcw className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="border-t border-white/10 bg-white/[0.03] px-5 py-5">
        <p className="premium-kicker">Live Summary</p>
        <div className="mt-4 grid gap-3 sm:grid-cols-3 lg:grid-cols-1">
          <div className="premium-card p-4">
            <div className="flex items-center gap-2 text-sm text-slate-300">
              <Activity className="h-4 w-4 text-emerald-300" />
              Active floats
            </div>
            <p className="mt-3 font-display text-3xl font-bold text-white">3,847</p>
          </div>

          <div className="premium-card p-4">
            <div className="flex items-center gap-2 text-sm text-slate-300">
              <Globe className="h-4 w-4 text-sky-300" />
              Coverage
            </div>
            <p className="mt-3 font-display text-3xl font-bold text-white">78%</p>
          </div>

          <div className="premium-card p-4">
            <div className="flex items-center gap-2 text-sm text-slate-300">
              <Waves className="h-4 w-4 text-cyan-300" />
              Fresh profiles
            </div>
            <p className="mt-3 font-display text-3xl font-bold text-white">+156</p>
          </div>
        </div>
      </div>
    </aside>
  );
}
