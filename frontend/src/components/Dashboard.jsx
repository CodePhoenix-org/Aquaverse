import { useEffect, useMemo, useState } from "react";
import { useLocation } from "react-router-dom";
import {
  Activity,
  ArrowRight,
  BarChart3,
  Database,
  Eye,
  FileText,
  GitCompare,
  Globe,
  Map,
  MessageCircle,
  Table,
  Thermometer,
  Waves,
  X,
} from "lucide-react";
import { transformApiData } from "../utils/dataTransformers";
import Navbar from "./Navbar";
import Sidebar from "./Sidebar";
import ChatInterface from "./chat/ChatInterface";
import FloatMap from "./maps/FloatMap";
import DataPlots from "./plots/DataPlots";
import ProfileComparison from "./plots/ProfileComparison";
import DataTable from "./dashboard/DataTable";
import ErrorBoundary from "./ErrorBoundary";
import PageShell from "./ui/PageShell";

const viewOptions = [
  { id: "overview", label: "Overview", icon: Eye },
  { id: "map", label: "Map View", icon: Map },
  { id: "plots", label: "Data Plots", icon: BarChart3 },
  { id: "comparison", label: "Comparison", icon: GitCompare },
  { id: "table", label: "Data Table", icon: Table },
];

function DashboardPanel({ icon: Icon, title, description, children, className = "" }) {
  return (
    <section className={`premium-panel overflow-hidden p-5 sm:p-6 ${className}`}>
      <div className="mb-5 flex items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.06]">
              <Icon className="h-5 w-5 text-cyan-100" />
            </div>
            <h3 className="font-display text-xl font-semibold text-white">{title}</h3>
          </div>
          {description ? (
            <p className="mt-2 text-sm leading-6 text-slate-300">{description}</p>
          ) : null}
        </div>
      </div>
      <div>{children}</div>
    </section>
  );
}

export default function Dashboard() {
  const location = useLocation();
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [activeView, setActiveView] = useState(location.state?.vizTab || "overview");
  const [chatData, setChatData] = useState(location.state?.vizData || null);

  useEffect(() => {
    if (location.state?.vizData) {
      setChatData(location.state.vizData);
    }
    if (location.state?.vizTab) {
      setActiveView(location.state.vizTab);
    }
  }, [location.state]);

  const transformedData = useMemo(() => {
    if (!chatData) return null;
    return transformApiData(chatData);
  }, [chatData]);

  useEffect(() => {
    if (!transformedData) return;

    if (transformedData.type?.includes("profile")) {
      setActiveView("plots");
      return;
    }

    if (transformedData.floats) {
      setActiveView("map");
    }
  }, [transformedData]);

  const handleDataReceived = (data) => {
    setChatData(data);

    if (data?.type?.includes("profile")) {
      setActiveView("plots");
    } else if (data?.type?.includes("map")) {
      setActiveView("map");
    }
  };

  const metrics = [
    {
      title: "Active Floats",
      value: "3,847",
      sub: "Tracking across major basins",
      icon: Activity,
    },
    {
      title: "Global Coverage",
      value: "78%",
      sub: "Cross-ocean visibility",
      icon: Globe,
    },
    {
      title: "Profiles Today",
      value: "1,234",
      sub: "Fresh observational profiles",
      icon: BarChart3,
    },
    {
      title: "Temperature Range",
      value: "2.1C - 25.2C",
      sub: "Current working dataset",
      icon: Thermometer,
    },
  ];

  const renderCurrentView = () => {
    switch (activeView) {
      case "map":
        return (
          <DashboardPanel
            icon={Map}
            title="Interactive Float Map"
            description="Track float positions and trajectories within a calmer premium surface."
          >
            <ErrorBoundary>
              <FloatMap data={transformedData} />
            </ErrorBoundary>
          </DashboardPanel>
        );
      case "plots":
        return (
          <DashboardPanel
            icon={BarChart3}
            title="Ocean Data Visualizations"
            description="Inspect depth profiles, 3D structures, and analysis-ready chart views."
          >
            <ErrorBoundary>
              <DataPlots data={transformedData} />
            </ErrorBoundary>
          </DashboardPanel>
        );
      case "comparison":
        return (
          <DashboardPanel
            icon={GitCompare}
            title="Profile Comparison"
            description="Compare regions, periods, and parameter shapes without leaving the workspace."
          >
            <ErrorBoundary>
              <ProfileComparison data={transformedData} />
            </ErrorBoundary>
          </DashboardPanel>
        );
      case "table":
        return (
          <DashboardPanel
            icon={Table}
            title="Measurement Table"
            description="Review structured rows, filter detail, and export a cleaner subset."
          >
            <ErrorBoundary>
              <DataTable data={transformedData} />
            </ErrorBoundary>
          </DashboardPanel>
        );
      default:
        return (
          <div className="grid gap-6 xl:grid-cols-2">
            <DashboardPanel
              icon={Map}
              title="Spatial Overview"
              description="A premium geographic surface for float locations and drift patterns."
            >
              <ErrorBoundary>
                <FloatMap data={transformedData} />
              </ErrorBoundary>
            </DashboardPanel>
            <DashboardPanel
              icon={BarChart3}
              title="Signal Overview"
              description="Profile and chart context for the latest transformed ocean query."
            >
              <ErrorBoundary>
                <DataPlots data={transformedData} />
              </ErrorBoundary>
            </DashboardPanel>
          </div>
        );
    }
  };

  return (
    <PageShell>
      <Navbar onOpenChat={() => setIsChatOpen(true)} />

      <main className="mx-auto w-full max-w-[1400px] px-4 pb-16 pt-6 sm:px-6 lg:px-8">
        <div className="grid gap-6 xl:grid-cols-[20rem_1fr]">
          <Sidebar onApply={() => {}} />

          <div className="space-y-6">
            <section className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
              <div className="premium-panel premium-panel-strong relative overflow-hidden p-6 sm:p-8">
                <div className="absolute inset-x-12 top-0 h-40 rounded-full bg-cyan-300/12 blur-3xl" />
                <div className="relative">
                  <span className="premium-badge">
                    <Waves className="h-3.5 w-3.5" />
                    Workspace Overview
                  </span>
                  <h1 className="mt-5 font-display text-4xl font-bold tracking-[-0.05em] text-white sm:text-5xl">
                    Ocean intelligence, now presented like a premium command deck.
                  </h1>
                  <p className="mt-5 max-w-2xl text-base leading-8 text-slate-300">
                    FloatChat can send fresh data directly into this dashboard. Review
                    the signal through maps, plots, comparisons, and tables while the
                    upgraded interface keeps context and hierarchy clean.
                  </p>

                  <div className="mt-8 flex flex-wrap gap-3">
                    <button
                      onClick={() => setIsChatOpen(true)}
                      className="premium-button"
                    >
                      <MessageCircle className="h-4 w-4" />
                      Launch FloatChat
                    </button>
                    <button
                      onClick={() => setActiveView("map")}
                      className="premium-button-secondary"
                    >
                      Open Map View
                      <ArrowRight className="h-4 w-4" />
                    </button>
                  </div>

                  <div className="mt-8 grid gap-4 sm:grid-cols-3">
                    {[
                      transformedData
                        ? { label: "Data status", value: "Ready" }
                        : { label: "Data status", value: "Waiting" },
                      { label: "Current view", value: activeView },
                      { label: "Assistant", value: "Connected" },
                    ].map((item) => (
                      <div key={item.label} className="premium-card px-4 py-4">
                        <p className="text-sm text-slate-300">{item.label}</p>
                        <p className="mt-2 font-display text-2xl font-semibold text-white capitalize">
                          {item.value}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                {metrics.map((metric) => {
                  const Icon = metric.icon;

                  return (
                    <div key={metric.title} className="premium-card p-5">
                      <div className="flex items-center justify-between gap-3">
                        <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.06]">
                          <Icon className="h-5 w-5 text-cyan-100" />
                        </div>
                        <span className="premium-chip">Live</span>
                      </div>
                      <p className="mt-4 text-sm text-slate-300">{metric.title}</p>
                      <p className="mt-2 font-display text-3xl font-bold text-white">
                        {metric.value}
                      </p>
                      <p className="mt-2 text-sm text-slate-400">{metric.sub}</p>
                    </div>
                  );
                })}
              </div>
            </section>

            <section className="premium-panel sticky top-24 p-3">
              <div className="flex flex-wrap gap-3">
                {viewOptions.map((option) => {
                  const Icon = option.icon;
                  const active = activeView === option.id;

                  return (
                    <button
                      key={option.id}
                      onClick={() => setActiveView(option.id)}
                      className={`inline-flex items-center gap-2 rounded-full px-4 py-3 text-sm font-semibold transition-all ${
                        active
                          ? "bg-gradient-to-r from-cyan-300 to-sky-400 text-slate-950 shadow-lg"
                          : "border border-white/10 bg-white/[0.04] text-slate-200 hover:bg-white/[0.08]"
                      }`}
                    >
                      <Icon className="h-4 w-4" />
                      {option.label}
                    </button>
                  );
                })}
              </div>
            </section>

            {renderCurrentView()}

            <DashboardPanel
              icon={Database}
              title="Recent Profiles"
              description="A premium activity strip for high-value records and export actions."
            >
              <div className="overflow-x-auto scrollbar-thin">
                <table className="min-w-full text-left text-sm text-slate-200">
                  <thead className="border-b border-white/10 text-slate-400">
                    <tr>
                      <th className="px-4 py-3 font-medium">Profile ID</th>
                      <th className="px-4 py-3 font-medium">Float</th>
                      <th className="px-4 py-3 font-medium">Region</th>
                      <th className="px-4 py-3 font-medium">Date</th>
                      <th className="px-4 py-3 font-medium">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/[0.06]">
                    {[1, 2, 3, 4, 5].map((id) => (
                      <tr key={id} className="transition-colors hover:bg-white/[0.03]">
                        <td className="px-4 py-4 font-medium text-cyan-100">PRF{id}</td>
                        <td className="px-4 py-4">Float {1240 + id}</td>
                        <td className="px-4 py-4 text-slate-300">North Atlantic</td>
                        <td className="px-4 py-4 text-slate-400">2024-06-0{id}</td>
                        <td className="px-4 py-4">
                          <button className="premium-button-secondary rounded-full px-4 py-2 text-xs">
                            <FileText className="h-3.5 w-3.5" />
                            Export
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </DashboardPanel>
          </div>
        </div>
      </main>

      <button
        onClick={() => setIsChatOpen(true)}
        className="fixed bottom-6 right-6 z-40 flex h-16 w-16 items-center justify-center rounded-full border border-cyan-200/20 bg-gradient-to-br from-cyan-300 to-sky-500 text-slate-950 shadow-[0_28px_55px_rgba(14,165,233,0.38)] transition-transform hover:scale-105"
        aria-label="Open chat"
      >
        <MessageCircle className="h-6 w-6" />
      </button>

      {isChatOpen ? (
        <div className="fixed inset-0 z-[999]">
          <button
            className="absolute inset-0 bg-slate-950/75 backdrop-blur-sm"
            onClick={() => setIsChatOpen(false)}
            aria-label="Close chat overlay"
          />
          <div className="absolute inset-x-4 bottom-4 top-4 mx-auto flex max-w-5xl flex-col overflow-hidden rounded-[30px] border border-white/[0.12] bg-[linear-gradient(180deg,rgba(4,15,28,0.96),rgba(7,24,42,0.96))] shadow-[0_45px_120px_rgba(0,10,25,0.62)]">
            <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
              <div>
                <p className="premium-kicker">FloatChat Overlay</p>
                <h3 className="mt-1 font-display text-2xl font-semibold text-white">
                  Ask, route, and visualize
                </h3>
              </div>
              <button
                onClick={() => setIsChatOpen(false)}
                className="inline-flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-white"
                aria-label="Close chat"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="min-h-0 flex-1">
              <ChatInterface
                onDataReceived={handleDataReceived}
                onCloseChat={() => setIsChatOpen(false)}
              />
            </div>
          </div>
        </div>
      ) : null}
    </PageShell>
  );
}
