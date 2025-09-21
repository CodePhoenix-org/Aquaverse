import React, { useEffect, useMemo, useState } from "react";
import { useLocation } from 'react-router-dom';
import { transformApiData } from "../utils/dataTransformers";
import Navbar from "./Navbar";
import Sidebar from "./Sidebar";
import ChatInterface from "./chat/ChatInterface";
import FloatMap from "./maps/FloatMap";
import DataPlots from "./plots/DataPlots";
import ProfileComparison from "./plots/ProfileComparison";
import DataTable from "./dashboard/DataTable";
import ErrorBoundary from "./ErrorBoundary";
import { 
  Activity, 
  Globe, 
  BarChart3, 
  Thermometer,
  Map,
  BarChart,
  GitCompare,
  Table,
  Eye,
  Download,
  FileText,
  Database,
  X
} from "lucide-react";

const Dashboard = () => {
  const location = useLocation();

  const [isChatOpen, setIsChatOpen] = useState(false);
  // If navigation state has vizTab, use it as initial tab, else 'overview'
  const [activeView, setActiveView] = useState(location.state?.vizTab || 'overview');
  // If navigation state has vizData, use it as initial data, else null
  const [chatData, setChatData] = useState(location.state?.vizData || null);
  useEffect(() => {
    // If navigation state changes (e.g., user comes from FloatChat), update state
    if (location.state?.vizData) setChatData(location.state.vizData);
    if (location.state?.vizTab) setActiveView(location.state.vizTab);
    // eslint-disable-next-line
  }, [location.state]);
  useEffect(() => {
    console.log('Chat data received:', chatData);
    console.log('Type of chat data:', typeof chatData);
  }, [chatData]);



  // Transform received data
  const transformedData = useMemo(() => {
    if (!chatData) return null;
    return transformApiData(chatData);
  }, [chatData]);

  // Handle data received from chat
  const handleDataReceived = (data) => {
    console.log('Raw data received:', data);
    setChatData(data);
    // Auto-switch view based on data type
    if (data && data.type) {
      if (data.type.includes('profile')) {
        setActiveView('plots');
      } else if (data.type.includes('map')) {
        setActiveView('map');
      }
    }
  };

  // Auto-switch view based on data type
  useEffect(() => {
    if (!transformedData) return;

    if (transformedData.type?.includes('profile')) {
      setActiveView('plots');
    } else if (transformedData.floats) {
      setActiveView('map');
    }
  }, [transformedData]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-900 text-white">
      <div className="relative z-50 pointer-events-auto">
        <Navbar onOpenChat={() => setIsChatOpen(true)} />
      </div>

      <div className="w-full mx-auto px-2 md:px-8 py-6 grid grid-cols-1 lg:grid-cols-[18rem_1fr] gap-8 relative z-10">
        <Sidebar onApply={() => {}} />

        <main className="space-y-8">
          {/* KPI Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-6">
            {[
              { 
                title: "Active Floats", 
                value: "3,847", 
                sub: "+12 today",
                icon: Activity,
                gradient: "from-emerald-500/20 to-emerald-600/30",
                iconColor: "text-emerald-400",
                valueColor: "text-emerald-300"
              },
              { 
                title: "Global Coverage", 
                value: "78%", 
                sub: "All oceans",
                icon: Globe,
                gradient: "from-blue-500/20 to-blue-600/30",
                iconColor: "text-blue-400",
                valueColor: "text-blue-300"
              },
              { 
                title: "Profiles Today", 
                value: "1,234", 
                sub: "+156 vs yesterday",
                icon: BarChart3,
                gradient: "from-purple-500/20 to-purple-600/30",
                iconColor: "text-purple-400",
                valueColor: "text-purple-300"
              },
              { 
                title: "Temperature Range", 
                value: "2.1°C - 25.2°C", 
                sub: "Current dataset",
                icon: Thermometer,
                gradient: "from-orange-500/20 to-orange-600/30",
                iconColor: "text-orange-400",
                valueColor: "text-orange-300"
              }
            ].map((c, i) => {
              const IconComponent = c.icon;
              return (
                <div 
                  key={i} 
                  className={`bg-gradient-to-br ${c.gradient} rounded-2xl p-6 backdrop-blur-md ring-1 ring-white/10 shadow-lg hover:shadow-xl hover:scale-[1.02] transition-all duration-300 group relative overflow-hidden`}
                >
                  {/* Subtle animated background pattern */}
                  <div className="absolute inset-0 opacity-5 group-hover:opacity-10 transition-opacity duration-300">
                    <div className="absolute top-0 right-0 w-20 h-20 bg-white rounded-full -translate-y-10 translate-x-10"></div>
                    <div className="absolute bottom-0 left-0 w-16 h-16 bg-white rounded-full translate-y-8 -translate-x-8"></div>
                  </div>
                  
                  <div className="relative z-10">
                    <div className="flex items-center justify-between mb-3">
                      <IconComponent className={`w-6 h-6 ${c.iconColor} group-hover:scale-110 transition-transform duration-300`} />
                      <div className="w-2 h-2 bg-white/20 rounded-full group-hover:bg-white/40 transition-colors duration-300"></div>
                    </div>
                    
                    <p className="text-sm text-blue-200 font-medium mb-2">{c.title}</p>
                    <p className={`text-3xl font-bold mb-1 ${c.valueColor} group-hover:scale-105 transition-transform duration-300`}>
                      {c.value}
                    </p>
                    <p className="text-xs text-purple-200 opacity-80">{c.sub}</p>
                  </div>
              </div>
              );
            })}
          </div>

          {/* Views Toolbar */}
          <div className="flex flex-wrap gap-3 sticky top-4 z-10 bg-slate-950/80 backdrop-blur-md py-4 -mx-2 px-2 rounded-lg border border-white/5">
            {[
              {id:'overview', label:'Overview', icon: Eye},
              {id:'map', label:'Map View', icon: Map},
              {id:'plots', label:'Data Plots', icon: BarChart},
              {id:'comparison', label:'Profile Comparison', icon: GitCompare},
              {id:'table', label:'Data Table', icon: Table}
            ].map(btn => {
              const IconComponent = btn.icon;
              return (
              <button
                key={btn.id}
                onClick={()=>setActiveView(btn.id)}
                  className={`px-4 py-3 rounded-xl text-sm font-medium transition-all duration-300 flex items-center gap-2 group ${
                    activeView === btn.id 
                      ? 'bg-gradient-to-r from-emerald-500 to-fuchsia-500 text-white shadow-lg shadow-emerald-500/25 scale-105' 
                      : 'bg-white/10 hover:bg-white/20 ring-1 ring-white/10 hover:ring-white/20 hover:scale-105'
                  }`}
                >
                  <IconComponent className={`w-4 h-4 transition-transform duration-300 ${
                    activeView === btn.id ? 'scale-110' : 'group-hover:scale-110'
                  }`} />
                  <span>{btn.label}</span>
                  {activeView === btn.id && (
                    <div className="w-1 h-1 bg-white rounded-full animate-pulse"></div>
                  )}
                </button>
              );
            })}
          </div>

          {/* Views Content */}



          {activeView === 'overview' && (
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
              <section className="relative z-0 bg-gradient-to-br from-white/10 to-white/5 rounded-2xl p-6 overflow-hidden ring-1 ring-white/10 shadow-lg hover:shadow-xl transition-all duration-300 group">
                <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <div className="relative z-10">
                  <div className="flex items-center gap-2 mb-4">
                    <Map className="w-5 h-5 text-emerald-400" />
                    <h3 className="text-lg font-semibold text-white">Global Float Map</h3>
                  </div>
                <ErrorBoundary>
                  <FloatMap key={`overview-${activeView}`} data={transformedData} />
                </ErrorBoundary>
                </div>
              </section>
              <section className="bg-gradient-to-br from-white/10 to-white/5 rounded-2xl p-6 overflow-hidden ring-1 ring-white/10 shadow-lg hover:shadow-xl transition-all duration-300 group">
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <div className="relative z-10">
                  <div className="flex items-center gap-2 mb-4">
                    <BarChart className="w-5 h-5 text-blue-400" />
                    <h3 className="text-lg font-semibold text-white">Data Visualization</h3>
                  </div>
                <ErrorBoundary>
                  <DataPlots data={transformedData} />
                </ErrorBoundary>
                </div>
              </section>
            </div>
          )}



          {activeView === 'map' && (
            <section className="relative z-0 bg-gradient-to-br from-white/10 to-white/5 rounded-2xl p-6 overflow-hidden ring-1 ring-white/10 shadow-lg hover:shadow-xl transition-all duration-300 group" style={{minHeight:'600px', height:'600px'}}>
              <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              <div className="relative z-10 h-full w-full">
                <div className="flex items-center gap-2 mb-4">
                  <Map className="w-5 h-5 text-emerald-400" />
                  <h3 className="text-lg font-semibold text-white">Interactive Float Map</h3>
                </div>
                <div className="h-[calc(100%-3rem)] w-full">
                <ErrorBoundary>
                  <FloatMap key={`map-${activeView}`} data={transformedData} />
                </ErrorBoundary>
                </div>
              </div>
            </section>
          )}



          {activeView === 'plots' && (
            <section className="bg-gradient-to-br from-white/10 to-white/5 rounded-2xl p-6 overflow-hidden ring-1 ring-white/10 shadow-lg hover:shadow-xl transition-all duration-300 group" style={{minHeight:'620px'}}>
              <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              <div className="relative z-10">
                <div className="flex items-center gap-2 mb-4">
                  <BarChart className="w-5 h-5 text-blue-400" />
                  <h3 className="text-lg font-semibold text-white">Data Analysis Plots</h3>
                </div>
              <ErrorBoundary>
                <DataPlots data={transformedData} />
              </ErrorBoundary>
              </div>
            </section>
          )}



          {activeView === 'comparison' && (
            <section className="bg-gradient-to-br from-white/10 to-white/5 rounded-2xl p-6 overflow-hidden ring-1 ring-white/10 shadow-lg hover:shadow-xl transition-all duration-300 group" style={{minHeight:'620px'}}>
              <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              <div className="relative z-10">
                <div className="flex items-center gap-2 mb-4">
                  <GitCompare className="w-5 h-5 text-purple-400" />
                  <h3 className="text-lg font-semibold text-white">Profile Comparison</h3>
                </div>
              <ErrorBoundary>
                <ProfileComparison data={transformedData} />
              </ErrorBoundary>
              </div>
            </section>
          )}



          {activeView === 'table' && (
            <section className="bg-gradient-to-br from-white/10 to-white/5 rounded-2xl p-6 overflow-hidden ring-1 ring-white/10 shadow-lg hover:shadow-xl transition-all duration-300 group" style={{minHeight:'620px'}}>
              <div className="absolute inset-0 bg-gradient-to-br from-orange-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              <div className="relative z-10">
                <div className="flex items-center gap-2 mb-4">
                  <Table className="w-5 h-5 text-orange-400" />
                  <h3 className="text-lg font-semibold text-white">Data Table</h3>
                </div>
              <ErrorBoundary>
                <DataTable data={transformedData} />
              </ErrorBoundary>
              </div>
            </section>
          )}

          <section className="bg-gradient-to-br from-white/10 to-white/5 border border-white/20 rounded-2xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 group">
            <div className="flex items-center gap-2 mb-6">
              <Database className="w-5 h-5 text-cyan-400" />
              <h3 className="text-lg font-semibold text-white">Recent Profiles</h3>
              <div className="ml-auto flex items-center gap-2">
                <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
                <span className="text-xs text-emerald-300">Live Data</span>
              </div>
            </div>
            
            <div className="overflow-x-auto scrollbar-thin">
              <table className="min-w-full text-sm">
                <thead className="text-blue-200">
                  <tr className="border-b border-white/10">
                    <th className="text-left px-4 py-3 font-medium">Profile ID</th>
                    <th className="text-left px-4 py-3 font-medium">Float</th>
                    <th className="text-left px-4 py-3 font-medium">Region</th>
                    <th className="text-left px-4 py-3 font-medium">Date</th>
                    <th className="text-left px-4 py-3 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {[1,2,3,4,5].map((id)=> (
                    <tr key={id} className="hover:bg-white/5 transition-colors duration-200 group/row">
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 bg-emerald-400 rounded-full"></div>
                          <span className="font-mono text-emerald-300">PRF{id}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className="text-white">Float {1240 + id}</span>
                      </td>
                      <td className="px-4 py-3">
                        <span className="text-blue-200">North Atlantic</span>
                      </td>
                      <td className="px-4 py-3">
                        <span className="text-purple-200">2024-06-0{id}</span>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex gap-2">
                          <button className="px-3 py-1 rounded-lg bg-emerald-500/20 hover:bg-emerald-500/30 text-xs text-emerald-300 border border-emerald-500/30 hover:border-emerald-500/50 transition-all duration-200 flex items-center gap-1">
                            <FileText className="w-3 h-3" />
                            CSV
                          </button>
                          <button className="px-3 py-1 rounded-lg bg-blue-500/20 hover:bg-blue-500/30 text-xs text-blue-300 border border-blue-500/30 hover:border-blue-500/50 transition-all duration-200 flex items-center gap-1">
                            <Database className="w-3 h-3" />
                            NetCDF
                          </button>
                          <button className="px-3 py-1 rounded-lg bg-purple-500/20 hover:bg-purple-500/30 text-xs text-purple-300 border border-purple-500/30 hover:border-purple-500/50 transition-all duration-200 flex items-center gap-1">
                            <Download className="w-3 h-3" />
                            ASCII
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            <div className="mt-4 pt-4 border-t border-white/10">
              <div className="flex items-center justify-between">
                <span className="text-xs text-blue-200">Showing 5 of 1,234 profiles</span>
                <button className="px-4 py-2 rounded-lg bg-gradient-to-r from-emerald-500/20 to-blue-500/20 hover:from-emerald-500/30 hover:to-blue-500/30 text-sm text-white border border-white/20 hover:border-white/30 transition-all duration-200">
                  View All Profiles
                </button>
              </div>
            </div>
          </section>
        </main>
      </div>

      {/* Floating Chat Widget */}
      <button
        onClick={() => setIsChatOpen(true)}
        className="fixed bottom-6 right-6 w-16 h-16 rounded-full bg-gradient-to-r from-cyan-500 to-blue-600 shadow-2xl border border-white/20 flex items-center justify-center hover:scale-110 hover:shadow-cyan-500/25 transition-all duration-300 group z-50"
        aria-label="Open Chat"
      >
        <div className="relative">
          <div className="text-2xl group-hover:scale-110 transition-transform duration-300">💬</div>
          <div className="absolute -top-1 -right-1 w-4 h-4 bg-emerald-400 rounded-full animate-pulse border-2 border-white"></div>
        </div>
      </button>

      {/* Chat Modal (uses your ChatInterface and returns data) */}
      {isChatOpen && (
        <div className="fixed inset-0 z-[9999] animate-in fade-in duration-300">
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setIsChatOpen(false)} />
          <div className="absolute inset-x-2 md:inset-x-auto md:right-8 top-8 bottom-8 md:w-[880px] bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-2xl flex flex-col overflow-hidden border border-gray-200">
            <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-gradient-to-r from-emerald-600 to-purple-600 text-white">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center">
                  <span className="text-lg">🤖</span>
                </div>
                <div>
                  <h4 className="font-semibold text-lg">FloatChat Assistant</h4>
                  <p className="text-xs text-emerald-100 flex items-center gap-1">
                    <div className="w-2 h-2 bg-emerald-300 rounded-full animate-pulse"></div>
                    AI Powered Ocean Data Discovery
                  </p>
                </div>
              </div>
              <button 
                onClick={() => setIsChatOpen(false)} 
                className="p-2 rounded-lg bg-white/20 hover:bg-white/30 transition-all duration-200 hover:scale-105 group"
                title="Close chat"
              >
                <X className="w-5 h-5 text-white group-hover:text-red-200 transition-colors duration-200" />
              </button>
            </div>
            <div className="flex-1 overflow-hidden">
              <ChatInterface onDataReceived={handleDataReceived} onCloseChat={() => setIsChatOpen(false)} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
