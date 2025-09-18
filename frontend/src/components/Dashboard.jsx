import React, { useState } from "react";
import Navbar from "./Navbar";
import Sidebar from "./Sidebar";
import ChatInterface from "./chat/ChatInterface";
import FloatMap from "./maps/FloatMap";
import DataPlots from "./plots/DataPlots";
import ProfileComparison from "./plots/ProfileComparison";
import DataTable from "./dashboard/DataTable";

const Dashboard = () => {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [activeView, setActiveView] = useState('overview');
  const [chatData, setChatData] = useState(null);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-900 text-white">
      <Navbar onOpenChat={() => setIsChatOpen(true)} />

      <div className="w-full mx-auto px-2 md:px-8 py-6 grid grid-cols-1 lg:grid-cols-[18rem_1fr] gap-8">
        <Sidebar onApply={() => {}} />

        <main className="space-y-8">
          {/* KPI Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-6">
            {[
              { title: "Active Floats", value: "3,847", sub: "+12 today" },
              { title: "Global Coverage", value: "78%", sub: "All oceans" },
              { title: "Profiles Today", value: "1,234", sub: "+156 vs yesterday" },
              { title: "Temperature Range", value: "2.1°C - 25.2°C", sub: "Current dataset" }
            ].map((c, i) => (
              <div key={i} className="bg-gradient-to-br from-emerald-800/30 to-fuchsia-800/30 rounded-2xl p-6 backdrop-blur-md ring-1 ring-white/10 shadow-lg">
                <p className="text-sm text-blue-200">{c.title}</p>
                <p className="text-3xl font-bold mt-2 text-emerald-300">{c.value}</p>
                <p className="text-xs text-purple-200 mt-1">{c.sub}</p>
              </div>
            ))}
          </div>

          {/* Views Toolbar */}
          <div className="flex flex-wrap gap-3 sticky top-0 z-10 bg-transparent py-2">
            {[
              {id:'overview', label:'Overview'},
              {id:'map', label:'Map View'},
              {id:'plots', label:'Data Plots'},
              {id:'comparison', label:'Profile Comparison'},
              {id:'table', label:'Data Table'}
            ].map(btn => (
              <button
                key={btn.id}
                onClick={()=>setActiveView(btn.id)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition ${activeView===btn.id? 'bg-gradient-to-r from-emerald-500 to-fuchsia-500 text-white shadow' : 'bg-white/10 hover:bg-white/20 ring-1 ring-white/10'}`}
              >{btn.label}</button>
            ))}
          </div>

          {/* Views Content */}
          {activeView === 'overview' && (
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
              <section className="relative z-0 bg-white/10 rounded-2xl p-4 overflow-hidden ring-1 ring-white/10 shadow-md">
                <FloatMap key={`overview-${activeView}`} data={chatData} />
              </section>
              <section className="bg-white/10 rounded-2xl p-4 overflow-hidden ring-1 ring-white/10 shadow-md">
                <DataPlots data={chatData} />
              </section>
            </div>
          )}

          {activeView === 'map' && (
            <section className="relative z-0 bg-white/10 rounded-2xl p-6 overflow-hidden ring-1 ring-white/10 shadow-md" style={{minHeight:'600px', height:'600px'}}>
              <div className="h-full w-full">
                <FloatMap key={`map-${activeView}`} data={chatData} />
              </div>
            </section>
          )}

          {activeView === 'plots' && (
            <section className="bg-white/10 rounded-2xl p-6 overflow-hidden ring-1 ring-white/10 shadow-md" style={{minHeight:'620px'}}>
              <DataPlots data={chatData} />
            </section>
          )}

          {activeView === 'comparison' && (
            <section className="bg-white/10 rounded-2xl p-6 overflow-hidden ring-1 ring-white/10 shadow-md" style={{minHeight:'620px'}}>
              <ProfileComparison data={chatData} />
            </section>
          )}

          {activeView === 'table' && (
            <section className="bg-white/10 rounded-2xl p-6 overflow-hidden ring-1 ring-white/10 shadow-md" style={{minHeight:'620px'}}>
              <DataTable data={chatData} />
            </section>
          )}

          <section className="bg-white/10 border border-white/20 rounded-2xl p-6 shadow-md">
            <h3 className="text-lg font-semibold mb-3">Recent Profiles</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead className="text-blue-200">
                  <tr>
                    <th className="text-left px-3 py-2">Profile ID</th>
                    <th className="text-left px-3 py-2">Float</th>
                    <th className="text-left px-3 py-2">Region</th>
                    <th className="text-left px-3 py-2">Date</th>
                    <th className="text-left px-3 py-2">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/10">
                  {[1,2,3,4,5].map((id)=> (
                    <tr key={id}>
                      <td className="px-3 py-2">PRF{id}</td>
                      <td className="px-3 py-2">Float {1240 + id}</td>
                      <td className="px-3 py-2">North Atlantic</td>
                      <td className="px-3 py-2">2024-06-0{id}</td>
                      <td className="px-3 py-2 flex gap-2">
                        <button className="px-2 py-1 rounded bg-white/10 hover:bg-white/20 text-xs">CSV</button>
                        <button className="px-2 py-1 rounded bg-white/10 hover:bg-white/20 text-xs">NetCDF</button>
                        <button className="px-2 py-1 rounded bg-white/10 hover:bg-white/20 text-xs">ASCII</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </main>
      </div>

      {/* Floating Chat Widget */}
      <button
        onClick={() => setIsChatOpen(true)}
        className="fixed bottom-6 right-6 w-14 h-14 rounded-full bg-gradient-to-r from-cyan-500 to-blue-600 shadow-2xl border border-white/20 flex items-center justify-center hover:scale-105 transition"
        aria-label="Open Chat"
      >
        💬
      </button>

      {/* Chat Modal (uses your ChatInterface and returns data) */}
      {isChatOpen && (
        <div className="fixed inset-0 z-[10000]">
          <div className="absolute inset-0 bg-black/50" onClick={() => setIsChatOpen(false)} />
          <div className="absolute inset-x-2 md:inset-x-auto md:right-8 top-8 bottom-8 md:w-[880px] bg-white rounded-2xl shadow-2xl flex flex-col overflow-hidden">
            <div className="flex items-center justify-between p-3 border-b border-gray-200 bg-gradient-to-r from-emerald-600 to-purple-600 text-white">
              <div>
                <h4 className="font-semibold">FloatChat Assistant</h4>
                <p className="text-xs text-emerald-100">AI Powered</p>
              </div>
              <button onClick={() => setIsChatOpen(false)} className="px-3 py-1 rounded bg-white/20 hover:bg-white/30 text-sm">Close</button>
            </div>
          <div className="flex-1 overflow-hidden">
            <ChatInterface onDataReceived={setChatData} />
          </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
