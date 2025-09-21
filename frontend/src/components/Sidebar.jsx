import { useState } from "react";
import { CalendarDaysIcon, BeakerIcon, MapPinIcon, HashtagIcon, ArrowPathIcon } from '@heroicons/react/24/outline';
import { 
  Filter, 
  TrendingUp, 
  Activity, 
  Globe, 
  Database,
  BarChart3,
  Map,
  Thermometer,
  Droplets,
  Wind
} from 'lucide-react';

function Sidebar({ onApply }) {
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [variable, setVariable] = useState("");
  const [region, setRegion] = useState("");
  const [floatId, setFloatId] = useState("");

  const apply = () => {
    onApply?.({ dateFrom, dateTo, variable, region, floatId });
  };

  return (
    <aside className="w-full lg:w-72 text-white bg-white/10 border border-white/20 rounded-2xl p-0 h-fit lg:sticky top-8 self-start shadow-lg hover:shadow-xl transition-all duration-300 relative z-20 pointer-events-auto">
      <div className="bg-gradient-to-b from-slate-900/70 to-slate-800/60 rounded-2xl ring-0 overflow-hidden">
        {/* Header */}
        <div className="px-4 py-4 border-b border-white/10 bg-gradient-to-r from-emerald-500/10 to-blue-500/10">
          <div className="flex items-center gap-2 mb-2">
            <Filter className="w-5 h-5 text-emerald-400" />
            <h2 className="text-sm uppercase tracking-wide text-emerald-300 font-semibold">Data Filters</h2>
          </div>
          <p className="text-xs text-blue-200">Refine your ocean data exploration</p>
        </div>

        <div className="p-4 space-y-4">
          {/* Date Range */}
          <div className="bg-white/5 rounded-lg p-3 border border-white/10">
            <label className="flex items-center gap-2 text-xs text-blue-200 mb-3 font-medium">
              <CalendarDaysIcon className="w-4 h-4" /> Date Range
            </label>
            <div className="grid grid-cols-2 gap-2">
              <input 
                type="date" 
                value={dateFrom} 
                onChange={(e)=>setDateFrom(e.target.value)} 
                className="bg-slate-900/80 border border-white/20 rounded-lg px-3 py-2 focus:ring-2 focus:ring-emerald-400 outline-none text-sm hover:border-white/30 transition-colors duration-200" 
              />
              <input 
                type="date" 
                value={dateTo} 
                onChange={(e)=>setDateTo(e.target.value)} 
                className="bg-slate-900/80 border border-white/20 rounded-lg px-3 py-2 focus:ring-2 focus:ring-emerald-400 outline-none text-sm hover:border-white/30 transition-colors duration-200" 
              />
            </div>
          </div>

          {/* Variable */}
          <div className="bg-white/5 rounded-lg p-3 border border-white/10">
            <label className="flex items-center gap-2 text-xs text-blue-200 mb-3 font-medium">
              <BeakerIcon className="w-4 h-4" /> Ocean Variables
            </label>
            <select 
              value={variable} 
              onChange={(e)=>setVariable(e.target.value)} 
              className="w-full bg-slate-900/80 text-white border border-white/20 rounded-lg px-3 py-2 focus:ring-2 focus:ring-emerald-400 outline-none text-sm hover:border-white/30 transition-colors duration-200"
            >
              <option value="">Select variable</option>
              <option className="text-slate-900" value="temperature">🌡️ Temperature</option>
              <option className="text-slate-900" value="salinity">🧂 Salinity</option>
              <option className="text-slate-900" value="oxygen">💨 Oxygen</option>
              <option className="text-slate-900" value="bgc">🔬 BGC</option>
            </select>
          </div>

          {/* Region */}
          <div className="bg-white/5 rounded-lg p-3 border border-white/10">
            <label className="flex items-center gap-2 text-xs text-blue-200 mb-3 font-medium">
              <MapPinIcon className="w-4 h-4" /> Ocean Regions
            </label>
            <select 
              value={region} 
              onChange={(e)=>setRegion(e.target.value)} 
              className="w-full bg-slate-900/80 text-white border border-white/20 rounded-lg px-3 py-2 focus:ring-2 focus:ring-emerald-400 outline-none text-sm hover:border-white/30 transition-colors duration-200"
            >
              <option value="">Select region</option>
              <option className="text-slate-900" value="north-atlantic">🌊 North Atlantic</option>
              <option className="text-slate-900" value="pacific">🌊 Pacific</option>
              <option className="text-slate-900" value="indian">🌊 Indian Ocean</option>
              <option className="text-slate-900" value="southern">🌊 Southern Ocean</option>
            </select>
          </div>

          {/* Float ID */}
          <div className="bg-white/5 rounded-lg p-3 border border-white/10">
            <label className="flex items-center gap-2 text-xs text-blue-200 mb-3 font-medium">
              <HashtagIcon className="w-4 h-4" /> Float ID
            </label>
            <input 
              value={floatId} 
              onChange={(e)=>setFloatId(e.target.value)} 
              placeholder="Search Float ID" 
              className="w-full bg-slate-900/80 border border-white/20 rounded-lg px-3 py-2 focus:ring-2 focus:ring-emerald-400 outline-none placeholder-blue-300/70 text-sm hover:border-white/30 transition-colors duration-200" 
            />
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2 pt-2">
            <button 
              onClick={apply} 
              className="flex-1 py-3 rounded-lg bg-gradient-to-r from-emerald-500 to-fuchsia-500 hover:from-emerald-400 hover:to-fuchsia-400 shadow-lg hover:shadow-xl transition-all duration-200 flex items-center justify-center gap-2 text-sm font-medium"
            >
              <Filter className="w-4 h-4" />
              Apply Filters
            </button>
            <button 
              onClick={()=>{setDateFrom('');setDateTo('');setVariable('');setRegion('');setFloatId('');}} 
              className="px-4 py-3 rounded-lg bg-white/10 hover:bg-white/20 ring-1 ring-white/10 hover:ring-white/20 transition-all duration-200" 
              title="Clear filters"
            >
              <ArrowPathIcon className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Quick Stats Section */}
        <div className="border-t border-white/10 bg-gradient-to-r from-blue-500/10 to-purple-500/10 p-4">
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="w-4 h-4 text-blue-400" />
            <h3 className="text-xs uppercase tracking-wide text-blue-300 font-semibold">Quick Stats</h3>
          </div>
          
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-white/5 rounded-lg p-2 border border-white/10">
              <div className="flex items-center gap-1 mb-1">
                <Activity className="w-3 h-3 text-emerald-400" />
                <span className="text-xs text-emerald-300 font-medium">Active</span>
              </div>
              <p className="text-lg font-bold text-emerald-300">3,847</p>
              <p className="text-xs text-emerald-200">Floats</p>
            </div>
            
            <div className="bg-white/5 rounded-lg p-2 border border-white/10">
              <div className="flex items-center gap-1 mb-1">
                <Globe className="w-3 h-3 text-blue-400" />
                <span className="text-xs text-blue-300 font-medium">Coverage</span>
              </div>
              <p className="text-lg font-bold text-blue-300">78%</p>
              <p className="text-xs text-blue-200">Global</p>
            </div>
          </div>
          
          <div className="mt-3 pt-3 border-t border-white/10">
            <div className="flex items-center justify-between text-xs">
              <span className="text-blue-200">Recent Activity</span>
              <span className="text-emerald-300 font-medium">+156 profiles</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}

export default Sidebar;


