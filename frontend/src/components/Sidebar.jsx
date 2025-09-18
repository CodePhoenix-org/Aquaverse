import { useState } from "react";
import { CalendarDaysIcon, BeakerIcon, MapPinIcon, HashtagIcon, ArrowPathIcon } from '@heroicons/react/24/outline';

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
    <aside className="w-full lg:w-72 text-white bg-white/10 border border-white/20 rounded-2xl p-0 h-fit lg:sticky top-8 self-start shadow-md">
      <div className="bg-gradient-to-b from-slate-900/70 to-slate-800/60 rounded-2xl ring-0 overflow-hidden">
        <div className="px-4 py-3 border-b border-white/10 sticky top-0 backdrop-blur supports-[backdrop-filter]:bg-slate-900/60">
          <h2 className="text-sm uppercase tracking-wide text-emerald-300">Data Filters</h2>
        </div>

        <div className="p-4 space-y-4">
          {/* Date Range */}
          <div>
            <label className="flex items-center gap-2 text-xs text-blue-200 mb-2">
              <CalendarDaysIcon className="w-4 h-4" /> Date Range
            </label>
            <div className="grid grid-cols-2 gap-2">
              <input type="date" value={dateFrom} onChange={(e)=>setDateFrom(e.target.value)} className="bg-slate-900/80 border border-white/20 rounded-md px-2 py-2 focus:ring-2 focus:ring-emerald-400 outline-none" />
              <input type="date" value={dateTo} onChange={(e)=>setDateTo(e.target.value)} className="bg-slate-900/80 border border-white/20 rounded-md px-2 py-2 focus:ring-2 focus:ring-emerald-400 outline-none" />
            </div>
          </div>

          {/* Variable */}
          <div>
            <label className="flex items-center gap-2 text-xs text-blue-200 mb-2">
              <BeakerIcon className="w-4 h-4" /> Variable
            </label>
            <select value={variable} onChange={(e)=>setVariable(e.target.value)} className="w-full bg-slate-900/80 text-white border border-white/20 rounded-md px-2 py-2 focus:ring-2 focus:ring-emerald-400 outline-none">
              <option value="">Select variable</option>
              <option className="text-slate-900" value="temperature">Temperature</option>
              <option className="text-slate-900" value="salinity">Salinity</option>
              <option className="text-slate-900" value="oxygen">Oxygen</option>
              <option className="text-slate-900" value="bgc">BGC</option>
            </select>
          </div>

          {/* Region */}
          <div>
            <label className="flex items-center gap-2 text-xs text-blue-200 mb-2">
              <MapPinIcon className="w-4 h-4" /> Region
            </label>
            <select value={region} onChange={(e)=>setRegion(e.target.value)} className="w-full bg-slate-900/80 text-white border border-white/20 rounded-md px-2 py-2 focus:ring-2 focus:ring-emerald-400 outline-none">
              <option value="">Select region</option>
              <option className="text-slate-900" value="north-atlantic">North Atlantic</option>
              <option className="text-slate-900" value="pacific">Pacific</option>
              <option className="text-slate-900" value="indian">Indian Ocean</option>
              <option className="text-slate-900" value="southern">Southern Ocean</option>
            </select>
          </div>

          {/* Float ID */}
          <div>
            <label className="flex items-center gap-2 text-xs text-blue-200 mb-2">
              <HashtagIcon className="w-4 h-4" /> Float ID
            </label>
            <input value={floatId} onChange={(e)=>setFloatId(e.target.value)} placeholder="Search Float ID" className="w-full bg-slate-900/80 border border-white/20 rounded-md px-2 py-2 focus:ring-2 focus:ring-emerald-400 outline-none placeholder-blue-300/70" />
          </div>

          <div className="flex gap-2 pt-2">
            <button onClick={apply} className="flex-1 py-2 rounded-md bg-gradient-to-r from-emerald-500 to-fuchsia-500 hover:from-emerald-400 hover:to-fuchsia-400 shadow">
              Apply Filters
            </button>
            <button onClick={()=>{setDateFrom('');setDateTo('');setVariable('');setRegion('');setFloatId('');}} className="px-3 rounded-md bg-white/10 hover:bg-white/20 ring-1 ring-white/10" title="Clear filters">
              <ArrowPathIcon className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
    </aside>
  );
}

export default Sidebar;


