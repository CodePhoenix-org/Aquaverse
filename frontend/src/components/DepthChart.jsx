import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";

// data shape: [{ depth: number, temperature?: number, salinity?: number }]
function DepthChart({ data, variable = "temperature" }) {
  return (
    <div className="h-[320px] rounded-lg bg-slate-800/60 border border-white/10 p-2">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="depth" stroke="#cbd5e1" label={{ value: "Depth (dbar)", position: "insideBottom", offset: -2, fill: "#cbd5e1" }} />
          <YAxis stroke="#cbd5e1" label={{ value: variable === "temperature" ? "Temp (°C)" : "Salinity (PSU)", angle: -90, position: "insideLeft", fill: "#cbd5e1" }} />
          <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid rgba(255,255,255,0.1)", color: "#e2e8f0" }} />
          <Legend />
          <Line type="monotone" dataKey={variable} stroke={variable === "temperature" ? "#f59e0b" : "#38bdf8"} dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default DepthChart;


