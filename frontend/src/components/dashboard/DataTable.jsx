import { useEffect, useMemo, useState } from "react";
import { ArrowDownTrayIcon, FunnelIcon } from "@heroicons/react/24/outline";

const sampleTableData = [
  {
    id: 1,
    floatId: "ARGO_001",
    date: "2023-03-15",
    latitude: -10.5,
    longitude: 75.2,
    depth: 0,
    temperature: 28.5,
    salinity: 34.2,
    oxygen: 220,
    status: "Good",
  },
  {
    id: 2,
    floatId: "ARGO_001",
    date: "2023-03-15",
    latitude: -10.5,
    longitude: 75.2,
    depth: 100,
    temperature: 22.3,
    salinity: 34.8,
    oxygen: 180,
    status: "Good",
  },
  {
    id: 3,
    floatId: "ARGO_002",
    date: "2023-03-14",
    latitude: -15.3,
    longitude: 82.1,
    depth: 0,
    temperature: 29.1,
    salinity: 34.1,
    oxygen: 215,
    status: "Good",
  },
  {
    id: 4,
    floatId: "ARGO_002",
    date: "2023-03-14",
    latitude: -15.3,
    longitude: 82.1,
    depth: 200,
    temperature: 18.7,
    salinity: 35.1,
    oxygen: 150,
    status: "Good",
  },
  {
    id: 5,
    floatId: "ARGO_003",
    date: "2023-02-28",
    latitude: -8.7,
    longitude: 70.8,
    depth: 50,
    temperature: 25.1,
    salinity: 34.6,
    oxygen: 200,
    status: "Questionable",
  },
];

const columns = [
  { key: "floatId", label: "Float ID" },
  { key: "date", label: "Date" },
  { key: "latitude", label: "Latitude" },
  { key: "longitude", label: "Longitude" },
  { key: "depth", label: "Depth (m)" },
  { key: "temperature", label: "Temp (C)" },
  { key: "salinity", label: "Salinity (PSU)" },
  { key: "oxygen", label: "O2 (umol/kg)" },
  { key: "status", label: "Status" },
];

function downloadFile(name, content, type = "text/plain;charset=utf-8") {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export default function DataTable({ data }) {
  const [tableData, setTableData] = useState([]);
  const [sortConfig, setSortConfig] = useState({ key: null, direction: "asc" });
  const [filters, setFilters] = useState({
    floatId: "",
    dateRange: "",
    parameter: "all",
  });
  const [showFilters, setShowFilters] = useState(false);

  useEffect(() => {
    if (data?.table) {
      setTableData(data.table);
    } else {
      setTableData(sampleTableData);
    }
  }, [data]);

  const filteredData = useMemo(() => {
    let filtered = [...tableData];

    if (filters.floatId) {
      filtered = filtered.filter((row) =>
        row.floatId?.toLowerCase().includes(filters.floatId.toLowerCase())
      );
    }

    if (filters.dateRange) {
      filtered = filtered.filter((row) => row.date === filters.dateRange);
    }

    if (sortConfig.key) {
      filtered.sort((a, b) => {
        if (a[sortConfig.key] < b[sortConfig.key]) {
          return sortConfig.direction === "asc" ? -1 : 1;
        }
        if (a[sortConfig.key] > b[sortConfig.key]) {
          return sortConfig.direction === "asc" ? 1 : -1;
        }
        return 0;
      });
    }

    return filtered;
  }, [filters, sortConfig, tableData]);

  const handleSort = (key) => {
    setSortConfig((current) => ({
      key,
      direction:
        current.key === key && current.direction === "asc" ? "desc" : "asc",
    }));
  };

  const handleExport = (format) => {
    if (!format) return;

    if (format === "csv") {
      const rows = [
        columns.map((column) => column.label).join(","),
        ...filteredData.map((row) =>
          columns.map((column) => JSON.stringify(row[column.key] ?? "")).join(",")
        ),
      ].join("\n");
      downloadFile("aquaverse-data.csv", rows, "text/csv;charset=utf-8");
      return;
    }

    if (format === "ascii") {
      const rows = filteredData
        .map((row) => columns.map((column) => `${column.label}: ${row[column.key] ?? ""}`).join(" | "))
        .join("\n");
      downloadFile("aquaverse-data.txt", rows);
      return;
    }

    if (format === "json") {
      downloadFile(
        "aquaverse-data.json",
        JSON.stringify(filteredData, null, 2),
        "application/json;charset=utf-8"
      );
    }
  };

  const getSortIcon = (columnKey) => {
    if (sortConfig.key !== columnKey) return "";
    return sortConfig.direction === "asc" ? "↑" : "↓";
  };

  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="premium-kicker">Structured View</p>
          <h3 className="mt-2 font-display text-2xl font-semibold text-white">
            Tabular measurement detail
          </h3>
          <p className="mt-2 text-sm leading-6 text-slate-300">
            Filter down to a clean subset, sort by signal, and export without leaving the dashboard.
          </p>
        </div>

        <div className="flex flex-wrap gap-3">
          <button
            onClick={() => setShowFilters((value) => !value)}
            className="premium-button-secondary"
          >
            <FunnelIcon className="h-4 w-4" />
            {showFilters ? "Hide filters" : "Show filters"}
          </button>
          <div className="relative">
            <select
              onChange={(event) => handleExport(event.target.value)}
              className="premium-select min-w-[10rem]"
              defaultValue=""
            >
              <option value="" disabled>
                Export
              </option>
              <option value="csv">CSV</option>
              <option value="ascii">ASCII</option>
              <option value="json">JSON</option>
            </select>
          </div>
        </div>
      </div>

      {showFilters ? (
        <div className="grid gap-4 md:grid-cols-3">
          <label className="space-y-2">
            <span className="text-sm font-medium text-slate-200">Float ID</span>
            <input
              type="text"
              value={filters.floatId}
              onChange={(event) =>
                setFilters((current) => ({ ...current, floatId: event.target.value }))
              }
              placeholder="Enter float ID"
              className="premium-input"
            />
          </label>

          <label className="space-y-2">
            <span className="text-sm font-medium text-slate-200">Parameter</span>
            <select
              value={filters.parameter}
              onChange={(event) =>
                setFilters((current) => ({ ...current, parameter: event.target.value }))
              }
              className="premium-select"
            >
              <option value="all">All parameters</option>
              <option value="temperature">Temperature</option>
              <option value="salinity">Salinity</option>
              <option value="oxygen">Oxygen</option>
            </select>
          </label>

          <label className="space-y-2">
            <span className="text-sm font-medium text-slate-200">Date</span>
            <input
              type="date"
              value={filters.dateRange}
              onChange={(event) =>
                setFilters((current) => ({ ...current, dateRange: event.target.value }))
              }
              className="premium-input"
            />
          </label>
        </div>
      ) : null}

      <div className="overflow-hidden rounded-[24px] border border-white/10">
        <div className="overflow-auto scrollbar-thin">
          <table className="min-w-full text-left text-sm text-slate-200">
            <thead className="bg-white/[0.05] text-slate-400">
              <tr>
                {columns.map((column) => (
                  <th
                    key={column.key}
                    className="cursor-pointer px-4 py-3 font-medium transition-colors hover:bg-white/[0.04]"
                    onClick={() => handleSort(column.key)}
                  >
                    <div className="flex items-center gap-2">
                      {column.label}
                      <span className="text-xs text-cyan-200">{getSortIcon(column.key)}</span>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.06]">
              {filteredData.map((row) => (
                <tr key={row.id} className="transition-colors hover:bg-white/[0.03]">
                  <td className="px-4 py-3 font-medium text-cyan-100">{row.floatId}</td>
                  <td className="px-4 py-3">{row.date}</td>
                  <td className="px-4 py-3">{row.latitude?.toFixed?.(2) ?? row.latitude}</td>
                  <td className="px-4 py-3">{row.longitude?.toFixed?.(2) ?? row.longitude}</td>
                  <td className="px-4 py-3">{row.depth}</td>
                  <td className="px-4 py-3">{row.temperature?.toFixed?.(1) ?? row.temperature}</td>
                  <td className="px-4 py-3">{row.salinity?.toFixed?.(1) ?? row.salinity}</td>
                  <td className="px-4 py-3">{row.oxygen}</td>
                  <td className="px-4 py-3">
                    <span
                      className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold ${
                        row.status === "Good"
                          ? "bg-emerald-300/12 text-emerald-200"
                          : "bg-amber-300/12 text-amber-200"
                      }`}
                    >
                      {row.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="premium-card p-4">
          <p className="text-sm text-slate-300">Visible rows</p>
          <p className="mt-2 font-display text-2xl font-semibold text-white">
            {filteredData.length}
          </p>
        </div>
        <div className="premium-card p-4">
          <p className="text-sm text-slate-300">Total rows</p>
          <p className="mt-2 font-display text-2xl font-semibold text-white">
            {tableData.length}
          </p>
        </div>
        <div className="premium-card p-4">
          <p className="text-sm text-slate-300">Export</p>
          <button onClick={() => handleExport("csv")} className="mt-3 premium-button-secondary">
            <ArrowDownTrayIcon className="h-4 w-4" />
            Download CSV
          </button>
        </div>
      </div>
    </div>
  );
}
