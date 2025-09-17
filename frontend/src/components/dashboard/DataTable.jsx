import React, { useState, useEffect } from 'react';
import { ArrowDownTrayIcon, FunnelIcon } from '@heroicons/react/24/outline';
import apiService from '../../services/apiService';

const DataTable = ({ data }) => {
  const [tableData, setTableData] = useState([]);
  const [filteredData, setFilteredData] = useState([]);
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });
  const [filters, setFilters] = useState({
    floatId: '',
    dateRange: '',
    parameter: 'all'
  });
  const [showFilters, setShowFilters] = useState(false);

  // Sample data for demonstration
  const sampleTableData = [
    {
      id: 1,
      floatId: 'ARGO_001',
      date: '2023-03-15',
      latitude: -10.5,
      longitude: 75.2,
      depth: 0,
      temperature: 28.5,
      salinity: 34.2,
      oxygen: 220,
      status: 'Good'
    },
    {
      id: 2,
      floatId: 'ARGO_001',
      date: '2023-03-15',
      latitude: -10.5,
      longitude: 75.2,
      depth: 100,
      temperature: 22.3,
      salinity: 34.8,
      oxygen: 180,
      status: 'Good'
    },
    {
      id: 3,
      floatId: 'ARGO_002',
      date: '2023-03-14',
      latitude: -15.3,
      longitude: 82.1,
      depth: 0,
      temperature: 29.1,
      salinity: 34.1,
      oxygen: 215,
      status: 'Good'
    },
    {
      id: 4,
      floatId: 'ARGO_002',
      date: '2023-03-14',
      latitude: -15.3,
      longitude: 82.1,
      depth: 200,
      temperature: 18.7,
      salinity: 35.1,
      oxygen: 150,
      status: 'Good'
    },
    {
      id: 5,
      floatId: 'ARGO_003',
      date: '2023-02-28',
      latitude: -8.7,
      longitude: 70.8,
      depth: 50,
      temperature: 25.1,
      salinity: 34.6,
      oxygen: 200,
      status: 'Questionable'
    }
  ];

  useEffect(() => {
    if (data && data.table) {
      setTableData(data.table);
      setFilteredData(data.table);
    } else {
      setTableData(sampleTableData);
      setFilteredData(sampleTableData);
    }
  }, [data]);

  useEffect(() => {
    applyFilters();
  }, [filters, tableData]);

  const applyFilters = () => {
    let filtered = [...tableData];

    if (filters.floatId) {
      filtered = filtered.filter(row => 
        row.floatId.toLowerCase().includes(filters.floatId.toLowerCase())
      );
    }

    if (filters.parameter !== 'all') {
      // This would be more meaningful with actual parameter filtering logic
    }

    setFilteredData(filtered);
  };

  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }

    const sortedData = [...filteredData].sort((a, b) => {
      if (a[key] < b[key]) return direction === 'asc' ? -1 : 1;
      if (a[key] > b[key]) return direction === 'asc' ? 1 : -1;
      return 0;
    });

    setFilteredData(sortedData);
    setSortConfig({ key, direction });
  };

  const handleExport = async (format) => {
    try {
      await apiService.exportData(format, {
        data: filteredData,
        filters: filters
      });
    } catch (error) {
      alert('Export failed. Please try again.');
    }
  };

  const getSortIcon = (columnKey) => {
    if (sortConfig.key === columnKey) {
      return sortConfig.direction === 'asc' ? '↑' : '↓';
    }
    return '';
  };

  return (
    <div className="h-full flex flex-col">
      <div className="bg-gray-50 px-4 py-2 border-b border-gray-200">
        <div className="flex justify-between items-center">
          <div>
            <h3 className="text-lg font-medium text-gray-900">Data Table</h3>
            <p className="text-sm text-gray-600">
              Tabular view of ARGO float measurements
            </p>
          </div>
          <div className="flex space-x-2">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="flex items-center space-x-1 px-3 py-1 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200"
            >
              <FunnelIcon className="w-4 h-4" />
              <span>Filter</span>
            </button>
            <div className="relative">
              <select
                onChange={(e) => handleExport(e.target.value)}
                className="bg-blue-500 text-white px-3 py-1 rounded-md cursor-pointer"
                defaultValue=""
              >
                <option value="" disabled>Export</option>
                <option value="csv">CSV</option>
                <option value="ascii">ASCII</option>
                <option value="netcdf">NetCDF</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Filters */}
      {showFilters && (
        <div className="bg-white border-b border-gray-200 p-3">
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Float ID
              </label>
              <input
                type="text"
                value={filters.floatId}
                onChange={(e) => setFilters({ ...filters, floatId: e.target.value })}
                placeholder="Enter float ID"
                className="w-full px-3 py-1 border border-gray-300 rounded-md text-sm"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Parameter
              </label>
              <select
                value={filters.parameter}
                onChange={(e) => setFilters({ ...filters, parameter: e.target.value })}
                className="w-full px-3 py-1 border border-gray-300 rounded-md text-sm"
              >
                <option value="all">All Parameters</option>
                <option value="temperature">Temperature</option>
                <option value="salinity">Salinity</option>
                <option value="oxygen">Oxygen</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Date Range
              </label>
              <input
                type="date"
                value={filters.dateRange}
                onChange={(e) => setFilters({ ...filters, dateRange: e.target.value })}
                className="w-full px-3 py-1 border border-gray-300 rounded-md text-sm"
              />
            </div>
          </div>
        </div>
      )}

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-100 sticky top-0">
            <tr>
              {[
                { key: 'floatId', label: 'Float ID' },
                { key: 'date', label: 'Date' },
                { key: 'latitude', label: 'Latitude' },
                { key: 'longitude', label: 'Longitude' },
                { key: 'depth', label: 'Depth (m)' },
                { key: 'temperature', label: 'Temp (°C)' },
                { key: 'salinity', label: 'Salinity (PSU)' },
                { key: 'oxygen', label: 'O₂ (μmol/kg)' },
                { key: 'status', label: 'Status' }
              ].map((column) => (
                <th
                  key={column.key}
                  className="px-4 py-2 text-left font-medium text-gray-700 cursor-pointer hover:bg-gray-200"
                  onClick={() => handleSort(column.key)}
                >
                  <div className="flex items-center space-x-1">
                    <span>{column.label}</span>
                    <span className="text-xs">{getSortIcon(column.key)}</span>
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filteredData.map((row) => (
              <tr key={row.id} className="border-b border-gray-200 hover:bg-gray-50">
                <td className="px-4 py-2 font-medium text-blue-600">{row.floatId}</td>
                <td className="px-4 py-2">{row.date}</td>
                <td className="px-4 py-2">{row.latitude?.toFixed(2)}</td>
                <td className="px-4 py-2">{row.longitude?.toFixed(2)}</td>
                <td className="px-4 py-2">{row.depth}</td>
                <td className="px-4 py-2">{row.temperature?.toFixed(1)}</td>
                <td className="px-4 py-2">{row.salinity?.toFixed(1)}</td>
                <td className="px-4 py-2">{row.oxygen}</td>
                <td className="px-4 py-2">
                  <span className={`px-2 py-1 rounded-full text-xs ${
                    row.status === 'Good' 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-yellow-100 text-yellow-800'
                  }`}>
                    {row.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Footer */}
      <div className="bg-gray-50 border-t border-gray-200 px-4 py-2">
        <div className="text-sm text-gray-600">
          Showing {filteredData.length} of {tableData.length} records
        </div>
      </div>
    </div>
  );
};

export default DataTable;