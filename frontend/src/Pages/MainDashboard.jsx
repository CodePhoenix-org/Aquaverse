import React, { useState } from 'react';
import ChatInterface from '../components/chat/ChatInterface';
import FloatMap from '../components/maps/FloatMap';
import DataPlots from '../components/plots/DataPlots';
import ProfileComparison from '../components/plots/ProfileComparison';
import DataTable from '../components/dashboard/DataTable';

const MainDashboard = () => {
  const [activeView, setActiveView] = useState('overview');
  const [chatData, setChatData] = useState(null);

  return (
    <div className="flex h-screen">
      {/* Left Panel - Chat Interface */}
      <div className="w-1/3 border-r border-gray-200 bg-white">
        <ChatInterface onDataReceived={setChatData} />
      </div>
      
      {/* Right Panel - Dashboard */}
      <div className="w-2/3 flex flex-col">
        {/* Dashboard Header */}
        <div className="bg-white border-b border-gray-200 p-4">
          <div className="flex space-x-4">
            <button
              onClick={() => setActiveView('overview')}
              className={`px-4 py-2 rounded-md ${
                activeView === 'overview' 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveView('map')}
              className={`px-4 py-2 rounded-md ${
                activeView === 'map' 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Map View
            </button>
            <button
              onClick={() => setActiveView('plots')}
              className={`px-4 py-2 rounded-md ${
                activeView === 'plots' 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Data Plots
            </button>
            <button
              onClick={() => setActiveView('comparison')}
              className={`px-4 py-2 rounded-md ${
                activeView === 'comparison' 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Profile Comparison
            </button>
            <button
              onClick={() => setActiveView('table')}
              className={`px-4 py-2 rounded-md ${
                activeView === 'table' 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Data Table
            </button>
          </div>
        </div>
        
        {/* Dashboard Content */}
        <div className="flex-1 p-4 bg-gray-50">
          {activeView === 'overview' && (
            <div className="grid grid-cols-2 gap-4 h-full">
              <div className="bg-white rounded-lg shadow-sm border border-gray-200">
                <FloatMap data={chatData} />
              </div>
              <div className="bg-white rounded-lg shadow-sm border border-gray-200">
                <DataPlots data={chatData} />
              </div>
            </div>
          )}
          {activeView === 'map' && (
            <div className="h-full bg-white rounded-lg shadow-sm border border-gray-200">
              <FloatMap data={chatData} />
            </div>
          )}
          {activeView === 'plots' && (
            <div className="h-full bg-white rounded-lg shadow-sm border border-gray-200">
              <DataPlots data={chatData} />
            </div>
          )}
          {activeView === 'comparison' && (
            <div className="h-full bg-white rounded-lg shadow-sm border border-gray-200">
              <ProfileComparison data={chatData} />
            </div>
          )}
          {activeView === 'table' && (
            <div className="h-full bg-white rounded-lg shadow-sm border border-gray-200">
              <DataTable data={chatData} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MainDashboard;