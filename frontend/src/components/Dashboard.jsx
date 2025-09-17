import React from "react";
import Chatbot from "./Chatbot";

const Dashboard = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white">
      <div className="container mx-auto px-6 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
            🌊 AquaVerse Dashboard
          </h1>
          <p className="text-blue-200 text-lg">
            AI-Powered Ocean Data Discovery Platform
          </p>
        </div>
        
        <div className="max-w-4xl mx-auto">
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 shadow-2xl">
            <h2 className="text-2xl font-semibold mb-6 text-center text-cyan-300">
              Chat with ARGO Assistant
            </h2>
            <Chatbot />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
