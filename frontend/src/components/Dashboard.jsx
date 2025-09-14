import React from "react";
import Chatbot from "./Chatbot";
const Dashboard = () => {
  return (
    <div className="min-h-screen text-white p-6">
      <h1 className="text-2xl font-bold mb-6">🌊 AquaVerse Dashboard</h1>
      <Chatbot />
    </div>
  );
};

export default Dashboard;
