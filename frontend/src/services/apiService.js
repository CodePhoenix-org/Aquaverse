// src/services/apiService.js
// Simple API service for chat and export endpoints

const API_BASE_URL = "http://127.0.0.1:8000";

const apiService = {
  async sendChatMessage(message) {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: message })
    });
    if (!response.ok) throw new Error("Failed to fetch chat response");
    return await response.json();
  },

  async exportData(format, payload) {
    // Example: POST to /export?format=csv or /export?format=ascii or /export?format=netcdf
    const response = await fetch(`${API_BASE_URL}/export?format=${format}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!response.ok) throw new Error("Export failed");
    // For file download, you may want to handle blob response here
    return await response.json();
  }
};

export default apiService;
