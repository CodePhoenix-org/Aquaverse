
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainDashboard from "./Pages/MainDashboard";
import AuthPage from "./Pages/AuthPage";
import Dashboard from "./components/Dashboard";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<AuthPage />} />
        <Route path="/dashboard" element={<MainDashboard />} />
        <Route path="/chatbot" element={<Dashboard />} />
      </Routes>
    </Router>
  );
}

export default App;
