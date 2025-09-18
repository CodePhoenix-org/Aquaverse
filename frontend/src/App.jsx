
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import AuthPage from "./Pages/AuthPage";
import Dashboard from "./components/Dashboard";
import Profile from "./Pages/Profile";
import History from "./Pages/History";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<AuthPage />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/history" element={<History />} />
      </Routes>
    </Router>
  );
}

export default App;
