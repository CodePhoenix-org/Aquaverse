import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
  useParams,
} from "react-router-dom";
import AuthPage from "./Pages/AuthPage";
import Dashboard from "./components/Dashboard";
import Profile from "./Pages/Profile";
import History from "./Pages/History";
import ProtectedRoute from "./components/ProtectedRoute";
import { AuthProvider } from "./context/Authcontext";
import FloatChat from "./components/FloatChat";
import Predictor from "./components/Predictor";
import Home from "./components/Home";
import "./i18n";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { useEffect } from "react";
import ArgoGlobe from "./components/maps/ArgoGlobe";
function App() {
  console.log("App.jsx loaded"); // Debugging to confirm file is loaded
  return (
    <AuthProvider>
      <Router>
        <Routes>
          {/* Public routes */}
          <Route path="/auth" element={<AuthPage />} />
          <Route path="/home" element={<Home />} />
          <Route path="/predict" element={<Predictor />} />
          <Route path="/floatchat" element={<FloatChat />} />
          <Route path="/visuals" element={<ArgoGlobe />} />
          <Route path="/" element={<Navigate to="/home" replace />} />

          {/* Protected routes */}
          <Route element={<ProtectedRoute />}>
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/profile" element={<Profile />} />
            <Route path="/history" element={<History />} />
          </Route>

          {/* Catch-all → go home */}
          <Route path="*" element={<Navigate to="/home" replace />} />
        </Routes>

        <ToastContainer
          position="top-right"
          autoClose={3000}
          hideProgressBar={false}
          newestOnTop={true}
          closeOnClick
          rtl={false}
          pauseOnFocusLoss
          draggable
          pauseOnHover
          theme="colored"
        />
      </Router>
    </AuthProvider>
  );
}

export default App;
