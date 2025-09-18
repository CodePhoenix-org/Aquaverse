import { Link, useNavigate } from "react-router-dom";
import {
  LogOut,
  User,
  History as HistoryIcon,
  MessageCircle,
} from "lucide-react";
const Navbar = ({ onOpenChat }) => {
  const navigate = useNavigate();

  const handleLogout = () => {
    try {
      localStorage.removeItem("auth_token");
      localStorage.removeItem("user");
      console.log("User logged out successfully");
    } catch (err) {
      console.error("Error during logout:", err);
    }
    navigate("/"); // redirect to login or landing page
  };

  return (
    <header className="w-full bg-gradient-to-r from-slate-900 via-blue-900 to-cyan-900 text-white shadow-xl">
      <div className="max-w-7xl mx-auto px-4 lg:px-6 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-lg bg-white/10 border border-white/20 flex items-center justify-center">
            <span className="text-xl">🌊</span>
          </div>
          <div>
            <h1 className="text-xl font-bold leading-tight">FloatChat</h1>
            <p className="text-xs text-blue-200">ARGO Ocean Data Discovery</p>
          </div>
        </div>

        <nav className="flex items-center space-x-2">
          <Link
            to="/dashboard"
            className="px-3 py-2 rounded-md text-sm hover:bg-white/10"
          >
            Dashboard
          </Link>

          <Link
            to="/profile"
            className="px-3 py-2 rounded-md text-sm hover:bg-white/10 inline-flex items-center space-x-2"
          >
            <User className="w-4 h-4" />
            <span>Profile</span>
          </Link>

          <Link
            to="/history"
            className="px-3 py-2 rounded-md text-sm hover:bg-white/10 inline-flex items-center space-x-2"
          >
            <HistoryIcon className="w-4 h-4" />
            <span>History</span>
          </Link>

          {/* Keep as button since it's triggering a chat open */}
          <button
            onClick={onOpenChat}
            className="px-3 py-2 rounded-md text-sm hover:bg-white/10 inline-flex items-center space-x-2"
          >
            <MessageCircle className="w-4 h-4" />
            <span>Chat</span>
          </button>

          {/* Logout button */}
          <button
            onClick={handleLogout}
            className="ml-2 px-3 py-2 rounded-md text-sm bg-white/10 hover:bg-white/20 border border-white/20 inline-flex items-center space-x-2"
          >
            <LogOut className="w-4 h-4" />
            <span>Logout</span>
          </button>
        </nav>
      </div>
    </header>
  );
};

export default Navbar;
