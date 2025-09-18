import { useNavigate } from "react-router-dom";
import { LogOut, User, History as HistoryIcon, MessageCircle } from "lucide-react";

function Navbar({ onOpenChat }) {
  const navigate = useNavigate();

  const handleLogout = () => {
    try {
      localStorage.removeItem("auth_token");
    } catch (err) {
      // noop
    }
    navigate("/");
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
          <button
            onClick={() => navigate("/dashboard")}
            className="px-3 py-2 rounded-md text-sm hover:bg-white/10"
          >
            Dashboard
          </button>
          <button
            onClick={() => navigate("/profile")}
            className="px-3 py-2 rounded-md text-sm hover:bg-white/10 inline-flex items-center space-x-2"
            title="View Profile"
          >
            <User className="w-4 h-4" />
            <span>Profile</span>
          </button>
          <button
            onClick={() => navigate("/history")}
            className="px-3 py-2 rounded-md text-sm hover:bg-white/10 inline-flex items-center space-x-2"
            title="History"
          >
            <HistoryIcon className="w-4 h-4" />
            <span>History</span>
          </button>
          <button
            onClick={onOpenChat}
            className="px-3 py-2 rounded-md text-sm hover:bg-white/10 inline-flex items-center space-x-2"
            title="Open Chat"
          >
            <MessageCircle className="w-4 h-4" />
            <span>Chat</span>
          </button>
          <button
            onClick={handleLogout}
            className="ml-2 px-3 py-2 rounded-md text-sm bg-white/10 hover:bg-white/20 border border-white/20 inline-flex items-center space-x-2"
            title="Logout"
          >
            <LogOut className="w-4 h-4" />
            <span>Logout</span>
          </button>
        </nav>
      </div>
    </header>
  );
}

export default Navbar;


