import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { toast } from "react-toastify";
import {
  History as HistoryIcon,
  LayoutDashboard,
  LogOut,
  Menu,
  MessageCircle,
  Sparkles,
  User,
  Waves,
  X,
} from "lucide-react";
import { useAuth } from "../context/Authcontext";
import BrandMark from "./ui/BrandMark";

const navItems = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/floatchat", label: "FloatChat", icon: MessageCircle },
  { to: "/predict", label: "Predictor", icon: Sparkles },
  { to: "/history", label: "History", icon: HistoryIcon },
  { to: "/profile", label: "Profile", icon: User },
  { to: "/visuals", label: "3D Globe", icon: Waves },
];

export default function Navbar({ onOpenChat }) {
  const navigate = useNavigate();
  const location = useLocation();
  const { logout } = useAuth();
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleLogout = () => {
    logout();
    toast.success("You have been logged out.");
    navigate("/home");
  };

  return (
    <header className="sticky top-0 z-40 px-4 pt-4 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-[1400px]">
        <div className="premium-panel premium-panel-strong flex items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <Link to="/home" className="shrink-0">
            <BrandMark compact={location.pathname !== "/home"} />
          </Link>

          <nav className="hidden min-w-0 items-center gap-2 xl:flex">
            {navItems.map((item) => {
              const Icon = item.icon;
              const active = location.pathname === item.to;

              return (
                <Link
                  key={item.to}
                  to={item.to}
                  className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition-all ${
                    active
                      ? "bg-white/10 text-white shadow-lg shadow-cyan-500/10"
                      : "text-slate-300 hover:bg-white/[0.06] hover:text-white"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </Link>
              );
            })}
          </nav>

          <div className="hidden items-center gap-3 xl:flex">
            {onOpenChat ? (
              <button onClick={onOpenChat} className="premium-button-secondary">
                <MessageCircle className="h-4 w-4" />
                Open Chat
              </button>
            ) : null}
            <button onClick={handleLogout} className="premium-button">
              <LogOut className="h-4 w-4" />
              Logout
            </button>
          </div>

          <button
            type="button"
            onClick={() => setMobileOpen((open) => !open)}
            className="inline-flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-white xl:hidden"
            aria-label="Toggle menu"
          >
            {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>

        {mobileOpen ? (
          <div className="premium-panel premium-panel-strong mt-3 space-y-3 px-4 py-4 xl:hidden">
            {navItems.map((item) => {
              const Icon = item.icon;
              const active = location.pathname === item.to;

              return (
                <Link
                  key={item.to}
                  to={item.to}
                  onClick={() => setMobileOpen(false)}
                  className={`flex items-center gap-3 rounded-2xl px-4 py-3 text-sm font-medium ${
                    active ? "bg-white/10 text-white" : "bg-white/5 text-slate-300"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </Link>
              );
            })}

            {onOpenChat ? (
              <button
                onClick={() => {
                  setMobileOpen(false);
                  onOpenChat();
                }}
                className="premium-button-secondary w-full"
              >
                <MessageCircle className="h-4 w-4" />
                Open Chat
              </button>
            ) : null}

            <button onClick={handleLogout} className="premium-button w-full">
              <LogOut className="h-4 w-4" />
              Logout
            </button>
          </div>
        ) : null}
      </div>
    </header>
  );
}
