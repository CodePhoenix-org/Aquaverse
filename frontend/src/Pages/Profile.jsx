import { useEffect, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { CalendarDays, LogOut, Mail, UserCircle } from "lucide-react";
import Navbar from "../components/Navbar";
import PageShell from "../components/ui/PageShell";

export default function Profile() {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchUser = async () => {
      const token = localStorage.getItem("auth_token");
      if (!token) {
        setError("You are not logged in.");
        setLoading(false);
        navigate("/auth");
        return;
      }

      try {
        const response = await axios.get("http://127.0.0.1:8000/auth/me", {
          headers: { Authorization: `Bearer ${token}` },
        });
        setUser(response.data);
      } catch (requestError) {
        setError(requestError.response?.data?.detail || "Failed to load profile information.");
      } finally {
        setLoading(false);
      }
    };

    fetchUser();
  }, [navigate]);

  if (loading) {
    return (
      <PageShell>
        <div className="flex min-h-screen items-center justify-center px-4">
          <div className="premium-panel premium-panel-strong px-8 py-6 text-center">
            <p className="font-display text-3xl font-semibold text-white">Loading profile...</p>
          </div>
        </div>
      </PageShell>
    );
  }

  if (error) {
    return (
      <PageShell>
        <div className="flex min-h-screen items-center justify-center px-4">
          <div className="premium-panel premium-panel-strong max-w-xl px-8 py-6 text-center">
            <p className="font-display text-3xl font-semibold text-white">Profile unavailable</p>
            <p className="mt-4 text-base leading-7 text-slate-300">{error}</p>
            <button onClick={() => navigate("/auth")} className="premium-button mt-6">
              Go to Auth
            </button>
          </div>
        </div>
      </PageShell>
    );
  }

  return (
    <PageShell>
      <Navbar />
      <main className="mx-auto max-w-6xl px-4 pb-16 pt-6 sm:px-6 lg:px-8">
        <section className="premium-panel premium-panel-strong overflow-hidden p-6 sm:p-8">
          <div className="grid gap-8 lg:grid-cols-[0.9fr_1.1fr]">
            <div className="space-y-5">
              <div className="flex flex-col items-center rounded-[30px] border border-white/10 bg-white/[0.04] px-6 py-8 text-center sm:items-start sm:text-left">
                {user.avatar_url ? (
                  <img
                    src={user.avatar_url}
                    alt="Profile avatar"
                    className="h-28 w-28 rounded-full border border-cyan-300/20 object-cover"
                  />
                ) : (
                  <UserCircle className="h-28 w-28 text-cyan-100" />
                )}
                <p className="premium-kicker mt-6">Member Profile</p>
                <h1 className="mt-2 font-display text-4xl font-bold text-white">
                  {user.name}
                </h1>
                <p className="mt-2 text-base text-slate-300">{user.email}</p>
                <span className="premium-chip mt-5">
                  Premium AquaVerse workspace enabled
                </span>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="premium-card p-4">
                  <p className="text-sm text-slate-300">Member since</p>
                  <p className="mt-2 font-display text-2xl font-semibold text-white">
                    {new Date(user.created_at).toLocaleDateString()}
                  </p>
                </div>
                <div className="premium-card p-4">
                  <p className="text-sm text-slate-300">Workspace</p>
                  <p className="mt-2 font-display text-2xl font-semibold text-white">
                    Premium
                  </p>
                </div>
              </div>
            </div>

            <div className="space-y-5">
              {[
                {
                  label: "Full name",
                  value: user.name,
                  icon: UserCircle,
                },
                {
                  label: "Email",
                  value: user.email,
                  icon: Mail,
                },
                {
                  label: "Joined",
                  value: new Date(user.created_at).toLocaleString(),
                  icon: CalendarDays,
                },
              ].map((item) => {
                const Icon = item.icon;
                return (
                  <div key={item.label} className="premium-card flex items-center gap-4 p-5">
                    <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.06]">
                      <Icon className="h-5 w-5 text-cyan-100" />
                    </div>
                    <div>
                      <p className="text-sm text-slate-300">{item.label}</p>
                      <p className="mt-1 text-lg font-semibold text-white">{item.value}</p>
                    </div>
                  </div>
                );
              })}

              <div className="premium-card p-5">
                <p className="premium-kicker">Workspace Actions</p>
                <div className="mt-5 flex flex-col gap-3 sm:flex-row">
                  <button
                    onClick={() => navigate("/history")}
                    className="premium-button-secondary"
                  >
                    View History
                  </button>
                  <button
                    onClick={() => {
                      localStorage.removeItem("auth_token");
                      localStorage.removeItem("user");
                      navigate("/home");
                    }}
                    className="premium-button"
                  >
                    <LogOut className="h-4 w-4" />
                    Logout
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </PageShell>
  );
}
