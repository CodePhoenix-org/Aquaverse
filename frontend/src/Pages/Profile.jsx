import { useState, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { UserCircle } from "lucide-react";
const Profile = () => {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchUser = async () => {
      const token = localStorage.getItem("auth_token");
      if (!token) {
        setError("You are not logged in");
        setLoading(false);
        navigate("/");
        return;
      }

      try {
        const res = await axios.get("http://127.0.0.1:8000/auth/me", {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        setUser(res.data);
      } catch (err) {
        console.error("Failed to fetch user:", err.response?.data || err.message);
        setError(err.response?.data?.detail || "Failed to fetch user info");
      } finally {
        setLoading(false);
      }
    };

    fetchUser();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white">
        <p className="text-lg font-medium animate-pulse">Loading profile...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-red-500">
        <p className="text-lg font-medium">{error}</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white flex items-center justify-center py-10">
      <div className="max-w-lg w-full bg-white/10 backdrop-blur-md border border-white/20 rounded-2xl shadow-xl p-8 space-y-6">
        {/* Profile Header */}
        <div className="flex flex-col items-center space-y-4">
          <div className="text-6xl text-blue-400">
            {user.avatar_url ? (
              <img
                src={user.avatar_url}
                alt="Profile Avatar"
                className="w-24 h-24 rounded-full border-4 border-blue-400 object-cover"
              />
            ) : (
              <UserCircle className="w-24 h-24 text-blue-400" />
            )}
          </div>
          <h1 className="text-2xl font-bold text-white">{user.name}</h1>
          <p className="text-sm text-white/70">Member since {new Date(user.created_at).toLocaleDateString()}</p>
        </div>

        {/* Profile Info */}
        <div className="space-y-4">
          <div className="flex justify-between items-center bg-white/5 p-4 rounded-xl hover:bg-white/10 transition">
            <span className="font-semibold text-blue-300">Name</span>
            <span className="text-white">{user.name}</span>
          </div>
          <div className="flex justify-between items-center bg-white/5 p-4 rounded-xl hover:bg-white/10 transition">
            <span className="font-semibold text-blue-300">Email</span>
            <span className="text-white">{user.email}</span>
          </div>
          <div className="flex justify-between items-center bg-white/5 p-4 rounded-xl hover:bg-white/10 transition">
            <span className="font-semibold text-blue-300">Joined</span>
            <span className="text-white">{new Date(user.created_at).toLocaleString()}</span>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex justify-center space-x-4 mt-6">
          <button
            onClick={() => navigate("/edit-profile")}
            className="bg-blue-500 hover:bg-blue-600 transition px-6 py-2 rounded-xl font-semibold"
          >
            Edit Profile
          </button>
          <button
            onClick={() => {
              localStorage.removeItem("auth_token");
              navigate("/");
            }}
            className="bg-red-500 hover:bg-red-600 transition px-6 py-2 rounded-xl font-semibold"
          >
            Logout
          </button>
        </div>
      </div>
    </div>
  );
};

export default Profile;
