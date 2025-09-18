import { useState, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
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
      <div className="min-h-screen flex items-center justify-center text-white">
        Loading profile...
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center text-red-500">
        {error}
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white">
      <div className="max-w-4xl mx-auto p-6">
        <h1 className="text-3xl font-bold mb-4">Your Profile</h1>
        <div className="bg-white/10 border border-white/20 rounded-xl p-6 space-y-2">
          <p>
            <span className="font-semibold text-blue-300">Name:</span> {user.name}
          </p>
          <p>
            <span className="font-semibold text-blue-300">Email:</span> {user.email}
          </p>
          <p>
            <span className="font-semibold text-blue-300">Joined:</span>{" "}
            {new Date(user.created_at).toLocaleString()}
          </p>
        </div>
      </div>
    </div>
  );
};

export default Profile;
