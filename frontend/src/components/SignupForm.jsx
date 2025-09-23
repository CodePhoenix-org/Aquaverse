import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { Eye, EyeOff, User, Mail, Lock, ArrowRight } from "lucide-react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
const SignupForm = () => {
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [apiUrl, setApiUrl] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    setApiUrl("http://127.0.0.1:8000/auth/signup");
  }, []);

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
  } = useForm();
  const password = watch("password");

  const onSubmit = async (data) => {
    try {
      const payload = {
        name: `${data.firstName} ${data.lastName}`,
        email: data.email,
        password: data.password,
        confirm_password: data.confirmPassword,
      };

      console.log("Payload to send:", payload);

      const res = await axios.post(apiUrl, payload, {
        headers: { "Content-Type": "application/json" },
      });

      console.log("Signup success:", res.data);
      localStorage.setItem("user", JSON.stringify(res.data));

      toast.success("Signed up successfully ✅");
      navigate("/floatchat");
    } catch (err) {
      console.error("Signup Axios error object:", err);
      toast.error(err.response?.data?.detail || "Signup failed!");
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">

      <div className="grid grid-cols-2 gap-4">
        <div className="relative group">
          <User className="absolute left-4 top-1/2 -translate-y-1/2 text-blue-300 w-5 h-5" />
          <input
            {...register("firstName", { required: "First Name is required" })}
            type="text"
            placeholder="First Name"
            className="w-full pl-12 pr-4 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-blue-200 focus:ring-2 focus:ring-cyan-400 focus:outline-none"
          />
          {errors.firstName && (
            <span className="text-red-500 text-xs">
              {errors.firstName.message}
            </span>
          )}
        </div>
        <div className="relative group">
          <User className="absolute left-4 top-1/2 -translate-y-1/2 text-blue-300 w-5 h-5" />
          <input
            {...register("lastName", { required: "Last Name is required" })}
            type="text"
            placeholder="Last Name"
            className="w-full pl-12 pr-4 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-blue-200 focus:ring-2 focus:ring-cyan-400 focus:outline-none"
          />
          {errors.lastName && (
            <span className="text-red-500 text-xs">
              {errors.lastName.message}
            </span>
          )}
        </div>
      </div>

      {/* Email */}
      <div className="relative group">
        <Mail className="absolute left-4 top-1/2 -translate-y-1/2 text-blue-300 w-5 h-5" />
        <input
          {...register("email", {
            required: "Email is required",
            pattern: {
              value: /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/,
              message: "Invalid email address",
            },
          })}
          type="email"
          placeholder="Email Address"
          className="w-full pl-12 pr-4 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-blue-200 focus:ring-2 focus:ring-cyan-400 focus:outline-none"
        />
        {errors.email && (
          <span className="text-red-500 text-xs">{errors.email.message}</span>
        )}
      </div>

      {/* Password */}
      <div className="relative group">
        <Lock className="absolute left-4 top-1/2 -translate-y-1/2 text-blue-300 w-5 h-5" />
        <input
          {...register("password", {
            required: "Password is required",
            minLength: {
              value: 6,
              message: "Password must be at least 6 characters",
            },
          })}
          type={showPassword ? "text" : "password"}
          placeholder="Password"
          className="w-full pl-12 pr-12 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-blue-200 focus:ring-2 focus:ring-cyan-400 focus:outline-none"
        />
        <button
          type="button"
          onClick={() => setShowPassword(!showPassword)}
          className="absolute right-4 top-1/2 -translate-y-1/2 text-blue-300 hover:text-cyan-400"
        >
          {showPassword ? (
            <EyeOff className="w-5 h-5" />
          ) : (
            <Eye className="w-5 h-5" />
          )}
        </button>
        {errors.password && (
          <span className="text-red-500 text-xs">
            {errors.password.message}
          </span>
        )}
      </div>

      {/* Confirm Password */}
      <div className="relative group">
        <Lock className="absolute left-4 top-1/2 -translate-y-1/2 text-blue-300 w-5 h-5" />
        <input
          {...register("confirmPassword", {
            required: "Please confirm your password",
            validate: (value) => value === password || "Passwords do not match",
          })}
          type={showConfirmPassword ? "text" : "password"}
          placeholder="Confirm Password"
          className="w-full pl-12 pr-12 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-blue-200 focus:ring-2 focus:ring-cyan-400 focus:outline-none"
        />
        <button
          type="button"
          onClick={() => setShowConfirmPassword(!showConfirmPassword)}
          className="absolute right-4 top-1/2 -translate-y-1/2 text-blue-300 hover:text-cyan-400"
        >
          {showConfirmPassword ? (
            <EyeOff className="w-5 h-5" />
          ) : (
            <Eye className="w-5 h-5" />
          )}
        </button>
        {errors.confirmPassword && (
          <span className="text-red-500 text-xs">
            {errors.confirmPassword.message}
          </span>
        )}
      </div>

      {/* Submit */}
      <button
        type="submit"
        className="w-full py-3 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-semibold rounded-xl shadow-lg transform hover:scale-105 transition-all flex items-center justify-center group"
      >
        Join the Exploration
        <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
      </button>
    </form>
  );
};

export default SignupForm;
