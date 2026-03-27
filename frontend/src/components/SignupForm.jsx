import { useState } from "react";
import { useForm } from "react-hook-form";
import axios from "axios";
import {
  ArrowRight,
  Eye,
  EyeOff,
  Lock,
  Mail,
  User,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { toast } from "react-toastify";
import { useAuth } from "../context/Authcontext";

export default function SignupForm() {
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const navigate = useNavigate();
  const { login } = useAuth();

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors, isSubmitting },
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

      const res = await axios.post("http://127.0.0.1:8000/auth/signup", payload, {
        headers: { "Content-Type": "application/json" },
      });

      const userData = res.data.user || res.data;
      if (res.data.access_token) {
        localStorage.setItem("auth_token", res.data.access_token);
      }
      localStorage.setItem("user", JSON.stringify(userData));
      login(userData);

      toast.success("Account created. Welcome to AquaVerse.");
      navigate("/dashboard");
    } catch (error) {
      toast.error(error.response?.data?.detail || "Signup failed. Please try again.");
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-5">
      <div className="grid gap-5 sm:grid-cols-2">
        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-200">First name</label>
          <div className="relative">
            <User className="pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-400" />
            <input
              {...register("firstName", { required: "First name is required" })}
              type="text"
              placeholder="Ariel"
              className="premium-input pl-12"
            />
          </div>
          {errors.firstName ? (
            <p className="text-sm text-rose-300">{errors.firstName.message}</p>
          ) : null}
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-200">Last name</label>
          <div className="relative">
            <User className="pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-400" />
            <input
              {...register("lastName", { required: "Last name is required" })}
              type="text"
              placeholder="Rivera"
              className="premium-input pl-12"
            />
          </div>
          {errors.lastName ? (
            <p className="text-sm text-rose-300">{errors.lastName.message}</p>
          ) : null}
        </div>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium text-slate-200">Email</label>
        <div className="relative">
          <Mail className="pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-400" />
          <input
            {...register("email", {
              required: "Email is required",
              pattern: {
                value: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
                message: "Enter a valid email address",
              },
            })}
            type="email"
            placeholder="you@aquaverse.ai"
            className="premium-input pl-12"
          />
        </div>
        {errors.email ? (
          <p className="text-sm text-rose-300">{errors.email.message}</p>
        ) : null}
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium text-slate-200">Password</label>
        <div className="relative">
          <Lock className="pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-400" />
          <input
            {...register("password", {
              required: "Password is required",
              minLength: {
                value: 6,
                message: "Password must be at least 6 characters",
              },
            })}
            type={showPassword ? "text" : "password"}
            placeholder="Create a password"
            className="premium-input pl-12 pr-12"
          />
          <button
            type="button"
            onClick={() => setShowPassword((value) => !value)}
            className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 transition-colors hover:text-white"
            aria-label={showPassword ? "Hide password" : "Show password"}
          >
            {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
          </button>
        </div>
        {errors.password ? (
          <p className="text-sm text-rose-300">{errors.password.message}</p>
        ) : null}
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium text-slate-200">Confirm password</label>
        <div className="relative">
          <Lock className="pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-400" />
          <input
            {...register("confirmPassword", {
              required: "Please confirm your password",
              validate: (value) => value === password || "Passwords do not match",
            })}
            type={showConfirmPassword ? "text" : "password"}
            placeholder="Confirm your password"
            className="premium-input pl-12 pr-12"
          />
          <button
            type="button"
            onClick={() => setShowConfirmPassword((value) => !value)}
            className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 transition-colors hover:text-white"
            aria-label={showConfirmPassword ? "Hide password" : "Show password"}
          >
            {showConfirmPassword ? (
              <EyeOff className="h-5 w-5" />
            ) : (
              <Eye className="h-5 w-5" />
            )}
          </button>
        </div>
        {errors.confirmPassword ? (
          <p className="text-sm text-rose-300">{errors.confirmPassword.message}</p>
        ) : null}
      </div>

      <div className="rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 py-3 text-sm text-slate-300">
        Create your account to unlock the upgraded dashboard, FloatChat history,
        and premium visualization workflow.
      </div>

      <button type="submit" disabled={isSubmitting} className="premium-button w-full">
        {isSubmitting ? "Creating account..." : "Create Account"}
        <ArrowRight className="h-4 w-4" />
      </button>
    </form>
  );
}
