import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  ArrowRight,
  ShieldCheck,
  Sparkles,
  Waves,
} from "lucide-react";
import LoginForm from "../components/LoginForm";
import SignupForm from "../components/SignupForm";
import PageShell from "../components/ui/PageShell";
import BrandMark from "../components/ui/BrandMark";
import AppLogo from "../components/ui/AppLogo";

const fadeUp = {
  hidden: { opacity: 0, y: 22 },
  show: { opacity: 1, y: 0 },
};

export default function AuthPage() {
  const [isLogin, setIsLogin] = useState(true);

  const highlights = useMemo(
    () => [
      "Live access to FloatChat, dashboard analytics, and premium 3D globe views.",
      "Unified ocean-tech interface tuned for clarity on desktop and mobile.",
      "Fast route from sign-in to research-ready exploration.",
    ],
    []
  );

  const stats = useMemo(
    () => [
      { value: "24/7", label: "Monitoring cadence" },
      { value: "150+", label: "Research sources" },
      { value: "4", label: "Core analysis views" },
    ],
    []
  );

  return (
    <PageShell backdropVariant="dense">
      <main className="mx-auto grid min-h-screen max-w-7xl items-center gap-8 px-4 py-8 sm:px-6 lg:grid-cols-[1.05fr_0.95fr] lg:px-8">
        <motion.section
          variants={fadeUp}
          initial="hidden"
          animate="show"
          transition={{ duration: 0.65 }}
          className="relative overflow-hidden rounded-[34px] border border-white/10 bg-slate-950/40 shadow-ocean"
        >
          <video
            className="absolute inset-0 h-full w-full object-cover"
            src="/videos/authVid.mp4"
            autoPlay
            loop
            muted
            playsInline
          />
          <div className="absolute inset-0 bg-[linear-gradient(135deg,rgba(2,10,23,0.35),rgba(2,10,23,0.78)_45%,rgba(2,10,23,0.96))]" />
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(125,211,252,0.18),transparent_30%)]" />

          <div className="relative z-10 flex min-h-[42rem] flex-col justify-between p-6 sm:p-8 lg:p-10">
            <div className="flex items-center justify-between gap-4">
              <BrandMark subtitle="Premium ocean data access" />
              <span className="premium-chip">
                <ShieldCheck className="h-3.5 w-3.5" />
                Secure Sign-In
              </span>
            </div>

            <div className="max-w-2xl">
              <span className="premium-badge">
                <Sparkles className="h-3.5 w-3.5" />
                Ocean Intelligence, Refined
              </span>
              <h1 className="mt-6 font-display text-4xl font-bold tracking-[-0.05em] text-white sm:text-5xl lg:text-6xl">
                Explore marine data through a more premium, immersive workspace.
              </h1>
              <p className="mt-5 max-w-xl text-base leading-8 text-slate-200 sm:text-lg">
                Sign in to access FloatChat, dashboards, and anomaly-aware ocean
                views inside AquaVerse&apos;s upgraded command experience.
              </p>

              <div className="mt-8 grid gap-3">
                {highlights.map((item) => (
                  <div
                    key={item}
                    className="premium-chip w-fit max-w-full rounded-2xl px-4 py-3 text-left text-sm text-slate-100"
                  >
                    <ArrowRight className="h-4 w-4 flex-none" />
                    <span>{item}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-3">
              {stats.map((stat) => (
                <div key={stat.label} className="premium-card p-5">
                  <p className="font-display text-3xl font-bold text-white">
                    {stat.value}
                  </p>
                  <p className="mt-1 text-sm text-slate-300">{stat.label}</p>
                </div>
              ))}
            </div>
          </div>
        </motion.section>

        <motion.section
          variants={fadeUp}
          initial="hidden"
          animate="show"
          transition={{ duration: 0.7, delay: 0.1 }}
          className="premium-panel premium-panel-strong relative overflow-hidden p-6 sm:p-8"
        >
          <div className="absolute inset-x-12 top-0 h-32 rounded-full bg-cyan-300/12 blur-3xl" />
          <div className="relative z-10">
            <div className="flex items-center gap-4">
              <AppLogo size="lg" />
              <div>
                <p className="premium-kicker">Access Portal</p>
                <h2 className="mt-2 font-display text-3xl font-semibold text-white">
                  {isLogin ? "Welcome back" : "Create your account"}
                </h2>
                <p className="mt-2 text-sm leading-6 text-slate-300">
                  {isLogin
                    ? "Sign in to continue exploring the upgraded AquaVerse experience."
                    : "Join AquaVerse to unlock chat, data views, and premium prediction tools."}
                </p>
              </div>
            </div>

            <div className="mt-8 grid grid-cols-2 gap-3 rounded-full border border-white/10 bg-slate-950/40 p-2">
              <button
                onClick={() => setIsLogin(true)}
                className={`rounded-full px-4 py-3 text-sm font-semibold transition-all ${
                  isLogin
                    ? "bg-gradient-to-r from-cyan-300 to-sky-400 text-slate-950 shadow-lg"
                    : "text-slate-300"
                }`}
              >
                Sign In
              </button>
              <button
                onClick={() => setIsLogin(false)}
                className={`rounded-full px-4 py-3 text-sm font-semibold transition-all ${
                  !isLogin
                    ? "bg-gradient-to-r from-cyan-300 to-sky-400 text-slate-950 shadow-lg"
                    : "text-slate-300"
                }`}
              >
                Create Account
              </button>
            </div>

            <div className="mt-8">
              {isLogin ? <LoginForm /> : <SignupForm />}
            </div>

            <div className="mt-8 premium-divider" />

            <div className="mt-6 flex flex-col gap-3 text-sm text-slate-300 sm:flex-row sm:items-center sm:justify-between">
              <div className="premium-chip rounded-full px-4 py-2">
                <Waves className="h-4 w-4" />
                Premium ocean-tech UI now live
              </div>
              <button
                onClick={() => setIsLogin((value) => !value)}
                className="text-left font-semibold text-cyan-200 transition-colors hover:text-white sm:text-right"
              >
                {isLogin
                  ? "Need an account? Create one now."
                  : "Already registered? Sign back in."}
              </button>
            </div>
          </div>
        </motion.section>
      </main>
    </PageShell>
  );
}
