import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import {
  ArrowRight,
  Bot,
  Brain,
  Globe2,
  LineChart,
  Menu,
  Orbit,
  Radar,
  ShieldCheck,
  Sparkles,
  Waves,
  X,
} from "lucide-react";
import PageShell from "./ui/PageShell";
import BrandMark from "./ui/BrandMark";

const fadeUp = {
  hidden: { opacity: 0, y: 26 },
  show: { opacity: 1, y: 0 },
};

function SectionHeading({ kicker, title, description }) {
  return (
    <div className="max-w-3xl">
      <p className="premium-kicker">{kicker}</p>
      <h2 className="mt-3 font-display text-3xl font-bold tracking-[-0.04em] text-white sm:text-4xl">
        {title}
      </h2>
      <p className="mt-4 text-base leading-7 text-slate-300 sm:text-lg">
        {description}
      </p>
    </div>
  );
}

export default function Home() {
  const navigate = useNavigate();
  const [mobileOpen, setMobileOpen] = useState(false);

  const stats = useMemo(
    () => [
      { value: "3,847", label: "Active floats" },
      { value: "85+", label: "Countries contributing" },
      { value: "24/7", label: "Signal refresh cadence" },
    ],
    []
  );

  const features = useMemo(
    () => [
      {
        icon: <Radar className="h-6 w-6 text-cyan-100" />,
        title: "Live ocean telemetry",
        description:
          "Track float fleets, profile updates, and regional conditions through a polished real-time command view.",
      },
      {
        icon: <Brain className="h-6 w-6 text-cyan-100" />,
        title: "AI-assisted exploration",
        description:
          "Ask natural-language questions and let FloatChat route you to the right visualization, map, or comparison.",
      },
      {
        icon: <LineChart className="h-6 w-6 text-cyan-100" />,
        title: "Research-grade visuals",
        description:
          "Move from profiles and tables to spatial plots and anomaly surfaces without leaving the same premium workspace.",
      },
      {
        icon: <ShieldCheck className="h-6 w-6 text-cyan-100" />,
        title: "Decision-ready insights",
        description:
          "Support climate monitoring, marine science, and early-warning workflows with cleaner signal and faster context.",
      },
    ],
    []
  );

  const modules = useMemo(
    () => [
      {
        title: "FloatChat",
        description:
          "A conversational surface for querying the ocean like a pro, without memorizing APIs or raw schemas.",
        icon: <Bot className="h-5 w-5 text-cyan-100" />,
      },
      {
        title: "Visual analytics",
        description:
          "Dive from surface trends into layered profile plots, comparison panels, and immersive global views.",
        icon: <Orbit className="h-5 w-5 text-cyan-100" />,
      },
      {
        title: "Prediction suite",
        description:
          "Test environmental conditions against anomaly and aquatic-life models with a sharper, more legible workflow.",
        icon: <Sparkles className="h-5 w-5 text-cyan-100" />,
      },
    ],
    []
  );

  const navItems = [
    { id: "features", label: "Features" },
    { id: "modules", label: "Modules" },
    { id: "workflow", label: "Workflow" },
  ];

  const scrollToSection = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
    setMobileOpen(false);
  };

  return (
    <PageShell>
      <header className="sticky top-0 z-40 px-4 pt-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <div className="premium-panel premium-panel-strong flex items-center justify-between px-4 py-3 sm:px-6">
            <BrandMark />

            <nav className="hidden items-center gap-8 xl:flex">
              {navItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => scrollToSection(item.id)}
                  className="text-sm font-medium text-slate-300 transition-colors hover:text-white"
                >
                  {item.label}
                </button>
              ))}
            </nav>

            <div className="hidden items-center gap-3 xl:flex">
              <button
                onClick={() => navigate("/visuals")}
                className="premium-button-secondary"
              >
                Explore Globe
              </button>
              <button onClick={() => navigate("/auth")} className="premium-button">
                Enter Platform
              </button>
            </div>

            <button
              onClick={() => setMobileOpen((open) => !open)}
              className="inline-flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-white xl:hidden"
              aria-label="Toggle menu"
            >
              {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>

          {mobileOpen ? (
            <div className="premium-panel premium-panel-strong mt-3 space-y-3 px-4 py-4 xl:hidden">
              {navItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => scrollToSection(item.id)}
                  className="block w-full rounded-2xl border border-white/[0.08] bg-white/5 px-4 py-3 text-left text-sm text-slate-200"
                >
                  {item.label}
                </button>
              ))}
              <button
                onClick={() => navigate("/visuals")}
                className="premium-button-secondary w-full"
              >
                Explore Globe
              </button>
              <button onClick={() => navigate("/auth")} className="premium-button w-full">
                Enter Platform
              </button>
            </div>
          ) : null}
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 pb-20 pt-10 sm:px-6 lg:px-8">
        <section className="grid items-center gap-8 py-10 lg:grid-cols-[1.15fr_0.85fr] lg:py-16">
          <motion.div
            variants={fadeUp}
            initial="hidden"
            animate="show"
            transition={{ duration: 0.65 }}
          >
            <span className="premium-badge">
              <Waves className="h-3.5 w-3.5" />
              Premium Ocean Command Center
            </span>
            <h1 className="mt-6 font-display text-5xl font-bold tracking-[-0.06em] text-balance text-white sm:text-6xl lg:text-7xl">
              A calmer, sharper way to work with ocean intelligence.
            </h1>
            <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-300">
              AquaVerse turns fragmented marine data into a refined operational
              experience for researchers, analysts, and ocean-tech teams. Query,
              visualize, compare, and predict from one unified surface.
            </p>

            <div className="mt-8 flex flex-col gap-3 sm:flex-row">
              <button onClick={() => navigate("/auth")} className="premium-button">
                Start Exploring
                <ArrowRight className="h-4 w-4" />
              </button>
              <button
                onClick={() => navigate("/dashboard")}
                className="premium-button-secondary"
              >
                View Dashboard
              </button>
            </div>

            <div className="mt-10 grid gap-4 sm:grid-cols-3">
              {stats.map((stat) => (
                <div key={stat.label} className="premium-card px-5 py-4">
                  <p className="font-display text-3xl font-bold text-white">
                    {stat.value}
                  </p>
                  <p className="mt-1 text-sm text-slate-300">{stat.label}</p>
                </div>
              ))}
            </div>
          </motion.div>

          <motion.div
            variants={fadeUp}
            initial="hidden"
            animate="show"
            transition={{ duration: 0.75, delay: 0.1 }}
            className="premium-panel premium-panel-strong relative overflow-hidden p-6 sm:p-8"
          >
            <div className="absolute inset-x-10 top-0 h-40 rounded-full bg-cyan-300/12 blur-3xl" />
            <div className="relative space-y-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="premium-kicker">Mission Signal</p>
                  <h2 className="mt-2 font-display text-3xl font-semibold text-white">
                    Designed like a premium research cockpit.
                  </h2>
                </div>
                <div className="rounded-2xl border border-cyan-300/20 bg-cyan-300/10 p-3 text-cyan-100">
                  <Globe2 className="h-6 w-6" />
                </div>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="premium-card p-5">
                  <p className="text-sm text-slate-300">Signal quality</p>
                  <p className="mt-3 font-display text-4xl font-bold text-white">
                    98.4%
                  </p>
                  <p className="mt-2 text-sm text-slate-400">
                    Cleaned pipeline for charts, maps, and profile comparisons.
                  </p>
                </div>
                <div className="premium-card p-5">
                  <p className="text-sm text-slate-300">AI routing</p>
                  <p className="mt-3 font-display text-4xl font-bold text-white">
                    4 views
                  </p>
                  <p className="mt-2 text-sm text-slate-400">
                    FloatChat can pivot results into maps, tables, plots, or comparison views.
                  </p>
                </div>
              </div>

              <div className="premium-card space-y-4 p-5">
                <div className="flex items-center justify-between">
                  <span className="premium-chip">Platform stack</span>
                  <span className="text-sm text-slate-400">React, AI, 3D, Maps</span>
                </div>
                <div className="premium-divider" />
                <div className="grid gap-3 sm:grid-cols-3">
                  {["FloatChat", "Globe", "Predictions"].map((item) => (
                    <div
                      key={item}
                      className="rounded-2xl border border-white/[0.08] bg-slate-950/40 px-4 py-3 text-sm text-slate-200"
                    >
                      {item}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        </section>

        <section id="features" className="py-16">
          <SectionHeading
            kicker="Core Experience"
            title="Every major workflow feels elevated, faster, and easier to read."
            description="The UI system focuses on depth, clarity, and confidence: premium surfaces, better hierarchy, cleaner interactions, and a visual language built for ocean-tech storytelling."
          />
          <div className="mt-10 grid gap-5 lg:grid-cols-2">
            {features.map((feature, index) => (
              <motion.article
                key={feature.title}
                variants={fadeUp}
                initial="hidden"
                whileInView="show"
                viewport={{ once: true, margin: "-80px" }}
                transition={{ duration: 0.55, delay: index * 0.08 }}
                className="premium-card p-6"
              >
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-cyan-300/18 bg-cyan-300/10">
                  {feature.icon}
                </div>
                <h3 className="mt-5 font-display text-2xl font-semibold text-white">
                  {feature.title}
                </h3>
                <p className="mt-3 text-base leading-7 text-slate-300">
                  {feature.description}
                </p>
              </motion.article>
            ))}
          </div>
        </section>

        <section id="modules" className="py-16">
          <div className="grid gap-8 lg:grid-cols-[0.92fr_1.08fr]">
            <SectionHeading
              kicker="Product Modules"
              title="One brand language across chat, dashboards, prediction, and immersive visuals."
              description="Instead of treating each route like a different app, AquaVerse now behaves like a coherent premium platform from the first click through the deepest view."
            />
            <div className="grid gap-5">
              {modules.map((module, index) => (
                <motion.div
                  key={module.title}
                  variants={fadeUp}
                  initial="hidden"
                  whileInView="show"
                  viewport={{ once: true, margin: "-100px" }}
                  transition={{ duration: 0.55, delay: index * 0.08 }}
                  className="premium-panel p-6"
                >
                  <div className="flex items-start gap-4">
                    <div className="rounded-2xl border border-white/10 bg-white/[0.06] p-3">
                      {module.icon}
                    </div>
                    <div>
                      <h3 className="font-display text-2xl font-semibold text-white">
                        {module.title}
                      </h3>
                      <p className="mt-2 text-base leading-7 text-slate-300">
                        {module.description}
                      </p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        <section id="workflow" className="py-16">
          <div className="premium-panel premium-panel-strong p-6 sm:p-8">
            <SectionHeading
              kicker="Workflow"
              title="A premium sequence from question to action."
              description="Start with a conversational prompt, shape the signal in a dashboard, inspect the globe, and validate conditions in the prediction suite without losing context."
            />

            <div className="mt-10 grid gap-5 lg:grid-cols-3">
              {[
                "Ask FloatChat for a map, profile, or anomaly-focused question.",
                "Validate spatial or temporal patterns inside the dashboard surfaces.",
                "Open the prediction and 3D visual routes for deeper situational context.",
              ].map((step, index) => (
                <div key={step} className="premium-card p-6">
                  <p className="premium-kicker">Step 0{index + 1}</p>
                  <p className="mt-4 text-lg leading-8 text-slate-100">{step}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="py-16">
          <div className="premium-panel premium-panel-strong flex flex-col gap-6 p-6 sm:p-8 lg:flex-row lg:items-center lg:justify-between">
            <div className="max-w-2xl">
              <p className="premium-kicker">Launch Ready</p>
              <h2 className="mt-3 font-display text-3xl font-bold tracking-[-0.04em] text-white sm:text-4xl">
                Enter AquaVerse with a UI that finally feels worthy of the data.
              </h2>
              <p className="mt-4 text-base leading-7 text-slate-300">
                The refreshed interface is built to feel premium on desktop and mobile,
                while still staying practical for real analysis work.
              </p>
            </div>
            <div className="flex flex-col gap-3 sm:flex-row">
              <button onClick={() => navigate("/auth")} className="premium-button">
                Open Auth
                <ArrowRight className="h-4 w-4" />
              </button>
              <button
                onClick={() => navigate("/visuals")}
                className="premium-button-secondary"
              >
                Open 3D Globe
              </button>
            </div>
          </div>
        </section>
      </main>
    </PageShell>
  );
}
