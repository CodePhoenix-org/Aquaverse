import React, { useState, useEffect, useMemo, useCallback } from "react";
import { motion, useTime, useTransform } from "framer-motion";
import { useNavigate } from "react-router-dom";

// Inline Waves Icon
const WavesIcon = () => (
  <svg
    className="w-5 h-5 text-primary-foreground"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M3 12c2.5-2 6.5-2 9 0s6.5 2 9 0"
    />
  </svg>
);

// Generate Wave Path
const generateWavePath = (width, height, frequency, amplitude, phase, time) => {
  let path = `M0 ${height}`;
  const points = 100;
  for (let i = 0; i <= points; i++) {
    const x = (width * i) / points;
    const y =
      height * 0.8 +
      amplitude * Math.sin(frequency * x + phase + time) +
      amplitude * 0.5 * Math.sin(frequency * x * 2 + phase * 2 + time * 1.5);
    path += ` L${x} ${y}`;
  }
  path += ` L${width} ${height} Z`;
  return path;
};

// Memoized WaveLayer
const WaveLayer = React.memo(({ wave, windowWidth, windowHeight }) => {
  const time = useTime();
  const pathD = useTransform(time, (latestTime) =>
    generateWavePath(
      windowWidth,
      windowHeight,
      wave.frequency,
      wave.amplitude,
      wave.phase,
      latestTime * 0.001 * wave.speed
    )
  );

  return (
    <div
      className="absolute inset-0"
      style={{ bottom: `${wave.layer * 20}px`, overflow: "visible" }}
    >
      <svg
        viewBox={`0 0 ${windowWidth} ${windowHeight}`}
        preserveAspectRatio="none"
        style={{ width: "100%", height: "100%" }}
      >
        <defs>
          <linearGradient
            id={`wave-gradient-${wave.layer}`}
            x1="0%"
            y1="0%"
            x2="100%"
            y2="0%"
          >
            <stop
              offset="0%"
              stopColor={wave.color}
              stopOpacity={wave.opacity}
            />
            <stop
              offset="50%"
              stopColor={wave.color}
              stopOpacity={wave.opacity * 0.7}
            />
            <stop
              offset="100%"
              stopColor={wave.color}
              stopOpacity={wave.opacity * 0.4}
            />
          </linearGradient>
        </defs>
        <motion.path
          d={pathD}
          fill={`url(#wave-gradient-${wave.layer})`}
          stroke="none"
          style={{ filter: `blur(${wave.layer * 0.5 + 1}px)` }}
        />
      </svg>
    </div>
  );
});

// Sinusoidal Wave Background
function SinusoidalWaveBackground({ opacity = 1, className = "" }) {
  const [windowSize, setWindowSize] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  });

  const waves = useMemo(
    () =>
      Array.from({ length: 6 }, (_, i) => ({
        id: `wave-${i}`,
        layer: i,
        frequency: 0.01 + i * 0.005,
        amplitude: 40 - i * 6,
        speed: 0.5 + i * 0.3,
        phase: i * Math.PI * 0.3,
        color: [
          "#38bdf8",
          "#0ea5e9",
          "#0284c7",
          "#0369a1",
          "#1e40af",
          "#1e3a8a",
        ][i],
        opacity: 0.8 - i * 0.12,
      })),
    []
  );

  useEffect(() => {
    const handleResize = () =>
      setWindowSize({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div
      className={`absolute inset-0 overflow-hidden ${className}`}
      style={{ opacity }}
    >
      <div
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(180deg, #0f172a 0%, #1e293b 30%, #1e40af 60%, #0f172a 100%)",
        }}
      />
      {waves.map((wave) => (
        <WaveLayer
          key={wave.id}
          wave={wave}
          windowWidth={windowSize.width}
          windowHeight={windowSize.height}
        />
      ))}
    </div>
  );
}

// Navigation
const Navigation = React.memo(() => {
  const navigate = useNavigate();
  const scrollToSection = (sectionId) =>
    document
      .getElementById(sectionId)
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  const navigatetologin = useCallback(() => navigate("/auth"), [navigate]);

  return (
    <motion.nav
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6 }}
      className="fixed top-0 w-full z-50 glassmorphic"
      style={{
        background: "rgba(20,30,50,0.95)",
        backdropFilter: "blur(20px)",
        borderBottom: "1px solid rgba(255,255,255,0.08)",
      }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex justify-between items-center h-16">
        <motion.div
          className="flex items-center space-x-2"
          whileHover={{ scale: 1.05 }}
        >
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{
              background: "linear-gradient(135deg,#38bdf8 0%,#06b6d4 100%)",
              boxShadow: "0 4px 15px rgba(56,189,248,0.4)",
            }}
          >
            <WavesIcon />
          </div>
          <span className="font-bold text-xl text-white whitespace-nowrap">
            Aquaverse
          </span>{" "}
        </motion.div>
        <div className="hidden md:flex items-center space-x-8">
          {["features", "architecture", "technology"].map((id) => (
            <motion.button
              key={id}
              onClick={() => scrollOrNavigate(id)}
              className="text-gray-300 hover:text-white transition-colors duration-300"
            >
              {id.charAt(0).toUpperCase() + id.slice(1)}
            </motion.button>
          ))}
          <motion.button
            onClick={navigatetologin}
            className="font-semibold text-blue-400 hover:text-cyan-400"
          >
            Get Started
          </motion.button>
        </div>
      </div>
    </motion.nav>
  );
});

// Hero Section
function HeroSection() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-20">
        <motion.h1
          initial={{ y: 50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="text-6xl md:text-8xl lg:text-9xl font-black mb-8 leading-none"
        >
          <motion.span
            className="block"
            style={{ color: "#38bdf8" }}
            animate={{ y: [0, -3, 0], scale: [1, 1.02, 1] }}
            transition={{ duration: 3, repeat: Infinity }}
          >
            Aqua
          </motion.span>
          <motion.span
            className="block bg-gradient-to-r from-cyan-400 via-blue-500 to-sky-600 bg-clip-text text-transparent"
            animate={{ y: [0, -3, 0], scale: [1, 1.02, 1] }}
            transition={{ duration: 3, repeat: Infinity, delay: 0.5 }}
          >
            verse
          </motion.span>
        </motion.h1>
        <p className="text-2xl md:text-3xl font-light text-slate-200 mb-4 max-w-4xl mx-auto">
          Ocean Data Intelligence Platform
        </p>
      </div>
    </section>
  );
}

// Features Section
function FeaturesSection() {
  const features = useMemo(
    () => [
      {
        title: "Real-time Ocean Data",
        desc: "Access millions of Argo float profiles and oceanographic datasets in real time.",
        icon: "🌊",
      },
      {
        title: "AI-Powered Analytics",
        desc: "Leverage advanced AI models for data visualization, anomaly detection, and predictions.",
        icon: "🤖",
      },
      {
        title: "Interactive Visualizations",
        desc: "Explore interactive charts, maps, and 3D plots for deep ocean insights.",
        icon: "📊",
      },
      {
        title: "Global Coverage",
        desc: "Data from 150+ research institutions and 85+ countries, updated 24/7.",
        icon: "🌍",
      },
    ],
    []
  );

  return (
    <section
      id="features"
      className="py-32 relative"
      style={{
        background:
          "linear-gradient(135deg, #1e3a8a 0%, #1e40af 30%, #0284c7 60%, #0ea5e9 100%)",
      }}
    >
      <div className="max-w-6xl mx-auto px-4 text-center relative z-10">
        <motion.h2
          className="text-5xl font-bold mb-8"
          initial={{ y: -20, opacity: 0 }}
          whileInView={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          style={{ color: "#38bdf8" }}
        >
          Features
        </motion.h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 mt-16">
          {features.map((f, i) => (
            <motion.div
              key={i}
              className="bg-white/10 p-8 rounded-3xl backdrop-blur-sm shadow-lg"
              whileHover={{ scale: 1.05 }}
              initial={{ y: 20, opacity: 0 }}
              whileInView={{ y: 0, opacity: 1 }}
              transition={{ duration: 0.6, delay: i * 0.2 }}
              viewport={{ once: true }}
            >
              <div className="text-4xl mb-4">{f.icon}</div>
              <h3 className="text-2xl font-semibold mb-2">{f.title}</h3>
              <p className="text-gray-200">{f.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

// Architecture Section
function ArchitectureSection() {
  return (
    <section
      id="architecture"
      className="py-32 relative bg-gradient-to-b from-slate-900 to-slate-800 text-white"
      style={{
        background:
          "linear-gradient(135deg, #1e3a8a 0%, #1e40af 30%, #0284c7 60%, #0ea5e9 100%)",
      }}
    >
      <div className="max-w-6xl mx-auto px-4 text-center">
        <motion.h2
          initial={{ y: -20, opacity: 0 }}
          whileInView={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8 }}
          className="text-5xl font-bold mb-8"
        >
          Architecture
        </motion.h2>
        <p className="text-xl md:text-2xl max-w-4xl mx-auto">
          Our platform integrates real-time Argo float datasets, AI-powered
          analytics engines, and interactive visualizations for ocean research
          and climate monitoring.
        </p>
      </div>
    </section>
  );
}

// Technology Section
function TechnologySection() {
  const tech = [
    "React.js",
    "TailwindCSS",
    "Framer Motion",
    "FastAPI",
    "PostgreSQL",
    "Supabase",
    "Python AI/ML",
    "Docker",
  ];
  return (
    <section
      id="technology"
      className="py-32 relative bg-gradient-to-b from-slate-800 to-slate-900 text-white"
      style={{
        background:
          "linear-gradient(135deg, #1e3a8a 0%, #1e40af 30%, #0284c7 60%, #0ea5e9 100%)",
      }}
    >
      <div className="max-w-6xl mx-auto px-4 text-center">
        <motion.h2
          initial={{ y: -20, opacity: 0 }}
          whileInView={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8 }}
          className="text-5xl font-bold mb-8"
        >
          Technology
        </motion.h2>
        <div className="flex flex-wrap justify-center gap-6 mt-12">
          {tech.map((t, i) => (
            <motion.div
              key={i}
              className="bg-white/10 px-6 py-3 rounded-2xl text-lg font-medium"
              whileHover={{ scale: 1.05 }}
              initial={{ y: 20, opacity: 0 }}
              whileInView={{ y: 0, opacity: 1 }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              viewport={{ once: true }}
            >
              {t}
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

// CTA Section
function CTASection() {
  const navigate = useNavigate();
  const handleGetStarted = useCallback(() => navigate("/auth"), [navigate]);
  return (
    <section
      id="cta"
      className="py-32 relative text-center"
      style={{
        background:
          "linear-gradient(135deg, #1e3a8a 0%, #1e40af 30%, #0284c7 60%, #0ea5e9 100%)",
      }}
    >
      <div className="max-w-6xl mx-auto px-4">
        <motion.h2
          initial={{ y: 50, opacity: 0 }}
          whileInView={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8 }}
          className="text-5xl font-bold mb-8 text-white"
        >
          Ready to Dive Into Ocean Data?
        </motion.h2>
        <motion.button
          whileHover={{ scale: 1.05 }}
          className="bg-gradient-to-r from-blue-500 to-cyan-500 text-white px-12 py-6 rounded-3xl font-bold"
          onClick={handleGetStarted}
        >
          Get Started
        </motion.button>
      </div>
    </section>
  );
}

// Footer
function Footer() {
  return (
    <footer
      className="py-12 text-center text-white bg-slate-900"
      style={{
        background:
          "linear-gradient(135deg, #1e3a8a 0%, #1e40af 30%, #0284c7 60%, #0ea5e9 100%)",
      }}
    >
      <p>&copy; 2025 Aquaverse. All Rights Reserved.</p>
    </footer>
  );
}

// Main Landing Component
export default function AquaverseLanding() {
  return (
    <div style={{ background: "#0f172a", minHeight: "100vh" }}>
      <SinusoidalWaveBackground
        opacity={0.9}
        className="fixed top-0 left-0 w-full h-full z-0"
      />
      <Navigation />
      <main style={{ position: "relative", zIndex: 10 }}>
        <HeroSection />
        <FeaturesSection />
        <ArchitectureSection />
        <TechnologySection />
        <CTASection />
      </main>
      <Footer />
    </div>
  );
}
