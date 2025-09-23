import React, { useState, useEffect, useMemo, useCallback } from "react";
import { motion, useTime, useTransform } from "framer-motion";
import { useNavigate } from "react-router-dom";

// Inline SVG icons
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

// Helper function to generate wave path
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

// Memoized WaveLayer component for performance
const WaveLayer = React.memo(({ wave, windowWidth, windowHeight }) => {
  const time = useTime();

  const pathD = useTransform(time, (latestTime) => {
    return generateWavePath(
      windowWidth,
      windowHeight,
      wave.frequency,
      wave.amplitude,
      wave.phase,
      latestTime * 0.001 * wave.speed
    );
  });

  return (
    <div
      className="absolute inset-0"
      style={{
        bottom: `${wave.layer * 20}px`,
        width: "100%",
        height: "100%",
        overflow: "visible",
      }}
    >
      <svg
        viewBox={`0 0 ${windowWidth} ${windowHeight}`}
        preserveAspectRatio="none"
        style={{
          position: "absolute",
          width: "100%",
          height: "100%",
          overflow: "visible",
        }}
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
          style={{
            filter: `blur(${wave.layer * 0.5 + 1}px)`,
            mixBlendMode: wave.layer > 2 ? "soft-light" : "normal",
          }}
        />
      </svg>
    </div>
  );
});

// ENHANCED & OPTIMIZED SinusoidalWaveBackground
function SinusoidalWaveBackground({ opacity = 1, className = "" }) {
  const [windowWidth, setWindowWidth] = useState(1920);
  const [windowHeight, setWindowHeight] = useState(1080);

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
    const updateDimensions = () => {
      setWindowWidth(window.innerWidth);
      setWindowHeight(window.innerHeight);
    };
    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
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
          windowWidth={windowWidth}
          windowHeight={windowHeight}
        />
      ))}

      <div className="absolute inset-0 pointer-events-none">
        <motion.svg
          className="absolute bottom-0 w-full h-32"
          viewBox="0 0 100 32"
          preserveAspectRatio="none"
          animate={{ x: [0, -50, 0], scaleX: [1, 1.05, 1] }}
          transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
        >
          <defs>
            <linearGradient
              id="foam-gradient"
              x1="0%"
              y1="0%"
              x2="100%"
              y2="0%"
            >
              <stop offset="0%" stopColor="rgba(255,255,255,0.8)" />
              <stop offset="50%" stopColor="rgba(255,255,255,0.4)" />
              <stop offset="100%" stopColor="rgba(56,189,248,0.6)" />
            </linearGradient>
          </defs>
          <motion.path
            d="M0 32 Q25 20 50 32 T100 32 L100 32 Z"
            fill="url(#foam-gradient)"
            animate={{ d: "M0 32 Q25 15 50 32 T100 32 L100 32 Z" }}
            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
          />
        </motion.svg>

        {[...Array(30)].map((_, i) => (
          <motion.div
            key={`particle-${i}`}
            className="absolute rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${20 + Math.random() * 80}%`,
              width: `${0.5 + Math.random() * 4}px`,
              height: `${0.5 + Math.random() * 4}px`,
              background: `radial-gradient(circle, hsl(${
                180 + Math.random() * 60
              }, 80%, ${60 + Math.random() * 40}%) 0%, transparent 70%)`,
              boxShadow: `0 0 ${5 + Math.random() * 10}px hsl(${
                180 + Math.random() * 60
              }, 70%, ${50 + Math.random() * 30}%)`,
            }}
            animate={{
              y: [-10, 20 + Math.random() * 30, -10],
              x: [-8, 8, -8],
              opacity: [0.4, 1, 0.4],
              rotate: [0, 360 * (Math.random() > 0.5 ? 1 : -1), 0],
              scale: [0.8, 1.2, 0.8],
            }}
            transition={{
              duration: 12 + Math.random() * 8,
              repeat: Infinity,
              delay: Math.random() * 3,
              ease: [0.25, 0.1, 0.25, 1],
            }}
          />
        ))}
      </div>

      <div className="absolute inset-0 opacity-30">
        {[...Array(8)].map((_, i) => (
          <motion.div
            key={`caustic-${i}`}
            className="absolute"
            style={{
              left: `${10 + i * 12}%`,
              top: "60%",
              width: "80px",
              height: "80px",
              background: `radial-gradient(circle, rgba(255, 255, 255, ${
                0.5 + i * 0.1
              }) 0%, transparent 70%)`,
              borderRadius: "50%",
              filter: "blur(10px)",
            }}
            animate={{
              y: [0, -40, 0],
              x: [-20, 20, -20],
              scale: [0.8, 1.4, 0.8],
              rotate: [0, 180, 360],
            }}
            transition={{
              duration: 15 + i * 3,
              repeat: Infinity,
              delay: i * 0.5,
              ease: "easeInOut",
            }}
          />
        ))}
      </div>
    </div>
  );
}

const Navigation = React.memo(() => {
  const navigate = useNavigate();
  const scrollToSection = (sectionId) =>
    document
      .getElementById(sectionId)
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  const navigatetologin = useCallback(() => navigate("/"), [navigate]);

  return (
    <motion.nav
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6 }}
      className="fixed top-0 w-full z-50 glassmorphic"
      style={{
        background: "rgba(20, 30, 50, 0.95)",
        backdropFilter: "blur(20px)",
        borderBottom: "1px solid rgba(255,255,255,0.08)",
        boxShadow: "0 4px 30px rgba(56, 189, 248, 0.1)",
      }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <motion.div
            className="flex items-center space-x-2"
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 400, damping: 17 }}
          >
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center relative overflow-hidden"
              style={{
                background: "linear-gradient(135deg, #38bdf8 0%, #06b6d4 100%)",
                boxShadow: "0 4px 15px rgba(56, 189, 248, 0.4)",
              }}
            >
              <WavesIcon />
            </div>
            <span className="font-bold text-xl text-white">Aquaverse</span>
          </motion.div>
          <div className="hidden md:flex items-center space-x-8">
            {[
              { label: "Features", id: "features" },
              { label: "Architecture", id: "architecture" },
              { label: "Technology", id: "technology" },
            ].map((item) => (
              <motion.button
                key={item.id}
                onClick={() => scrollToSection(item.id)}
                className="text-gray-300 hover:text-white transition-colors duration-300 relative group"
                whileHover={{ y: -2 }}
              >
                {item.label}
                <motion.div
                  className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full"
                  initial={false}
                  whileHover={{ width: "100%" }}
                  transition={{ duration: 0.3 }}
                />
              </motion.button>
            ))}
            <motion.button
              onClick={navigatetologin}
              className="font-semibold text-blue-400 hover:text-cyan-400 transition-colors duration-300 relative group"
              whileHover={{ y: -2 }}
            >
              Get Started
              <motion.div
                className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full"
                initial={false}
                whileHover={{ width: "100%" }}
                transition={{ duration: 0.3 }}
              />
            </motion.button>
          </div>
        </div>
      </div>
    </motion.nav>
  );
});

function HeroSection() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-20">
        <motion.div
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="mb-8 mt-24"
        >
          <motion.div
            className="inline-flex items-center px-4 py-2 rounded-full glassmorphic"
            style={{
              background: "rgba(255,255,255,0.1)",
              color: "#38bdf8",
              border: "1px solid rgba(255,255,255,0.1)",
              backdropFilter: "blur(8px)",
              boxShadow: "0 4px 20px rgba(56, 189, 248, 0.2)",
            }}
            whileHover={{ scale: 1.02 }}
          >
            <span
              className="w-2 h-2 rounded-full mr-2"
              style={{ background: "#38bdf8" }}
            >
              <motion.div
                animate={{ scale: [1, 1.3, 1] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              />
            </span>
            Powered by AI & Ocean Intelligence
          </motion.div>
        </motion.div>
        <motion.div
          initial={{ y: 50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="relative"
        >
          <h1 className="text-6xl md:text-8xl lg:text-9xl font-black mb-8 leading-none">
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
          </h1>
          <p className="text-2xl md:text-3xl font-light text-slate-200 mb-4 max-w-4xl mx-auto">
            Ocean Data Intelligence Platform
          </p>
        </motion.div>
      </div>
    </section>
  );
}

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
      {" "}
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
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 mt-12">
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ y: 40, opacity: 0 }}
              whileInView={{ y: 0, opacity: 1 }}
              transition={{ duration: 0.7, delay: i * 0.1 }}
              viewport={{ once: true }}
              className="bg-slate-800/80 backdrop-blur-sm rounded-2xl p-8 shadow-lg flex flex-col items-center border border-white/10"
              whileHover={{
                scale: 1.02,
                boxShadow: "0 20px 40px rgba(56, 189, 248, 0.2)",
              }}
            >
              <div className="text-5xl mb-4">{f.icon}</div>
              <h3
                className="text-2xl font-bold mb-2"
                style={{ color: "#38bdf8" }}
              >
                {f.title}
              </h3>
              <p className="text-lg text-slate-300">{f.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

function ArchitectureSection() {
  return (
    <section
      id="architecture"
      className="py-32 relative"
      style={{
        background:
          "linear-gradient(135deg, #1e3a8a 0%, #1e40af 30%, #0284c7 60%, #0ea5e9 100%)",
      }}
    >
      {" "}
      <div className="max-w-6xl mx-auto px-4 text-center relative z-10">
        <motion.h2
          className="text-5xl font-bold mb-8 text-white"
          initial={{ y: -20, opacity: 0 }}
          whileInView={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          Architecture
        </motion.h2>
        <motion.p
          className="text-xl text-slate-200 mb-12"
          initial={{ y: 20, opacity: 0 }}
          whileInView={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          viewport={{ once: true }}
        >
          Aquaverse is built on a scalable, cloud-native architecture,
          integrating real-time data ingestion, AI analytics, and interactive
          visualization layers.
        </motion.p>
        <div className="flex flex-col md:flex-row justify-center items-center gap-12">
          {[
            {
              title: "Data Pipeline",
              items: [
                "Argo float data ingestion",
                "Data cleaning & transformation",
                "AI-powered analytics",
                "Visualization & API layer",
              ],
            },
            {
              title: "Tech Stack",
              items: [
                "React, Tailwind CSS, Framer Motion",
                "FastAPI, PostgreSQL, ChromaDB",
                "Cloud-native deployment",
              ],
            },
          ].map((section, index) => (
            <motion.div
              key={section.title}
              className="bg-white/15 backdrop-blur-sm rounded-2xl p-8 shadow-lg w-full md:w-1/2 border border-white/20"
              initial={{ y: 20, opacity: 0 }}
              whileInView={{ y: 0, opacity: 1 }}
              whileHover={{ scale: 1.02 }}
              transition={{ duration: 0.8, delay: 0.2 + index * 0.1 }}
              viewport={{ once: true }}
            >
              <h3 className="text-2xl font-bold mb-4 text-white">
                {section.title}
              </h3>
              <ul className="text-left list-disc list-inside text-lg text-slate-200 space-y-2">
                {section.items.map((item, i) => (
                  <motion.li
                    key={item}
                    initial={{ x: -20, opacity: 0 }}
                    whileInView={{ x: 0, opacity: 1 }}
                    transition={{ duration: 0.5, delay: 0.3 + i * 0.1 }}
                    viewport={{ once: true }}
                  >
                    {item}
                  </motion.li>
                ))}
              </ul>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

function TechnologySection() {
  const techs = useMemo(
    () => [
      { name: "React", color: "#61dafb" },
      { name: "Tailwind CSS", color: "#06b6d4" },
      { name: "Framer Motion", color: "#ff6b6b" },
      { name: "FastAPI", color: "#10b981" },
      { name: "PostgreSQL", color: "#4169e1" },
      { name: "ChromaDB", color: "#f472b6" },
    ],
    []
  );
  return (
    <section
      id="technology"
      className="py-32 relative"
      style={{
        background:
          "linear-gradient(135deg, #1e3a8a 0%, #1e40af 30%, #0284c7 60%, #0ea5e9 100%)",
      }}
    >
      {" "}
      <div className="max-w-6xl mx-auto px-4 text-center relative z-10">
        <motion.h2
          className="text-5xl font-bold mb-8"
          initial={{ y: -20, opacity: 0 }}
          whileInView={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          style={{ color: "#38bdf8" }}
        >
          Technology
        </motion.h2>
        <div className="flex flex-wrap justify-center gap-8 mt-12">
          {techs.map((t, i) => (
            <motion.div
              key={t.name}
              initial={{ y: 40, opacity: 0 }}
              whileInView={{ y: 0, opacity: 1 }}
              transition={{ duration: 0.7, delay: i * 0.1 }}
              viewport={{ once: true }}
              whileHover={{
                scale: 1.1,
                y: -10,
                boxShadow: `0 20px 40px ${t.color}30`,
              }}
              className="rounded-2xl px-8 py-6 shadow-lg text-2xl font-bold"
              style={{ background: `${t.color}22`, color: t.color }}
            >
              {t.name}
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

function CTASection() {
  return (
    <section
      id="cta"
      className="py-32 relative"
      style={{
        background:
          "linear-gradient(135deg, #1e3a8a 0%, #1e40af 30%, #0284c7 60%, #0ea5e9 100%)",
      }}
    >
      {" "}
      <div className="relative z-10 max-w-6xl mx-auto text-center px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ y: 80, opacity: 0 }}
          whileInView={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="text-6xl md:text-7xl lg:text-8xl font-black mb-8 leading-none">
            <span className="block bg-gradient-to-r from-sky-400 to-cyan-400 bg-clip-text text-transparent">
              Ready to Dive
            </span>
            <span className="block bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
              Into Ocean Data?
            </span>
          </h2>
          <motion.button
            whileHover={{
              scale: 1.05,
              boxShadow: "0 25px 50px rgba(56,189,248,0.4)",
              y: -5,
            }}
            whileTap={{ scale: 0.95 }}
            className="bg-gradient-to-r from-blue-500 to-cyan-500 text-white px-12 py-6 rounded-3xl text-xl font-bold transition-all duration-400 shadow-2xl"
            onClick={() => (window.location.href = "/")}
          >
            Start Your Ocean Journey
          </motion.button>
        </motion.div>
      </div>
    </section>
  );
}

const Footer = React.memo(() => {
  return (
    <footer className="py-12 relative text-center" style={{
    background: "linear-gradient(135deg, #1e3a8a 0%, #1e40af 30%, #0284c7 60%, #0ea5e9 100%)"}}>
      <div className="max-w-6xl mx-auto px-4 relative z-10">
        <div className="mb-4 font-bold text-xl" style={{ color: "#38bdf8" }}>
          Aquaverse
        </div>
        <div className="text-slate-300 mb-2">
          © {new Date().getFullYear()} Aquaverse. All rights reserved.
        </div>
        <div className="flex justify-center gap-4 mt-4">
          {[
            {
              href: "https://github.com/CodePhoenix-org/Aquaverse",
              label: "GitHub",
            },
            { href: "https://argo.ucsd.edu/", label: "Argo" },
          ].map((link) => (
            <motion.a
              key={link.label}
              href={link.href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-400 transition-colors duration-300"
              whileHover={{ y: -2, color: "#38bdf8" }}
              whileTap={{ scale: 0.95 }}
            >
              {link.label}
            </motion.a>
          ))}
        </div>
      </div>
    </footer>
  );
});

export default function AquaverseLandingStandalone() {
  useEffect(() => {
    if (!document.getElementById("aquaverse-landing-styles")) {
      const style = document.createElement("style");
      style.id = "aquaverse-landing-styles";
      style.innerHTML = `body { margin: 0; font-family: 'Inter', sans-serif; background-color: #0f172a; color: #e0f2fe; } .glassmorphic { background: rgba(255,255,255,0.07); backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.08); }`;
      document.head.appendChild(style);
    }
  }, []);

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
      <div style={{ position: "relative", zIndex: 10 }}>
        <Footer />
      </div>
    </div>
  );
}
