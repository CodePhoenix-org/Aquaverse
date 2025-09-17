import { useState } from "react";
import { motion } from "framer-motion";
import { Waves } from "lucide-react";
import LoginForm from "../components/LoginForm";
import SignupForm from "../components/SignupForm";
import { ImagesSlider } from "../components/ImagesSlider";
import { Boxes } from "../components/Boxes";

export default function Auth() {
  const [isLogin, setIsLogin] = useState(true);
  const images = [
    "/images/ocean1.png",
    "/images/ocean2.png",
    "/images/ocean3.png",
    "/images/ocean4.png",
  ];

  return (
    <div className="min-h-screen flex flex-col lg:flex-row">
      {/* Left Half - Image Slider */}
      <div className="w-full lg:w-1/2 h-screen lg:h-screen relative overflow-hidden">
        <ImagesSlider
          images={images}
          overlay={true}
          overlayClassName="bg-gradient-to-br from-black/40 to-black/60"
          className="h-full w-full"
          autoplay={true}
          direction="up"
        />
        
        {/* Overlay Content on Images */}
        <div className="absolute inset-0 z-10 flex flex-col justify-center items-center text-white p-4 lg:p-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-center"
          >
            <div className="inline-flex items-center justify-center w-16 h-16 lg:w-24 lg:h-24 rounded-full bg-gradient-to-r from-cyan-400/20 to-blue-500/20 backdrop-blur-sm mb-4 lg:mb-6 border border-white/20">
              <Waves className="w-8 h-8 lg:w-12 lg:h-12 text-white animate-pulse" />
            </div>
            <h1 className="text-3xl lg:text-5xl font-bold mb-2 lg:mb-4 tracking-wider bg-gradient-to-r from-cyan-300 to-blue-300 bg-clip-text text-transparent">
              AquaVerse
            </h1>
            <p className="text-lg lg:text-xl text-blue-100 mb-1 lg:mb-2">
              Explore the depths of marine science
            </p>
            <p className="text-sm lg:text-lg text-blue-200/80">
              AI-Powered Ocean Data Discovery
            </p>
          </motion.div>
        </div>
      </div>

      {/* Right Half - Auth Form */}
  <div className="w-full lg:w-1/2 h-screen lg:h-screen relative flex items-center justify-center p-4 lg:p-12 overflow-hidden bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
        {/* Animated Boxes Background */}
        <Boxes />
        {/* Form Content Overlay */}
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
          className="w-full max-w-md lg:max-w-lg z-10"
        >
          <div className="bg-white/5 backdrop-blur-lg rounded-2xl lg:rounded-3xl p-6 lg:p-8 border border-white/20 shadow-2xl">
            {/* Header */}
            <div className="text-center mb-6 lg:mb-8">
              <div className="inline-flex items-center justify-center w-12 h-12 lg:w-16 lg:h-16 rounded-full bg-gradient-to-r from-cyan-400 to-blue-500 mb-3 lg:mb-4 shadow-lg shadow-blue-500/25">
                <Waves className="w-6 h-6 lg:w-8 lg:h-8 text-white animate-pulse" />
              </div>
              <h2 className="text-xl lg:text-2xl font-bold text-white mb-2 tracking-wider">
                {isLogin ? "Welcome Back" : "Join AquaVerse"}
              </h2>
              <p className="text-blue-200 text-xs lg:text-sm">
                {isLogin ? "Sign in to continue your exploration" : "Start your oceanographic journey"}
              </p>
            </div>

            {/* Forms */}
            {isLogin ? <LoginForm /> : <SignupForm />}

            {/* Toggle */}
            <div className="text-center mt-4 lg:mt-6">
              <p className="text-blue-200 text-xs lg:text-sm">
                {isLogin ? "New to oceanography? " : "Already exploring with us? "}
                <button
                  onClick={() => setIsLogin(!isLogin)}
                  className="text-cyan-400 hover:text-cyan-300 underline font-medium transition-colors"
                >
                  {isLogin ? "Start your journey" : "Welcome back"}
                </button>
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
