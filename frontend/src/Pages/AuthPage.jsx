import { useState } from "react";
import { motion } from "framer-motion";
import { Waves } from "lucide-react";
import LoginForm from "../components/LoginForm";
import SignupForm from "../components/SignupForm";
import { ImagesSlider } from "../components/ImagesSlider";

export default function Auth() {
  const [isLogin, setIsLogin] = useState(true);
  const images = [
    "/images/ocean1.png",
    "/images/ocean2.png",
    "/images/ocean3.png",
    "/images/ocean4.png",
  ];

  return (
    <div className="min-h-screen relative">
      <ImagesSlider
        images={images}
        overlay={true}
        overlayClassName="bg-black/50"
        className="min-h-screen"
        autoplay={true}
        direction="up"
      >
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="relative z-50 bg-white/10 backdrop-blur-lg rounded-3xl p-8 max-w-md w-full border border-white/20 mx-auto mt-16 shadow-2xl"
        >
          {/* Header */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-r from-cyan-400 to-blue-500 mb-4 shadow-lg shadow-blue-500/25">
              <Waves className="w-10 h-10 text-white animate-pulse" />
            </div>
            <h1 className="text-3xl font-bold text-white mb-2 tracking-wider">
              AquaVerse
            </h1>
            <p className="text-blue-200 text-sm">
              Explore the depths of marine science
            </p>
          </div>

          {/* Forms */}
          {isLogin ? <LoginForm /> : <SignupForm />}

          {/* Toggle */}
          <div className="text-center mt-6">
            <p className="text-blue-200 text-sm">
              {isLogin ? "New to oceanography? " : "Already exploring with us? "}
              <button
                onClick={() => setIsLogin(!isLogin)}
                className="text-cyan-400 hover:text-cyan-300 underline"
              >
                {isLogin ? "Start your journey" : "Welcome back"}
              </button>
            </p>
          </div>
        </motion.div>
      </ImagesSlider>
    </div>
  );
}
