import { Button } from "@/components/ui/button";
import { ArrowRight, Play } from "lucide-react";
import heroWaves from "@/assets/hero-waves.jpg";
import bgVideoUrl from "@/assets/Bg-video.mp4";
// import Typed from "react-typed";

const HeroSection = () => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background video */}
      <div className="absolute inset-0 z-0">
        <video
          className="w-full h-full object-cover"
          src={bgVideoUrl}
          poster={heroWaves}
          autoPlay
          muted
          loop
          playsInline
          aria-hidden="true"
        />
        <div className="absolute inset-0 bg-gradient-hero opacity-50" />
      </div>

      {/* Content */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <div className="space-y-8 animate-fade-in">
          {/* Main headline */}
        {/* <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold text-primary-foreground leading-tight animate-slide-fade">
  Unlock the secrets of our{" "}
  <span className="bg-gradient-to-r from-[#00C9FF] to-[#000080] bg-clip-text text-transparent">
    oceans
  </span>{" "}
  with AI
</h1> */}

<h1 className="text-5xl md:text-7xl lg:text-8xl font-bold font-serif leading-tight mb-5 text-white glow-text ">
  <span className="typing-effect" style={{ "--delay": "0s" }}>
    Unlock the secrets of  
  </span>
  <br />
  <span className="typing-effect" style={{ "--delay": "4s" }}>
   oceans with AI
  </span>
</h1>



{/* <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold text-primary-foreground leading-tight">
  <Typed
    strings={[
      "Unlock the secrets of our oceans with AI",
      "Explore ARGO float data with conversational AI",
      "Transform complex oceanographic data into actionable insights"
    ]}
    typeSpeed={50}
    backSpeed={30}
    loop
  />
</h1> */}


          {/* Subheading */}
          <p className="text-xl md:text-2xl text-primary-foreground/90 max-w-4xl mx-auto leading-relaxed">
            Explore ARGO float data with conversational AI, geospatial dashboards, 
            and intuitive visualizations. Transform complex oceanographic data into actionable insights.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
            <Button variant="cta" size="lg" className="text-lg px-8 py-4 h-auto">
              Start Exploring
              <ArrowRight className="w-5 h-5 ml-2" />
            </Button>
            {/* <Button 
              variant="outline" 
              size="lg" 
              className="text-lg px-8 py-4 h-auto bg-primary-foreground/10 border-primary-foreground/20 text-primary-foreground hover:bg-primary-foreground/20"
            >
              <Play className="w-5 h-5 mr-2" />
              Watch Demo
            </Button> */}
          </div>

          {/* Floating stats preview */}
          {/* <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-3xl mx-auto pt-12">
            {[
              { value: "3000+", label: "Active ARGO Floats" },
              { value: "50M+", label: "Data Profiles" },
              { value: "20+", label: "Years of Data" }
            ].map((stat, index) => (
              <div 
                key={index}
                className="bg-primary-foreground/10 backdrop-blur-sm rounded-lg p-6 border border-primary-foreground/20 hover:bg-primary-foreground/20 transition-smooth"
              >
                <div className="text-2xl font-bold text-primary-foreground">{stat.value}</div>
                <div className="text-primary-foreground/80 text-sm">{stat.label}</div>
              </div>
            ))}
          </div> */}
        </div>
      </div>

      {/* Scroll indicator */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
        <div className="w-6 h-10 border-2 border-primary-foreground/40 rounded-full flex justify-center">
          <div className="w-1 h-3 bg-primary-foreground/60 rounded-full mt-2 animate-pulse" />
        </div>
      </div>
    </section>
  );
};

export default HeroSection;

