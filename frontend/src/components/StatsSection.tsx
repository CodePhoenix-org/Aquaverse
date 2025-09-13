import { useEffect, useRef, useState } from "react";
import { Database, Globe, TrendingUp } from "lucide-react";

const StatsSection = () => {
  const sectionRef = useRef(null);
  const [isVisible, setIsVisible] = useState(false);
  const [currentCardIndex, setCurrentCardIndex] = useState(0);

  const stats = [
    {
      icon: Database,
      value: "3000+",
      label: "Active ARGO Floats",
    },
    {
      icon: TrendingUp,     
      value: "50M+",   
      label: "Data Profiles",
    },
    {
      icon: Globe,
      value: "20+",
      label: "Years of Data",
    }
  ];

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.3 }
    );

    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }

    return () => observer.disconnect();
  }, []);

  // Slideshow effect
  useEffect(() => {
    if (!isVisible) return;

    const interval = setInterval(() => {
      setCurrentCardIndex((prevIndex) => (prevIndex + 1) % stats.length);
    }, 3000); // Change card every 3 seconds

    return () => clearInterval(interval);
  }, [isVisible, stats.length]);

  return (
    <section ref={sectionRef} className="py-20 bg-gradient-ocean-light relative overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Left-aligned content */}
          <div className="max-w-2xl">
            {/* Subtitle */}
            <p className="text-sm md:text-base font-medium text-primary mb-2 uppercase tracking-wide">
              THE BLUE PLANET
            </p>
            
            {/* Main title */}
            <h2 className="text-4xl md:text-6xl lg:text-7xl font-bold text-foreground mb-4 font-serif leading-tight">
              OCEANS
            </h2>
            
            {/* Divider line */}
            <div className="w-24 h-0.5 bg-primary mb-8"></div>
            
            {/* Description */}
            <p className="text-lg md:text-xl text-muted-foreground max-w-2xl leading-relaxed mb-8">
              Dive deep into the world's largest autonomous oceanographic monitoring network. 
              Explore real-time data from thousands of ARGO floats, uncovering the mysteries 
              of our planet's most vital ecosystem through cutting-edge AI technology.
            </p>
            
            {/* Action buttons */}
            <div className="flex flex-col sm:flex-row items-start gap-4">
              <button className="bg-primary hover:bg-primary/90 text-primary-foreground px-8 py-3 rounded-full font-medium transition-all duration-300 hover:scale-105">
                EXPLORE DATA
              </button>
              <button className="w-12 h-12 bg-primary/20 hover:bg-primary/30 text-primary rounded-full flex items-center justify-center transition-all duration-300 hover:scale-105">
                <svg className="w-5 h-5 ml-0.5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z"/>
                </svg>
              </button>
            </div>
          </div>

          {/* Right side - Slideshow container */}
          <div className="flex items-center justify-center lg:justify-end">
            <div 
              className="relative w-80 h-96 overflow-hidden"
            >
              {stats.map((stat, index) => (
                <div
                  key={index}
                  className={`
                    absolute inset-0 bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-8
                    transition-all duration-1000 ease-in-out
                    ${isVisible 
                      ? (index === currentCardIndex 
                          ? 'translate-x-0 opacity-100 scale-100' 
                          : 'translate-x-full opacity-0 scale-95')
                      : 'translate-x-full opacity-0 scale-95'
                    }
                  `}
                >
                  <div className="flex flex-col items-center justify-center h-full text-center space-y-6">
                    <div className="w-20 h-20 bg-primary/20 rounded-2xl flex items-center justify-center">
                      <stat.icon className="w-10 h-10 text-primary" />
                    </div>
                    <div>
                      <div className="text-5xl font-bold text-foreground mb-2">{stat.value}</div>
                      <div className="text-lg text-muted-foreground">{stat.label}</div>
                    </div>
                  </div>
                </div>
              ))}
              
              {/* Slideshow indicators */}
              <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex space-x-2">
                {stats.map((_, index) => (
                  <button
                    key={index}
                    className={`
                      w-2 h-2 rounded-full transition-all duration-300
                      ${index === currentCardIndex 
                        ? 'bg-primary scale-125' 
                        : 'bg-white/40 hover:bg-white/60'
                      }
                    `}
                    onClick={() => setCurrentCardIndex(index)}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default StatsSection;