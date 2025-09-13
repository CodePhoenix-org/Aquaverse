import { MessageSquare, BarChart3, Layers } from "lucide-react";

const FeaturesListSection = () => {
  const features = [
    {
      icon: MessageSquare,
      title: "Conversational AI",
      description: "Ask questions in natural language and get instant insights from ocean data. Our AI understands complex oceanographic concepts and relationships."
    },
    {
      icon: BarChart3,
      title: "Interactive Dashboards",
      description: "Explore data through customizable visualizations. Create maps, time series, depth profiles, and comparative analyses with intuitive controls."
    },
    {
      icon: Layers,
      title: "BGC Data Support",
      description: "Access biogeochemical parameters including oxygen, chlorophyll, nitrate, and pH measurements from advanced ARGO-BGC floats."
    }
  ];

  return (
    <section className="py-20 bg-gradient-ocean-light">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl lg:text-6xl font-bold text-foreground mb-4 font-serif leading-tight">
            Powerful features for marine research
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Advanced tools designed for oceanographers, researchers, and data scientists to unlock insights from global ocean observations.
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div 
              key={index}
              className="bg-gradient-card rounded-xl p-8 shadow-soft hover:shadow-ocean transition-smooth group hover:scale-105"
            >
              {/* Icon */}
              <div className="flex items-center justify-center w-16 h-16 bg-gradient-ocean rounded-xl mb-6 group-hover:shadow-glow transition-smooth">
                <feature.icon className="w-8 h-8 text-primary-foreground" />
              </div>

              {/* Content */}
              <h3 className="text-xl font-bold text-foreground mb-4">{feature.title}</h3>
              <p className="text-muted-foreground leading-relaxed">{feature.description}</p>

              {/* Learn more link */}
              <div className="mt-6">
                <a 
                  href="#" 
                  className="text-primary font-medium hover:text-primary-glow transition-smooth inline-flex items-center group"
                >
                  Learn more
                  <svg 
                    className="w-4 h-4 ml-2 transform group-hover:translate-x-1 transition-smooth" 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturesListSection;