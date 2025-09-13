import { Button } from "@/components/ui/button";
import { ArrowRight, MessageSquare, BarChart3, Download, Bot } from "lucide-react";
import { useState } from "react";
import dashboardMockup from "@/assets/dashboard-mockup.jpg";

const FeatureSection = () => {
  const [activeDashboard, setActiveDashboard] = useState(null);

  // Original 4 features for the interactive dashboard cards
  const dashboardCards = [
    {
      id: 'nlp',
      title: 'Natural Language Processing',
      description: 'Show me temperature anomalies in the North Atlantic during El Niño years',
      icon: MessageSquare,
      content: {
        title: 'Natural Language Processing',
        subtitle: 'Query ARGO data in plain English',
        description: 'Ask natural language questions and instantly generate visualizations. Our AI understands complex oceanographic queries and delivers publication-ready charts and insights.',
        example: 'Show me temperature anomalies in the North Atlantic during El Niño years',
        features: [
          'Advanced NLP algorithms',
          'Context-aware query understanding',
          'Multi-language support',
          'Real-time processing'
        ]
      }
    },
    {
      id: 'visualizations',
      title: 'Instant Visualizations',
      description: 'Generate maps, time series, and 3D profiles automatically',
      icon: BarChart3,
      content: {
        title: 'Instant Visualizations',
        subtitle: 'Generate maps, time series, and 3D profiles automatically',
        description: 'Transform your queries into stunning visual representations. Create interactive maps, time series charts, and 3D oceanographic profiles with a single command.',
        example: 'Generate a 3D temperature profile of the Pacific Ocean',
        features: [
          'Interactive 3D visualizations',
          'Real-time data mapping',
          'Customizable chart types',
          'Publication-ready exports'
        ]
      }
    },
    {
      id: 'export',
      title: 'Export & Share',
      description: 'Download data in multiple formats: CSV, NetCDF, or images',
      icon: Download,
      content: {
        title: 'Export & Share',
        subtitle: 'Download data in multiple formats',
        description: 'Export your oceanographic data and visualizations in various formats. Share insights with colleagues or integrate data into your research workflow.',
        example: 'Export North Atlantic temperature data as NetCDF',
        features: [
          'Multiple export formats',
          'High-resolution images',
          'Collaborative sharing',
          'API integration'
        ]
      }
    },
    {
      id: 'assistant',
      title: 'Try the AI Assistant',
      description: 'Interactive AI-powered oceanographic research assistant',
      icon: Bot,
      content: {
        title: 'AI Assistant',
        subtitle: 'Your intelligent oceanographic research partner',
        description: 'Interact with our advanced AI assistant to explore ARGO data, generate insights, and discover patterns in oceanographic phenomena.',
        example: 'Help me analyze seasonal temperature variations',
        features: [
          'Conversational interface',
          'Predictive analytics',
          'Research recommendations',
          'Data interpretation'
        ]
      }
    }
  ];

  const defaultContent = {
    title: 'Query ARGO data in plain English',
    subtitle: 'Advanced Oceanographic AI Platform',
    description: 'Ask natural language questions and instantly generate visualizations. Our AI understands complex oceanographic queries and delivers publication-ready charts and insights.',
    example: 'Show me temperature anomalies in the North Atlantic during El Niño years',
    features: [
      'Natural Language Processing',
      'Instant Visualizations', 
      'Export & Share',
      'AI Assistant'
    ]
  };

  const currentContent = activeDashboard ? 
    dashboardCards.find(card => card.id === activeDashboard)?.content : 
    defaultContent;

  return (
    <section className="py-20 bg-background relative overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center min-h-[600px]">
          {/* Left Column - Dynamic Content */}
          <div className="space-y-8 transition-all duration-700 ease-in-out">
            <div className="space-y-4">
              <h2 className="text-4xl md:text-6xl lg:text-7xl font-bold text-foreground mb-4 font-serif leading-tight">
                {currentContent.title}
              </h2>
              <p className="text-xl text-muted-foreground leading-relaxed">
                {currentContent.description}
              </p>
            </div>

            {currentContent.example && (
              <div className="bg-primary/5 border-l-4 border-primary p-4 rounded-r-lg">
                <p className="text-sm text-primary font-medium mb-2">Example Query:</p>
                <p className="text-muted-foreground italic">"{currentContent.example}"</p>
              </div>
            )}

            {currentContent.features && (
              <div className="space-y-4">
                {currentContent.features.map((feature, index) => (
                  <div key={index} className="flex items-start space-x-4">
                <div className="w-8 h-8 bg-gradient-ocean rounded-lg flex items-center justify-center mt-1">
                  <div className="w-2 h-2 bg-primary-foreground rounded-full" />
                </div>
                <div>
                      <h3 className="font-semibold text-foreground mb-1">{feature}</h3>
                      {activeDashboard && (
                        <p className="text-muted-foreground text-sm">
                          {dashboardCards.find(card => card.id === activeDashboard)?.content.features[index]}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}

            <Button variant="hero" size="lg" className="text-lg">
              {activeDashboard === 'assistant' ? 'Start Conversation' : 'Try the AI Assistant'}
              <ArrowRight className="w-5 h-5 ml-2" />
            </Button>
          </div>

          {/* Right Column - Interactive Dashboard Cards */}
          <div className="relative h-[600px] flex items-center justify-center">
            <div className="grid grid-cols-2 gap-8 w-full max-w-2xl">
              {dashboardCards.map((card, index) => {
                const Icon = card.icon;
                return (
                  <div
                    key={card.id}
                    onClick={() => setActiveDashboard(card.id)}
                    className={`
                      relative cursor-pointer group transition-all duration-500 ease-out
                      ${activeDashboard === card.id 
                        ? 'transform scale-105 rotate-1 z-10' 
                        : 'transform hover:scale-102 hover:rotate-0.5'
                      }
                    `}
                    style={{
                      transformStyle: 'preserve-3d',
                      animationDelay: `${index * 100}ms`
                    }}
                  >
                    <div className={`
                      bg-gradient-to-br from-primary/10 to-accent/10 backdrop-blur-sm 
                      border border-white/20 rounded-2xl p-8 h-64 w-full flex flex-col items-center justify-center
                      transition-all duration-500 ease-out
                      ${activeDashboard === card.id 
                        ? 'shadow-2xl shadow-primary/25 bg-gradient-to-br from-primary/20 to-accent/20' 
                        : 'shadow-lg hover:shadow-xl'
                      }
                    `}>
                      <div className="mb-6">
                        <div className={`
                          w-20 h-20 rounded-2xl flex items-center justify-center
                          transition-all duration-500
                          ${activeDashboard === card.id 
                            ? 'bg-primary/30 shadow-lg' 
                            : 'bg-primary/20 group-hover:bg-primary/25'
                          }
                        `}>
                          <Icon className={`
                            w-10 h-10 transition-all duration-500
                            ${activeDashboard === card.id 
                              ? 'text-primary scale-110' 
                              : 'text-primary/80 group-hover:text-primary'
                            }
                          `} />
                </div>
              </div>

                      <h3 className={`
                        text-xl font-semibold text-center mb-3 transition-all duration-500
                        ${activeDashboard === card.id 
                          ? 'text-foreground scale-105' 
                          : 'text-foreground/90 group-hover:text-foreground'
                        }
                      `}>
                        {card.title}
                      </h3>
                      
                      <p className={`
                        text-base text-muted-foreground text-center leading-relaxed transition-all duration-500
                        ${activeDashboard === card.id 
                          ? 'text-muted-foreground/80' 
                          : 'text-muted-foreground/70 group-hover:text-muted-foreground/90'
                        }
                      `}>
                        {card.description}
                  </p>
                </div>
                  </div>
                );
              })}
              </div>

            {/* Background Dashboard Mockup */}
            <div className="absolute inset-0 -z-10 opacity-20">
              <div className="relative overflow-hidden rounded-2xl shadow-ocean h-full">
                <img 
                  src={dashboardMockup} 
                  alt="Interactive oceanographic dashboard"
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-tr from-primary/10 to-transparent" />
              </div>
                </div>
                </div>
              </div>
            </div>

    </section>
  );
};

export default FeatureSection;