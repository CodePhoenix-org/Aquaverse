import { Button } from "@/components/ui/button";
import { ArrowRight, Sparkles } from "lucide-react";
import footerImg from "@/assets/footer_img1.jpg";

const CTASection = () => {
  return (
    <section className="py-20 relative overflow-hidden">
      {/* Background Image */}
      <div className="absolute inset-0">
        <img 
          src={footerImg} 
          alt="Ocean background"
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-primary/20 via-primary/10 to-primary/30" />
      </div>

      <div className="relative z-10 max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <div className="space-y-8">
          {/* Icon */}
          <div className="flex justify-center">
            <div className="w-16 h-16 bg-primary-foreground/10 rounded-2xl flex items-center justify-center backdrop-blur-sm border border-primary-foreground/20">
              <Sparkles className="w-8 h-8 text-primary-foreground" />
            </div>
          </div>

          {/* Headline */}
          <h2 className="text-3xl md:text-5xl font-bold text-primary-foreground leading-tight">
            Ready to explore ocean data?
          </h2>

          {/* Subtext */}
          <p className="text-xl text-primary-foreground/90 max-w-2xl mx-auto leading-relaxed">
            Join thousands of researchers, oceanographers, and data scientists using our platform 
            to discover insights hidden beneath the waves.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
            <Button 
              variant="cta" 
              size="lg" 
              className="text-lg px-8 py-4 h-auto bg-primary-foreground text-primary hover:bg-primary-foreground/90"
            >
              Start Exploring Now
              <ArrowRight className="w-5 h-5 ml-2" />
            </Button>
            <Button 
              variant="outline" 
              size="lg" 
              className="text-lg px-8 py-4 h-auto bg-transparent border-primary-foreground/20 text-primary-foreground hover:bg-primary-foreground/10"
            >
              Book a Demo
            </Button>
          </div>

          {/* Trust indicators */}
          <div className="pt-8 border-t border-primary-foreground/20">
            <p className="text-primary-foreground/70 text-sm mb-4">Trusted by leading institutions</p>
            <div className="flex items-center justify-center space-x-8 text-primary-foreground/50">
              <span className="font-semibold">NOAA</span>
              <span className="font-semibold">SCRIPPS</span>
              <span className="font-semibold">WHOI</span>
              <span className="font-semibold">IFREMER</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default CTASection;