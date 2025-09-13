import Navigation from "@/components/Navigation";
import HeroSection from "@/components/HeroSection";
import StatsSection from "@/components/StatsSection";
import FeatureSection from "@/components/FeatureSection";
import FeaturesListSection from "@/components/FeaturesListSection";
import CTASection from "@/components/CTASection";
import Footer from "@/components/Footer";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <main>
        <HeroSection />
        <StatsSection />
        <FeatureSection />
        <FeaturesListSection />
        <CTASection />
      </main>
      <Footer />
    </div>
  );
};

export default Index;
