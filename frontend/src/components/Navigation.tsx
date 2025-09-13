import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Menu, X } from "lucide-react";
import Logo from "../assets/logo.png"; 
import { FloatingDock } from "@/components/ui/floating-dock";
import { Home, BarChart2, Bot, Info, Mail } from "lucide-react";

const Navigation = () => {
  const [isOpen, setIsOpen] = useState(false);

  const navLinks = [
    { name: "Home", href: "#" },
    { name: "Dashboards", href: "#dashboards" },
    { name: "Chatbot", href: "#chatbot" },
    { name: "About", href: "#about" },
    { name: "Contact", href: "#contact" },
  ];

  const dockItems = [
    { title: "Home", href: "#", icon: <Home className="w-6 h-6" /> },
    { title: "Dashboards", href: "#dashboards", icon: <BarChart2 className="w-6 h-6" /> },
    { title: "Chatbot", href: "#chatbot", icon: <Bot className="w-6 h-6" /> },
    { title: "About", href: "#about", icon: <Info className="w-6 h-6" /> },
    { title: "Contact", href: "#contact", icon: <Mail className="w-6 h-6" /> },
  ];

  return (
//     <nav className="sticky top-0 z-50 bg-background/80 backdrop-blur-md border-b border-border">
//       <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
//         <div className="flex justify-between items-center h-16">
//           {/* Logo */}
//           <a href="#" className="flex items-center space-x-2">
//             <div className="w-8 h-8 rounded-lg flex items-center justify-center">
//               <img src={Logo} alt="AquaVerse Logo" className="w-8 h-8 object-contain" />
//             </div>
//             <span className="text-xl font-bold text-foreground">AquaVerse</span>
//           </a>

//           {/* Desktop Navigation - icons only */}
//           <div className="md:flex items-center space-x-8 ml-auto">
//   <FloatingDock items={dockItems} />
//   <Button
//   variant="hero"
//   size="default"
//   className="
//     text-base font-semibold 
//     transition-all duration-300 
//     hover:text-lg hover:scale-110
//     hover:shadow-[0_0_15px_#3B82F6,0_0_30px_#60A5FA,0_0_45px_#93C5FD]
//   "
// >
//   Get Started
// </Button>

// </div>


//           {/* Mobile menu button */}
//           <div className="md:hidden ml-auto">
//             <Button
//               variant="ghost"
//               size="icon"
//               onClick={() => setIsOpen(!isOpen)}
//             >
//               {isOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
//             </Button>
//           </div>
//         </div>

//         {/* Mobile Navigation */}
//         {isOpen && (
//           <div className="md:hidden">
//             <div className="px-2 pt-0 pb-0 bg-card rounded-lg mt-0 shadow-soft flex items-center justify-between">
//               <FloatingDock items={dockItems} mobileClassName="" />
//               <Button variant="hero" size="default">Get Started</Button>
//             </div>
//           </div>
//         )}
//       </div>
//       {/* Global floating dock (optional): remove if you only want it embedded in navbar) */}
//       {/* <FloatingDock items={dockItems} /> */}
//     </nav>
<nav className="sticky top-0 z-50 bg-background/80 backdrop-blur-md border-b border-border">
  <div className="w-full px-4 sm:px-6 lg:px-8">
    <div className="flex justify-between items-center h-16">
      
      {/* Logo */}
     <a href="#" className="flex items-center space-x-3">
  <div className="w-12 h-12 rounded-lg flex items-center justify-center">
    <img src={Logo} alt="AquaVerse Logo" className="w-13 h-13 object-contain" />
  </div>
  <span className="text-2xl font-bold text-foreground">AquaVerse</span>
</a>

      {/* Desktop Navigation */}
      <div className="hidden md:flex items-center space-x-8">
        <FloatingDock items={dockItems} />
        <Button
          variant="hero"
          size="default"
          className="
            text-base font-semibold 
            transition-all duration-300 
            hover:text-lg hover:scale-110
            hover:shadow-[0_0_15px_#3B82F6,0_0_30px_#60A5FA,0_0_45px_#93C5FD]
          "
        >
          Get Started
        </Button>
      </div>

      {/* Mobile menu button */}
      <div className="md:hidden">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setIsOpen(!isOpen)}
        >
          {isOpen ? <X className="w-10 h-10" /> : <Menu className="w-10 h-10" />}
        </Button>
      </div>
    </div>

    {/* Mobile Navigation */}
    {isOpen && (
  <div className="md:hidden">
    <div className="px-2 py-2 bg-card rounded-lg mt-2 shadow-soft flex items-center justify-between">
      <FloatingDock items={dockItems} mobileClassName="" />
      <Button
        variant="hero"
        size="default"
        className="glow-button px-6 py-3 rounded-lg"
      >
        Get Started
      </Button>
    </div>
  </div>
)}
  </div>
</nav>

  );
};

export default Navigation;
