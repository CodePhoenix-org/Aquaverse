import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/Authcontext';
import { 
  AlertTriangle, 
  ThermometerSun, 
  Droplets, 
  Leaf, 
  Wind, 
  MapPin, 
  Calendar, 
  Zap, 
  Shield, 
  BarChart3, 
  AlertCircle,
  CheckCircle,
  X,
  Send,
  Globe,
  Waves,
  Search,
  ChevronLeft,
  Menu,
  Moon,
  Sun,
  Sparkles,
  MessageCircle
} from 'lucide-react';

// Inject custom scrollbar styles globally (only once)
if (typeof window !== 'undefined') {
  const styleId = 'disaster-predictor-scrollbar-style';
  if (!document.getElementById(styleId)) {
    const style = document.createElement('style');
    style.id = styleId;
    style.innerHTML = `
      .custom-scrollbar::-webkit-scrollbar { width: 8px; border-radius: 8px; }
      .custom-scrollbar::-webkit-scrollbar-thumb { background: #374151; border-radius: 8px; }
      .custom-scrollbar.light::-webkit-scrollbar-thumb { background: #e5e7eb; }
      .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
      
      @media (max-width: 768px) {
        .sidebar-enter {
          transform: translateX(0);
        }
        .sidebar-leave {
          transform: translateX(-100%);
        }
      }
    `;
    document.head.appendChild(style);
  }
}

// Utility function to calculate Tropical Cyclone Heat Potential (simplified)
const calculateTCHP = (tempProfile, depthThreshold = 100) => {
  const avgTemp = tempProfile.reduce((sum, t) => sum + t, 0) / tempProfile.length;
  return Math.max(0, (avgTemp - 26) * depthThreshold * 0.1);
};

// Risk assessment based on thresholds (from NOAA/AOML research)
const assessRisk = (tchp, sst, salinity, chl, oxy) => {
  const baseRisk = tchp > 100 ? 0.8 : tchp > 50 ? 0.5 : 0.2;
  const sstMod = sst > 28.5 ? 0.3 : sst > 26.5 ? 0.1 : 0;
  const salMod = salinity < 34 ? 0.2 : 0;
  const chlMod = chl > 0.3 ? -0.1 : 0;
  const oxyMod = oxy < 50 ? 0.1 : 0;

  const totalRisk = Math.min(1, baseRisk + sstMod + salMod + chlMod + oxyMod);
  const level = totalRisk > 0.7 ? 'High' : totalRisk > 0.4 ? 'Medium' : 'Low';
  const probability = Math.round(totalRisk * 100);

  return { level, probability, totalRisk };
};

// Fetch mock ARGO-like data (simulated API call)
const fetchOceanData = async (lat, lon, date) => {
  await new Promise(resolve => setTimeout(resolve, 1500));
  
  const baseTemp = [29.5, 29.0, 28.5, 27.0, 20.0, 15.0];
  const baseSal = 34.3;
  const baseChl = 0.1;
  const baseOxy = 58.9;
  
  const sst = 29.9 + (Math.random() - 0.5) * 2;
  const salinity = baseSal + (Math.random() - 0.5) * 1;
  const chl = baseChl + Math.random() * 0.2;
  const oxy = baseOxy + (Math.random() - 0.5) * 20;
  
  const tempProfile = baseTemp.map(t => t + (Math.random() - 0.5) * 1);
  const tchp = calculateTCHP(tempProfile);
  
  return { sst, salinity, chl, oxy, tempProfile, tchp };
};

const renderRiskCard = (data, risk, themeClasses) => (
  <div className={`p-4 sm:p-6 rounded-2xl ${themeClasses.border} border ${
    risk ? 
      (risk.level === 'High' ? 'bg-red-500/10 border-red-500/20' : 
       risk.level === 'Medium' ? 'bg-yellow-500/10 border-yellow-500/20' : 
       'bg-green-500/10 border-green-500/20')
      : `${themeClasses.cardBg}`
  }`}>
    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4 gap-3">
      <h2 className={`text-xl sm:text-2xl font-bold ${themeClasses.text}`}>Cyclone Risk Assessment</h2>
      {risk && (
        <div className={`px-3 py-1 rounded-full text-sm font-semibold w-fit ${
          risk.level === 'High' ? 'bg-red-500 text-white' : 
          risk.level === 'Medium' ? 'bg-yellow-500 text-white' : 
          'bg-green-500 text-white'
        }`}>
          {risk.level} Risk
        </div>
      )}
    </div>
    {risk ? (
      <div className="space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
          <div className="text-center p-3 rounded-lg bg-white/10 col-span-2 md:col-span-1">
            <div className="text-xl sm:text-2xl font-bold text-blue-400">{risk.probability}%</div>
            <div className={`${themeClasses.textMuted} text-xs sm:text-sm`}>Rapid Intensification Probability</div>
          </div>
          <div className="text-center p-3 rounded-lg bg-white/10">
            <ThermometerSun className="w-5 h-5 sm:w-6 sm:h-6 mx-auto mb-2 text-orange-400" />
            <div className={`${themeClasses.text} font-medium text-sm`}>SST: {data.sst.toFixed(1)}°C</div>
            <div className={`${themeClasses.textMuted} text-xs`}>High if greater than 28.5°C</div>
          </div>
          <div className="text-center p-3 rounded-lg bg-white/10">
            <Droplets className="w-5 h-5 sm:w-6 sm:h-6 mx-auto mb-2 text-blue-400" />
            <div className={`${themeClasses.text} font-medium text-sm`}>Salinity: {data.salinity.toFixed(1)} PSU</div>
            <div className={`${themeClasses.textMuted} text-xs`}>Low promotes mixing</div>
          </div>
          <div className="text-center p-3 rounded-lg bg-white/10">
            <Leaf className="w-5 h-5 sm:w-6 sm:h-6 mx-auto mb-2 text-green-400" />
            <div className={`${themeClasses.text} font-medium text-sm`}>Chl-a: {data.chl.toFixed(2)} mg/m³</div>
            <div className={`${themeClasses.textMuted} text-xs`}>Indicates productivity</div>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
          <div>
            <h3 className={`font-semibold mb-2 flex items-center ${themeClasses.text} text-sm sm:text-base`}>
              <Wind className="w-4 h-4 sm:w-5 sm:h-5 mr-2 flex-shrink-0" /> Tropical Cyclone Heat Potential (TCHP)
            </h3>
            <div className={`${themeClasses.textMuted} text-sm`}>{data.tchp.toFixed(0)} kJ/cm²</div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-2">
              <div className="bg-blue-600 h-2 rounded-full transition-all" 
                   style={{ width: `${Math.min((data.tchp / 150) * 100, 100)}%` }}></div>
            </div>
          </div>
          <div>
            <h3 className={`font-semibold mb-2 flex items-center ${themeClasses.text} text-sm sm:text-base`}>
              <Zap className="w-4 h-4 sm:w-5 sm:h-5 mr-2 flex-shrink-0" /> Oxygen Levels
            </h3>
            <div className={`${themeClasses.textMuted} text-sm`}>{data.oxy.toFixed(1)} µmol/kg</div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-2">
              <div className={`h-2 rounded-full transition-all ${data.oxy < 50 ? 'bg-red-600' : 'bg-purple-600'}`} 
                   style={{ width: `${Math.min((data.oxy / 200) * 100, 100)}%` }}></div>
            </div>
          </div>
        </div>
        <div className="mt-6 p-3 sm:p-4 rounded-lg bg-white/5">
          <p className={`${themeClasses.textSecondary} text-xs sm:text-sm leading-relaxed`}>
            Based on ARGO-like profile data, this assessment uses subsurface temperature, salinity stratification, and biogeochemical indicators to predict cyclone intensification potential. High risk indicates favorable conditions for rapid intensification (≥30 knots/24h).
          </p>
        </div>
      </div>
    ) : (
      <p className={`${themeClasses.textMuted} text-center py-8 text-sm sm:text-base`}>Enter coordinates and date to assess cyclone risk.</p>
    )}
  </div>
);

const Predictor = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  
  const [lat, setLat] = useState(9);
  const [lon, setLon] = useState(68);
  const [date, setDate] = useState('2025-09-22');
  const [isLoading, setIsLoading] = useState(false);
  const [data, setData] = useState(null);
  const [risk, setRisk] = useState(null);
  const [darkMode, setDarkMode] = useState(() => {
    try {
      return localStorage.getItem('darkMode') === 'true';
    } catch {
      return true;
    }
  });
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const themeClasses = {
    bg: darkMode ? 'bg-gray-900' : 'bg-white',
    sidebarBg: darkMode ? 'bg-gray-800' : 'bg-gray-50',
    cardBg: darkMode ? 'bg-gray-800' : 'bg-white',
    text: darkMode ? 'text-white' : 'text-gray-900',
    textSecondary: darkMode ? 'text-gray-300' : 'text-gray-600',
    textMuted: darkMode ? 'text-gray-400' : 'text-gray-500',
    border: darkMode ? 'border-gray-700' : 'border-gray-200',
    borderLight: darkMode ? 'border-gray-600' : 'border-gray-100',
    hoverBg: darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100',
    inputBg: darkMode ? 'bg-gray-800' : 'bg-gray-50',
    inputBorder: darkMode ? 'border-gray-600' : 'border-gray-200',
    inputText: darkMode ? 'text-white' : 'text-gray-900',
    inputFocus: darkMode ? 'ring-blue-400/20' : 'ring-blue-50',
    buttonText: darkMode ? 'text-white' : 'text-gray-900'
  };

  useEffect(() => {
    localStorage.setItem('darkMode', darkMode.toString());
  }, [darkMode]);

  const handlePredict = async () => {
    if (!lat || !lon || !date) return;
    setIsLoading(true);
    try {
      const fetchedData = await fetchOceanData(lat, lon, date);
      setData(fetchedData);
      const assessedRisk = assessRisk(fetchedData.tchp, fetchedData.sst, fetchedData.salinity, fetchedData.chl, fetchedData.oxy);
      setRisk(assessedRisk);
    } catch (error) {
      console.error('Prediction failed:', error);
    }
    setIsLoading(false);
  };

  if (!user) {
    return (
      <div className={`h-screen ${themeClasses.bg} flex items-center justify-center p-4`}>
        <div className="text-center max-w-md w-full">
          <div className="w-16 h-16 bg-gradient-to-br from-red-500 to-orange-500 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg">
            <AlertTriangle className="w-8 h-8 text-white" />
          </div>
          <h1 className={`text-xl sm:text-2xl font-bold ${themeClasses.text} mb-4`}>Disaster Predictor</h1>
          <p className={`${themeClasses.textSecondary} mb-6 text-sm sm:text-base`}>
            Login required for advanced oceanographic risk assessment.
          </p>
          <button 
            onClick={() => navigate('/login')} 
            className="w-full sm:w-auto bg-red-500 hover:bg-red-600 text-white px-6 py-3 rounded-lg font-medium transition-colors"
          >
            Login
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`h-screen ${themeClasses.bg} flex relative overflow-hidden`}>
      {/* Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 w-64 ${themeClasses.sidebarBg} ${themeClasses.border} border-r transition-transform ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      } lg:translate-x-0 lg:static lg:z-auto`}>
        <div className="h-full flex flex-col">
          <div className="p-4 flex-shrink-0">
            <div className="flex items-center justify-between mb-6">
              <h2 className={`text-lg font-semibold ${themeClasses.text}`}>Navigation</h2>
              <button 
                onClick={() => setSidebarOpen(false)} 
                className="lg:hidden p-1 rounded hover:bg-gray-700"
              >
                <X className={`w-5 h-5 ${themeClasses.text}`} />
              </button>
            </div>
            <nav className="space-y-2">
              <button 
                onClick={() => navigate('/floatchat')} 
                className={`w-full text-left p-3 rounded-lg ${themeClasses.hoverBg} ${themeClasses.textSecondary} flex items-center space-x-3 text-sm`}
              >
                <MessageCircle className="w-4 h-4 sm:w-5 sm:h-5 flex-shrink-0" />
                <span className={`${themeClasses.textSecondary}`}>FloatChat</span>
              </button>
              <button 
                onClick={() => navigate('/dashboard')} 
                className={`w-full text-left p-3 rounded-lg ${themeClasses.hoverBg} ${themeClasses.textSecondary} flex items-center space-x-3 text-sm`}
              >
                <BarChart3 className="w-4 h-4 sm:w-5 sm:h-5 flex-shrink-0" />
                <span className={`${themeClasses.textSecondary}`}>Dashboard</span>
              </button>
            </nav>
          </div>
          {/* Sidebar spacer for lg screens */}
          <div className="flex-1 lg:block hidden"></div>
        </div>
      </div>

      {/* Sidebar Overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className={`p-3 sm:p-4 ${themeClasses.border} border-b flex items-center justify-between ${themeClasses.bg} shadow-sm`}>
          <div className="flex items-center space-x-2 sm:space-x-3 flex-1">
            <button 
              onClick={() => setSidebarOpen(true)}
              className={`p-2 ${themeClasses.hoverBg} rounded-lg transition-colors flex-shrink-0`}
            >
              <Menu className={`w-5 h-5 sm:w-6 sm:h-6 ${themeClasses.text}`} />
            </button>
            <div className="flex items-center space-x-2 flex-1 min-w-0">
              <AlertTriangle className="w-5 h-5 sm:w-6 sm:h-6 text-red-500 flex-shrink-0" />
              <h1 className={`text-lg sm:text-xl font-bold ${themeClasses.text} truncate`}>Disaster Predictor</h1>
            </div>
          </div>
          <div className="flex items-center space-x-1 sm:space-x-2">
            <button 
              onClick={() => setDarkMode(!darkMode)} 
              className={`p-2 ${themeClasses.hoverBg} rounded-lg transition-colors`}
            >
              {darkMode ? <Sun className={`w-4 h-4 sm:w-5 sm:h-5 ${themeClasses.text}`} /> : <Moon className={`w-4 h-4 sm:w-5 sm:h-5 ${themeClasses.text}`} />}
            </button>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-4 sm:p-6 custom-scrollbar">
          <div className="max-w-full sm:max-w-4xl mx-auto space-y-4 sm:space-y-6">
            {/* Input Form */}
            <div className={`${themeClasses.cardBg} p-4 sm:p-6 rounded-2xl ${themeClasses.border} border shadow-lg`}>
              <h2 className={`text-lg font-semibold mb-4 flex items-center ${themeClasses.text}`}>
                <MapPin className="w-5 h-5 mr-2 flex-shrink-0" /> Location & Time
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4">
                <div>
                  <label className={`${themeClasses.textMuted} block text-sm mb-1`}>Latitude (°N)</label>
                  <input
                    type="number"
                    value={lat}
                    onChange={e => setLat(parseFloat(e.target.value) || 0)}
                    step="0.1"
                    min="-90"
                    max="90"
                    className={`${themeClasses.inputBg} ${themeClasses.inputBorder} w-full px-3 py-2 rounded-lg ${themeClasses.inputText} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm`}
                    placeholder="e.g., 9.0"
                  />
                </div>
                <div>
                  <label className={`${themeClasses.textMuted} block text-sm mb-1`}>Longitude (°E)</label>
                  <input
                    type="number"
                    value={lon}
                    onChange={e => setLon(parseFloat(e.target.value) || 0)}
                    step="0.1"
                    min="-180"
                    max="180"
                    className={`${themeClasses.inputBg} ${themeClasses.inputBorder} w-full px-3 py-2 rounded-lg ${themeClasses.inputText} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm`}
                    placeholder="e.g., 68.0"
                  />
                </div>
                <div>
                  <label className={`${themeClasses.textMuted} block text-sm mb-1`}>Date</label>
                  <input
                    type="date"
                    value={date}
                    onChange={e => setDate(e.target.value)}
                    className={`${themeClasses.inputBg} ${themeClasses.inputBorder} w-full px-3 py-2 rounded-lg ${themeClasses.inputText} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm`}
                  />
                </div>
              </div>
              <button
                onClick={handlePredict}
                disabled={isLoading || !lat || !lon || !date}
                className={`mt-4 w-full flex items-center justify-center space-x-2 px-4 sm:px-6 py-3 rounded-lg font-medium transition-all text-sm ${
                  isLoading || !lat || !lon || !date
                    ? 'bg-gray-400 dark:bg-gray-600 cursor-not-allowed'
                    : 'bg-gradient-to-r from-red-500 to-orange-500 hover:from-red-600 hover:to-orange-600'
                } text-white`}
              >
                {isLoading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4" />
                    <span>Predict Cyclone Risk</span>
                  </>
                )}
              </button>
            </div>

            {/* Results */}
            {data && renderRiskCard(data, risk, themeClasses)}

            {/* Info Panel */}
            <div className={`${themeClasses.cardBg} p-4 sm:p-6 rounded-2xl ${themeClasses.border} border shadow-lg`}>
              <h3 className={`font-semibold mb-4 ${themeClasses.text} text-base sm:text-lg`}>How It Works</h3>
              <ul className="space-y-2 text-xs sm:text-sm">
                <li className={`${themeClasses.textSecondary}`}>
                  • Uses ARGO float-like profiles for temperature, salinity, oxygen, chlorophyll
                </li>
                <li className={`${themeClasses.textSecondary}`}>
                  • Calculates TCHP for heat available to fuel cyclones
                </li>
                <li className={`${themeClasses.textSecondary}`}>
                  • Assesses rapid intensification risk based on ocean conditions
                </li>
                <li className={`${themeClasses.textSecondary}`}>
                  • Incorporates salinity effects on mixing & stratification
                </li>
              </ul>
            </div>

            {/* Mobile bottom spacer */}
            <div className="h-16 lg:hidden"></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Predictor;