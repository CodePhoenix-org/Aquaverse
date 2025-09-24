import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/Authcontext';
import { 
  AlertTriangle, 
  ThermometerSun, 
  Droplets, 
  Leaf, 
  MapPin, 
  Zap, 
  BarChart3, 
  AlertCircle,
  X,
  Zap as ZapIcon,
  Menu,
  Moon,
  Sun,
  MessageCircle
} from 'lucide-react';

// Inject custom scrollbar styles globally
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
        .sidebar-enter { transform: translateX(0); }
        .sidebar-leave { transform: translateX(-100%); }
      }
    `;
    document.head.appendChild(style);
  }
}

// Render risk card
const renderRiskCard = (data, risk, themeClasses, error) => {
  console.log('renderRiskCard called with:', { data, risk, error });
  return (
    <div className={`p-4 sm:p-6 rounded-2xl ${themeClasses.border} border ${risk ? (risk.level === 'High' ? 'bg-red-500/10 border-red-500/20' : 'bg-green-500/10 border-green-500/20') : `${themeClasses.cardBg}`}`}>
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4 gap-3">
        <h2 className={`text-xl sm:text-2xl font-bold ${themeClasses.text}`}>Disaster Risk Assessment</h2>
        {risk && (
          <div className={`px-3 py-1 rounded-full text-sm font-semibold w-fit ${risk.level === 'High' ? 'bg-red-500 text-white' : 'bg-green-500 text-white'}`}>
            {risk.level} Risk
          </div>
        )}
      </div>
      {error ? (
        <div className="flex items-center justify-center p-4 bg-red-500/10 rounded-lg">
          <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
          <p className={`${themeClasses.textMuted} text-sm`}>{error}</p>
        </div>
      ) : data && risk ? (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
            <div className="text-center p-3 rounded-lg bg-white/10 col-span-2 md:col-span-1">
              <div className="text-xl sm:text-2xl font-bold text-blue-400">{risk.probability}%</div>
              <div className={`${themeClasses.textMuted} text-xs sm:text-sm`}>Prediction Confidence</div>
            </div>
            <div className="text-center p-3 rounded-lg bg-white/10">
              <ThermometerSun className="w-5 h-5 sm:w-6 sm:h-6 mx-auto mb-2 text-orange-400" />
              <div className={`${themeClasses.text} font-medium text-sm`}>Temperature: {data.temperature.toFixed(1)}°C</div>
              <div className={`${themeClasses.textMuted} text-xs`}>Ocean temperature</div>
            </div>
            <div className="text-center p-3 rounded-lg bg-white/10">
              <Droplets className="w-5 h-5 sm:w-6 sm:h-6 mx-auto mb-2 text-blue-400" />
              <div className={`${themeClasses.text} font-medium text-sm`}>Salinity: {data.salinity.toFixed(1)} PSU</div>
              <div className={`${themeClasses.textMuted} text-xs`}>Ocean salinity</div>
            </div>
            <div className="text-center p-3 rounded-lg bg-white/10">
              <Leaf className="w-5 h-5 sm:w-6 sm:h-6 mx-auto mb-2 text-green-400" />
              <div className={`${themeClasses.text} font-medium text-sm`}>Chl-a: {data.chlorophyll.toFixed(2)} mg/m³</div>
              <div className={`${themeClasses.textMuted} text-xs`}>Chlorophyll-a</div>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
            <div>
              <h3 className={`font-semibold mb-2 flex items-center ${themeClasses.text} text-sm sm:text-base`}>
                <MapPin className="w-4 h-4 sm:w-5 sm:h-5 mr-2 flex-shrink-0" /> Location
              </h3>
              <div className={`${themeClasses.textMuted} text-sm`}>Lat: {data.latitude.toFixed(1)}°N, Lon: {data.longitude.toFixed(1)}°E</div>
            </div>
            <div>
              <h3 className={`font-semibold mb-2 flex items-center ${themeClasses.text} text-sm sm:text-base`}>
                <Zap className="w-4 h-4 sm:w-5 sm:h-5 mr-2 flex-shrink-0" /> Oxygen & Depth
              </h3>
              <div className={`${themeClasses.textMuted} text-sm`}>Oxygen: {data.oxygen.toFixed(1)} µmol/kg, Depth: {data.depth.toFixed(0)} m</div>
            </div>
          </div>
          <div className="mt-6 p-3 sm:p-4 rounded-lg bg-white/5">
            <p className={`${themeClasses.textSecondary} text-xs sm:text-sm leading-relaxed`}>
              Based on ARGO float data, this assessment uses a machine learning model to predict potential oceanic disasters. High risk indicates conditions favorable for anomalies such as extreme weather events.
            </p>
          </div>
        </div>
      ) : (
        <p className={`${themeClasses.textMuted} text-center py-8 text-sm sm:text-base`}>Enter ocean data to assess disaster risk.</p>
      )}
    </div>
  );
};

// Render survival card (mock)
const renderSurvivalCard = (data, survival, themeClasses, error) => {
  console.log('renderSurvivalCard called with:', { data, survival, error });
  let bgClass = `${themeClasses.cardBg}`;
  let borderClass = themeClasses.border;
  let badgeBg = 'bg-gray-500';
  if (survival) {
    const level = survival.level.split(' ')[0];
    if (level === 'High') {
      bgClass = 'bg-green-500/10 border-green-500/20';
      badgeBg = 'bg-green-500 text-white';
    } else if (level === 'Medium') {
      bgClass = 'bg-yellow-500/10 border-yellow-500/20';
      badgeBg = 'bg-yellow-500 text-white';
    } else {
      bgClass = 'bg-red-500/10 border-red-500/20';
      badgeBg = 'bg-red-500 text-white';
    }
  }
  return (
    <div className={`p-4 sm:p-6 rounded-2xl ${borderClass} border ${bgClass}`}>
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4 gap-3">
        <h2 className={`text-xl sm:text-2xl font-bold ${themeClasses.text}`}>Aquatic Life Survival Assessment</h2>
        {survival && (
          <div className={`px-3 py-1 rounded-full text-sm font-semibold w-fit ${badgeBg}`}>
            {survival.level}
          </div>
        )}
      </div>
      {error ? (
        <div className="flex items-center justify-center p-4 bg-red-500/10 rounded-lg">
          <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
          <p className={`${themeClasses.textMuted} text-sm`}>{error}</p>
        </div>
      ) : data && survival ? (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
            <div className="text-center p-3 rounded-lg bg-white/10 col-span-2 md:col-span-1">
              <div className="text-xl sm:text-2xl font-bold text-blue-400">{survival.probability}%</div>
              <div className={`${themeClasses.textMuted} text-xs sm:text-sm`}>Estimated Survival Rate</div>
            </div>
            <div className="text-center p-3 rounded-lg bg-white/10">
              <ThermometerSun className="w-5 h-5 sm:w-6 sm:h-6 mx-auto mb-2 text-orange-400" />
              <div className={`${themeClasses.text} font-medium text-sm`}>Temperature: {data.temperature.toFixed(1)}°C</div>
              <div className={`${themeClasses.textMuted} text-xs`}>Ocean temperature</div>
            </div>
            <div className="text-center p-3 rounded-lg bg-white/10">
              <Zap className="w-5 h-5 sm:w-6 sm:h-6 mx-auto mb-2 text-purple-400" />
              <div className={`${themeClasses.text} font-medium text-sm`}>Oxygen: {data.oxygen.toFixed(1)} µmol/kg</div>
              <div className={`${themeClasses.textMuted} text-xs`}>Dissolved oxygen</div>
            </div>
            <div className="text-center p-3 rounded-lg bg-white/10">
              <Leaf className="w-5 h-5 sm:w-6 sm:h-6 mx-auto mb-2 text-green-400" />
              <div className={`${themeClasses.text} font-medium text-sm`}>Chl-a: {data.chlorophyll.toFixed(2)} mg/m³</div>
              <div className={`${themeClasses.textMuted} text-xs`}>Chlorophyll-a</div>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
            <div>
              <h3 className={`font-semibold mb-2 flex items-center ${themeClasses.text} text-sm sm:text-base`}>
                <MapPin className="w-4 h-4 sm:w-5 sm:h-5 mr-2 flex-shrink-0" /> Location
              </h3>
              <div className={`${themeClasses.textMuted} text-sm`}>Lat: {data.latitude.toFixed(1)}°N, Lon: {data.longitude.toFixed(1)}°E</div>
            </div>
            <div>
              <h3 className={`font-semibold mb-2 flex items-center ${themeClasses.text} text-sm sm:text-base`}>
                <Droplets className="w-4 h-4 sm:w-5 sm:h-5 mr-2 flex-shrink-0" /> Other Metrics
              </h3>
              <div className={`${themeClasses.textMuted} text-sm`}>Salinity: {data.salinity.toFixed(1)} PSU, Depth: {data.depth.toFixed(0)} m</div>
            </div>
          </div>
          <div className="mt-6 p-3 sm:p-4 rounded-lg bg-white/5">
            <p className={`${themeClasses.textSecondary} text-xs sm:text-sm leading-relaxed`}>
              This is a mock assessment for aquatic life survival based on temperature, dissolved oxygen, and chlorophyll levels (as a proxy for food availability). High survival indicates optimal environmental conditions for marine organisms.
            </p>
          </div>
        </div>
      ) : (
        <p className={`${themeClasses.textMuted} text-center py-8 text-sm sm:text-base`}>Enter ocean data to assess survival rate.</p>
      )}
    </div>
  );
};

const Predictor = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  
  // State for input fields
  const [lat, setLat] = useState(9);
  const [lon, setLon] = useState(68);
  const [depth, setDepth] = useState(980);
  const [temperature, setTemperature] = useState(29.5);
  const [salinity, setSalinity] = useState(34.3);
  const [oxygen, setOxygen] = useState(58.9);
  const [chlorophyll, setChlorophyll] = useState(0.1);
  const [isLoading, setIsLoading] = useState(false);
  const [data, setData] = useState(null);
  const [risk, setRisk] = useState(null);
  const [survival, setSurvival] = useState(null);
  const [error, setError] = useState(null);
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

  // Handle prediction API call
  const handlePredict = async () => {
    setIsLoading(true);
    setError(null);
    setData(null);
    setRisk(null);
    setSurvival(null);
    const payload = {
      latitude: parseFloat(lat),
      longitude: parseFloat(lon),
      depth: parseFloat(depth),
      temperature: parseFloat(temperature),
      salinity: parseFloat(salinity),
      oxygen: parseFloat(oxygen),
      chlorophyll: parseFloat(chlorophyll),
    };
    console.log('Sending payload:', payload);
    try {
      const response = await fetch('http://localhost:8000/predict/disaster', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error ${response.status}: ${errorText}`);
      }
      const result = await response.json();
      console.log('Received response:', result);
      setData(result);
      setRisk({
        level: result.disaster_prediction === 'Anomaly' ? 'High' : 'Low',
        probability: Math.round(result.prediction_confidence * 100),
      });

      // Mock survival rate calculation
      const temp = parseFloat(temperature);
      const oxy = parseFloat(oxygen);
      const chl = parseFloat(chlorophyll);
      let surv = 100;

      // Penalize temperature outside 20-25°C
      if (temp < 20 || temp > 25) {
        surv -= Math.abs(temp - 22.5) * 2;
      }

      // Penalize low oxygen
      if (oxy < 50) {
        surv -= (50 - oxy) * 2;
      } else if (oxy < 100) {
        surv -= (100 - oxy) * 0.5;
      }

      // Penalize chlorophyll outside 0.5-1.0 mg/m³
      if (chl < 0.5) {
        surv -= (0.5 - chl) * 100;
      } else if (chl > 1.0) {
        surv -= (chl - 1.0) * 50;
      }

      surv = Math.max(0, Math.min(100, Math.round(surv)));
      const survLevel = surv > 70 ? 'High Survival' : surv > 40 ? 'Medium Survival' : 'Low Survival';
      setSurvival({ level: survLevel, probability: surv });
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.message || 'Failed to fetch prediction');
    } finally {
      setIsLoading(false);
    }
  };

  if (!user) {
    return (
      <div className={`h-screen ${themeClasses.bg} flex items-center justify-center`}>
        <div className="text-center p-6 sm:p-8 rounded-2xl bg-gray-800/50 max-w-md w-full mx-4">
          <AlertTriangle className="w-12 h-12 mx-auto mb-4 text-red-500" />
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
                <MapPin className="w-5 h-5 mr-2 flex-shrink-0" /> Ocean Data Input
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
                <div>
                  <label className={`${themeClasses.textMuted} block text-sm mb-1`}>Latitude (°N)</label>
                  <input
                    type="number"
                    value={lat}
                    onChange={e => setLat(e.target.value ? parseFloat(e.target.value) : '')}
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
                    onChange={e => setLon(e.target.value ? parseFloat(e.target.value) : '')}
                    step="0.1"
                    min="-180"
                    max="180"
                    className={`${themeClasses.inputBg} ${themeClasses.inputBorder} w-full px-3 py-2 rounded-lg ${themeClasses.inputText} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm`}
                    placeholder="e.g., 68.0"
                  />
                </div>
                <div>
                  <label className={`${themeClasses.textMuted} block text-sm mb-1`}>Depth (m)</label>
                  <input
                    type="number"
                    value={depth}
                    onChange={e => setDepth(e.target.value ? parseFloat(e.target.value) : '')}
                    step="1"
                    min="0"
                    className={`${themeClasses.inputBg} ${themeClasses.inputBorder} w-full px-3 py-2 rounded-lg ${themeClasses.inputText} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm`}
                    placeholder="e.g., 980"
                  />
                </div>
                <div>
                  <label className={`${themeClasses.textMuted} block text-sm mb-1`}>Temperature (°C)</label>
                  <input
                    type="number"
                    value={temperature}
                    onChange={e => setTemperature(e.target.value ? parseFloat(e.target.value) : '')}
                    step="0.1"
                    className={`${themeClasses.inputBg} ${themeClasses.inputBorder} w-full px-3 py-2 rounded-lg ${themeClasses.inputText} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm`}
                    placeholder="e.g., 29.5"
                  />
                </div>
                <div>
                  <label className={`${themeClasses.textMuted} block text-sm mb-1`}>Salinity (PSU)</label>
                  <input
                    type="number"
                    value={salinity}
                    onChange={e => setSalinity(e.target.value ? parseFloat(e.target.value) : '')}
                    step="0.1"
                    className={`${themeClasses.inputBg} ${themeClasses.inputBorder} w-full px-3 py-2 rounded-lg ${themeClasses.inputText} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm`}
                    placeholder="e.g., 34.3"
                  />
                </div>
                <div>
                  <label className={`${themeClasses.textMuted} block text-sm mb-1`}>Oxygen (µmol/kg)</label>
                  <input
                    type="number"
                    value={oxygen}
                    onChange={e => setOxygen(e.target.value ? parseFloat(e.target.value) : '')}
                    step="0.1"
                    className={`${themeClasses.inputBg} ${themeClasses.inputBorder} w-full px-3 py-2 rounded-lg ${themeClasses.inputText} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm`}
                    placeholder="e.g., 58.9"
                  />
                </div>
                <div>
                  <label className={`${themeClasses.textMuted} block text-sm mb-1`}>Chlorophyll (mg/m³)</label>
                  <input
                    type="number"
                    value={chlorophyll}
                    onChange={e => setChlorophyll(e.target.value ? parseFloat(e.target.value) : '')}
                    step="0.01"
                    className={`${themeClasses.inputBg} ${themeClasses.inputBorder} w-full px-3 py-2 rounded-lg ${themeClasses.inputText} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm`}
                    placeholder="e.g., 0.1"
                  />
                </div>
              </div>
              <button
                onClick={handlePredict}
                disabled={isLoading || lat === '' || lon === '' || depth === '' || temperature === '' || salinity === '' || oxygen === '' || chlorophyll === ''}
                className={`mt-4 w-full flex items-center justify-center space-x-2 px-4 sm:px-6 py-3 rounded-lg font-medium transition-all text-sm ${
                  isLoading || lat === '' || lon === '' || depth === '' || temperature === '' || salinity === '' || oxygen === '' || chlorophyll === ''
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
                    <ZapIcon className="w-4 h-4" />
                    <span>Predict Disaster Risk</span>
                  </>
                )}
              </button>
            </div>

            {/* Results */}
            {renderRiskCard(data, risk, themeClasses, error)}
            {renderSurvivalCard(data, survival, themeClasses, error)}

            {/* Info Panel */}
            <div className={`${themeClasses.cardBg} p-4 sm:p-6 rounded-2xl ${themeClasses.border} border shadow-lg`}>
              <h3 className={`font-semibold mb-4 ${themeClasses.text} text-base sm:text-lg`}>How It Works</h3>
              <ul className="space-y-2 text-xs sm:text-sm">
                <li className={`${themeClasses.textSecondary}`}>
                  • Uses ARGO float data for temperature, salinity, oxygen, and chlorophyll
                </li>
                <li className={`${themeClasses.textSecondary}`}>
                  • Machine learning model predicts oceanic anomalies
                </li>
                <li className={`${themeClasses.textSecondary}`}>
                  • High risk indicates potential for extreme oceanic events
                </li>
                <li className={`${themeClasses.textSecondary}`}>
                  • Integrates with backend for real-time predictions
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