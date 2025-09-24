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
const renderRiskCard = (data, risk, themeClasses, error) => (
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
    ) : risk ? (
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

const Predictor = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  
  // State for input fields
  const [lat, setLat] = useState(-40.3);
  const [lon, setLon] = useState(73.4);
  const [depth, setDepth] = useState(980);
  const [temperature, setTemperature] = useState(3.8);
  const [salinity, setSalinity] = useState(34.5);
  const [oxygen, setOxygen] = useState(210);
  const [chlorophyll, setChlorophyll] = useState(0.4);
  const [isLoading, setIsLoading] = useState(false);
  const [data, setData] = useState(null);
  const [risk, setRisk] = useState(null);
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
                    onChange={e => setLat(parseFloat(e.target.value) || 0)}
                    step="0.1"
                    min="-90"
                    max="90"
                    className={`${themeClasses.inputBg} ${themeClasses.inputBorder} w-full px-3 py-2 rounded-lg ${themeClasses.inputText} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm`}
                    placeholder="e.g., -40.3"
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
                    placeholder="e.g., 73.4"
                  />
                </div>
                <div>
                  <label className={`${themeClasses.textMuted} block text-sm mb-1`}>Depth (m)</label>
                  <input
                    type="number"
                    value={depth}
                    onChange={e => setDepth(parseFloat(e.target.value) || 0)}
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
                    onChange={e => setTemperature(parseFloat(e.target.value) || 0)}
                    step="0.1"
                    className={`${themeClasses.inputBg} ${themeClasses.inputBorder} w-full px-3 py-2 rounded-lg ${themeClasses.inputText} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm`}
                    placeholder="e.g., 3.8"
                  />
                </div>
                <div>
                  <label className={`${themeClasses.textMuted} block text-sm mb-1`}>Salinity (PSU)</label>
                  <input
                    type="number"
                    value={salinity}
                    onChange={e => setSalinity(parseFloat(e.target.value) || 0)}
                    step="0.1"
                    className={`${themeClasses.inputBg} ${themeClasses.inputBorder} w-full px-3 py-2 rounded-lg ${themeClasses.inputText} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm`}
                    placeholder="e.g., 34.5"
                  />
                </div>
                <div>
                  <label className={`${themeClasses.textMuted} block text-sm mb-1`}>Oxygen (µmol/kg)</label>
                  <input
                    type="number"
                    value={oxygen}
                    onChange={e => setOxygen(parseFloat(e.target.value) || 0)}
                    step="0.1"
                    className={`${themeClasses.inputBg} ${themeClasses.inputBorder} w-full px-3 py-2 rounded-lg ${themeClasses.inputText} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm`}
                    placeholder="e.g., 210"
                  />
                </div>
                <div>
                  <label className={`${themeClasses.textMuted} block text-sm mb-1`}>Chlorophyll (mg/m³)</label>
                  <input
                    type="number"
                    value={chlorophyll}
                    onChange={e => setChlorophyll(parseFloat(e.target.value) || 0)}
                    step="0.01"
                    className={`${themeClasses.inputBg} ${themeClasses.inputBorder} w-full px-3 py-2 rounded-lg ${themeClasses.inputText} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm`}
                    placeholder="e.g., 0.4"
                  />
                </div>
              </div>
              <button
                onClick={handlePredict}
                disabled={isLoading || !lat || !lon || !depth || !temperature || !salinity || !oxygen || !chlorophyll}
                className={`mt-4 w-full flex items-center justify-center space-x-2 px-4 sm:px-6 py-3 rounded-lg font-medium transition-all text-sm ${
                  isLoading || !lat || !lon || !depth || !temperature || !salinity || !oxygen || !chlorophyll
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