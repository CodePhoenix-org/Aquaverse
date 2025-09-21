
import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  MessageCircle, 
  Mic, 
  MicOff, 
  Send, 
  History, 
  BarChart3, 
  Settings, 
  Upload,
  Search,
  Globe,
  Waves,
  ChevronLeft,
  ChevronRight,
  Plus,
  Trash2,
  User,
  Bot,
  Menu,
  X,
  Star,
  Share,
  MoreVertical,
  Edit3,
  Sparkles,
  Moon,
  Sun,
  Paperclip
} from 'lucide-react';

// Inject custom scrollbar styles globally (only once)
if (typeof window !== 'undefined') {
  const styleId = 'floatchat-scrollbar-style';
  if (!document.getElementById(styleId)) {
    const style = document.createElement('style');
    style.id = styleId;
    style.innerHTML = `
      .custom-scrollbar::-webkit-scrollbar {
        width: 8px;
        border-radius: 8px;
      }
      .custom-scrollbar::-webkit-scrollbar-thumb {
        background: #374151;
        border-radius: 8px;
      }
      .custom-scrollbar.light::-webkit-scrollbar-thumb {
        background: #e5e7eb;
      }
      .custom-scrollbar::-webkit-scrollbar-track {
        background: transparent;
      }
    `;
    document.head.appendChild(style);
  }
}

// Handle Enter key press in textarea
function handleKeyPress(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    handleSendMessage();
  }
}
// Handle file input change
function handleFileChange(event) {
  const file = event.target.files && event.target.files[0];
  if (file) {
    // You can add file processing logic here, e.g., upload or parse
    // For now, just log the file name
    console.log('Selected file:', file.name);
  }
}

const FloatChat = () => {
  const navigate = useNavigate();
  // Restore state from localStorage if available
  const getInitialState = (key, fallback) => {
    try {
      const stored = localStorage.getItem(key);
      if (stored) return JSON.parse(stored);
    } catch {}
    return fallback;
  };
  const [messages, setMessages] = useState(() => getInitialState('floatchat_messages', []));
  const [lastVizData, setLastVizData] = useState(() => getInitialState('floatchat_lastVizData', null));
  const [lastVizTab, setLastVizTab] = useState(() => getInitialState('floatchat_lastVizTab', null));
  const [inputText, setInputText] = useState(() => getInitialState('floatchat_inputText', ''));
  const [isRecording, setIsRecording] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const [chatHistory, setChatHistory] = useState([
    { id: 1, title: "Ocean temperature analysis", snippet: "Show me temperature profiles...", date: "Today", starred: false },
    { id: 2, title: "Salinity data Arabian Sea", snippet: "Compare salinity levels in...", date: "Yesterday", starred: true },
    { id: 3, title: "ARGO float trajectories", snippet: "Display float paths for...", date: "2 days ago", starred: false },
    { id: 4, title: "BGC parameter analysis", snippet: "Analyze bio-geo-chemical...", date: "1 week ago", starred: false }
  ]);
  // Persist chat state to localStorage on change
  useEffect(() => {
    localStorage.setItem('floatchat_messages', JSON.stringify(messages));
  }, [messages]);
  useEffect(() => {
    localStorage.setItem('floatchat_lastVizData', JSON.stringify(lastVizData));
  }, [lastVizData]);
  useEffect(() => {
    localStorage.setItem('floatchat_lastVizTab', JSON.stringify(lastVizTab));
  }, [lastVizTab]);
  useEffect(() => {
    localStorage.setItem('floatchat_inputText', JSON.stringify(inputText));
  }, [inputText]);
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const toggleRecording = () => {
    setIsRecording(!isRecording);
    if (!isRecording) {
      setTimeout(() => {
        setInputText("Show me temperature profiles in the Indian Ocean for the last month");
        setIsRecording(false);
      }, 2000);
    }
  };

  const handleFileUpload = () => {
    fileInputRef.current?.click();
  };

  const handleSendMessage = () => {
    if (inputText.trim() === '') return;

    const userMessage = {
      id: messages.length + 1,
      type: 'user',
      content: inputText,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsTyping(true);

    // Show canned message first
    setTimeout(() => {
      const cannedBotMessage = {
        id: messages.length + 2,
        type: 'bot',
        content: `I'll analyze your ocean data query: "${inputText}"
Based on our ARGO float database, I can help you explore temperature, salinity, and BGC parameters across different ocean regions. Let me process this request and generate the appropriate visualizations and insights.`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, cannedBotMessage]);

      // Now fetch real backend response
      import('axios').then(({ default: axios }) => {
        axios.post('http://127.0.0.1:8000/chat', { query: inputText })
          .then(res => {
            // Support both array and object responses
            let backendMsg = '';
            let vizData = null;
            let vizTab = null;
            if (Array.isArray(res.data)) {
              backendMsg = res.data[0] || 'No response from backend.';
              vizData = res.data[1] || null;
            } else if (typeof res.data === 'object') {
              backendMsg = res.data.message || res.data.answer || 'No response from backend.';
              vizData = res.data.data || null;
            } else {
              backendMsg = String(res.data);
            }
            // Heuristic: set tab type based on vizData.type
            if (vizData && vizData.type) {
              if (vizData.type.includes('profile')) vizTab = 'plots';
              else if (vizData.type.includes('map')) vizTab = 'map';
              else if (vizData.type.includes('comparison')) vizTab = 'comparison';
              else if (vizData.type.includes('table')) vizTab = 'table';
            }
            setLastVizData(vizData);
            setLastVizTab(vizTab);
            setMessages(prev => [...prev, {
              id: prev.length + 1,
              type: 'bot',
              content: backendMsg,
              timestamp: new Date()
            }]);
          })
          .catch(err => {
            setMessages(prev => [...prev, {
              id: prev.length + 1,
              type: 'bot',
              content: '⚠️ Error fetching backend response.',
              timestamp: new Date()
            }]);
          })
          .finally(() => setIsTyping(false));
      });
    }, 1500);
  };
  const samplePrompts = [
    {
      icon: <Globe className="w-5 h-5" />,
      title: "Ocean Data Analysis",
      description: "Show me salinity profiles near the equator in March 2023"
    },
    {
      icon: <BarChart3 className="w-5 h-5" />,
      title: "Compare Parameters",
      description: "Compare BGC parameters in the Arabian Sea for the last 6 months"
    },
    {
      icon: <Search className="w-5 h-5" />,
      title: "Find Floats",
      description: "What are the nearest ARGO floats to coordinates 15°N, 68°E?"
    },
    {
      icon: <Waves className="w-5 h-5" />,
      title: "Temperature Trends",
      description: "Display temperature anomalies in the Indian Ocean basin"
    }
  ];

  // Theme classes
  const themeClasses = {
    bg: darkMode ? 'bg-gray-900' : 'bg-white',
    sidebarBg: darkMode ? 'bg-gray-800' : 'bg-gray-50',
    cardBg: darkMode ? 'bg-gray-800' : 'bg-white',
    cardHoverBg: darkMode ? 'bg-gray-700' : 'bg-white',
    promptBg: darkMode ? 'bg-gray-800' : 'bg-gray-50',
    promptHoverBg: darkMode ? 'bg-gray-700' : 'bg-gray-100',
    text: darkMode ? 'text-white' : 'text-gray-900',
    textSecondary: darkMode ? 'text-gray-300' : 'text-gray-600',
    textMuted: darkMode ? 'text-gray-400' : 'text-gray-500',
    border: darkMode ? 'border-gray-700' : 'border-gray-200',
    borderLight: darkMode ? 'border-gray-600' : 'border-gray-100',
    inputBg: darkMode ? 'bg-gray-800' : 'bg-gray-50',
    inputBorder: darkMode ? 'border-gray-600' : 'border-gray-200',
    inputFocus: darkMode ? 'border-blue-400 ring-blue-400/20' : 'border-blue-300 ring-blue-50',
    botMessageBg: darkMode ? 'bg-gray-800' : 'bg-gray-50',
    hoverBg: darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100',
    buttonHoverBg: darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-100'
  };

  return (
    <div className={`h-screen ${themeClasses.bg} flex relative transition-colors duration-200`}>
      {/* Hidden file input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept=".nc,.csv,.txt,.json"
        className="hidden"
      />

      {/* Sidebar */}
      <div
        className={`
          fixed inset-y-0 left-0 z-50
          ${sidebarOpen ? 'w-80' : 'w-16'}
          ${themeClasses.sidebarBg} ${themeClasses.border} border-r
          transition-all duration-300 ease-in-out
          overflow-hidden
          flex flex-col
          lg:relative lg:z-auto
          ${!sidebarOpen ? 'items-center py-4' : ''}
        `}
        style={{ minWidth: sidebarOpen ? '20rem' : '4rem', maxWidth: sidebarOpen ? '20rem' : '4rem' }}
      >
        {/* Sidebar Header */}
        <div className={`flex items-center justify-between p-4 ${themeClasses.border} border-b ${!sidebarOpen ? 'flex-col space-y-4' : ''}`}>
          <div className={`flex items-center ${sidebarOpen ? 'space-x-3' : 'justify-center w-full'}`}>
            <div className={`w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-2xl flex items-center justify-center shadow-lg border-2 border-cyan-400/30 ${!sidebarOpen ? 'mx-auto' : ''}`}>
              <Waves className="w-6 h-6 text-white" />
            </div>
            {sidebarOpen && (
              <div>
                <h1 className={`text-lg font-semibold ${themeClasses.text}`}>FloatChat</h1>
                <p className={`text-xs ${themeClasses.textMuted}`}>ARGO Data Discovery</p>
              </div>
            )}
          </div>
          <button
            onClick={() => setSidebarOpen((open) => !open)}
            className={`p-2 ${themeClasses.hoverBg} rounded-lg transition-colors border border-transparent hover:border-cyan-400/40 ${!sidebarOpen ? 'mx-auto mt-2' : ''}`}
            aria-label={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
          >
            {sidebarOpen ? <X className={`w-5 h-5 ${themeClasses.text}`} /> : <Menu className={`w-5 h-5 ${themeClasses.text}`} />}
          </button>
        </div>

        {/* New Chat Button */}
        <div className={`p-4 ${!sidebarOpen ? 'flex justify-center' : ''}`}>
          <button
            className={`
              ${sidebarOpen ? 'w-full' : 'w-12 h-12'}
              ${themeClasses.cardBg} ${themeClasses.border} border ${themeClasses.hoverBg} ${themeClasses.textSecondary}
              py-3 px-4 rounded-xl flex items-center justify-center space-x-2 transition-colors shadow-sm
              ${!sidebarOpen ? 'justify-center' : ''}
              hover:scale-105
            `}
            onClick={() => {
              setMessages([]);
              setInputText('');
              setLastVizData(null);
              setLastVizTab(null);
              // Also clear persisted state
              localStorage.removeItem('floatchat_messages');
              localStorage.removeItem('floatchat_lastVizData');
              localStorage.removeItem('floatchat_lastVizTab');
              localStorage.removeItem('floatchat_inputText');
            }}
          >
            <Edit3 className="w-5 h-5" />
            {sidebarOpen && <span>New chat</span>}
          </button>
        </div>

        {/* Chat History */}
        <div className={`flex-1 ${sidebarOpen ? 'px-4' : 'px-1'} overflow-y-auto custom-scrollbar`}>
          {sidebarOpen && (
            <div className="space-y-1">
              <h3 className={`text-sm font-medium ${themeClasses.textMuted} px-2 py-1`}>Recent</h3>
              {chatHistory.map((chat) => (
                <div key={chat.id} className={`group p-3 ${themeClasses.cardHoverBg} rounded-xl cursor-pointer transition-colors border border-transparent hover:border-cyan-400/40 ${darkMode ? 'hover:border-cyan-600' : ''} hover:shadow-sm`}>
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <p className={`text-sm font-medium ${themeClasses.text} truncate`}>{chat.title}</p>
                      <p className={`text-xs ${themeClasses.textMuted} truncate`}>{chat.snippet}</p>
                      <p className={`text-xs ${themeClasses.textMuted} mt-1`}>{chat.date}</p>
                    </div>
                    <div className="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button className={`p-1.5 ${themeClasses.buttonHoverBg} rounded-lg transition-colors`}>
                        <Star className={`w-3.5 h-3.5 ${chat.starred ? 'text-yellow-500 fill-yellow-500' : themeClasses.textMuted}`} />
                      </button>
                      <button className={`p-1.5 ${themeClasses.buttonHoverBg} rounded-lg transition-colors`}>
                        <MoreVertical className={`w-3.5 h-3.5 ${themeClasses.textMuted}`} />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Sidebar Footer */}
        <div className={`p-4 ${themeClasses.border} border-t space-y-2`}>
          {/* Dashboard Button */}
          <button
            className={`w-full text-left p-3 ${themeClasses.cardHoverBg} rounded-xl flex items-center space-x-3 transition-colors ${themeClasses.textSecondary} ${!sidebarOpen ? 'justify-center' : ''}`}
            onClick={() => navigate('/dashboard')}
          >
            <BarChart3 className="w-5 h-5" />
            {sidebarOpen && <span>Dashboard</span>}
          </button>
          {/* Settings Button */}
          <button className={`w-full text-left p-3 ${themeClasses.cardHoverBg} rounded-xl flex items-center space-x-3 transition-colors ${themeClasses.textSecondary} ${!sidebarOpen ? 'justify-center' : ''}`}>
            <Settings className="w-5 h-5" />
            {sidebarOpen && <span>Settings</span>}
          </button>
        </div>
      </div>

      {/* Sidebar Overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-25 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className={`${themeClasses.bg} ${themeClasses.borderLight} border-b px-4 py-3 flex items-center justify-between`}>
          <div className="flex items-center space-x-3">
            <button 
              onClick={() => setSidebarOpen(true)}
              className={`lg:hidden p-2 ${themeClasses.hoverBg} rounded-lg transition-colors`}
            >
              <Menu className={`w-5 h-5 ${themeClasses.text}`} />
            </button>
            <div className="flex items-center space-x-2">
              <Sparkles className="w-5 h-5 text-blue-500" />
              <span className={`font-semibold ${themeClasses.text}`}>FloatChat</span>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button 
              onClick={() => setDarkMode(!darkMode)}
              className={`p-2 ${themeClasses.hoverBg} rounded-lg ${themeClasses.textMuted} transition-colors`}
            >
              {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
            <button className={`p-2 ${themeClasses.hoverBg} rounded-lg ${themeClasses.textMuted} transition-colors`}>
              <Share className="w-5 h-5" />
            </button>
            <button className={`p-2 ${themeClasses.hoverBg} rounded-lg ${themeClasses.textMuted} transition-colors`}>
              <MoreVertical className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Messages Area */}
  <div className="flex-1 overflow-y-auto custom-scrollbar">
          {messages.length === 0 ? (
            /* Welcome Screen */
            <div className="max-w-4xl mx-auto px-6 py-12">
              <div className="text-center mb-12">
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-2xl flex items-center justify-center mx-auto mb-6">
                  <Waves className="w-8 h-8 text-white" />
                </div>
                <h1 className={`text-4xl font-bold ${themeClasses.text} mb-4`}>
                  Hello, I'm FloatChat
                </h1>
                <p className={`text-xl ${themeClasses.textSecondary} max-w-2xl mx-auto`}>
                  Your AI assistant for discovering and analyzing ARGO ocean data. 
                  Ask me anything about temperature, salinity, BGC parameters, or float trajectories.
                </p>
              </div>

              {/* Sample Prompts */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
                {samplePrompts.map((prompt, index) => (
                  <button
                    key={index}
                    onClick={() => setInputText(prompt.description)}
                    className={`text-left p-6 ${themeClasses.promptBg} ${themeClasses.promptHoverBg} rounded-2xl ${themeClasses.border} border hover:border-gray-300 ${darkMode ? 'hover:border-gray-600' : ''} transition-all duration-200 group`}
                  >
                    <div className="flex items-start space-x-4">
                      <div className={`w-10 h-10 ${themeClasses.cardBg} rounded-xl flex items-center justify-center text-blue-500 group-hover:bg-blue-50 ${darkMode ? 'group-hover:bg-blue-900/50' : ''} transition-colors`}>
                        {prompt.icon}
                      </div>
                      <div className="flex-1">
                        <h3 className={`font-semibold ${themeClasses.text} mb-1`}>{prompt.title}</h3>
                        <p className={`${themeClasses.textSecondary} text-sm leading-relaxed`}>{prompt.description}</p>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            /* Chat Messages */
            <div className="max-w-4xl mx-auto px-6 py-6 space-y-6">
              {messages.map((message, idx) => {
                const isLastBotMsg =
                  message.type === 'bot' &&
                  idx === messages.length - 1 &&
                  /check the visualization/i.test(message.content);
                return (
                  <div key={message.id} className={`flex items-start space-x-4 ${message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      message.type === 'user' 
                        ? 'bg-blue-500 text-white' 
                        : 'bg-gradient-to-br from-purple-500 to-pink-500 text-white'
                    }`}>
                      {message.type === 'user' ? <User className="w-4 h-4" /> : <Sparkles className="w-4 h-4" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className={`prose prose-gray max-w-none ${message.type === 'user' ? 'text-right' : ''}`}>
                        <div className={`inline-block p-4 rounded-2xl ${
                          message.type === 'user' 
                            ? 'bg-blue-500 text-white' 
                            : `${themeClasses.botMessageBg} ${themeClasses.text}`
                        }`}>
                          <p className="whitespace-pre-wrap m-0">{message.content}</p>
                          {isLastBotMsg && (
                            <button
                              onClick={() => {
                                navigate('/dashboard', {
                                  state: lastVizData ? { vizData: lastVizData, vizTab: lastVizTab } : undefined
                                });
                              }}
                              className="mt-4 px-4 py-2 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl shadow hover:scale-105 transition font-semibold text-sm"
                            >
                              View Visualization
                            </button>
                          )}
                        </div>
                      </div>
                      <p className={`text-xs ${themeClasses.textMuted} mt-2 ${message.type === 'user' ? 'text-right' : ''}`}>
                        {(() => {
                          let dateObj = message.timestamp;
                          if (typeof dateObj === 'string') {
                            dateObj = new Date(dateObj);
                          }
                          return dateObj && typeof dateObj.toLocaleTimeString === 'function'
                            ? dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                            : '';
                        })()}
                      </p>
                    </div>
                  </div>
                );
              })}

              {isTyping && (
                <div className="flex items-start space-x-4">
                  <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                    <Sparkles className="w-4 h-4 text-white" />
                  </div>
                  <div className={`${themeClasses.botMessageBg} p-4 rounded-2xl`}>
                    <div className="flex space-x-2">
                      <div className={`w-2 h-2 ${themeClasses.textMuted} rounded-full animate-bounce`}></div>
                      <div className={`w-2 h-2 ${themeClasses.textMuted} rounded-full animate-bounce`} style={{ animationDelay: '0.1s' }}></div>
                      <div className={`w-2 h-2 ${themeClasses.textMuted} rounded-full animate-bounce`} style={{ animationDelay: '0.2s' }}></div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className={`${themeClasses.borderLight} border-t ${themeClasses.bg}`}>
          <div className="max-w-4xl mx-auto px-6 py-4">
            <div className="relative">
              <div className={`flex items-end space-x-3 ${themeClasses.inputBg} rounded-3xl p-4 ${themeClasses.inputBorder} border focus-within:${themeClasses.inputFocus} focus-within:ring-4 transition-all`}>
                <textarea
                  ref={inputRef}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask FloatChat about ocean data..."
                  className={`flex-1 bg-transparent border-none resize-none focus:outline-none placeholder-gray-500 ${themeClasses.text}`}
                  rows={1}
                  style={{
                    minHeight: '24px',
                    maxHeight: '200px',
                    resize: 'none'
                  }}
                />
                <div className="flex items-center space-x-2">
                  <button
                    onClick={handleFileUpload}
                    className={`p-2 rounded-full transition-colors ${themeClasses.textMuted} hover:text-gray-600 ${themeClasses.buttonHoverBg}`}
                    title="Upload ARGO data file"
                  >
                    <Paperclip className="w-5 h-5" />
                  </button>
                  <button
                    onClick={toggleRecording}
                    className={`p-2 rounded-full transition-colors ${
                      isRecording 
                        ? 'bg-red-500 text-white animate-pulse' 
                        : `${themeClasses.textMuted} hover:text-gray-600 ${themeClasses.buttonHoverBg}`
                    }`}
                  >
                    {isRecording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
                  </button>
                  <button
                    onClick={handleSendMessage}
                    disabled={inputText.trim() === ''}
                    className="p-2 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-200 disabled:text-gray-400 text-white rounded-full transition-colors"
                  >
                    <Send className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
            <div className="flex items-center justify-center mt-3">
              <p className={`text-xs ${themeClasses.textMuted}`}>
                FloatChat can make mistakes. Consider checking important ocean data insights.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FloatChat;
