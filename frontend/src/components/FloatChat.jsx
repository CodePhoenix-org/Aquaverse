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
  Paperclip,
  Info
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

// Enhanced Speech Recognition Detection
const detectSpeechRecognition = () => {
  if (typeof window === 'undefined') return false;
  
  // Check if running on secure context (required for SpeechRecognition)
  const isSecureContext = window.isSecureContext || 
    (window.location.protocol === 'https:' || window.location.hostname === 'localhost');
  
  if (!isSecureContext) {
    console.warn('SpeechRecognition requires HTTPS or localhost');
    return { supported: false, reason: 'insecure-context' };
  }

  const SpeechRecognition = 
    window.SpeechRecognition || 
    window.webkitSpeechRecognition || 
    window.mozSpeechRecognition || 
    window.msSpeechRecognition;

  const supported = !!SpeechRecognition;
  
  if (!supported) {
    console.warn('SpeechRecognition API not supported in this browser');
    return { supported: false, reason: 'not-supported' };
  }

  return { supported: true, reason: 'supported' };
};

// Speech Recognition Utility
class VoiceInput {
  constructor(onTranscript, onError, onEnd) {
    this.recognition = null;
    this.onTranscript = onTranscript;
    this.onError = onError;
    this.onEnd = onEnd;
    this.detection = detectSpeechRecognition();
    this.isSupported = this.detection.supported;
    this.isListening = false;
  }

  startListening() {
    if (!this.isSupported) {
      this.onError(`Speech recognition not available: ${this.detection.reason}`);
      return;
    }

    if (this.isListening) {
      this.stopListening();
    }

    try {
      this.recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
      this.recognition.continuous = false;
      this.recognition.interimResults = true;
      this.recognition.lang = 'en-US';
      this.recognition.maxAlternatives = 1;

      // Add grammar for better accuracy (optional)
      if (window.SpeechGrammarList) {
        const grammar = '#JSGF V1.0; grammar ocean; public <query> = show me | display | analyze | compare | find | what are | temperature | salinity | pressure | float | argo | profile | map | trend | anomaly;';
        const speechRecognitionList = new (window.SpeechGrammarList || window.webkitSpeechGrammarList)();
        speechRecognitionList.addFromString(grammar, 1);
        this.recognition.grammars = speechRecognitionList;
      }

      this.recognition.onstart = () => {
        this.isListening = true;
        console.log('Voice recognition started');
      };

      this.recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
          } else {
            interimTranscript += transcript;
          }
        }

        if (interimTranscript) {
          this.onTranscript(interimTranscript, false);
        }

        if (finalTranscript) {
          this.onTranscript(finalTranscript, true);
        }
      };

      this.recognition.onerror = (event) => {
        this.isListening = false;
        let errorMessage = 'Speech recognition error';
        
        switch (event.error) {
          case 'no-speech':
            errorMessage = 'No speech detected. Try speaking louder.';
            break;
          case 'audio-capture':
            errorMessage = 'Microphone access denied. Please allow microphone access.';
            break;
          case 'not-allowed':
            errorMessage = 'Microphone permission denied. Please enable microphone access.';
            break;
          case 'network':
            errorMessage = 'Network error. Please check your internet connection.';
            break;
          case 'service-not-allowed':
            errorMessage = 'Speech service not available. Please check your browser settings.';
            break;
          case 'aborted':
            errorMessage = 'Speech recognition was aborted.';
            break;
          default:
            errorMessage = `Speech recognition error: ${event.error}`;
        }
        
        console.error('Speech recognition error:', event.error);
        this.onError(errorMessage);
      };

      this.recognition.onend = () => {
        this.isListening = false;
        console.log('Voice recognition ended');
        this.onEnd();
      };

      this.recognition.start();
      
    } catch (error) {
      console.error('Failed to initialize speech recognition:', error);
      this.onError('Failed to initialize speech recognition: ' + error.message);
    }
  }

  stopListening() {
    if (this.recognition && this.isListening) {
      this.recognition.stop();
      this.isListening = false;
    }
  }

  destroy() {
    this.stopListening();
    this.recognition = null;
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
  const [isVoiceActive, setIsVoiceActive] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState('idle'); // 'idle', 'listening', 'processing', 'error', 'requesting'
  const [voiceEnabled, setVoiceEnabled] = useState(() => {
    return getInitialState('floatchat_voiceEnabled', false);
  });
  
  // Voice support detection
  const [voiceSupport, setVoiceSupport] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const [chatHistory, setChatHistory] = useState([
    { id: 1, title: "Ocean temperature analysis", snippet: "Show me temperature profiles...", date: "Today", starred: false },
    { id: 2, title: "Salinity data Arabian Sea", snippet: "Compare salinity levels in...", date: "Yesterday", starred: true },
    { id: 3, title: "ARGO float trajectories", snippet: "Display float paths for...", date: "2 days ago", starred: false },
    { id: 4, title: "BGC parameter analysis", snippet: "Analyze bio-geo-chemical...", date: "1 week ago", starred: false }
  ]);

  // Voice recognition instance
  const voiceInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);

  // Initialize voice support detection
  useEffect(() => {
    const detection = detectSpeechRecognition();
    setVoiceSupport(detection);
    console.log('Voice support detection:', detection);
  }, []);

  // Initialize voice recognition
  useEffect(() => {
    if (!voiceSupport?.supported) return;

    const voiceInput = new VoiceInput(
      (transcript, isFinal) => {
        setInputText(prev => {
          if (isFinal) {
            return transcript;
          } else {
            return prev ? prev + ' ' + transcript : transcript;
          }
        });
        
        setTimeout(() => {
          inputRef.current?.scrollIntoView({ behavior: 'smooth' });
        }, 0);
      },
      (error) => {
        console.error('Voice recognition error:', error);
        setVoiceStatus('error');
        setTimeout(() => setVoiceStatus('idle'), 3000);
      },
      () => {
        setIsVoiceActive(false);
        setVoiceStatus('idle');
      }
    );

    voiceInputRef.current = voiceInput;

    return () => {
      voiceInput.destroy();
    };
  }, [voiceSupport]);

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

  // Persist voice enabled state
  useEffect(() => {
    if (voiceSupport?.supported) {
      localStorage.setItem('floatchat_voiceEnabled', JSON.stringify(voiceEnabled));
    }
  }, [voiceEnabled, voiceSupport]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  // Voice recording functions
  const toggleVoiceRecording = () => {
    if (!voiceSupport?.supported || !voiceInputRef.current?.isSupported) {
      setVoiceStatus('error');
      return;
    }

    if (isVoiceActive) {
      voiceInputRef.current.stopListening();
      setIsVoiceActive(false);
      setVoiceStatus('idle');
      setIsRecording(false);
    } else {
      setIsVoiceActive(true);
      setVoiceStatus('listening');
      setIsRecording(true);
      setInputText('');
      
      navigator.mediaDevices.getUserMedia({ audio: true })
        .then(() => {
          voiceInputRef.current.startListening();
        })
        .catch((err) => {
          console.error('Microphone access denied:', err);
          setVoiceStatus('error');
          setIsVoiceActive(false);
          setIsRecording(false);
          setTimeout(() => setVoiceStatus('idle'), 3000);
        });
    }
  };

  // Enable voice input
  const enableVoiceInput = () => {
    if (!voiceSupport?.supported) {
      setVoiceStatus('error');
      return;
    }

    setVoiceStatus('requesting');
    
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(() => {
        setVoiceEnabled(true);
        setVoiceStatus('idle');
      })
      .catch((err) => {
        console.error('Microphone permission denied:', err);
        setVoiceStatus('error');
        setTimeout(() => setVoiceStatus('idle'), 5000);
      });
  };

  // Auto-send when voice input is finalized and has content
  useEffect(() => {
    if (inputText.trim() && !isVoiceActive && voiceStatus === 'idle' && isRecording) {
      const timer = setTimeout(() => {
        if (inputText.trim() && voiceStatus === 'idle') {
          handleSendMessage();
        }
      }, 500);

      return () => clearTimeout(timer);
    }
  }, [inputText, isVoiceActive, voiceStatus, isRecording]);

  const handleFileUpload = () => {
    fileInputRef.current?.click();
  };

  const handleSendMessage = () => {
    if (inputText.trim() === '') return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputText,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsTyping(true);
    setIsRecording(false);
    setVoiceStatus('idle');

    setTimeout(() => {
      const cannedBotMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: `🔍 Processing your query: "${inputText.substring(0, 50)}${inputText.length > 50 ? '...' : ''}"\n\nI'll analyze this using our ARGO float database and generate the appropriate ocean data visualizations.`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, cannedBotMessage]);

      import('axios').then(({ default: axios }) => {
        axios.post('http://127.0.0.1:8000/chat/query', { query: inputText })
          .then(res => {
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

            if (vizData && vizData.type) {
              if (vizData.type.includes('profile')) vizTab = 'plots';
              else if (vizData.type.includes('map')) vizTab = 'map';
              else if (vizData.type.includes('comparison')) vizTab = 'comparison';
              else if (vizData.type.includes('table')) vizTab = 'table';
            }

            setLastVizData(vizData);
            setLastVizTab(vizTab);
            
            const finalBotMessage = {
              id: Date.now() + 2,
              type: 'bot',
              content: backendMsg,
              timestamp: new Date(),
              hasVisualization: !!vizData
            };

            setMessages(prev => [...prev, finalBotMessage]);
          })
          .catch(err => {
            console.error('Backend error:', err);
            setMessages(prev => [...prev, {
              id: Date.now() + 2,
              type: 'bot',
              content: '⚠️ Sorry, I encountered an error while processing your request. Please try again or check your connection.',
              timestamp: new Date()
            }]);
          })
          .finally(() => setIsTyping(false));
      });
    }, 1000);
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
    cardHoverBg: darkMode ? 'bg-gray-700' : 'bg-gray-100',
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

  // Voice status styling
  const getVoiceButtonClasses = () => {
    if (voiceStatus === 'error') {
      return 'bg-red-500 text-white animate-pulse';
    } else if (isVoiceActive) {
      return 'bg-gradient-to-r from-purple-500 to-pink-500 text-white animate-pulse';
    } else if (!voiceEnabled && voiceStatus !== 'requesting' && voiceSupport?.supported) {
      return `${themeClasses.textMuted} opacity-50 cursor-not-allowed`;
    } else {
      return `${themeClasses.textMuted} hover:text-purple-500 ${themeClasses.buttonHoverBg}`;
    }
  };

  const getVoiceIcon = () => {
    if (voiceStatus === 'error') {
      return <X className="w-5 h-5" />;
    } else if (isVoiceActive) {
      return <Mic className="w-5 h-5" />;
    } else if (!voiceEnabled) {
      return <MicOff className="w-5 h-5" />;
    } else {
      return <MicOff className="w-5 h-5" />;
    }
  };

  const isMicDisabled = () => {
    return !voiceEnabled || !voiceSupport?.supported || !voiceInputRef.current?.isSupported || voiceStatus === 'requesting';
  };

  // Get voice support message
  const getVoiceSupportMessage = () => {
    if (!voiceSupport) return null;
    
    if (!voiceSupport.supported) {
      if (voiceSupport.reason === 'insecure-context') {
        return (
          <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-3 mt-3">
            <div className="flex items-start space-x-2">
              <Info className="w-5 h-5 text-yellow-500 mt-0.5 flex-shrink-0" />
              <div className="text-sm">
                <p className={`${themeClasses.textSecondary}`}>
                  🔒 Voice input requires a secure connection (HTTPS). 
                  For development, you can use <code className="bg-yellow-200 dark:bg-yellow-800/50 px-1 py-0.5 rounded text-xs font-mono">localhost</code> or deploy to HTTPS.
                </p>
              </div>
            </div>
          </div>
        );
      } else {
        return (
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3 mt-3">
            <div className="flex items-start space-x-2">
              <Info className="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" />
              <div className="text-sm">
                <p className={`${themeClasses.textSecondary}`}>
                  💬 Voice input works best in Chrome, Edge, and Safari. 
                  Firefox has limited support. You can still use typing for all ocean data queries!
                </p>
              </div>
            </div>
          </div>
        );
      }
    }
    
    return null;
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
              setIsVoiceActive(false);
              setVoiceStatus('idle');
              setIsRecording(false);
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
          <button
            className={`w-full text-left p-3 ${themeClasses.cardHoverBg} rounded-xl flex items-center space-x-3 transition-colors ${themeClasses.textSecondary} ${!sidebarOpen ? 'justify-center' : ''}`}
            onClick={() => navigate('/dashboard')}
          >
            <BarChart3 className="w-5 h-5" />
            {sidebarOpen && <span>Dashboard</span>}
          </button>
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
                <p className={`text-xl ${themeClasses.textSecondary} max-w-2xl mx-auto mb-6`}>
                  Your AI assistant for discovering and analyzing ARGO ocean data.
                </p>
                
                <div className="flex flex-col sm:flex-row justify-center items-center space-y-2 sm:space-y-0 sm:space-x-4 text-sm mb-4">
                  <span className={`${themeClasses.textMuted}`}>💬 Type your query or</span>
                  
                  {voiceSupport?.supported ? (
                    voiceEnabled ? (
                      <button
                        onClick={toggleVoiceRecording}
                        className={`inline-flex items-center space-x-1 px-3 py-1 rounded-lg ${getVoiceButtonClasses()} transition-colors`}
                      >
                        <Mic className="w-4 h-4" />
                        <span className="font-medium">Speak</span>
                      </button>
                    ) : (
                      <button
                        onClick={enableVoiceInput}
                        className="inline-flex items-center space-x-1 px-3 py-1 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:from-purple-600 hover:to-pink-600 transition-all duration-200 font-medium shadow-sm hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
                        disabled={voiceStatus === 'requesting'}
                      >
                        {voiceStatus === 'requesting' ? (
                          <>
                            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                            <span>Enabling...</span>
                          </>
                        ) : (
                          <>
                            <Mic className="w-4 h-4" />
                            <span>Enable Voice</span>
                          </>
                        )}
                      </button>
                    )
                  ) : (
                    <button
                      className="inline-flex items-center space-x-1 px-3 py-1 rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-500 rounded-lg cursor-not-allowed"
                      disabled
                    >
                      <MicOff className="w-4 h-4" />
                      <span className="font-medium">Voice Unavailable</span>
                    </button>
                  )}
                </div>

                {/* Voice Support Info */}
                {getVoiceSupportMessage()}
                
                {!voiceEnabled && voiceSupport?.supported && (
                  <p className={`text-xs ${themeClasses.textMuted} mt-2 max-w-md mx-auto`}>
                    🔒 Voice input is disabled by default for privacy. Click "Enable Voice" to allow microphone access for hands-free ocean data queries.
                  </p>
                )}
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
                const isLastBotMsg = message.type === 'bot' && 
                  idx === messages.length - 1 && 
                  (message.hasVisualization || /visualization|chart|plot|map|graph/i.test(message.content));
                
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
                          {isLastBotMsg && lastVizData && (
                            <button
                              onClick={() => {
                                navigate('/dashboard', {
                                  state: { vizData: lastVizData, vizTab: lastVizTab }
                                });
                              }}
                              className="mt-4 px-4 py-2 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl shadow hover:scale-105 transition font-semibold text-sm"
                            >
                              📊 View Visualization
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
                  placeholder={isVoiceActive ? "🎤 Listening... Speak your ocean data query" : !voiceSupport?.supported ? "Ask FloatChat about ocean data..." : !voiceEnabled ? "Ask FloatChat about ocean data... (Enable voice input above)" : "Ask FloatChat about ocean data... or tap the mic to speak"}
                  className={`flex-1 bg-transparent border-none resize-none focus:outline-none placeholder-gray-500 ${themeClasses.text} ${
                    isVoiceActive ? 'border-l-2 border-purple-400 pl-3' : ''
                  }`}
                  rows={1}
                  style={{
                    minHeight: '24px',
                    maxHeight: '200px',
                    resize: 'none'
                  }}
                  disabled={isVoiceActive}
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
                    onClick={voiceSupport?.supported ? (voiceEnabled ? toggleVoiceRecording : enableVoiceInput) : null}
                    disabled={isMicDisabled()}
                    className={`p-2 rounded-full transition-all duration-200 ${getVoiceButtonClasses()} disabled:opacity-50 disabled:cursor-not-allowed`}
                    title={isVoiceActive ? "Stop listening" : !voiceSupport?.supported ? "Voice not supported in this browser" : !voiceEnabled ? "Enable voice input first" : "Start voice input"}
                  >
                    {getVoiceIcon()}
                  </button>
                  <button
                    onClick={handleSendMessage}
                    disabled={inputText.trim() === '' || isVoiceActive}
                    className={`p-2 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-200 disabled:text-gray-400 text-white rounded-full transition-colors ${
                      inputText.trim() !== '' && !isVoiceActive ? 'animate-pulse' : ''
                    }`}
                  >
                    <Send className="w-5 h-5" />
                  </button>
                </div>
              </div>
              
              {/* Voice Status Indicator */}
              {voiceStatus === 'listening' && (
                <div className="flex items-center justify-center mt-2 space-x-2">
                  <div className="w-2 h-2 bg-purple-500 rounded-full animate-ping"></div>
                  <span className={`text-xs ${themeClasses.textSecondary}`}>
                    🎤 Listening... Speak clearly about ocean data queries
                  </span>
                </div>
              )}
              
              {voiceStatus === 'error' && (
                <div className="flex items-center justify-center mt-2">
                  <span className={`text-xs text-red-500`}>
                    ❌ Microphone access required. Please allow access and try again.
                  </span>
                </div>
              )}
              
              {voiceStatus === 'requesting' && (
                <div className="flex items-center justify-center mt-2">
                  <div className="w-3 h-3 border-2 border-purple-500 border-t-transparent rounded-full animate-spin mr-2"></div>
                  <span className={`text-xs ${themeClasses.textSecondary}`}>
                    Requesting microphone permission...
                  </span>
                </div>
              )}
            </div>
            <div className="flex items-center justify-center mt-3">
              <p className={`text-xs ${themeClasses.textMuted}`}>
                {isVoiceActive ? '🎤 Voice input active - speak your query!' : !voiceSupport?.supported ? '💬 FloatChat works great with typing too!' : !voiceEnabled ? '🔒 Voice input disabled. Click the mic icon to enable microphone access.' : 'FloatChat can make mistakes. Consider checking important ocean data insights.'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FloatChat;