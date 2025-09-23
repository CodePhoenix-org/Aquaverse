import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
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
  Info,
  Download
} from 'lucide-react';

// Inject custom scrollbar and animation styles globally (only once)
if (typeof window !== 'undefined') {
  const styleId = 'floatchat-styles';
  if (!document.getElementById(styleId)) {
    const style = document.createElement('style');
    style.id = styleId;
    style.innerHTML = `
      .custom-scrollbar::-webkit-scrollbar {
        width: 6px;
        border-radius: 10px;
      }
      .custom-scrollbar::-webkit-scrollbar-thumb {
        background: #374151;
        border-radius: 10px;
        transition: background 0.3s ease;
      }
      .custom-scrollbar.light::-webkit-scrollbar-thumb {
        background: #e5e7eb;
      }
      .custom-scrollbar::-webkit-scrollbar-track {
        background: transparent;
      }
      .animate-pulse-slow {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
      }
      .animate-slide-in {
        animation: slideIn 0.4s ease-out;
      }
      .animate-float {
        animation: float 3s ease-in-out infinite;
      }
      @keyframes slideIn {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
      }
      @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
      }
      .glassmorphism {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
      }
      .dark .glassmorphism {
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
      }
    `;
    document.head.appendChild(style);
  }
}

// Enhanced Speech Recognition Detection
const detectSpeechRecognition = () => {
  if (typeof window === 'undefined') return false;
  
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
  this.recognition.lang = 'hi-IN'; // Prioritize Hindi
      this.recognition.maxAlternatives = 1;

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
        this.onTranscript(interimTranscript, false);
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

// Handle file input change with direct download
function handleFileChange(event, setFileStatus, setMessages) {
  const file = event.target.files?.[0];
  if (!file) {
    setFileStatus({ status: 'error', message: 'No file selected.' });
    toast.error('No file selected.');
    return;
  }

  const validExtensions = ['.nc', '.csv', '.txt', '.json'];
  const fileExtension = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
  if (!validExtensions.includes(fileExtension)) {
    setFileStatus({
      status: 'error',
      message: `Invalid file type. Please upload a ${validExtensions.join(', ')} file.`,
    });
    toast.error(`Invalid file type. Please upload a ${validExtensions.join(', ')} file.`);
    return;
  }

  setFileStatus({ status: 'uploading', message: 'Uploading and converting file...' });

  const formData = new FormData();
  formData.append('file', file);

  import('axios').then(({ default: axios }) => {
    axios
      .post('http://127.0.0.1:8000/convert', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      .then((res) => {
        const { convertedData, filename } = res.data;
        setFileStatus({ status: 'success', message: 'File converted successfully!' });
        toast.success('File converted successfully!');

        const botMessage = {
          id: Date.now(),
          type: 'bot',
          content: `📁 File "${file.name}" uploaded and converted successfully. Download your CSV below.`,
          timestamp: new Date(),
          filename,
        };
        setMessages((prev) => [...prev, botMessage]);
      })
      .catch((err) => {
        console.error('File conversion error:', err);
        setFileStatus({
          status: 'error',
          message: '⚠️ Error converting file. Please try again.',
        });
        toast.error('Error converting file. Please try again.');
        const errorMessage = {
          id: Date.now(),
          type: 'bot',
          content: '⚠️ Sorry, there was an error processing your file. Please check the file and try again.',
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errorMessage]);
      });
  });
}

const FloatChat = () => {
  const navigate = useNavigate();
  
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
  const [voiceError, setVoiceError] = useState(null);

  // Multilingual translation for voice input
  const translateText = async (text) => {
    if (!text) return;
    try {
      const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q=${encodeURI(text)}`;
      const response = await fetch(url);
      const data = await response.json();
      const translatedText = data[0][0][0];
      if (translatedText && translatedText.trim() !== '') {
        setInputText(translatedText);
      } else {
        setInputText(text); // Fallback to original text
      }
    } catch (error) {
      console.error('Translation error:', error);
      setInputText(text); // Fallback to original text
    }
  };
  const [isRecording, setIsRecording] = useState(false);
  const [isVoiceActive, setIsVoiceActive] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState('idle');
  const [voiceEnabled, setVoiceEnabled] = useState(() => {
    return getInitialState('floatchat_voiceEnabled', false);
  });
  
  const [fileStatus, setFileStatus] = useState({ status: 'idle', message: '' });
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

  const voiceInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  useEffect(() => {
    const detection = detectSpeechRecognition();
    setVoiceSupport(detection);
    console.log('Voice support detection:', detection);
  }, []);

  useEffect(() => {
    if (!voiceSupport?.supported) return;

    const voiceInput = new VoiceInput(
      (transcript, isFinal) => {
        setInputText(transcript);
        if (isFinal) {
          translateText(transcript);
        }
        setTimeout(() => {
          inputRef.current?.scrollIntoView({ behavior: 'smooth' });
        }, 0);
      },
      (error) => {
        console.error('Voice recognition error:', error);
        setVoiceError(error);
        setVoiceStatus('error');
        setTimeout(() => setVoiceStatus('idle'), 3000);
        toast.error(error);
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

  useEffect(() => {
    if (voiceSupport?.supported) {
      localStorage.setItem('floatchat_voiceEnabled', JSON.stringify(voiceEnabled));
    }
  }, [voiceEnabled, voiceSupport]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const toggleVoiceRecording = () => {
    if (!voiceSupport?.supported || !voiceInputRef.current?.isSupported) {
      setVoiceStatus('error');
      toast.error('Speech recognition not available.');
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
          toast.error('Microphone access denied.');
        });
    }
  };

  const enableVoiceInput = () => {
    if (!voiceSupport?.supported) {
      setVoiceStatus('error');
      toast.error('Speech recognition not supported.');
      return;
    }

    setVoiceStatus('requesting');
    
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(() => {
        setVoiceEnabled(true);
        setVoiceStatus('idle');
        toast.success('Voice input enabled!');
      })
      .catch((err) => {
        console.error('Microphone permission denied:', err);
        setVoiceStatus('error');
        setTimeout(() => setVoiceStatus('idle'), 5000);
        toast.error('Microphone permission denied.');
      });
  };

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

  const handleDownload = (filename) => {
    import('axios').then(({ default: axios }) => {
      axios
        .get(`http://127.0.0.1:8000/download/${filename}`, {
          responseType: 'blob',
        })
        .then((response) => {
          const url = window.URL.createObjectURL(new Blob([response.data]));
          const link = document.createElement('a');
          link.href = url;
          link.setAttribute('download', filename);
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          window.URL.revokeObjectURL(url);
          toast.success(`Downloaded ${filename}`);
        })
        .catch((err) => {
          console.error('Download error:', err);
          toast.error('Failed to download file. Please try again.');
        });
    });
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
            toast.error('Error processing query.');
          })
          .finally(() => setIsTyping(false));
      });
    }, 1000);
  };

  const samplePrompts = [
    {
      icon: <Globe className="w-8 h-8 animate-float" />,
      title: "Ocean Data Analysis",
      description: "Show me salinity profiles near the equator in March 2023"
    },
    {
      icon: <BarChart3 className="w-8 h-8 animate-float" />,
      title: "Compare Parameters",
      description: "Compare BGC parameters in the Arabian Sea for the last 6 months"
    },
    {
      icon: <Search className="w-8 h-8 animate-float" />,
      title: "Find Floats",
      description: "What are the nearest ARGO floats to coordinates 15°N, 68°E?"
    },
    {
      icon: <Waves className="w-8 h-8 animate-float" />,
      title: "Temperature Trends",
      description: "Display temperature anomalies in the Indian Ocean basin"
    }
  ];

  const themeClasses = {
    bg: darkMode ? 'bg-gray-900' : 'bg-white',
    sidebarBg: darkMode ? 'bg-gray-800/95 backdrop-blur-md' : 'bg-gray-50/95 backdrop-blur-md',
    cardBg: darkMode ? 'bg-gray-800/50' : 'bg-white/80',
    cardHoverBg: darkMode ? 'hover:bg-gray-700/50' : 'hover:bg-gray-100/80',
    promptBg: darkMode ? 'bg-gray-800/50' : 'bg-gray-50/80',
    promptHoverBg: darkMode ? 'hover:bg-gray-700/50' : 'hover:bg-gray-100/80',
    text: darkMode ? 'text-white' : 'text-gray-900',
    textSecondary: darkMode ? 'text-gray-300' : 'text-gray-600',
    textMuted: darkMode ? 'text-gray-400' : 'text-gray-500',
    border: darkMode ? 'border-gray-700/50' : 'border-gray-200/50',
    borderLight: darkMode ? 'border-gray-600/50' : 'border-gray-100/50',
    inputBg: darkMode ? 'bg-gray-800/50 backdrop-blur-md' : 'bg-gray-50/80 backdrop-blur-md',
    inputBorder: darkMode ? 'border-gray-600/50' : 'border-gray-200/50',
    inputFocus: darkMode ? 'border-blue-400 ring-blue-400/30' : 'border-blue-300 ring-blue-300/30',
    botMessageBg: darkMode ? 'bg-gray-800/50 backdrop-blur-md' : 'bg-gray-50/80 backdrop-blur-md',
    hoverBg: darkMode ? 'hover:bg-gray-700/50' : 'hover:bg-gray-100/80',
    buttonHoverBg: darkMode ? 'hover:bg-gray-600/50' : 'hover:bg-gray-100/80',
    accent: 'bg-gradient-to-r from-blue-500 to-cyan-600',
    accentHover: 'hover:from-blue-600 hover:to-cyan-700'
  };

  const getVoiceButtonClasses = () => {
    if (voiceStatus === 'error') {
      return 'bg-red-500/90 text-white animate-pulse-slow backdrop-blur-md shadow-lg';
    } else if (isVoiceActive) {
      return 'bg-gradient-to-r from-blue-500 to-cyan-600 text-white animate-pulse-slow backdrop-blur-md shadow-lg';
    } else if (!voiceEnabled && voiceStatus !== 'requesting' && voiceSupport?.supported) {
      return `${themeClasses.textMuted} opacity-50 cursor-not-allowed`;
    } else {
      return `${themeClasses.textMuted} hover:text-blue-400 ${themeClasses.buttonHoverBg} backdrop-blur-md shadow-sm hover:shadow-md`;
    }
  };

  const getVoiceIcon = () => {
    if (voiceStatus === 'error') {
      return <X className="w-6 h-6" />;
    } else if (isVoiceActive) {
      return <Mic className="w-6 h-6 animate-pulse" />;
    } else if (!voiceEnabled) {
      return <MicOff className="w-6 h-6" />;
    } else {
      return <Mic className="w-6 h-6" />;
    }
  };

  const isMicDisabled = () => {
    return !voiceEnabled || !voiceSupport?.supported || !voiceInputRef.current?.isSupported || voiceStatus === 'requesting';
  };

  const getVoiceSupportMessage = () => {
    if (!voiceSupport) return null;
    
    if (!voiceSupport.supported) {
      if (voiceSupport.reason === 'insecure-context') {
        return (
          <div className="bg-yellow-50/90 dark:bg-yellow-900/20 border border-yellow-200/50 dark:border-yellow-800/50 rounded-xl p-4 mt-4 backdrop-blur-md animate-slide-in shadow-lg">
            <div className="flex items-start space-x-3">
              <Info className="w-6 h-6 text-yellow-500 mt-1 flex-shrink-0 animate-pulse-slow" />
              <div className="text-base">
                <p className={`${themeClasses.textSecondary} font-semibold`}>
                  🔒 Voice input requires a secure connection (HTTPS). 
                  For development, use <code className="bg-yellow-200/50 dark:bg-yellow-800/30 px-2 py-1 rounded text-sm font-mono">localhost</code> or deploy to HTTPS.
                </p>
              </div>
            </div>
          </div>
        );
      } else {
        return (
          <div className="bg-blue-50/90 dark:bg-blue-900/20 border border-blue-200/50 dark:border-blue-800/50 rounded-xl p-4 mt-4 backdrop-blur-md animate-slide-in shadow-lg">
            <div className="flex items-start space-x-3">
              <Info className="w-6 h-6 text-blue-500 mt-1 flex-shrink-0 animate-pulse-slow" />
              <div className="text-base">
                <p className={`${themeClasses.textSecondary} font-semibold`}>
                  💬 Voice input works best in Chrome, Edge, and Safari. 
                  Firefox has limited support. Use typing for seamless ocean data queries!
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
    <div className={`h-screen ${themeClasses.bg} flex relative transition-all duration-300 font-sans antialiased overflow-hidden`}>
      <input
        type="file"
        ref={fileInputRef}
        onChange={(e) => handleFileChange(e, setFileStatus, setMessages)}
        accept=".nc,.csv,.txt,.json"
        className="hidden"
      />

      <div
        className={`
          fixed inset-y-0 left-0 z-50 transform transition-transform duration-300 ease-in-out
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
          w-80 ${themeClasses.sidebarBg} ${themeClasses.border} border-r
          lg:translate-x-0 lg:static lg:z-auto
          overflow-hidden
          flex flex-col
          py-4
          shadow-2xl
        `}
      >
        <div className={`flex items-center justify-between p-6 ${themeClasses.border} border-b`}>
          <div className="flex items-center space-x-3">
            <div className={`w-14 h-14 ${themeClasses.accent} rounded-xl flex items-center justify-center shadow-xl animate-float`}>
              <Waves className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className={`text-2xl font-bold ${themeClasses.text} tracking-tight`}>FloatChat</h1>
              <p className={`text-base ${themeClasses.textMuted} font-semibold`}>ARGO Data Discovery</p>
            </div>
          </div>
          <button
            onClick={() => setSidebarOpen((open) => !open)}
            className={`p-3 ${themeClasses.hoverBg} rounded-xl transition-all duration-200 hover:scale-110 ${themeClasses.accentHover} shadow-md lg:hidden`}
            aria-label={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
          >
            <X className={`w-6 h-6 ${themeClasses.text}`} />
          </button>
        </div>

        <div className="p-6">
          <button
            className={`
              w-full ${themeClasses.cardBg} ${themeClasses.border} border ${themeClasses.accentHover}
              py-4 px-5 rounded-xl flex items-center justify-center space-x-3 transition-all duration-200 shadow-lg
              hover:scale-105 hover:shadow-2xl
              ${themeClasses.textSecondary}
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
            <Edit3 className="w-6 h-6" />
            <span className="text-base font-semibold">New Chat</span>
          </button>
        </div>

        <div className="flex-1 px-6 overflow-y-auto custom-scrollbar">
          <div className="space-y-3">
            <h3 className={`text-lg font-semibold ${themeClasses.textMuted} px-2 py-2 tracking-wide`}>Recent Chats</h3>
            {chatHistory.map((chat, index) => (
              <div 
                key={chat.id} 
                className={`group p-4 ${themeClasses.cardHoverBg} rounded-xl cursor-pointer transition-all duration-300 border ${themeClasses.border} hover:shadow-2xl hover:scale-[1.02] animate-slide-in ${index % 2 === 0 ? 'animate-delay-200' : 'animate-delay-400'}`}
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <p className={`text-lg font-semibold ${themeClasses.text} truncate`}>{chat.title}</p>
                    <p className={`text-base ${themeClasses.textMuted} truncate mt-1`}>{chat.snippet}</p>
                    <p className={`text-base ${themeClasses.textMuted} mt-1`}>{chat.date}</p>
                  </div>
                  <div className="flex items-center space-x-2 opacity-0 group-hover:opacity-100 transition-all duration-300">
                    <button className={`p-2 ${themeClasses.buttonHoverBg} rounded-lg transition-all duration-200 hover:scale-110`}>
                      <Star className={`w-5 h-5 ${chat.starred ? 'text-yellow-500 fill-yellow-500 animate-pulse-slow' : themeClasses.textMuted}`} />
                    </button>
                    <button className={`p-2 ${themeClasses.buttonHoverBg} rounded-lg transition-all duration-200 hover:scale-110`}>
                      <MoreVertical className={`w-5 h-5 ${themeClasses.textMuted}`} />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className={`p-6 ${themeClasses.border} border-t space-y-3`}>
          <button
            className={`w-full text-left p-4 ${themeClasses.cardHoverBg} rounded-xl flex items-center space-x-3 transition-all duration-200 ${themeClasses.textSecondary} hover:shadow-xl hover:scale-[1.02]`}
            onClick={() => navigate('/dashboard')}
          >
            <BarChart3 className="w-6 h-6 animate-float" />
            <span className="text-base font-semibold">Dashboard</span>
          </button>
          <button onClick={() => navigate("/profile")} className={`w-full text-left p-4 ${themeClasses.cardHoverBg} rounded-xl flex items-center space-x-3 transition-all duration-200 ${themeClasses.textSecondary} hover:shadow-xl hover:scale-[1.02]`}>
            <Settings className="w-6 h-6 animate-float" />
            <span className="text-base font-semibold">Settings</span>
          </button>
        </div>
      </div>

      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40 lg:hidden transition-all duration-300"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <div className="flex-1 flex flex-col">
        <div className={`${themeClasses.bg} ${themeClasses.borderLight} border-b px-6 py-5 flex items-center justify-between shadow-lg`}>
          <div className="flex items-center space-x-3">
            <button 
              onClick={() => setSidebarOpen(true)}
              className={`lg:hidden p-3 ${themeClasses.hoverBg} rounded-xl transition-all duration-200 hover:scale-110 shadow-md`}
            >
              <Menu className={`w-7 h-7 ${themeClasses.text}`} />
            </button>
            <div className="flex items-center space-x-3">
              <Sparkles className="w-7 h-7 text-blue-500 animate-pulse-slow" />
              <span className={`text-2xl font-bold ${themeClasses.text} tracking-tight`}>FloatChat</span>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <button 
              onClick={() => setDarkMode(!darkMode)}
              className={`p-3 ${themeClasses.hoverBg} rounded-xl ${themeClasses.textMuted} transition-all duration-200 hover:scale-110 shadow-md`}
            >
              {darkMode ? <Sun className="w-6 h-6" /> : <Moon className="w-6 h-6" />}
            </button>
            <button className={`p-3 ${themeClasses.hoverBg} rounded-xl ${themeClasses.textMuted} transition-all duration-200 hover:scale-110 shadow-md`}>
              <Share className="w-6 h-6" />
            </button>
            <button className={`p-3 ${themeClasses.hoverBg} rounded-xl ${themeClasses.textMuted} transition-all duration-200 hover:scale-110 shadow-md`}>
              <MoreVertical className="w-6 h-6" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar">
          {messages.length === 0 ? (
            <div className="max-w-5xl mx-auto px-6 py-16">
              <div className="text-center mb-12 animate-slide-in">
                <div className={`w-24 h-24 ${themeClasses.accent} rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-2xl animate-float`}>
                  <Waves className="w-12 h-12 text-white" />
                </div>
                <h1 className={`text-4xl font-extrabold ${themeClasses.text} mb-4 tracking-tight`}>
                  Welcome to FloatChat
                </h1>
                <p className={`text-xl ${themeClasses.textSecondary} max-w-2xl mx-auto mb-6 font-semibold leading-relaxed`}>
                  Your AI-powered assistant for exploring and analyzing ARGO ocean data with precision and ease.
                </p>
                
                <div className="flex flex-col sm:flex-row justify-center items-center space-y-4 sm:space-y-0 sm:space-x-4 text-base mb-6">
                  <span className={`${themeClasses.textMuted} font-semibold`}>💬 Type your query or</span>
                  
                  {voiceSupport?.supported ? (
                    voiceEnabled ? (
                      <button
                        onClick={toggleVoiceRecording}
                        className={`inline-flex items-center space-x-2 px-5 py-3 rounded-xl ${getVoiceButtonClasses()} transition-all duration-200 font-semibold shadow-lg hover:shadow-2xl hover:scale-105 text-base`}
                      >
                        <Mic className="w-6 h-6" />
                        <span>Speak Now</span>
                      </button>
                    ) : (
                      <button
                        onClick={enableVoiceInput}
                        className={`inline-flex items-center space-x-2 px-5 py-3 ${themeClasses.accent} text-white rounded-xl font-semibold shadow-lg hover:shadow-2xl hover:scale-105 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed text-base`}
                        disabled={voiceStatus === 'requesting'}
                      >
                        {voiceStatus === 'requesting' ? (
                          <>
                            <div className="w-6 h-6 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                            <span>Enabling...</span>
                          </>
                        ) : (
                          <>
                            <Mic className="w-6 h-6" />
                            <span>Enable Voice</span>
                          </>
                        )}
                      </button>
                    )
                  ) : (
                    <button
                      className="inline-flex items-center space-x-1 px-3 py-1 rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-500 cursor-not-allowed"
                      disabled
                    >
                      <MicOff className="w-6 h-6" />
                      <span>Voice Unavailable</span>
                    </button>
                  )}
                </div>

                {getVoiceSupportMessage()}
                
                {!voiceEnabled && voiceSupport?.supported && (
                  <p className={`text-base ${themeClasses.textMuted} mt-3 max-w-md mx-auto font-semibold`}>
                    🔒 Voice input is disabled by default for privacy. Click "Enable Voice" to allow microphone access for hands-free ocean data queries.
                  </p>
                )}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                {samplePrompts.map((prompt, index) => (
                  <button
                    key={index}
                    onClick={() => setInputText(prompt.description)}
                    className={`text-left p-6 ${themeClasses.promptBg} ${themeClasses.promptHoverBg} rounded-2xl ${themeClasses.border} border hover:shadow-2xl hover:scale-[1.02] transition-all duration-300 group animate-slide-in`}
                    style={{ animationDelay: `${index * 150}ms` }}
                  >
                    <div className="flex items-start space-x-4">
                      <div className={`w-14 h-14 ${themeClasses.cardBg} rounded-xl flex items-center justify-center text-blue-500 group-hover:bg-blue-50 ${darkMode ? 'group-hover:bg-blue-900/50' : ''} transition-all duration-300 shadow-lg group-hover:shadow-xl`}>
                        {prompt.icon}
                      </div>
                      <div className="flex-1">
                        <h3 className={`text-xl font-semibold ${themeClasses.text} mb-2 tracking-tight`}>{prompt.title}</h3>
                        <p className={`${themeClasses.textSecondary} text-base leading-relaxed font-semibold`}>{prompt.description}</p>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="max-w-5xl mx-auto px-6 py-8 space-y-6">
              {messages.map((message, idx) => {
                const isLastBotMsg = message.type === 'bot' && 
                  idx === messages.length - 1 && 
                  (message.hasVisualization || /visualization|chart|plot|map|graph/i.test(message.content));
                
                return (
                  <div key={message.id} className={`flex items-start space-x-4 ${message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''} animate-slide-in`}>
                    <div className={`w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0 shadow-xl ${
                      message.type === 'user' 
                        ? 'bg-blue-500 text-white' 
                        : 'bg-gradient-to-br from-purple-500 to-pink-500 text-white'
                    } transition-all duration-200 hover:scale-110`}>
                      {message.type === 'user' ? <User className="w-6 h-6" /> : <Sparkles className="w-6 h-6 animate-pulse-slow" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className={`prose prose-gray max-w-none ${message.type === 'user' ? 'text-right' : ''}`}>
                        <div className={`inline-block p-6 rounded-2xl shadow-xl ${
                          message.type === 'user' 
                            ? 'bg-blue-500 text-white' 
                            : `${themeClasses.botMessageBg} ${themeClasses.text}`
                        } transition-all duration-200 hover:shadow-2xl`}>
                          <p className="whitespace-pre-wrap m-0 text-lg font-semibold leading-relaxed">{message.content}</p>
                          {isLastBotMsg && lastVizData && (
                            <button
                              onClick={() => {
                                navigate('/dashboard', {
                                  state: { vizData: lastVizData, vizTab: lastVizTab }
                                });
                              }}
                              className="mt-4 mr-2 px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl shadow-xl hover:shadow-2xl hover:scale-105 transition-all duration-200 font-semibold text-base"
                            >
                              📊 View Visualization
                            </button>
                          )}
                          {message.filename && (
                            <button
                              onClick={() => handleDownload(message.filename)}
                              className="mt-4 px-6 py-3 bg-gradient-to-r from-green-500 to-teal-600 text-white rounded-xl shadow-xl hover:shadow-2xl hover:scale-105 transition-all duration-200 font-semibold text-base"
                            >
                              <Download className="w-6 h-6 inline-block mr-2" />
                              Download Your CSV
                            </button>
                          )}
                        </div>
                      </div>
                      <p className={`text-base ${themeClasses.textMuted} mt-2 font-semibold ${message.type === 'user' ? 'text-right' : ''}`}>
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
                <div className="flex items-start space-x-4 animate-slide-in">
                  <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center shadow-xl">
                    <Sparkles className="w-6 h-6 text-white animate-pulse-slow" />
                  </div>
                  <div className={`${themeClasses.botMessageBg} p-6 rounded-2xl shadow-xl`}>
                    <div className="flex space-x-2">
                      <div className={`w-3 h-3 ${themeClasses.textMuted} rounded-full animate-bounce`}></div>
                      <div className={`w-3 h-3 ${themeClasses.textMuted} rounded-full animate-bounce`} style={{ animationDelay: '0.1s' }}></div>
                      <div className={`w-3 h-3 ${themeClasses.textMuted} rounded-full animate-bounce`} style={{ animationDelay: '0.2s' }}></div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className={`${themeClasses.borderLight} border-t ${themeClasses.bg} shadow-xl`}>
          <div className="max-w-5xl mx-auto px-6 py-8">
            <div className="relative">
              <div className={`flex items-end space-x-4 ${themeClasses.inputBg} rounded-3xl p-5 ${themeClasses.inputBorder} border focus-within:${themeClasses.inputFocus} focus-within:ring-4 transition-all duration-200 shadow-xl hover:shadow-2xl`}>
                <textarea
                  ref={inputRef}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={isVoiceActive ? "🎤 Listening... Speak your ocean data query" : !voiceSupport?.supported ? "Ask FloatChat about ocean data..." : !voiceEnabled ? "Ask FloatChat about ocean data... (Enable voice input above)" : "Ask FloatChat about ocean data... or tap the mic to speak"}
                  className={`flex-1 bg-transparent border-none resize-none focus:outline-none placeholder-gray-500 ${themeClasses.text} font-semibold text-lg ${
                    isVoiceActive ? 'border-l-2 border-purple-400 pl-4' : ''
                  }`}
                  rows={1}
                  style={{
                    minHeight: '28px',
                    maxHeight: '200px',
                    resize: 'none'
                  }}
                  disabled={isVoiceActive}
                />
                <div className="flex items-center space-x-4">
                  <button
                    onClick={handleFileUpload}
                    className={`p-3 rounded-full transition-all duration-200 ${themeClasses.textMuted} hover:text-gray-600 ${themeClasses.buttonHoverBg} hover:scale-110 shadow-md disabled:opacity-50`}
                    title="Upload ARGO data file"
                    disabled={fileStatus.status === 'uploading'}
                  >
                    <Paperclip className="w-6 h-6" />
                  </button>
                  <button
                    onClick={voiceSupport?.supported ? (voiceEnabled ? toggleVoiceRecording : enableVoiceInput) : null}
                    disabled={isMicDisabled()}
                    className={`p-3 rounded-full transition-all duration-200 ${getVoiceButtonClasses()} shadow-md hover:scale-110 disabled:opacity-50 disabled:cursor-not-allowed`}
                    title={isVoiceActive ? "Stop listening" : !voiceSupport?.supported ? "Voice not supported in this browser" : !voiceEnabled ? "Enable voice input first" : "Start voice input"}
                  >
                    {getVoiceIcon()}
                  </button>
                  <button
                    onClick={handleSendMessage}
                    disabled={inputText.trim() === '' || isVoiceActive}
                    className={`p-3 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-200 disabled:text-gray-400 text-white rounded-full transition-all duration-200 shadow-md hover:shadow-xl hover:scale-110 ${
                      inputText.trim() !== '' && !isVoiceActive ? 'animate-pulse-slow' : ''
                    }`}
                  >
                    <Send className="w-6 h-5" />
                  </button>
                </div>
              </div>
              
              {voiceStatus === 'listening' && (
                <div className="flex items-center justify-center mt-4 space-x-2">
                  <div className="w-3 h-3 bg-purple-500 rounded-full animate-ping"></div>
                  <span className={`text-base ${themeClasses.textSecondary} font-semibold`}>
                    🎤 Listening... Speak clearly about ocean data queries
                  </span>
                </div>
              )}
              
              {voiceStatus === 'error' && (
                <div className="flex items-center justify-center mt-4">
                  <span className={`text-base text-red-500 font-semibold`}>
                    ❌ Microphone access required. Please allow access and try again.
                  </span>
                </div>
              )}
              {voiceError && (
                <p className="text-red-500 text-xs mt-1">{voiceError}</p>
              )}
              
              {voiceStatus === 'requesting' && (
                <div className="flex items-center justify-center mt-4">
                  <div className="w-4 h-4 border-2 border-purple-500 border-t-transparent rounded-full animate-spin mr-2"></div>
                  <span className={`text-base ${themeClasses.textSecondary} font-semibold`}>
                    Requesting microphone permission...
                  </span>
                </div>
              )}

              {fileStatus.status !== 'idle' && (
                <div className="flex items-center justify-center mt-4">
                  {fileStatus.status === 'uploading' && (
                    <>
                      <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mr-2"></div>
                      <span className={`text-base ${themeClasses.textSecondary} font-semibold`}>{fileStatus.message}</span>
                    </>
                  )}
                  {fileStatus.status === 'success' && (
                    <span className={`text-base text-green-500 font-semibold`}>{fileStatus.message}</span>
                  )}
                  {fileStatus.status === 'error' && (
                    <span className={`text-base text-red-500 font-semibold`}>{fileStatus.message}</span>
                  )}
                </div>
              )}
            </div>
            <div className="flex items-center justify-center mt-4">
              <p className={`text-base ${themeClasses.textMuted} font-semibold`}>
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