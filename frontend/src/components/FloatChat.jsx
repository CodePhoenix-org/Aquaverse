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
  Loader2,
  AlertCircle
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

// Utility function to get JWT token
const getAuthToken = () => {
  try {
    return localStorage.getItem('auth_token');
  } catch {
    return null;
  }
};

// Check if backend chat history endpoints are available
const checkChatHistoryEndpoints = async (token) => {
  if (!token) return false;
  
  try {
    const response = await fetch('http://127.0.0.1:8000/chat/history', {
      method: 'GET',
      headers: { 
        'Authorization': `Bearer ${token}`, 
        'Content-Type': 'application/json' 
      },
    });
    return response.ok;
  } catch {
    return false;
  }
};

// Fetch chat history from backend (with fallback)
const fetchChatHistory = async (token, setMessages, setBackendAvailable) => {
  if (!token) return;

  try {
    const response = await fetch('http://127.0.0.1:8000/chat/history', {
      method: 'GET',
      headers: { 
        'Authorization': `Bearer ${token}`, 
        'Content-Type': 'application/json' 
      },
    });

    if (!response.ok) {
      if (response.status === 404) {
        setBackendAvailable(false);
        console.warn('Chat history endpoints not found. Using local storage.');
        return;
      }
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    const mappedMessages = data.map(msg => ({
      id: msg.id,
      type: msg.sender === 'user' ? 'user' : 'bot',
      content: msg.content,
      timestamp: new Date(msg.timestamp),
      viz_data: msg.viz_data || null,
      viz_tab: msg.viz_tab || null
    }));

    setMessages(mappedMessages);
    setBackendAvailable(true);
  } catch (error) {
    console.error('Error fetching chat history:', error);
    setBackendAvailable(false);
    // Fallback to local storage if backend fails
    const fallback = localStorage.getItem('floatchat_messages');
    if (fallback) {
      setMessages(JSON.parse(fallback));
    }
  }
};

// Save message to backend (with fallback)
const saveMessageToBackend = async (token, messageData, setBackendAvailable) => {
  if (!token) return;

  try {
    const response = await fetch('http://127.0.0.1:8000/chat/history', {
      method: 'POST',
      headers: { 
        'Authorization': `Bearer ${token}`, 
        'Content-Type': 'application/json' 
      },
      body: JSON.stringify({
        sender: messageData.type === 'user' ? 'user' : 'bot',
        content: messageData.content,
        viz_data: messageData.viz_data || null,
        viz_tab: messageData.viz_tab || null
      }),
    });

    if (!response.ok) {
      if (response.status === 404) {
        setBackendAvailable(false);
        console.warn('Chat history POST endpoint not found. Using local storage.');
        return;
      }
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error saving message:', error);
    setBackendAvailable(false);
    throw error;
  }
};

// Handle Enter key press in textarea
const handleKeyPress = (event, handleSendMessage) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    handleSendMessage();
  }
};

// Handle file input change
const handleFileChange = (event) => {
  const file = event.target.files && event.target.files[0];
  if (file) {
    console.log('Selected file:', file.name);
  }
};

const FloatChat = () => {
  const navigate = useNavigate();
  
  // Get auth token
  const token = getAuthToken();
  const isAuthenticated = !!token;
  
  // Backend availability state
  const [backendAvailable, setBackendAvailable] = useState(true);
  
  // Restore state from localStorage if available (only for unauthenticated users or if backend unavailable)
  const getInitialState = (key, fallback) => {
    if (isAuthenticated && backendAvailable) return fallback; // Don't use localStorage for auth users with working backend
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
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    try {
      return localStorage.getItem('darkMode') === 'true';
    } catch {
      return true;
    }
  });
  
  const [chatHistory] = useState([
    { id: 1, title: "Ocean temperature analysis", snippet: "Show me temperature profiles...", date: "Today", starred: false },
    { id: 2, title: "Salinity data Arabian Sea", snippet: "Compare salinity levels in...", date: "Yesterday", starred: true },
    { id: 3, title: "ARGO float trajectories", snippet: "Display float paths for...", date: "2 days ago", starred: false },
    { id: 4, title: "BGC parameter analysis", snippet: "Analyze bio-geo-chemical...", date: "1 week ago", starred: false }
  ]);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  // Check backend availability and load chat history on mount for authenticated users
  useEffect(() => {
    const initializeChat = async () => {
      if (isAuthenticated && token) {
        setIsLoadingHistory(true);
        // First check if endpoints exist
        const endpointsAvailable = await checkChatHistoryEndpoints(token);
        setBackendAvailable(endpointsAvailable);
        
        if (endpointsAvailable) {
          await fetchChatHistory(token, setMessages, setBackendAvailable);
        } else {
          // Fallback to local storage
          const fallback = localStorage.getItem('floatchat_messages');
          if (fallback) {
            setMessages(JSON.parse(fallback));
          }
        }
        setIsLoadingHistory(false);
      }
    };

    initializeChat();
  }, [token]);

  // Persist chat state to localStorage only when not using backend
  useEffect(() => { 
    if (!isAuthenticated || !backendAvailable) {
      localStorage.setItem('floatchat_messages', JSON.stringify(messages));
    }
  }, [messages, isAuthenticated, backendAvailable]);
  
  useEffect(() => { 
    if (!isAuthenticated || !backendAvailable) {
      localStorage.setItem('floatchat_lastVizData', JSON.stringify(lastVizData));
    }
  }, [lastVizData, isAuthenticated, backendAvailable]);
  
  useEffect(() => { 
    if (!isAuthenticated || !backendAvailable) {
      localStorage.setItem('floatchat_lastVizTab', JSON.stringify(lastVizTab));
    }
  }, [lastVizTab, isAuthenticated, backendAvailable]);
  
  useEffect(() => { 
    if (!isAuthenticated || !backendAvailable) {
      localStorage.setItem('floatchat_inputText', JSON.stringify(inputText));
    }
  }, [inputText, isAuthenticated, backendAvailable]);

  // Persist dark mode
  useEffect(() => {
    localStorage.setItem('darkMode', darkMode);
  }, [darkMode]);

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

  const handleSendMessage = async () => {
    if (inputText.trim() === '') return;

    const userMessage = {
      id: `temp-${Date.now()}`, // Temporary ID for optimistic update
      type: 'user',
      content: inputText,
      timestamp: new Date(),
      isOptimistic: true
    };

    // Optimistic update: show user message immediately
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsTyping(true);

    // Save user message to backend if authenticated and backend available
    if (isAuthenticated && token && backendAvailable) {
      try {
        await saveMessageToBackend(token, userMessage, setBackendAvailable);
      } catch (error) {
        console.error('Error saving user message:', error);
        // Remove optimistic message on error (if backend fails)
        if (error.message.includes('404')) {
          setBackendAvailable(false);
          setMessages(prev => prev.filter(msg => msg.id !== userMessage.id));
          setIsTyping(false);
          return;
        }
      }
    }

    // Add thinking message
    const thinkingMessageId = `temp-thinking-${Date.now()}`;
    const thinkingBotMessage = {
      id: thinkingMessageId,
      type: 'bot',
      content: `🤔 Thinking... Analyzing your ocean data query: "${inputText}"\n\nProcessing ARGO float database...`,
      timestamp: new Date(),
      isOptimistic: true
    };
    setMessages(prev => [...prev, thinkingBotMessage]);

    // Fetch real backend response for AI
    let apiResponse = null;
    let apiError = null;

    try {
      const { default: axios } = await import('axios');
      const config = token ? {
        headers: { 
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      } : {
        headers: { 'Content-Type': 'application/json' }
      };

      const endpoint = token ? 'http://127.0.0.1:8000/chat/query' : 'http://127.0.0.1:8000/chat';
      const res = await axios.post(endpoint, { query: inputText }, config);
      apiResponse = res;
    } catch (error) {
      console.error('AI API call failed:', error);
      apiError = error;
    }

    let backendMsg = '';
    let vizData = null;
    let vizTab = null;

    if (apiError) {
      backendMsg = '⚠️ Sorry, I encountered an error processing your request. Please try again.';
    } else {
      // Support both array and object responses
      if (Array.isArray(apiResponse.data)) {
        backendMsg = apiResponse.data[0] || 'No response from backend.';
        vizData = apiResponse.data[1] || null;
      } else if (typeof apiResponse.data === 'object') {
        backendMsg = apiResponse.data.message || apiResponse.data.answer || 'No response from backend.';
        vizData = apiResponse.data.data || null;
      } else {
        backendMsg = String(apiResponse.data);
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

      // Save bot response to backend if authenticated and backend available
      if (isAuthenticated && token && backendAvailable) {
        const botMessageData = {
          id: `temp-bot-${Date.now()}`,
          type: 'bot',
          content: backendMsg,
          timestamp: new Date(),
          viz_data: vizData,
          viz_tab: vizTab
        };
        try {
          await saveMessageToBackend(token, botMessageData, setBackendAvailable);
        } catch (error) {
          console.error('Error saving bot message:', error);
          if (error.message.includes('404')) {
            setBackendAvailable(false);
          }
        }
      }
    }

    // Replace thinking message with actual response
    setMessages(prev => prev.map(msg => 
      msg.id === thinkingMessageId 
        ? { 
            ...msg, 
            id: `bot-${Date.now()}`, // Give it a stable ID
            content: backendMsg,
            viz_data: vizData,
            viz_tab: vizTab,
            isOptimistic: false
          }
        : msg
    ));

    setIsTyping(false);

    // Refetch complete history from backend to ensure consistency (only if backend is available)
    if (isAuthenticated && token && backendAvailable) {
      setTimeout(() => {
        fetchChatHistory(token, setMessages, setBackendAvailable);
      }, 500);
    }
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

  // Show loading screen while fetching history
  if (isAuthenticated && isLoadingHistory) {
    return (
      <div className={`h-screen ${themeClasses.bg} flex items-center justify-center`}>
        <div className="flex flex-col items-center space-y-4">
          <div className="w-12 h-12 border-4 border-blue-500/20 border-t-blue-500 rounded-full animate-spin"></div>
          <p className={`text-lg ${themeClasses.text}`}>Loading your chat history...</p>
        </div>
      </div>
    );
  }

  // Show backend unavailable warning
  const showBackendWarning = isAuthenticated && !backendAvailable;

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

      {/* Backend Unavailable Warning Banner */}
      {showBackendWarning && (
        <div className="fixed top-0 left-0 right-0 z-50 bg-yellow-100 border-b border-yellow-300 dark:bg-yellow-900 dark:border-yellow-700">
          <div className="max-w-4xl mx-auto px-4 py-2 flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-yellow-700 dark:text-yellow-300" />
            <span className="text-sm text-yellow-800 dark:text-yellow-200">
              Chat history endpoints not found. Using local storage as fallback.
            </span>
          </div>
        </div>
      )}

      {/* Sidebar */}
      <div
        className={`
          fixed inset-y-0 left-0 z-50 ${showBackendWarning ? 'lg:top-12' : ''}
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
                <p className={`text-xs ${themeClasses.textMuted}`}>
                  {isAuthenticated 
                    ? (backendAvailable ? 'ARGO Data Discovery' : 'ARGO Data Discovery (Local)')
                    : 'ARGO Data Discovery (Guest)'
                  }
                </p>
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
            onClick={async () => {
              if (isAuthenticated && token && backendAvailable) {
                // For authenticated users with working backend, clear backend history
                try {
                  await fetch('http://127.0.0.1:8000/chat/history', {
                    method: 'DELETE',
                    headers: { 
                      'Authorization': `Bearer ${token}`, 
                      'Content-Type': 'application/json' 
                    },
                  });
                  await fetchChatHistory(token, setMessages, setBackendAvailable);
                } catch (err) {
                  console.error('Error clearing chat history:', err);
                  if (err.message.includes('404')) {
                    setBackendAvailable(false);
                  }
                }
              } else {
                // For guest users or backend unavailable, clear local state
                setMessages([]);
                setInputText('');
                setLastVizData(null);
                setLastVizTab(null);
                localStorage.removeItem('floatchat_messages');
                localStorage.removeItem('floatchat_lastVizData');
                localStorage.removeItem('floatchat_lastVizTab');
                localStorage.removeItem('floatchat_inputText');
              }
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
          className={`fixed inset-0 bg-black bg-opacity-25 z-40 lg:hidden ${showBackendWarning ? 'top-12' : ''}`}
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Content */}
      <div className={`flex-1 flex flex-col ${showBackendWarning ? 'lg:pt-12' : ''}`}>
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
                  (/check the visualization|view the chart|see the data|open dashboard/i.test(message.content) || message.viz_data);
                
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
                          {isLastBotMsg && message.viz_data && (
                            <button
                              onClick={() => {
                                navigate('/dashboard', {
                                  state: { vizData: message.viz_data, vizTab: message.viz_tab }
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
                    <div className="flex items-center space-x-3">
                      <Loader2 className={`w-4 h-4 ${themeClasses.textMuted} animate-spin`} />
                      <div className="flex space-x-2">
                        <div className={`w-2 h-2 ${themeClasses.textMuted} rounded-full animate-bounce`}></div>
                        <div className={`w-2 h-2 ${themeClasses.textMuted} rounded-full animate-bounce`} style={{ animationDelay: '0.1s' }}></div>
                        <div className={`w-2 h-2 ${themeClasses.textMuted} rounded-full animate-bounce`} style={{ animationDelay: '0.2s' }}></div>
                      </div>
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
                  onKeyPress={(e) => handleKeyPress(e, handleSendMessage)}
                  placeholder="Ask FloatChat about ocean data..."
                  className={`flex-1 bg-transparent border-none resize-none focus:outline-none placeholder-gray-500 ${themeClasses.text}`}
                  rows={1}
                  style={{
                    minHeight: '24px',
                    maxHeight: '200px',
                    resize: 'none'
                  }}
                  disabled={isTyping}
                />
                <div className="flex items-center space-x-2">
                  <button
                    onClick={handleFileUpload}
                    disabled={isTyping}
                    className={`p-2 rounded-full transition-colors ${themeClasses.textMuted} hover:text-gray-600 ${themeClasses.buttonHoverBg} ${isTyping ? 'opacity-50 cursor-not-allowed' : ''}`}
                    title="Upload ARGO data file"
                  >
                    <Paperclip className="w-5 h-5" />
                  </button>
                  <button
                    onClick={toggleRecording}
                    disabled={isTyping}
                    className={`p-2 rounded-full transition-colors ${
                      isRecording 
                        ? 'bg-red-500 text-white animate-pulse' 
                        : `${themeClasses.textMuted} hover:text-gray-600 ${themeClasses.buttonHoverBg}`
                    } ${isTyping ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    {isRecording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
                  </button>
                  <button
                    onClick={handleSendMessage}
                    disabled={inputText.trim() === '' || isTyping}
                    className={`p-2 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-200 disabled:text-gray-400 text-white rounded-full transition-colors ${isTyping ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    <Send className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
            <div className="flex items-center justify-center mt-3">
              <p className={`text-xs ${themeClasses.textMuted}`}>
                {isAuthenticated 
                  ? (backendAvailable 
                      ? 'FloatChat can make mistakes. Consider checking important ocean data insights.' 
                      : 'Chat history endpoints not found. Using local storage.')
                  : 'Guest mode: Your chat history will not be saved.'
                }
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FloatChat;