import { useState, useRef, useEffect } from 'react';
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

const FloatChat = () => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [chatHistory, setChatHistory] = useState([
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

    // Simulate AI response
    setTimeout(() => {
      const botMessage = {
        id: messages.length + 2,
        type: 'bot',
        content: `I'll analyze your ocean data query: "${inputText}"\n\nBased on our ARGO float database, I can help you explore temperature, salinity, and BGC parameters across different ocean regions. Let me process this request and generate the appropriate visualizations and insights.`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMessage]);
      setIsTyping(false);
    }, 1500);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

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

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Simulate file upload
      const fileMessage = {
        id: messages.length + 1,
        type: 'user',
        content: `📁 Uploaded file: ${file.name}\n\nPlease analyze this ARGO NetCDF data file and provide insights about the ocean parameters contained within.`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, fileMessage]);
      
      // Simulate AI response to file upload
      setTimeout(() => {
        const botMessage = {
          id: messages.length + 2,
          type: 'bot',
          content: `I've received your file "${file.name}". I can help you analyze this ARGO data file. Based on the filename, this appears to contain oceanographic measurements. I'll process the NetCDF format and extract relevant parameters like temperature, salinity, and depth profiles for visualization and analysis.`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
      }, 1000);
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
      <div className={`${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} fixed inset-y-0 left-0 z-50 w-80 ${themeClasses.sidebarBg} ${themeClasses.border} border-r transform transition-transform duration-300 ease-in-out lg:relative lg:translate-x-0 lg:flex lg:flex-col`}>
        {/* Sidebar Header */}
        <div className={`flex items-center justify-between p-4 ${themeClasses.border} border-b`}>
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center">
              <Waves className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className={`text-lg font-semibold ${themeClasses.text}`}>FloatChat</h1>
              <p className={`text-xs ${themeClasses.textMuted}`}>ARGO Data Discovery</p>
            </div>
          </div>
          <button 
            onClick={() => setSidebarOpen(false)}
            className={`lg:hidden p-2 ${themeClasses.hoverBg} rounded-lg transition-colors`}
          >
            <X className={`w-5 h-5 ${themeClasses.text}`} />
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-4">
          <button className={`w-full ${themeClasses.cardBg} ${themeClasses.border} border ${themeClasses.hoverBg} ${themeClasses.textSecondary} py-3 px-4 rounded-xl flex items-center justify-center space-x-2 transition-colors shadow-sm`}>
            <Edit3 className="w-4 h-4" />
            <span>New chat</span>
          </button>
        </div>

        {/* Chat History */}
        <div className="flex-1 px-4 overflow-y-auto">
          <div className="space-y-1">
            <h3 className={`text-sm font-medium ${themeClasses.textMuted} px-2 py-1`}>Recent</h3>
            {chatHistory.map((chat) => (
              <div key={chat.id} className={`group p-3 ${themeClasses.cardHoverBg} rounded-xl cursor-pointer transition-colors border border-transparent hover:border-gray-300 ${darkMode ? 'hover:border-gray-600' : ''} hover:shadow-sm`}>
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
        </div>

        {/* Sidebar Footer */}
        <div className={`p-4 ${themeClasses.border} border-t space-y-2`}>
          <button className={`w-full text-left p-3 ${themeClasses.cardHoverBg} rounded-xl flex items-center space-x-3 transition-colors ${themeClasses.textSecondary}`}>
            <Settings className="w-5 h-5" />
            <span>Settings</span>
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
        <div className="flex-1 overflow-y-auto">
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
              {messages.map((message) => (
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
                      </div>
                    </div>
                    <p className={`text-xs ${themeClasses.textMuted} mt-2 ${message.type === 'user' ? 'text-right' : ''}`}>
                      {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  </div>
                </div>
              ))}

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
