import React, { useState } from 'react';
import { 
  Search, 
  Filter, 
  Calendar, 
  Clock, 
  MessageCircle, 
  BarChart3, 
  Map, 
  Eye, 
  Download, 
  Star, 
  MoreVertical,
  ChevronRight,
  Bot,
  User,
  Waves,
  Thermometer,
  Droplets,
  Globe,
  Database,
  Trash2,
  Share,
  Copy,
  Archive
} from 'lucide-react';
import Navbar from '../components/Navbar';

function History() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFilter, setSelectedFilter] = useState('all');
  const [selectedChat, setSelectedChat] = useState(null);

  // Sample chat history data
  const chatHistory = [
    {
      id: 1,
      title: "Ocean Temperature Analysis",
      query: "Show me temperature profiles in the Indian Ocean for the last month",
      response: "I found temperature data from 15 ARGO floats in the Indian Ocean region. The data shows average surface temperatures ranging from 28.5°C to 30.2°C...",
      timestamp: "2024-01-15 14:30",
      date: "Today",
      type: "temperature",
      hasVisualization: true,
      visualizationType: "plots",
      starred: false,
      tags: ["temperature", "indian ocean", "argo floats"]
    },
    {
      id: 2,
      title: "Salinity Data Comparison",
      query: "Compare salinity levels in the Arabian Sea vs Bay of Bengal",
      response: "Based on recent ARGO float data, the Arabian Sea shows higher salinity levels (36.2-36.8 psu) compared to the Bay of Bengal (33.5-34.2 psu)...",
      timestamp: "2024-01-14 09:15",
      date: "Yesterday",
      type: "salinity",
      hasVisualization: true,
      visualizationType: "comparison",
      starred: true,
      tags: ["salinity", "arabian sea", "bay of bengal", "comparison"]
    },
    {
      id: 3,
      title: "ARGO Float Trajectories",
      query: "Display the paths of ARGO floats in the Pacific Ocean",
      response: "I've mapped the trajectories of 8 active ARGO floats in the Pacific region. The floats show interesting circulation patterns...",
      timestamp: "2024-01-13 16:45",
      date: "2 days ago",
      type: "trajectory",
      hasVisualization: true,
      visualizationType: "map",
      starred: false,
      tags: ["trajectory", "pacific", "argo floats", "circulation"]
    },
    {
      id: 4,
      title: "BGC Parameter Analysis",
      query: "Analyze bio-geo-chemical parameters in the Southern Ocean",
      response: "The Southern Ocean BGC data reveals significant seasonal variations in chlorophyll-a concentrations and nutrient levels...",
      timestamp: "2024-01-12 11:20",
      date: "3 days ago",
      type: "bgc",
      hasVisualization: true,
      visualizationType: "plots",
      starred: false,
      tags: ["bgc", "southern ocean", "chlorophyll", "nutrients"]
    },
    {
      id: 5,
      title: "Deep Water Mass Analysis",
      query: "What are the characteristics of deep water masses in the Atlantic?",
      response: "Deep water masses in the Atlantic show distinct temperature and salinity signatures. North Atlantic Deep Water (NADW) exhibits...",
      timestamp: "2024-01-11 13:10",
      date: "4 days ago",
      type: "analysis",
      hasVisualization: false,
      visualizationType: null,
      starred: true,
      tags: ["deep water", "atlantic", "nadw", "water masses"]
    }
  ];

  const getTypeIcon = (type) => {
    switch (type) {
      case 'temperature': return <Thermometer className="w-4 h-4" />;
      case 'salinity': return <Droplets className="w-4 h-4" />;
      case 'trajectory': return <Map className="w-4 h-4" />;
      case 'bgc': return <BarChart3 className="w-4 h-4" />;
      default: return <Database className="w-4 h-4" />;
    }
  };

  const getTypeColor = (type) => {
    switch (type) {
      case 'temperature': return 'text-orange-400 bg-orange-500/20';
      case 'salinity': return 'text-blue-400 bg-blue-500/20';
      case 'trajectory': return 'text-emerald-400 bg-emerald-500/20';
      case 'bgc': return 'text-purple-400 bg-purple-500/20';
      default: return 'text-gray-400 bg-gray-500/20';
    }
  };

  const filteredChats = chatHistory.filter(chat => {
    const matchesSearch = chat.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         chat.query.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         chat.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
    const matchesFilter = selectedFilter === 'all' || 
                         (selectedFilter === 'starred' && chat.starred) ||
                         (selectedFilter === 'visualization' && chat.hasVisualization) ||
                         chat.type === selectedFilter;
    
    return matchesSearch && matchesFilter;
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-900 text-white">
      {/* Navbar */}
      <div className="relative z-50 pointer-events-auto">
        <Navbar onOpenChat={() => {}} />
      </div>

      {/* Main Content Container */}
      <div className="max-w-7xl mx-auto px-4 lg:px-6 py-8">
        {/* Page Header Section */}
        <div className="mb-8">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-emerald-500/20 to-blue-500/20 border border-emerald-500/30 flex items-center justify-center shadow-lg">
              <Waves className="w-8 h-8 text-emerald-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">Chat History</h1>
              <p className="text-lg text-blue-200">Your ARGO ocean data exploration journey</p>
            </div>
          </div>

          {/* Search and Filter Bar */}
          <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-6 shadow-lg">
            <div className="flex flex-col lg:flex-row gap-4">
              <div className="flex-1 relative">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search conversations, queries, or tags..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-12 pr-4 py-4 bg-white/10 border border-white/20 rounded-xl focus:ring-2 focus:ring-emerald-400 focus:border-emerald-400 outline-none placeholder-gray-400 text-white hover:bg-white/15 transition-colors duration-200"
                />
              </div>
              
              <div className="flex gap-3">
                <select
                  value={selectedFilter}
                  onChange={(e) => setSelectedFilter(e.target.value)}
                  className="px-6 py-4 bg-white/10 border border-white/20 rounded-xl focus:ring-2 focus:ring-emerald-400 focus:border-emerald-400 outline-none text-white hover:bg-white/15 transition-colors duration-200 min-w-[200px]"
                  style={{ colorScheme: 'dark' }}
                >
                  <option value="all" className="bg-slate-800 text-white">All Conversations</option>
                  <option value="starred" className="bg-slate-800 text-white">⭐ Starred</option>
                  <option value="visualization" className="bg-slate-800 text-white">📊 With Visualizations</option>
                  <option value="temperature" className="bg-slate-800 text-white">🌡️ Temperature</option>
                  <option value="salinity" className="bg-slate-800 text-white">🧂 Salinity</option>
                  <option value="trajectory" className="bg-slate-800 text-white">🗺️ Trajectories</option>
                  <option value="bgc" className="bg-slate-800 text-white">🔬 BGC</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Chat List */}
          <div className="lg:col-span-2">
            <div className="space-y-6">
              {filteredChats.map((chat) => (
                <div
                  key={chat.id}
                  onClick={() => setSelectedChat(chat)}
                  className={`bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-6 cursor-pointer transition-all duration-300 hover:bg-white/10 hover:border-white/20 hover:shadow-xl hover:shadow-emerald-500/10 group ${
                    selectedChat?.id === chat.id ? 'bg-white/10 border-emerald-500/50 shadow-lg shadow-emerald-500/20' : ''
                  }`}
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${getTypeColor(chat.type)}`}>
                        {getTypeIcon(chat.type)}
                      </div>
                      <div>
                        <h3 className="font-semibold text-lg group-hover:text-emerald-300 transition-colors">
                          {chat.title}
                        </h3>
                        <div className="flex items-center gap-2 text-sm text-gray-400">
                          <Clock className="w-4 h-4" />
                          <span>{chat.date} • {chat.timestamp.split(' ')[1]}</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      {chat.starred && (
                        <Star className="w-5 h-5 text-yellow-400 fill-yellow-400" />
                      )}
                      {chat.hasVisualization && (
                        <div className="w-8 h-8 rounded-lg bg-emerald-500/20 flex items-center justify-center">
                          <Eye className="w-4 h-4 text-emerald-400" />
                        </div>
                      )}
                      <button className="p-2 rounded-lg bg-white/10 hover:bg-white/20 opacity-0 group-hover:opacity-100 transition-all">
                        <MoreVertical className="w-4 h-4" />
                      </button>
                    </div>
                  </div>

                  <div className="mb-4">
                    <div className="flex items-start gap-3 mb-3">
                      <div className="w-6 h-6 rounded-full bg-blue-500 flex items-center justify-center flex-shrink-0">
                        <User className="w-3 h-3 text-white" />
                      </div>
                      <p className="text-sm text-gray-300 leading-relaxed">{chat.query}</p>
                    </div>
                    
                    <div className="flex items-start gap-3">
                      <div className="w-6 h-6 rounded-full bg-gradient-to-r from-emerald-500 to-emerald-600 flex items-center justify-center flex-shrink-0">
                        <Bot className="w-3 h-3 text-white" />
                      </div>
                      <p className="text-sm text-gray-300 leading-relaxed line-clamp-2">
                        {chat.response}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex flex-wrap gap-2">
                      {chat.tags.slice(0, 3).map((tag, index) => (
                        <span
                          key={index}
                          className="px-2 py-1 text-xs bg-white/10 rounded-lg text-gray-300"
                        >
                          #{tag}
                        </span>
                      ))}
                      {chat.tags.length > 3 && (
                        <span className="px-2 py-1 text-xs bg-white/10 rounded-lg text-gray-300">
                          +{chat.tags.length - 3} more
                        </span>
                      )}
                    </div>
                    
                    <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-emerald-400 transition-colors" />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Chat Details Sidebar */}
          <div className="lg:col-span-1">
            {selectedChat ? (
              <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-6 sticky top-8 shadow-lg">
                <div className="flex items-center gap-3 mb-6">
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${getTypeColor(selectedChat.type)}`}>
                    {getTypeIcon(selectedChat.type)}
                  </div>
                  <div>
                    <h3 className="font-semibold text-lg">{selectedChat.title}</h3>
                    <p className="text-sm text-gray-400">{selectedChat.date}</p>
                  </div>
                </div>

                <div className="space-y-4 mb-6">
                  <div>
                    <h4 className="text-sm font-medium text-gray-300 mb-2">Your Query</h4>
                    <div className="bg-white/5 rounded-lg p-3">
                      <p className="text-sm text-gray-300">{selectedChat.query}</p>
                    </div>
                  </div>

                  <div>
                    <h4 className="text-sm font-medium text-gray-300 mb-2">AI Response</h4>
                    <div className="bg-white/5 rounded-lg p-3">
                      <p className="text-sm text-gray-300">{selectedChat.response}</p>
                    </div>
                  </div>
                </div>

                <div className="space-y-3">
                  <h4 className="text-sm font-medium text-gray-300">Tags</h4>
                  <div className="flex flex-wrap gap-2">
                    {selectedChat.tags.map((tag, index) => (
                      <span
                        key={index}
                        className="px-3 py-1 text-xs bg-emerald-500/20 text-emerald-300 rounded-lg"
                      >
                        #{tag}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="mt-6 pt-6 border-t border-white/10">
                  <div className="flex gap-2">
                    <button className="flex-1 px-4 py-2 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 rounded-lg transition-colors flex items-center justify-center gap-2">
                      <Share className="w-4 h-4" />
                      Share
                    </button>
                    <button className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors">
                      <Copy className="w-4 h-4" />
                    </button>
                    <button className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors">
                      <Archive className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-8 text-center shadow-lg">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 border border-blue-500/30 flex items-center justify-center mx-auto mb-6">
                  <MessageCircle className="w-8 h-8 text-blue-400" />
                </div>
                <h3 className="font-semibold text-xl mb-3 text-white">Select a Conversation</h3>
                <p className="text-gray-300 leading-relaxed">
                  Choose a chat from the list to view details, insights, and conversation history
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default History;


