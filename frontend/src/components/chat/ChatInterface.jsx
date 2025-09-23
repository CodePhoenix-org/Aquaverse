import React, { useState, useRef, useEffect } from 'react';
import { PaperAirplaneIcon, SparklesIcon } from '@heroicons/react/24/outline';
import { Send, Bot, User, Loader2, X } from 'lucide-react';
import axios from 'axios';

const ChatInterface = ({ onDataReceived, onCloseChat }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      content:
        'Welcome to FloatChat! I can help you explore ARGO ocean data. Try asking me something like "Show me salinity profiles near the equator in March 2023"',
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [lastVizData, setLastVizData] = useState(null);
  const [lastVizTab, setLastVizTab] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
  // Debug: log input value
  console.log('Sending query:', inputValue);
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Send POST request to your backend API
      const response = await axios.post('http://127.0.0.1:8000/chat/query', {
        query: inputValue
      });
      // Debug: log full response
      console.log('Full response:', response);
      console.log('Response data:', response.data);
      console.log('Type of response data:', typeof response.data);

      // Extract the message properly
      let messageContent = '';
      let responseData = null;
      let vizTab = null;

      // Handle both array and object responses
      if (Array.isArray(response.data)) {
        // This is the array format you're currently getting
        messageContent = response.data[0] || '⚠️ No response';
        responseData = response.data[1] || null;
      } else if (typeof response.data === 'object') {
        // This is the proper object format
        messageContent = response.data.message || response.data.answer || '⚠️ No response';
        responseData = response.data.data || null;
      } else {
        // Fallback for unexpected formats
        messageContent = String(response.data);
      }

      // Determine visualization tab based on data type
      if (responseData && responseData.type) {
        if (responseData.type.includes('profile')) vizTab = 'plots';
        else if (responseData.type.includes('map')) vizTab = 'map';
        else if (responseData.type.includes('comparison')) vizTab = 'comparison';
        else if (responseData.type.includes('table')) vizTab = 'table';
      }

      // Store visualization data for the "View Visualization" button
      setLastVizData(responseData);
      setLastVizTab(vizTab);

      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: messageContent,
        data: responseData,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, botMessage]);

      if (responseData) {
        onDataReceived(responseData);
      }
    } catch (error) {
      console.error('API Error:', error);
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-gray-50 to-white">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-thin">
        {messages.map((message, index) => {
          const isLastBotMsg = 
            message.type === 'bot' && 
            index === messages.length - 1 && 
            lastVizData;
          
          return (
            <div
              key={message.id}
              className={`flex ${
                message.type === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div className={`flex items-start gap-3 max-w-[80%] ${
                message.type === 'user' ? 'flex-row-reverse' : 'flex-row'
              }`}>
                {/* Avatar */}
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  message.type === 'user' 
                    ? 'bg-gradient-to-r from-blue-500 to-blue-600' 
                    : 'bg-gradient-to-r from-emerald-500 to-emerald-600'
                }`}>
                  {message.type === 'user' ? (
                    <User className="w-4 h-4 text-white" />
                  ) : (
                    <Bot className="w-4 h-4 text-white" />
                  )}
                </div>
                
                {/* Message Bubble */}
                <div className={`px-4 py-3 rounded-2xl shadow-sm ${
                  message.type === 'user'
                    ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white'
                    : 'bg-white text-gray-800 border border-gray-200'
                }`}>
                  <p className="text-sm leading-relaxed">
                    {typeof message.content === 'string'
                      ? message.content
                      : JSON.stringify(message.content)}
                  </p>
                  
                  {/* View Visualization Button */}
                  {isLastBotMsg && (
                    <button
                      onClick={() => {
                        onDataReceived(lastVizData);
                        onCloseChat && onCloseChat();
                      }}
                      className="mt-3 px-4 py-2 bg-gradient-to-r from-emerald-500 to-blue-600 text-white rounded-xl shadow hover:scale-105 transition-all duration-200 flex items-center gap-2 text-sm font-medium"
                    >
                      <Bot className="w-4 h-4" />
                      View Visualization
                    </button>
                  )}
                  
                  <p className={`text-xs mt-2 ${
                    message.type === 'user' ? 'text-blue-100' : 'text-gray-500'
                  }`}>
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            </div>
          );
        })}

        {isLoading && (
          <div className="flex justify-start">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-gradient-to-r from-emerald-500 to-emerald-600 flex items-center justify-center">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <div className="bg-white px-4 py-3 rounded-2xl shadow-sm border border-gray-200">
                <div className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin text-emerald-500" />
                  <span className="text-sm text-gray-600">Analyzing your request...</span>
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 bg-white p-4">
        <div className="flex gap-3">
          <div className="flex-1 relative">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about ARGO ocean data... (e.g., 'Show temperature profiles in the Indian Ocean')"
              className="w-full border border-gray-300 rounded-xl px-4 py-3 resize-none focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 text-gray-900 placeholder:text-gray-400 bg-gray-50 hover:bg-white transition-colors duration-200"
              rows="2"
              disabled={isLoading}
            />
            <div className="absolute bottom-2 right-2 text-xs text-gray-400">
              Press Enter to send
            </div>
          </div>
          <button
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isLoading}
            className="bg-gradient-to-r from-emerald-500 to-emerald-600 text-white px-4 py-3 rounded-xl hover:from-emerald-600 hover:to-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center min-w-[48px] shadow-sm hover:shadow-md"
          >
            {isLoading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Send className="h-5 w-5" />
            )}
          </button>
        </div>
        
        {/* Quick Actions */}
        <div className="mt-3 flex flex-wrap gap-2">
          <span className="text-xs text-gray-500">Quick actions:</span>
          {[
            "Show temperature profiles",
            "Find floats in Pacific",
            "Compare salinity data",
            "Map view of recent data"
          ].map((action, index) => (
            <button
              key={index}
              onClick={() => setInputValue(action)}
              className="px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors duration-200"
            >
              {action}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;
