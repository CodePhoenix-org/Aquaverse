import React, { useState, useRef, useEffect } from 'react';
import { PaperAirplaneIcon } from '@heroicons/react/24/outline';
import axios from 'axios';

const ChatInterface = ({ onDataReceived }) => {
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
      const response = await axios.post('http://127.0.0.1:8000/chat', {
        query: inputValue
      });
      // Debug: log full response
      console.log('Full response:', response);
      console.log('Response data:', response.data);
      console.log('Type of response data:', typeof response.data);

      // Extract the message properly
      let messageContent = '';
      let responseData = null;

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
    <div className="flex flex-col h-full">
      {/* Chat Header */}
      <div className="bg-gradient-to-r from-emerald-600 to-fuchsia-600 text-white p-3">
        <h2 className="text-base font-semibold">
          Ask me about ocean data, floats, and profiles
        </h2>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${
              message.type === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            <div
              className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${message.type === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 text-gray-800'
                }`}
            >
              {/* <p className="text-sm">{message.content}</p> */}
              <p className="text-sm">
                {typeof message.content === 'string'
                  ? message.content
                  : JSON.stringify(message.content)}
              </p>
              <p className="text-xs mt-1 opacity-70">
                {message.timestamp.toLocaleTimeString()}
              </p>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 px-4 py-2 rounded-lg">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div
                  className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                  style={{ animationDelay: '0.1s' }}
                ></div>
                <div
                  className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                  style={{ animationDelay: '0.2s' }}
                ></div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 p-4">
        <div className="flex space-x-2">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about ARGO data (e.g., 'Show temperature profiles in the Indian Ocean')"
            className="flex-1 border border-gray-300 rounded-lg px-3 py-2 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 placeholder:text-gray-400 bg-white"
            rows="2"
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isLoading}
            className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <PaperAirplaneIcon className="h-5 w-5" />
          </button>
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;
