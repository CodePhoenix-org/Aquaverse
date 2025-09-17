import { useState } from "react";

const Chatbot = () => {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hi! I’m your ARGO assistant🌊. Ask me anything." },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const newMessages = [...messages, { sender: "user", text: input }];
    setMessages(newMessages);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: input }),
      });

      const data = await res.json();

      setMessages([
        ...newMessages,
        { sender: "bot", text: data.text || data.reply || "⚠️ No response" },
      ]);
    } catch (err) {
      console.error("Error:", err);
      setMessages([
        ...newMessages,
        { sender: "bot", text: "⚠️ Server error." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[600px] max-w-2xl mx-auto bg-white/5 backdrop-blur-sm rounded-2xl shadow-2xl overflow-hidden border border-white/10">
      {/* Chat messages */}
      <div className="flex-1 p-6 space-y-4 overflow-y-auto scrollbar-thin scrollbar-thumb-blue-400 scrollbar-track-transparent">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${
              msg.sender === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`px-4 py-3 rounded-2xl max-w-md break-words shadow-lg ${
                msg.sender === "user"
                  ? "bg-gradient-to-r from-cyan-500 to-blue-600 text-white"
                  : "bg-white/10 text-gray-100 border border-white/20"
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="px-4 py-3 rounded-2xl bg-white/10 text-cyan-300 animate-pulse border border-white/20">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                <span className="ml-2">ARGO is thinking...</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input box */}
      <div className="flex p-4 bg-white/5 border-t border-white/10">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about ARGO data, oceanography, or marine science..."
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          className="flex-1 px-4 py-3 rounded-l-xl bg-white/10 text-white placeholder-blue-200 border border-white/20 focus:ring-2 focus:ring-cyan-400 focus:outline-none transition-all"
        />
        <button
          onClick={sendMessage}
          disabled={!input.trim()}
          className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed rounded-r-xl text-white font-semibold transition-all transform hover:scale-105 disabled:transform-none"
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default Chatbot;
