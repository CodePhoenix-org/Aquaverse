import { useEffect, useRef, useState } from "react";
import { Bot, Loader2, Send, Sparkles, User } from "lucide-react";
import axios from "axios";
import AppLogo from "../ui/AppLogo";

const quickActions = [
  "Show temperature profiles in the Indian Ocean",
  "Find floats in the Pacific",
  "Compare salinity data",
  "Open a map of recent float activity",
];

export default function ChatInterface({ onDataReceived, onCloseChat }) {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: "bot",
      content:
        'Welcome to FloatChat. Ask for ocean profiles, map views, comparisons, or anomaly-focused visuals, and I will route the result into the dashboard.',
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [lastVizData, setLastVizData] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: "user",
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages((current) => [...current, userMessage]);
    setInputValue("");
    setIsLoading(true);

    try {
      const response = await axios.post("http://127.0.0.1:8000/chat/query", {
        query: inputValue,
      });

      let messageContent = "";
      let responseData = null;

      if (Array.isArray(response.data)) {
        messageContent = response.data[0] || "No response received.";
        responseData = response.data[1] || null;
      } else if (typeof response.data === "object") {
        messageContent =
          response.data.message || response.data.answer || "No response received.";
        responseData = response.data.data || null;
      } else {
        messageContent = String(response.data);
      }

      setLastVizData(responseData);
      setMessages((current) => [
        ...current,
        {
          id: Date.now() + 1,
          type: "bot",
          content: messageContent,
          data: responseData,
          timestamp: new Date(),
        },
      ]);

      if (responseData) {
        onDataReceived(responseData);
      }
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: Date.now() + 1,
          type: "bot",
          content: "I hit an error while processing that request. Please try again.",
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex h-full flex-col bg-transparent">
      <div className="border-b border-white/10 px-5 py-4">
        <div className="flex items-center gap-4">
          <AppLogo size="md" alt="AquaVerse chat logo" />
          <div>
            <p className="premium-kicker">FloatChat</p>
            <h3 className="mt-2 font-display text-2xl font-semibold text-white">
              Conversational ocean analysis
            </h3>
            <p className="mt-2 text-sm leading-6 text-slate-300">
              Ask a question and route the answer into maps, plots, or comparison views.
            </p>
          </div>
        </div>
      </div>

      <div className="flex-1 space-y-5 overflow-y-auto px-5 py-5 scrollbar-thin">
        {messages.map((message, index) => {
          const isLastBotMessage =
            message.type === "bot" && index === messages.length - 1 && lastVizData;

          return (
            <div
              key={message.id}
              className={`flex ${message.type === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`flex max-w-[90%] items-start gap-3 ${
                  message.type === "user" ? "flex-row-reverse" : ""
                }`}
              >
                <div
                  className={`flex h-10 w-10 items-center justify-center rounded-2xl ${
                    message.type === "user"
                      ? "bg-gradient-to-br from-cyan-300 to-sky-400 text-slate-950"
                      : "border border-white/10 bg-white/[0.08] text-cyan-100"
                  }`}
                >
                  {message.type === "user" ? (
                    <User className="h-4 w-4" />
                  ) : (
                    <Bot className="h-4 w-4" />
                  )}
                </div>

                <div
                  className={`rounded-[24px] px-4 py-4 shadow-ocean ${
                    message.type === "user"
                      ? "bg-gradient-to-br from-cyan-300 to-sky-400 text-slate-950"
                      : "border border-white/10 bg-white/[0.05] text-slate-100"
                  }`}
                >
                  <p className="whitespace-pre-wrap text-sm leading-7">{message.content}</p>

                  {isLastBotMessage ? (
                    <button
                      onClick={() => {
                        onDataReceived(lastVizData);
                        onCloseChat?.();
                      }}
                      className="mt-4 premium-button-secondary px-4 py-2 text-xs"
                    >
                      <Sparkles className="h-3.5 w-3.5" />
                      Open in Dashboard
                    </button>
                  ) : null}

                  <p
                    className={`mt-3 text-xs ${
                      message.type === "user" ? "text-slate-900/70" : "text-slate-400"
                    }`}
                  >
                    {message.timestamp.toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </p>
                </div>
              </div>
            </div>
          );
        })}

        {isLoading ? (
          <div className="flex justify-start">
            <div className="flex items-start gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.08] text-cyan-100">
                <Bot className="h-4 w-4" />
              </div>
              <div className="rounded-[24px] border border-white/10 bg-white/[0.05] px-4 py-4 text-slate-200">
                <div className="flex items-center gap-2 text-sm">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Analyzing your query...
                </div>
              </div>
            </div>
          </div>
        ) : null}

        <div ref={messagesEndRef} />
      </div>

      <div className="border-t border-white/10 px-5 py-5">
        <div className="mb-3 flex flex-wrap gap-2">
          {quickActions.map((action) => (
            <button
              key={action}
              onClick={() => setInputValue(action)}
              className="premium-chip rounded-full px-3 py-2 text-left"
            >
              {action}
            </button>
          ))}
        </div>

        <div className="flex gap-3">
          <textarea
            value={inputValue}
            onChange={(event) => setInputValue(event.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Ask FloatChat for a map, profile, comparison, or anomaly insight..."
            className="premium-input min-h-[5.5rem] flex-1 resize-none"
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isLoading}
            className="premium-button h-fit disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </button>
        </div>
      </div>
    </div>
  );
}
