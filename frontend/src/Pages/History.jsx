import { useMemo, useState } from "react";
import {
  Archive,
  Bot,
  Clock,
  Copy,
  Eye,
  Map,
  MessageCircle,
  Search,
  Share,
  Star,
  Thermometer,
  Waves,
  Droplets,
  BarChart3,
  Database,
} from "lucide-react";
import Navbar from "../components/Navbar";
import PageShell from "../components/ui/PageShell";

const sampleHistory = [
  {
    id: 1,
    title: "Ocean Temperature Analysis",
    query: "Show me temperature profiles in the Indian Ocean for the last month",
    response:
      "I found temperature data from 15 ARGO floats in the Indian Ocean region. The average surface temperature ranges from 28.5 to 30.2 C with consistent warm-pool behavior.",
    timestamp: "2024-01-15 14:30",
    date: "Today",
    type: "temperature",
    hasVisualization: true,
    starred: false,
    tags: ["temperature", "indian ocean", "argo floats"],
  },
  {
    id: 2,
    title: "Salinity Data Comparison",
    query: "Compare salinity levels in the Arabian Sea vs Bay of Bengal",
    response:
      "Recent ARGO float data shows the Arabian Sea with higher salinity signatures than the Bay of Bengal, especially in near-surface layers.",
    timestamp: "2024-01-14 09:15",
    date: "Yesterday",
    type: "salinity",
    hasVisualization: true,
    starred: true,
    tags: ["salinity", "arabian sea", "bay of bengal", "comparison"],
  },
  {
    id: 3,
    title: "ARGO Float Trajectories",
    query: "Display the paths of ARGO floats in the Pacific Ocean",
    response:
      "Eight active floats in the Pacific reveal clear circulation traces and several route segments worth following in map view.",
    timestamp: "2024-01-13 16:45",
    date: "2 days ago",
    type: "trajectory",
    hasVisualization: true,
    starred: false,
    tags: ["trajectory", "pacific", "circulation"],
  },
  {
    id: 4,
    title: "BGC Parameter Analysis",
    query: "Analyze bio-geochemical parameters in the Southern Ocean",
    response:
      "The Southern Ocean BGC data reveals seasonal chlorophyll and nutrient shifts with strong biological variability.",
    timestamp: "2024-01-12 11:20",
    date: "3 days ago",
    type: "bgc",
    hasVisualization: true,
    starred: false,
    tags: ["bgc", "southern ocean", "chlorophyll", "nutrients"],
  },
  {
    id: 5,
    title: "Deep Water Mass Analysis",
    query: "What are the characteristics of deep water masses in the Atlantic?",
    response:
      "Deep Atlantic water masses show distinct temperature and salinity signatures, with NADW presenting a recognizable profile structure.",
    timestamp: "2024-01-11 13:10",
    date: "4 days ago",
    type: "analysis",
    hasVisualization: false,
    starred: true,
    tags: ["deep water", "atlantic", "nadw"],
  },
];

const typeIconMap = {
  temperature: Thermometer,
  salinity: Droplets,
  trajectory: Map,
  bgc: BarChart3,
  analysis: Database,
};

export default function History() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedFilter, setSelectedFilter] = useState("all");
  const [selectedChat, setSelectedChat] = useState(sampleHistory[0]);

  const filteredChats = useMemo(
    () =>
      sampleHistory.filter((chat) => {
        const matchesSearch =
          chat.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
          chat.query.toLowerCase().includes(searchQuery.toLowerCase()) ||
          chat.tags.some((tag) => tag.toLowerCase().includes(searchQuery.toLowerCase()));

        const matchesFilter =
          selectedFilter === "all" ||
          (selectedFilter === "starred" && chat.starred) ||
          (selectedFilter === "visualization" && chat.hasVisualization) ||
          chat.type === selectedFilter;

        return matchesSearch && matchesFilter;
      }),
    [searchQuery, selectedFilter]
  );

  return (
    <PageShell>
      <Navbar />

      <main className="mx-auto max-w-7xl px-4 pb-16 pt-6 sm:px-6 lg:px-8">
        <section className="premium-panel premium-panel-strong p-6 sm:p-8">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <span className="premium-badge">
                <Waves className="h-3.5 w-3.5" />
                Conversation Archive
              </span>
              <h1 className="mt-5 font-display text-4xl font-bold tracking-[-0.05em] text-white sm:text-5xl">
                FloatChat history, elevated.
              </h1>
              <p className="mt-4 max-w-3xl text-base leading-8 text-slate-300">
                Review past ocean investigations with a cleaner archive view built for
                scanning, filtering, and diving back into the right conversation fast.
              </p>
            </div>

            <div className="grid gap-4 sm:grid-cols-3">
              {[
                { label: "Conversations", value: sampleHistory.length },
                { label: "Starred", value: sampleHistory.filter((chat) => chat.starred).length },
                { label: "With visuals", value: sampleHistory.filter((chat) => chat.hasVisualization).length },
              ].map((metric) => (
                <div key={metric.label} className="premium-card p-4">
                  <p className="text-sm text-slate-300">{metric.label}</p>
                  <p className="mt-2 font-display text-2xl font-semibold text-white">
                    {metric.value}
                  </p>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-8 grid gap-8 lg:grid-cols-[0.95fr_1.05fr]">
            <div className="space-y-5">
              <div className="premium-card p-4">
                <div className="relative">
                  <Search className="pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-400" />
                  <input
                    type="text"
                    placeholder="Search titles, queries, or tags..."
                    value={searchQuery}
                    onChange={(event) => setSearchQuery(event.target.value)}
                    className="premium-input pl-12"
                  />
                </div>

                <div className="mt-4 flex flex-wrap gap-2">
                  {[
                    { id: "all", label: "All" },
                    { id: "starred", label: "Starred" },
                    { id: "visualization", label: "With visuals" },
                    { id: "temperature", label: "Temperature" },
                    { id: "salinity", label: "Salinity" },
                    { id: "trajectory", label: "Trajectory" },
                    { id: "bgc", label: "BGC" },
                  ].map((filter) => (
                    <button
                      key={filter.id}
                      onClick={() => setSelectedFilter(filter.id)}
                      className={`rounded-full px-4 py-2 text-sm font-semibold transition-all ${
                        selectedFilter === filter.id
                          ? "bg-gradient-to-r from-cyan-300 to-sky-400 text-slate-950 shadow-lg"
                          : "border border-white/10 bg-white/[0.04] text-slate-200 hover:bg-white/[0.08]"
                      }`}
                    >
                      {filter.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-4">
                {filteredChats.map((chat) => {
                  const Icon = typeIconMap[chat.type] || Database;
                  const active = selectedChat?.id === chat.id;

                  return (
                    <button
                      key={chat.id}
                      onClick={() => setSelectedChat(chat)}
                      className={`premium-card w-full p-5 text-left ${
                        active ? "border-cyan-300/20 bg-white/[0.08]" : ""
                      }`}
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex items-start gap-4">
                          <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.06]">
                            <Icon className="h-5 w-5 text-cyan-100" />
                          </div>
                          <div>
                            <h3 className="font-display text-xl font-semibold text-white">
                              {chat.title}
                            </h3>
                            <div className="mt-2 flex items-center gap-2 text-sm text-slate-400">
                              <Clock className="h-4 w-4" />
                              {chat.date} at {chat.timestamp.split(" ")[1]}
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center gap-2">
                          {chat.starred ? (
                            <Star className="h-4 w-4 fill-amber-300 text-amber-300" />
                          ) : null}
                          {chat.hasVisualization ? (
                            <span className="premium-chip">
                              <Eye className="h-3.5 w-3.5" />
                              Visual
                            </span>
                          ) : null}
                        </div>
                      </div>

                      <div className="mt-4 space-y-3 text-sm leading-7 text-slate-300">
                        <p>
                          <span className="font-semibold text-slate-100">Q:</span> {chat.query}
                        </p>
                        <p className="line-clamp-2">
                          <span className="font-semibold text-slate-100">A:</span> {chat.response}
                        </p>
                      </div>

                      <div className="mt-4 flex flex-wrap gap-2">
                        {chat.tags.map((tag) => (
                          <span key={tag} className="premium-chip rounded-full px-3 py-1.5">
                            #{tag}
                          </span>
                        ))}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="premium-card h-fit p-6 lg:sticky lg:top-28">
              {selectedChat ? (
                <div className="space-y-6">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="premium-kicker">Selected Conversation</p>
                      <h2 className="mt-2 font-display text-3xl font-semibold text-white">
                        {selectedChat.title}
                      </h2>
                      <p className="mt-2 text-sm text-slate-400">
                        {selectedChat.date} at {selectedChat.timestamp.split(" ")[1]}
                      </p>
                    </div>
                    {selectedChat.starred ? (
                      <Star className="h-5 w-5 fill-amber-300 text-amber-300" />
                    ) : null}
                  </div>

                  <div className="premium-divider" />

                  <div className="space-y-4">
                    <div className="rounded-[24px] border border-white/10 bg-white/[0.04] p-4">
                      <div className="mb-2 flex items-center gap-2 text-sm font-medium text-slate-200">
                        <MessageCircle className="h-4 w-4 text-cyan-100" />
                        Your query
                      </div>
                      <p className="text-sm leading-7 text-slate-300">{selectedChat.query}</p>
                    </div>

                    <div className="rounded-[24px] border border-white/10 bg-white/[0.04] p-4">
                      <div className="mb-2 flex items-center gap-2 text-sm font-medium text-slate-200">
                        <Bot className="h-4 w-4 text-cyan-100" />
                        FloatChat response
                      </div>
                      <p className="text-sm leading-7 text-slate-300">{selectedChat.response}</p>
                    </div>
                  </div>

                  <div>
                    <p className="text-sm font-medium text-slate-200">Tags</p>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {selectedChat.tags.map((tag) => (
                        <span key={tag} className="premium-chip rounded-full px-3 py-1.5">
                          #{tag}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="grid gap-3 sm:grid-cols-3">
                    <button className="premium-button-secondary">
                      <Share className="h-4 w-4" />
                      Share
                    </button>
                    <button className="premium-button-secondary">
                      <Copy className="h-4 w-4" />
                      Copy
                    </button>
                    <button className="premium-button-secondary">
                      <Archive className="h-4 w-4" />
                      Archive
                    </button>
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        </section>
      </main>
    </PageShell>
  );
}
