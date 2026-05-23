
import { Search, Activity, ShieldAlert, FileText, Database, User } from "lucide-react"
import { cn } from "../../lib/utils"

export type TabType = "rag" | "trends" | "meds" | "ingest"

interface SidebarProps {
  activeTab: TabType
  setActiveTab: (tab: TabType) => void
  selectedPatient: string
}

export function Sidebar({ activeTab, setActiveTab, selectedPatient }: SidebarProps) {
  const menuItems = [
    { id: "rag", label: "Clinical RAG Search", icon: Search },
    { id: "trends", label: "Patient Trends", icon: Activity },
    { id: "meds", label: "Medication Guard", icon: ShieldAlert },
    { id: "ingest", label: "Document Ingestion", icon: FileText },
  ] as const

  return (
    <div className="w-64 bg-white border-r border-borderwhisper flex flex-col h-screen select-none shrink-0">
      {/* App Logo/Header */}
      <div className="h-16 border-b border-borderwhisper flex items-center px-5 gap-2.5">
        <div className="bg-pine w-8 h-8 rounded-md flex items-center justify-center text-white font-mono font-bold text-base shadow-sm">
          H
        </div>
        <div>
          <h1 className="text-sm font-semibold tracking-tight text-ink">Hospital Copilot AI</h1>
          <span className="text-[10px] text-steel font-mono uppercase tracking-wider flex items-center gap-1">
            <Database className="w-3 h-3 text-pine" /> RAG Engine Active
          </span>
        </div>
      </div>

      {/* Selected Patient Mini Profile */}
      <div className="p-4 mx-3 my-4 bg-slate-50 rounded-lg border border-borderwhisper/60">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-full bg-teal-100 flex items-center justify-center text-pine">
            <User className="w-4 h-4" />
          </div>
          <div>
            <p className="text-[10px] text-steel font-medium font-mono">ACTIVE PATIENT</p>
            <p className="text-xs font-semibold text-ink">{selectedPatient}</p>
          </div>
        </div>
      </div>

      {/* Navigation Menu */}
      <nav className="flex-1 px-3 space-y-1">
        {menuItems.map((item) => {
          const Icon = item.icon
          const isActive = activeTab === item.id
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-xs font-medium transition-all duration-150 outline-none",
                isActive
                  ? "bg-pine text-white shadow-sm"
                  : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
              )}
            >
              <Icon className="w-4 h-4 shrink-0" />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Footer Info */}
      <div className="p-4 border-t border-borderwhisper text-[10px] text-steel font-mono">
        <p>GROQ MODEL: llama3-8b</p>
        <p className="mt-0.5">VECTORS: 32 Embeddings</p>
      </div>
    </div>
  )
}
