import * as React from "react"
import { Search, Brain, FileText, CheckCircle2, AlertTriangle, ArrowRight } from "lucide-react"
import { Button } from "../ui/button"
import { Card, CardHeader, CardTitle, CardContent } from "../ui/card"
import { Input } from "../ui/input"

interface SourceSnippet {
  id: string
  file: string
  score: number
  text: string
}

interface QAHistory {
  query: string
  response: string
  sources: SourceSnippet[]
}

const PRESET_QUERIES = [
  "Does the patient have a penicillin allergy or medication issues?",
  "Analyze stability trends: heart rate and temperature history.",
  "Check clinical handover notes for safety signals."
]

const MOCK_ANSWERS: Record<string, { answer: string; sources: SourceSnippet[] }> = {
  "Does the patient have a penicillin allergy or medication issues?": {
    answer: `### Clinical Safety Signal Detected
**WARNING:** Patient has a documented history of severe penicillin allergy (anaphylaxis reaction noted in 2024 handover file). 

**Key Insights:**
1. **Allergy Alert:** Avoid all penicillins (e.g., Amoxicillin, Piperacillin/Tazobactam). Cross-reactivity with cephalosporins (like Ceftriaxone) is a risk; monitor closely if cephalosporins are clinically indicated.
2. **Current Orders:** Verify active medication sheets to ensure no penicillin-class medications are actively prescribed.
3. **Clinical Recommendation:** Substitute with alternative agents (e.g., Macrolides or Fluoroquinolones) depending on the targeted infectious process.

*Note: This is a clinical decision support suggestion. Confirm with active chart details before executing changes.*`,
    sources: [
      {
        id: "src-1",
        file: "handover_note_vance.pdf",
        score: 0.12,
        text: "Patient Marcus Vance has a documented history of anaphylaxis following Penicillin-V administration. Reaction includes severe hives and throat swelling requiring epinephrine injection (July 2024)."
      },
      {
        id: "src-2",
        file: "discharge_summary.pdf",
        score: 0.34,
        text: "Past Medical History: Penicillin allergy (severe). Patient was switched to Levofloxacin during admission for respiratory infection."
      }
    ]
  },
  "Analyze stability trends: heart rate and temperature history.": {
    answer: `### Vital Signs Trend Analysis
**STATUS: Borderline Unstable (Monitoring Required)**

**Observations over last 5 Days:**
1. **Tachycardia Trend:** Heart rate showed a steady escalation from **85 bpm (Day 1)** to **100 bpm (Day 4)** before stabilizing slightly at **92 bpm (Day 5)**. This indicates border-zone cardiovascular activation.
2. **Low-Grade Fever:** Temperature tracked alongside heart rate, reaching **99.5°F (Day 4)**. 
3. **Clinical Interpretation:** The dual rise in temp and heart rate is consistent with a mild inflammatory or infectious response. The Day 5 stabilization (92 bpm, 98.9°F) is reassuring but requires continued vitals charting.

*Safety Guideline: If heart rate exceeds 105 bpm or temperature rises above 100.4°F, notify the attending physician immediately.*`,
    sources: [
      {
        id: "src-3",
        file: "handover_note_vance.pdf",
        score: 0.22,
        text: "Vitals trend show mild elevation in cardiovascular parameters. Heart rate touched 100bpm during peak temperature reading on Day 4."
      },
      {
        id: "src-4",
        file: "discharge_summary.pdf",
        score: 0.41,
        text: "Discharged with instructions to monitor twice-daily temperatures. Vitals stable at time of leaving."
      }
    ]
  },
  "Check clinical handover notes for safety signals.": {
    answer: `### Handover Safety Signals
**STATUS: High Alert Warnings Identified**

1. **Allergy Profile:** Documented anaphylactic allergy to Penicillin.
2. **Ambulation Status:** Patient Marcus Vance has a moderate fall risk. Assist of 1 required for all transfers.
3. **Cardiovascular:** Borderline tachycardic spikes noted during evening handovers.

*Recommendation: Update the electronic whiteboard in the patient room to reflect the penicillin allergy and fallback precautions.*`,
    sources: [
      {
        id: "src-1",
        file: "handover_note_vance.pdf",
        score: 0.15,
        text: "Fall risk status: Moderate. Needs assistance when getting out of bed due to mild orthostatic hypotension."
      }
    ]
  }
}

export function RagSearch() {
  const [query, setQuery] = React.useState("")
  const [loading, setLoading] = React.useState(false)
  const [history, setHistory] = React.useState<QAHistory[]>([])

  const handleSearch = (searchQuery: string) => {
    if (!searchQuery.trim()) return
    setQuery(searchQuery)
    setLoading(true)

    setTimeout(() => {
      const match = MOCK_ANSWERS[searchQuery] || {
        answer: `### Clinical Inquiry Answered
No direct matches found in local patient files for: *"${searchQuery}"*. 

**RAG Retrieval Context:**
- The embedding vector database (FAISS) was queried using model: \`all-MiniLM-L6-v2\`.
- Groq LLM (\`llama3-8b-8192\`) suggests reviewing the ingestion queue to make sure relevant patient history PDFs are properly loaded.`,
        sources: []
      }

      setHistory((prev) => [
        {
          query: searchQuery,
          response: match.answer,
          sources: match.sources
        },
        ...prev
      ])
      setLoading(false)
    }, 1200)
  }

  const latestResult = history[0]

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      {/* Top Banner */}
      <div className="px-6 py-4 border-b border-borderwhisper bg-white flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold tracking-tight text-ink">Clinical RAG Search</h2>
          <p className="text-xs text-steel">Query patient records using semantic search powered by SentenceTransformers and Groq LLM.</p>
        </div>
        <div className="flex items-center gap-1.5 px-3 py-1 bg-slate-50 border border-borderwhisper rounded-full text-[10px] font-mono text-steel">
          <Brain className="w-3.5 h-3.5 text-pine" /> RAG MODEL: GROQ-LLAMA3
        </div>
      </div>

      {/* Main Workspace: Asymmetric Layout */}
      <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
        {/* Left Side: Search Input and AI Output (2/3 width) */}
        <div className="flex-1 p-6 flex flex-col gap-5 overflow-y-auto border-r border-borderwhisper">
          {/* Query Box */}
          <div className="bg-white border border-borderwhisper rounded-lg p-4 flex flex-col gap-3.5 shadow-sm">
            <div className="flex gap-2.5">
              <Input
                placeholder="Ask a clinical question about Marcus Vance's history..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSearch(query)
                }}
                className="h-10"
              />
              <Button onClick={() => handleSearch(query)} disabled={loading} className="gap-2 shrink-0">
                <Search className="w-4 h-4" /> Search
              </Button>
            </div>

            {/* Presets */}
            <div>
              <p className="text-[10px] font-semibold text-slate-500 font-mono tracking-wider mb-2">SUGGESTED CLINICAL INQUIRIES</p>
              <div className="flex flex-col gap-1.5">
                {PRESET_QUERIES.map((preset, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSearch(preset)}
                    className="text-left text-xs text-pine hover:text-pine-light flex items-center gap-1.5 outline-none font-medium transition-colors"
                  >
                    <ArrowRight className="w-3 h-3 text-pine/70 shrink-0" />
                    <span className="truncate">{preset}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Loader or AI Response */}
          {loading ? (
            <Card className="animate-pulse border border-borderwhisper">
              <CardHeader className="bg-slate-50/50">
                <div className="h-4 bg-slate-200 rounded w-1/4"></div>
                <div className="h-3 bg-slate-200 rounded w-1/3 mt-2"></div>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="h-3 bg-slate-200 rounded w-full"></div>
                <div className="h-3 bg-slate-200 rounded w-5/6"></div>
                <div className="h-3 bg-slate-200 rounded w-4/5"></div>
              </CardContent>
            </Card>
          ) : latestResult ? (
            <div className="flex flex-col gap-4">
              <Card className="border border-borderwhisper shadow-sm">
                <CardHeader className="bg-slate-50/50 flex flex-row items-center gap-2.5 py-4">
                  <div className="w-8 h-8 rounded bg-teal-50 border border-teal-200 flex items-center justify-center text-pine">
                    <Brain className="w-4 h-4" />
                  </div>
                  <div>
                    <CardTitle className="text-xs font-mono uppercase tracking-wider text-steel">Groq LLM Response</CardTitle>
                    <p className="text-xs font-semibold text-ink">Inquiry: {latestResult.query}</p>
                  </div>
                </CardHeader>
                <CardContent className="pt-5 leading-relaxed text-slate-800 text-sm">
                  {/* Inline styling parser for markdown headers/bold */}
                  <div className="space-y-4">
                    {latestResult.response.split("\n\n").map((para, pIdx) => {
                      if (para.startsWith("###")) {
                        return (
                          <h4 key={pIdx} className="text-sm font-bold text-ink border-b border-borderwhisper pb-1 mt-2">
                            {para.replace("###", "").trim()}
                          </h4>
                        )
                      }
                      if (para.startsWith("**WARNING:**") || para.startsWith("**STATUS:")) {
                        const isWarning = para.includes("WARNING");
                        return (
                          <div
                            key={pIdx}
                            className={`p-3 rounded-md flex items-start gap-2 text-xs font-medium border ${
                              isWarning
                                ? "bg-red-50 border-red-200 text-critical"
                                : "bg-amber-50 border-amber-200 text-amber-950"
                            }`}
                          >
                            {isWarning ? <AlertTriangle className="w-4 h-4 shrink-0" /> : <Brain className="w-4 h-4 shrink-0" />}
                            <span>{para.replace("**WARNING:**", "").replace("**STATUS:", "STATUS:").trim()}</span>
                          </div>
                        )
                      }
                      return (
                        <p key={pIdx} className="text-xs leading-relaxed text-slate-700 whitespace-pre-line">
                          {para}
                        </p>
                      )
                    })}
                  </div>
                </CardContent>
              </Card>

              {/* History list if multiple searches */}
              {history.length > 1 && (
                <div>
                  <h3 className="text-[10px] font-semibold text-slate-500 font-mono tracking-wider mb-2.5">INQUIRY HISTORY</h3>
                  <div className="flex flex-col gap-2">
                    {history.slice(1).map((item, histIdx) => (
                      <div
                        key={histIdx}
                        onClick={() => {
                          setQuery(item.query);
                          setHistory([item, ...history.filter((_, i) => i !== histIdx + 1)]);
                        }}
                        className="p-3 bg-white border border-borderwhisper rounded-lg hover:border-slate-400 cursor-pointer flex items-center justify-between text-xs"
                      >
                        <span className="font-medium text-slate-700 truncate">{item.query}</span>
                        <span className="text-[10px] text-steel font-mono">1 inquiry ago</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-12 text-slate-400 border border-dashed border-borderwhisper rounded-lg bg-slate-50/50">
              <Brain className="w-10 h-10 text-slate-300 mb-3" />
              <p className="text-xs font-semibold text-slate-600">No active clinical query</p>
              <p className="text-[10px] text-steel mt-1 max-w-[280px]">Select a preset inquiry above or enter a custom query to perform retrieval reasoning.</p>
            </div>
          )}
        </div>

        {/* Right Side: Retrieved RAG Sources (1/3 width) */}
        <div className="w-full md:w-80 bg-slate-50/50 p-6 overflow-y-auto flex flex-col gap-4">
          <div className="flex items-center justify-between border-b border-borderwhisper pb-2">
            <h3 className="text-xs font-bold text-ink flex items-center gap-1.5">
              <FileText className="w-3.5 h-3.5 text-steel" /> Retrieved Context
            </h3>
            <span className="text-[10px] font-mono text-steel">FAISS DB</span>
          </div>

          {latestResult && latestResult.sources.length > 0 ? (
            <div className="flex flex-col gap-3">
              {latestResult.sources.map((src) => (
                <div key={src.id} className="p-3 bg-white border border-borderwhisper rounded-lg flex flex-col gap-2 shadow-sm">
                  <div className="flex items-center justify-between text-[10px]">
                    <span className="font-semibold text-slate-700 truncate max-w-[120px]">{src.file}</span>
                    <span className="text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded font-mono text-[9px] font-bold">
                      Dist: {src.score}
                    </span>
                  </div>
                  <p className="text-[11px] leading-relaxed text-slate-600 italic bg-slate-50 p-2 rounded border border-slate-100">
                    "{src.text}"
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-4 text-slate-400">
              <CheckCircle2 className="w-8 h-8 text-slate-300 mb-2" />
              <p className="text-[10px] font-semibold text-slate-600">No retrieved sources</p>
              <p className="text-[9px] text-steel mt-1">Submit a search query to load embedding match scores.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
