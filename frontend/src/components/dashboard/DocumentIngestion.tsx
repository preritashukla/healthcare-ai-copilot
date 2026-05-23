import * as React from "react"
import { UploadCloud, CheckCircle2, FileText, Database, Cpu } from "lucide-react"
import { Button } from "../ui/button"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "../ui/card"

interface DocumentItem {
  name: string
  size: string
  vectors: number
  status: "ready" | "processing"
  dateIndexed: string
}

export function DocumentIngestion() {
  const [documents, setDocuments] = React.useState<DocumentItem[]>([
    {
      name: "handover_note_vance.pdf",
      size: "24.5 KB",
      vectors: 16,
      status: "ready",
      dateIndexed: "2026-05-23 09:40:24"
    },
    {
      name: "discharge_summary.pdf",
      size: "112.8 KB",
      vectors: 32,
      status: "ready",
      dateIndexed: "2026-05-23 09:40:28"
    }
  ])

  const [logs, setLogs] = React.useState<string[]>([
    "System initialized with database config: FAISS L2 Flat index",
    "Loaded SentenceTransformer model: all-MiniLM-L6-v2 (dimensions: 384)",
    "RAG Server listening on internal context port..."
  ])

  const [uploading, setUploading] = React.useState(false)

  const handleSimulatedUpload = () => {
    setUploading(true)
    const newDocName = "progress_notes_icu.pdf"

    // Simulate logs in order
    setTimeout(() => {
      setLogs(prev => [...prev, `[File Upload] Ingested "${newDocName}" (${18.4} KB)`])
    }, 500)

    setTimeout(() => {
      setLogs(prev => [...prev, "[PDF Extraction] Successfully parsed text (564 words)"])
    }, 1500)

    setTimeout(() => {
      setLogs(prev => [...prev, "[Embedding Pipeline] Computed 8 sentence vectors using SentenceTransformers"])
    }, 2800)

    setTimeout(() => {
      setLogs(prev => [...prev, "[FAISS Index] Merged 8 vectors into active IndexFlatL2"])
      setDocuments(prev => [
        {
          name: newDocName,
          size: "18.4 KB",
          vectors: 8,
          status: "ready",
          dateIndexed: new Date().toISOString().replace('T', ' ').slice(0, 19)
        },
        ...prev
      ])
      setUploading(false)
    }, 3800)
  }

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      {/* Top Banner */}
      <div className="px-6 py-4 border-b border-borderwhisper bg-white flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold tracking-tight text-ink">Document Ingestion Queue</h2>
          <p className="text-xs text-steel">Upload patient records, clinical charts, and handover PDFs to ingest them into the active RAG vector database.</p>
        </div>
        <div className="flex items-center gap-1.5 px-3 py-1 bg-slate-50 border border-borderwhisper rounded-full text-[10px] font-mono text-steel">
          <Database className="w-3.5 h-3.5 text-pine" /> INDEX VECTOR DENSITY: 384
        </div>
      </div>

      {/* Workspace */}
      <div className="flex-1 p-6 overflow-y-auto space-y-6">
        {/* Upload Zone */}
        <div className="border border-dashed border-borderwhisper rounded-lg p-10 bg-white flex flex-col items-center justify-center text-center gap-4 transition-all duration-200 hover:border-pine">
          <div className="w-12 h-12 bg-slate-50 rounded-full flex items-center justify-center border border-borderwhisper text-slate-400">
            <UploadCloud className="w-6 h-6 text-slate-500" />
          </div>
          <div>
            <p className="text-sm font-semibold text-ink">Ingest New Patient PDFs</p>
            <p className="text-xs text-steel mt-1 max-w-[320px]">Drag clinical notes, laboratory reports, or discharge summaries here. Files are automatically split and embedded.</p>
          </div>
          <Button onClick={handleSimulatedUpload} disabled={uploading} className="text-xs mt-2">
            {uploading ? "Ingesting & Indexing..." : "Upload & Parse PDF"}
          </Button>
        </div>

        {/* Asymmetric layout: Document list (2/3) and Pipeline logs (1/3) */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Document list */}
          <Card className="lg:col-span-2 border border-borderwhisper shadow-sm">
            <CardHeader className="py-4">
              <CardTitle>Currently Indexed Clinical Files</CardTitle>
              <CardDescription>Vector store database matches are pulled from these sources.</CardDescription>
            </CardHeader>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-slate-50 border-b border-borderwhisper text-[10px] font-mono text-steel uppercase">
                      <th className="py-3 px-5">File Name</th>
                      <th className="py-3 px-5">Size</th>
                      <th className="py-3 px-5">Vectors Created</th>
                      <th className="py-3 px-5">Indexed Date</th>
                      <th className="py-3 px-5 text-right">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-borderwhisper text-xs">
                    {documents.map((doc) => (
                      <tr key={doc.name} className="hover:bg-slate-50/50 transition-colors">
                        <td className="py-3 px-5 font-semibold text-ink flex items-center gap-2">
                          <FileText className="w-3.5 h-3.5 text-steel shrink-0" />
                          <span>{doc.name}</span>
                        </td>
                        <td className="py-3 px-5 text-slate-600 font-mono">{doc.size}</td>
                        <td className="py-3 px-5 text-slate-600 font-mono">{doc.vectors} segments</td>
                        <td className="py-3 px-5 text-steel font-mono">{doc.dateIndexed}</td>
                        <td className="py-3 px-5 text-right">
                          <span className="inline-flex items-center gap-1 text-[10px] font-mono text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded font-semibold uppercase">
                            <CheckCircle2 className="w-3 h-3 text-stable" /> Ready
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Pipeline logs console */}
          <Card className="border border-borderwhisper bg-[#0F172A] text-slate-300 font-mono text-[11px] shadow-sm flex flex-col h-[280px]">
            <CardHeader className="py-3.5 border-b border-slate-800 bg-[#0F172A] flex flex-row items-center justify-between">
              <CardTitle className="text-white text-xs font-mono tracking-tight uppercase flex items-center gap-2">
                <Cpu className="w-3.5 h-3.5 text-pine-light" /> Pipeline Console Logs
              </CardTitle>
              <div className="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-pulse"></div>
            </CardHeader>
            <CardContent className="p-4 flex-1 overflow-y-auto space-y-2 select-text">
              {logs.map((log, idx) => (
                <div key={idx} className="leading-relaxed border-l-2 border-slate-700 pl-2.5 py-0.5">
                  <span className="text-slate-500">[{new Date().toLocaleTimeString()}]</span> {log}
                </div>
              ))}
              {uploading && (
                <div className="text-pine-light animate-pulse">
                  &gt; Computing embeddings, please hold...
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
