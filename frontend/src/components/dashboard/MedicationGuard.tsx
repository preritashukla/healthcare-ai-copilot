import * as React from "react"
import { ShieldAlert, AlertTriangle, CheckCircle2, FileText, Check, Ban } from "lucide-react"
import { Button } from "../ui/button"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "../ui/card"

interface Medication {
  name: string
  class: string
  dose: string
  route: string
  status: "critical" | "warning" | "stable"
  reason: string
}

const MOCK_MEDS: Medication[] = [
  {
    name: "Amoxicillin-Clavulanate",
    class: "Penicillin Beta-lactam",
    dose: "875/125 mg",
    route: "Oral",
    status: "critical",
    reason: "CRITICAL: Patient has a severe documented anaphylaxis allergy to penicillins."
  },
  {
    name: "Ceftriaxone",
    class: "Cephalosporin (3rd Gen)",
    dose: "1 g",
    route: "Intravenous",
    status: "warning",
    reason: "MODERATE: ~3-5% cross-reactivity risk with documented penicillin allergy. Monitor first dose closely."
  },
  {
    name: "Levofloxacin",
    class: "Fluoroquinolone",
    dose: "500 mg",
    route: "Oral",
    status: "stable",
    reason: "SAFE: Approved therapeutic substitution. No structural class-cross reactivity with penicillins."
  }
]

export function MedicationGuard() {
  const [meds, setMeds] = React.useState<Medication[]>(MOCK_MEDS)
  const [notified, setNotified] = React.useState(false)

  const handleDiscontinue = (medName: string) => {
    setMeds(meds.filter(m => m.name !== medName))
  }

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      {/* Top Banner */}
      <div className="px-6 py-4 border-b border-borderwhisper bg-white flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold tracking-tight text-ink">Medication Guard</h2>
          <p className="text-xs text-steel">Drug-allergy cross-checking engine scanning active prescriptions against RAG patient history logs.</p>
        </div>
        <div className="flex items-center gap-1.5 px-3 py-1 bg-red-50 border border-red-200 rounded-full text-[10px] font-mono text-critical">
          <ShieldAlert className="w-3.5 h-3.5 text-critical animate-pulse" /> MED SAFETY STATUS: ALERT ACTIVE
        </div>
      </div>

      <div className="flex-1 p-6 overflow-y-auto space-y-6">
        {/* Documented Allergy Registry */}
        <Card className="border border-borderwhisper shadow-sm">
          <CardHeader className="py-4 bg-slate-50/50">
            <CardTitle className="text-xs font-mono uppercase text-steel">Documented Patient Allergy Registry</CardTitle>
          </CardHeader>
          <CardContent className="pt-4">
            <div className="flex flex-wrap gap-3">
              <div className="px-3 py-2 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-critical shrink-0" />
                <div className="text-xs">
                  <span className="font-bold text-critical">Penicillins</span>
                  <span className="text-slate-600 ml-1.5">(Severe Anaphylaxis reaction documented)</span>
                </div>
              </div>
              <div className="px-3 py-2 bg-slate-50 border border-borderwhisper rounded-lg flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-stable shrink-0" />
                <div className="text-xs">
                  <span className="font-bold text-slate-700">Sulfa Drugs</span>
                  <span className="text-slate-500 ml-1.5">(No active history)</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Current Active Prescriptions Table */}
        <Card className="border border-borderwhisper shadow-sm">
          <CardHeader className="py-4 flex flex-row items-center justify-between">
            <div>
              <CardTitle>Active Orders Inspector</CardTitle>
              <CardDescription>Real-time clinical checking of patient prescriptions.</CardDescription>
            </div>
            <span className="text-[10px] text-steel font-mono">3 active drugs scanned</span>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-slate-50 border-b border-borderwhisper text-[10px] font-mono text-steel uppercase">
                    <th className="py-3 px-5">Medication Name</th>
                    <th className="py-3 px-5">Class</th>
                    <th className="py-3 px-5">Dosage / Route</th>
                    <th className="py-3 px-5">Safety Status</th>
                    <th className="py-3 px-5 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-borderwhisper text-xs">
                  {meds.map((med) => (
                    <tr key={med.name} className="hover:bg-slate-50/50 transition-colors">
                      <td className="py-3.5 px-5 font-semibold text-ink">{med.name}</td>
                      <td className="py-3.5 px-5 text-slate-600">{med.class}</td>
                      <td className="py-3.5 px-5 font-mono text-steel">{med.dose} ({med.route})</td>
                      <td className="py-3.5 px-5">
                        <div className="flex flex-col gap-1 max-w-[280px]">
                          <span
                            className={`inline-flex items-center gap-1 font-semibold text-[10px] uppercase font-mono px-2 py-0.5 rounded w-max ${
                              med.status === "critical"
                                ? "bg-red-50 text-critical"
                                : med.status === "warning"
                                ? "bg-amber-50 text-amber-700"
                                : "bg-emerald-50 text-stable"
                            }`}
                          >
                            {med.status === "critical" ? (
                              <Ban className="w-3 h-3" />
                            ) : med.status === "warning" ? (
                              <AlertTriangle className="w-3 h-3" />
                            ) : (
                              <Check className="w-3 h-3" />
                            )}
                            {med.status}
                          </span>
                          <span className="text-[10px] text-slate-500 leading-normal">{med.reason}</span>
                        </div>
                      </td>
                      <td className="py-3.5 px-5 text-right">
                        {med.status === "critical" && (
                          <Button
                            variant="critical"
                            size="sm"
                            onClick={() => handleDiscontinue(med.name)}
                            className="text-xs"
                          >
                            Discontinue
                          </Button>
                        )}
                        {med.status === "warning" && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              setMeds(meds.map(m => m.name === med.name ? { ...m, status: "stable", reason: "Override Approved: attending physician confirmed cephalosporin tolerability." } : m))
                            }}
                          >
                            Override
                          </Button>
                        )}
                        {med.status === "stable" && (
                          <span className="text-slate-400 font-mono text-[10px]">No Actions Needed</span>
                        )}
                      </td>
                    </tr>
                  ))}
                  {meds.length === 0 && (
                    <tr>
                      <td colSpan={5} className="py-8 text-center text-slate-400 italic">
                        All active critical orders discontinued.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {/* Source Text Verification from FAISS */}
        <div className="bg-slate-50 border border-borderwhisper rounded-lg p-5 flex flex-col gap-3">
          <div className="flex items-center gap-2 border-b border-borderwhisper pb-2">
            <FileText className="w-4 h-4 text-steel" />
            <h3 className="text-xs font-bold text-ink">RAG Cross-Reference Proof (PDF Vector Matching)</h3>
          </div>
          <div className="space-y-3 font-mono text-[11px] text-slate-600 leading-relaxed">
            <div className="bg-white p-3 rounded border border-borderwhisper/60">
              <p className="font-semibold text-slate-800 text-[10px] mb-1">FILE: handover_note_vance.pdf (Distance: 0.12)</p>
              <p className="italic">"...patient Marcus Vance has a documented history of anaphylaxis following Penicillin-V administration. Reaction includes severe hives..."</p>
            </div>
            <div className="bg-white p-3 rounded border border-borderwhisper/60">
              <p className="font-semibold text-slate-800 text-[10px] mb-1">FILE: discharge_summary.pdf (Distance: 0.34)</p>
              <p className="italic">"...history of severe drug allergy: penicillins. Attending physician swapped order for respiratory tract coverage..."</p>
            </div>
          </div>
        </div>

        {/* Quick Notify Attending Physician */}
        <div className="flex justify-between items-center bg-white border border-borderwhisper rounded-lg p-4 shadow-sm">
          <div className="text-xs text-slate-700">
            **Action Required:** Notify Attending Physician (Dr. Sarah Chen) to substitute Amoxicillin.
          </div>
          <Button
            variant={notified ? "outline" : "primary"}
            onClick={() => setNotified(true)}
            disabled={notified}
            className="text-xs"
          >
            {notified ? "✓ Notification Sent" : "Notify Dr. Chen"}
          </Button>
        </div>
      </div>
    </div>
  )
}
