import * as React from "react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from "recharts"
import { Activity, AlertCircle, Heart, Thermometer, UserCheck } from "lucide-react"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "../ui/card"

const MOCK_VITALS_DATA = [
  { day: "Day 1", heartRate: 85, temp: 98.4 },
  { day: "Day 2", heartRate: 90, temp: 98.7 },
  { day: "Day 3", heartRate: 95, temp: 99.1 },
  { day: "Day 4", heartRate: 100, temp: 99.5 },
  { day: "Day 5", heartRate: 92, temp: 98.9 },
]

export function PatientTrends() {
  const [hoveredData, setHoveredData] = React.useState<any>(null)

  const heartRateStats = {
    min: 85,
    max: 100,
    avg: 92.4,
    status: "Elevated"
  }

  const tempStats = {
    min: 98.4,
    max: 99.5,
    avg: 98.92,
    status: "Low-grade fever"
  }

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      {/* Top Banner */}
      <div className="px-6 py-4 border-b border-borderwhisper bg-white flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold tracking-tight text-ink">Patient Trends & Analytics</h2>
          <p className="text-xs text-steel">Interactive telemetry monitoring patient vital signs over time with safety threshold flagging.</p>
        </div>
        <div className="flex items-center gap-1.5 px-3 py-1 bg-amber-50 border border-amber-200 rounded-full text-[10px] font-mono text-amber-800">
          <Activity className="w-3.5 h-3.5 text-amber-600 animate-pulse" /> MONITORING: BORDERLINE INSTABILITY
        </div>
      </div>

      <div className="flex-1 p-6 overflow-y-auto space-y-6">
        {/* Patient Profile Summary Card */}
        <div className="bg-white border border-borderwhisper rounded-lg p-5 flex flex-col md:flex-row justify-between gap-4 shadow-sm">
          <div className="space-y-1.5">
            <h3 className="text-xs font-bold text-steel font-mono tracking-wider">PATIENT IDENTIFICATION</h3>
            <p className="text-base font-bold text-ink">Marcus Vance</p>
            <div className="flex items-center gap-4 text-xs text-slate-600">
              <span>**ID:** MRN-88291A</span>
              <span>**DOB:** Oct 12, 1968 (Age 57)</span>
              <span>**Room:** ICU-4B</span>
            </div>
          </div>
          <div className="flex items-center gap-6 self-start md:self-center font-mono">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-red-50 rounded-full flex items-center justify-center text-critical">
                <Heart className="w-4 h-4 fill-critical" />
              </div>
              <div>
                <p className="text-[10px] text-steel">CURRENT HR</p>
                <p className="text-sm font-bold text-ink">92 bpm</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-amber-50 rounded-full flex items-center justify-center text-amber-600">
                <Thermometer className="w-4 h-4" />
              </div>
              <div>
                <p className="text-[10px] text-steel">CURRENT TEMP</p>
                <p className="text-sm font-bold text-ink">98.9 °F</p>
              </div>
            </div>
          </div>
        </div>

        {/* Asymmetric grid: Charts (2/3 width) and Metrics Telemetry (1/3 width) */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-2 border border-borderwhisper shadow-sm">
            <CardHeader className="py-4">
              <CardTitle>Stability History</CardTitle>
              <CardDescription>Line chart representation of daily vitals. Red reference line defines safety threshold limits.</CardDescription>
            </CardHeader>
            <CardContent className="h-80 pt-4 pr-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={MOCK_VITALS_DATA}
                  margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                  onMouseMove={(state: any) => {
                    if (state && state.activePayload) {
                      setHoveredData(state.activePayload[0].payload)
                    }
                  }}
                  onMouseLeave={() => setHoveredData(null)}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                  <XAxis dataKey="day" tick={{ fontSize: 10, fill: "#64748B" }} stroke="#CBD5E1" />
                  <YAxis yAxisId="left" tick={{ fontSize: 10, fill: "#64748B" }} domain={[70, 110]} stroke="#CBD5E1" />
                  <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10, fill: "#64748B" }} domain={[97, 101]} stroke="#CBD5E1" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#FFFFFF",
                      borderColor: "#E2E8F0",
                      fontSize: "11px",
                      borderRadius: "6px",
                      boxShadow: "0 1px 3px rgba(0,0,0,0.05)"
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: "11px", marginTop: "10px" }} />
                  {/* Thresholds */}
                  <ReferenceLine yAxisId="left" y={95} stroke="#BE123C" strokeDasharray="3 3" label={{ value: "HR Limit", fill: "#BE123C", fontSize: 9, position: "top" }} />
                  <ReferenceLine yAxisId="right" y={99.0} stroke="#D97706" strokeDasharray="3 3" label={{ value: "Temp Limit", fill: "#D97706", fontSize: 9, position: "top" }} />
                  <Line yAxisId="left" type="monotone" dataKey="heartRate" name="Heart Rate (bpm)" stroke="#0F766E" strokeWidth={2} activeDot={{ r: 6 }} dot={{ r: 4 }} />
                  <Line yAxisId="right" type="monotone" dataKey="temp" name="Temperature (°F)" stroke="#BE123C" strokeWidth={2} activeDot={{ r: 6 }} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Telemetry Summary panel */}
          <div className="space-y-6">
            {/* Heart Rate Statistics */}
            <Card className="border border-borderwhisper shadow-sm">
              <CardHeader className="py-4 bg-slate-50/50">
                <CardTitle className="text-xs font-mono uppercase text-steel flex items-center gap-1.5">
                  <Heart className="w-3.5 h-3.5 text-pine" /> Cardiovascular Summary
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-4 space-y-4">
                <div className="grid grid-cols-3 gap-2 text-center font-mono">
                  <div className="bg-slate-50 p-2 rounded">
                    <p className="text-[9px] text-steel">MIN</p>
                    <p className="text-xs font-bold text-ink">{heartRateStats.min}</p>
                  </div>
                  <div className="bg-slate-50 p-2 rounded">
                    <p className="text-[9px] text-steel">MAX</p>
                    <p className="text-xs font-bold text-critical">{heartRateStats.max}</p>
                  </div>
                  <div className="bg-slate-50 p-2 rounded">
                    <p className="text-[9px] text-steel">AVG</p>
                    <p className="text-xs font-bold text-ink">{heartRateStats.avg}</p>
                  </div>
                </div>
                <div className="flex items-center justify-between text-xs pt-1 border-t border-borderwhisper/60">
                  <span className="text-steel">Vitals Status:</span>
                  <span className="text-amber-700 font-semibold px-2 py-0.5 bg-amber-50 rounded">
                    {heartRateStats.status}
                  </span>
                </div>
              </CardContent>
            </Card>

            {/* Temperature Statistics */}
            <Card className="border border-borderwhisper shadow-sm">
              <CardHeader className="py-4 bg-slate-50/50">
                <CardTitle className="text-xs font-mono uppercase text-steel flex items-center gap-1.5">
                  <Thermometer className="w-3.5 h-3.5 text-critical" /> Thermoregulation Summary
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-4 space-y-4">
                <div className="grid grid-cols-3 gap-2 text-center font-mono">
                  <div className="bg-slate-50 p-2 rounded">
                    <p className="text-[9px] text-steel">MIN</p>
                    <p className="text-xs font-bold text-ink">{tempStats.min}</p>
                  </div>
                  <div className="bg-slate-50 p-2 rounded">
                    <p className="text-[9px] text-steel">MAX</p>
                    <p className="text-xs font-bold text-critical">{tempStats.max}</p>
                  </div>
                  <div className="bg-slate-50 p-2 rounded">
                    <p className="text-[9px] text-steel">AVG</p>
                    <p className="text-xs font-bold text-ink">{tempStats.avg}</p>
                  </div>
                </div>
                <div className="flex items-center justify-between text-xs pt-1 border-t border-borderwhisper/60">
                  <span className="text-steel">Vitals Status:</span>
                  <span className="text-amber-700 font-semibold px-2 py-0.5 bg-amber-50 rounded">
                    {tempStats.status}
                  </span>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Hovered Day Insights Card */}
        {hoveredData && (
          <div className="bg-slate-50 border border-borderwhisper rounded-lg p-4 flex items-center justify-between text-xs animate-fade-in">
            <div className="flex items-center gap-2">
              <UserCheck className="w-4 h-4 text-pine" />
              <span>Telemetry telemetry selection: **{hoveredData.day}**</span>
            </div>
            <div className="font-mono flex gap-4">
              <span>Heart Rate: **{hoveredData.heartRate} bpm**</span>
              <span>Temperature: **{hoveredData.temp} °F**</span>
            </div>
          </div>
        )}

        {/* Clinical Stability Summary */}
        <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-lg flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-stable shrink-0 mt-0.5" />
          <div className="text-xs">
            <h4 className="font-bold text-emerald-950">Safety Summary (FAISS Retrospective Match)</h4>
            <p className="text-emerald-800 leading-relaxed mt-0.5">
              The patient's temperature peaked on Day 4 (99.5°F) corresponding with a heart rate of 100 bpm. Day 5 displays stable downward shifts. Vitals mapping matches standard postoperative recovery patterns documented in patient history files.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
