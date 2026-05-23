import React from 'react';
import { ArrowUp, BarChart2, Bell, Brain, CheckCircle, Eye, FileText, FlaskConical, Heart, History, Microscope, Paperclip, Plus, Settings, Shield, User, Users, Zap } from 'lucide-react';


export const WorkspaceScreen: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#FAFAF9]">
<div>
  {/* Sidebar: Fixed Clinical Left Panel */}
  <aside className="flex flex-col h-full w-72 bg-surface-container-low border-r border-outline-variant/20 z-40">
    {/* Brand Header from JSON */}
    <div className="p-6 flex flex-col gap-1">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center text-on-primary">
          <Microscope />
        </div>
        <div>
          <h1 className="font-headline-sm text-headline-sm font-semibold text-primary">Copilot OS</h1>
          <p className="font-body-sm text-body-sm text-muted-slate">Clinical Intelligence</p>
        </div>
      </div>
      <button className="mt-6 w-full py-3 bg-soft-graphite text-white rounded-xl font-body-sm flex items-center justify-center gap-2 hover:opacity-90 transition-all">
        <Plus className="text-[18px]" />
        New Analysis
      </button>
    </div>
    {/* Navigation Context */}
    <nav className="flex-1 px-4 py-2 space-y-1 overflow-y-auto scrollbar-hide">
      <div className="mb-6">
        <label className="px-2 font-label-caps text-label-caps text-muted-slate uppercase mb-3 block">Primary Navigation</label>
        <div className="space-y-1">
          <a className="flex items-center gap-3 px-3 py-2.5 bg-secondary-container text-on-secondary-container rounded-lg font-medium transition-all" href="#">
            <Users />
            <span className="font-body-sm">Patients</span>
          </a>
          <a className="flex items-center gap-3 px-3 py-2.5 text-muted-slate hover:bg-surface-variant/50 hover:text-soft-graphite rounded-lg transition-all" href="#">
            <History />
            <span className="font-body-sm">Recent Nodes</span>
          </a>
          <a className="flex items-center gap-3 px-3 py-2.5 text-muted-slate hover:bg-surface-variant/50 hover:text-soft-graphite rounded-lg transition-all" href="#">
            <BarChart2 />
            <span className="font-body-sm">Clinical Data</span>
          </a>
        </div>
      </div>
      {/* Patient List (Specific Requirement) */}
      <div>
        <label className="px-2 font-label-caps text-label-caps text-muted-slate uppercase mb-3 block">Active Patients</label>
        <div className="space-y-2">
          <div className="p-3 bg-white soft-touch-shadow rounded-xl active-glow transition-all cursor-pointer border border-transparent hover:border-outline-variant/30">
            <div className="flex justify-between items-center">
              <span className="font-body-sm font-semibold text-soft-graphite">P-882 (AJ)</span>
              <span className="w-2 h-2 rounded-full bg-blue-400" />
            </div>
            <p className="text-[12px] text-muted-slate mt-1">Stable • Post-Op Day 2</p>
          </div>
          <div className="p-3 bg-transparent hover:bg-white/50 rounded-xl transition-all cursor-pointer border border-transparent">
            <div className="flex justify-between items-center">
              <span className="font-body-sm text-soft-graphite">P-914 (MT)</span>
              <span className="w-2 h-2 rounded-full bg-clinical-error" />
            </div>
            <p className="text-[12px] text-muted-slate mt-1">Elevated HR • Labs Pending</p>
          </div>
          <div className="p-3 bg-transparent hover:bg-white/50 rounded-xl transition-all cursor-pointer border border-transparent">
            <div className="flex justify-between items-center">
              <span className="font-body-sm text-soft-graphite">P-702 (SK)</span>
              <span className="w-2 h-2 rounded-full bg-slate-300" />
            </div>
            <p className="text-[12px] text-muted-slate mt-1">Discharge Planning</p>
          </div>
        </div>
      </div>
    </nav>
    {/* Footer from JSON */}
    <div className="p-4 mt-auto border-t border-outline-variant/10">
      <div className="flex items-center gap-3 px-3 py-2 text-muted-slate hover:text-soft-graphite cursor-pointer transition-all">
        <CheckCircle className="text-[18px]" />
        <span className="font-label-caps text-label-caps uppercase">System Operational</span>
      </div>
      <div className="mt-4 flex flex-col gap-1 px-3 text-[10px] text-muted-slate/60">
        <span>© 2024 Copilot OS.</span>
        <div className="flex gap-2">
          <a className="hover:underline" href="#">Privacy</a>
          <a className="hover:underline" href="#">Terms</a>
          <a className="hover:underline" href="#">HIPAA</a>
        </div>
      </div>
    </div>
  </aside>
  {/* Main Workspace */}
  <main className="flex-1 flex flex-col min-w-0 bg-background relative">
    {/* Patient Context Top Bar (Sticky/Smoke Glass) */}
    <header className="sticky top-0 z-30 h-16 w-full glass-panel border-b border-outline-variant/30 px-8 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <div className="flex flex-col">
          <h2 className="font-headline-sm text-headline-sm font-bold text-soft-graphite leading-tight">P-882 (AJ)</h2>
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
            <span className="font-label-caps text-label-caps text-muted-slate">Inpatient • Room 402-B</span>
          </div>
        </div>
        <div className="h-8 w-px bg-outline-variant/40 mx-2" />
        <div className="flex gap-4">
          <div className="flex flex-col">
            <span className="font-label-caps text-label-caps text-muted-slate">BP</span>
            <span className="font-body-sm font-medium text-soft-graphite">118/74</span>
          </div>
          <div className="flex flex-col">
            <span className="font-label-caps text-label-caps text-muted-slate">HR</span>
            <span className="font-body-sm font-medium text-soft-graphite">72 bpm</span>
          </div>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <button className="p-2 hover:bg-surface-container-high rounded-full transition-colors text-muted-slate">
          <Bell />
        </button>
        <button className="p-2 hover:bg-surface-container-high rounded-full transition-colors text-muted-slate">
          <Settings />
        </button>
        <div className="h-8 w-8 rounded-full bg-surface-container-highest flex items-center justify-center border border-outline-variant/30 ml-2 overflow-hidden">
          <img alt="Clinician Profile" className="w-full h-full object-cover" data-alt="A professional close-up portrait of a medical doctor in clinical attire, set against a blurred medical facility background with soft, natural morning light. The aesthetic is clean, trustworthy, and high-key, reflecting a modern healthcare environment with a calm and focused atmosphere." src="https://lh3.googleusercontent.com/aida-public/AB6AXuCnL9w_Da-OdISN1YR-0LkOQDuSv7qvMISyLbofnFzbPB2hEh818SbeDnQaP3EIB4mv3x8VI9lyex3TGhsLy8zCCiFO8gfY9vnsof-w0LlP75FNgBxHlubXj-RDWa6aqXZy4ELxLpFtxIvaBkmdpOYfWGmsqoagXERw0K0nA7N1L5PW7m2wFPhmcIOmSQFoE5Gy7jIKlk1TIGOtcd2mgqPtdt6TqfMHE4V3s2wLHReI7Jegj9F-GywGkQTW-dwuzia42J9cB9GcFatR" />
        </div>
      </div>
    </header>
    {/* Conversational Canvas */}
    <div className="flex-1 overflow-y-auto scrollbar-hide px-panel-margin py-10 max-w-container-max mx-auto w-full flex flex-col gap-10">
      {/* User Query */}
      <div className="flex flex-col items-center">
        <div className="w-full max-w-3xl border-b border-outline-variant/20 pb-8">
          <div className="flex gap-4 items-start">
            <div className="w-8 h-8 rounded-lg bg-surface-container-high flex items-center justify-center text-muted-slate mt-1 shrink-0">
              <User className="text-[18px]" />
            </div>
            <div className="space-y-1">
              <h3 className="font-body-lg text-body-lg text-soft-graphite">Review the last 24 hours of P-882's vital stability and laboratory trends for potential discharge criteria.</h3>
              <span className="font-body-sm text-body-sm text-muted-slate/60">Today at 10:45 AM</span>
            </div>
          </div>
        </div>
      </div>
      {/* AI Response Card */}
      <div className="flex flex-col items-center">
        <div className="w-full max-w-3xl bg-white soft-touch-shadow rounded-[24px] p-8 space-y-6 border border-white">
          <div className="flex gap-4 items-start">
            <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center text-on-primary mt-1 shrink-0">
              <FileText className="text-[18px]" />
            </div>
            <div className="space-y-4 flex-1">
              <header className="flex justify-between items-center">
                <h3 className="font-headline-sm text-headline-sm font-semibold text-primary">Clinical Reasoning Analysis</h3>
                <div className="flex gap-2">
                  <span className="px-2.5 py-1 bg-surface-container-low text-soft-graphite font-label-caps text-label-caps rounded-full border border-outline-variant/10">Confidence: 98%</span>
                </div>
              </header>
              <p className="font-body-lg text-body-lg text-soft-graphite leading-relaxed">
                Patient P-882 (AJ) demonstrates strong physiological recovery post-orthopedic intervention. Current metrics suggest readiness for transition to outpatient care, pending surgical team review of the final imaging.
              </p>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-background rounded-xl border border-outline-variant/10">
                  <h4 className="font-label-caps text-label-caps text-muted-slate mb-2">VITAL TRENDS</h4>
                  <div className="h-16 w-full flex items-end gap-1 mb-2">
                    {/* Mock Chart */}
                    <div className="w-full h-[60%] bg-blue-100 rounded-sm" />
                    <div className="w-full h-[65%] bg-blue-200 rounded-sm" />
                    <div className="w-full h-[62%] bg-blue-300 rounded-sm" />
                    <div className="w-full h-[70%] bg-blue-400 rounded-sm" />
                    <div className="w-full h-[68%] bg-blue-500 rounded-sm" />
                  </div>
                  <p className="text-[12px] text-soft-graphite font-medium">Normotensive (avg 116/72)</p>
                </div>
                <div className="p-4 bg-background rounded-xl border border-outline-variant/10">
                  <h4 className="font-label-caps text-label-caps text-muted-slate mb-2">LABORATORY</h4>
                  <div className="flex flex-col gap-2">
                    <div className="flex justify-between items-center">
                      <span className="text-[12px] text-muted-slate">WBC Count</span>
                      <span className="text-[12px] text-soft-graphite font-medium">8.2 (Normal)</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-[12px] text-muted-slate">Hgb</span>
                      <span className="text-[12px] text-soft-graphite font-medium">13.4 g/dL</span>
                    </div>
                  </div>
                </div>
              </div>
              <div className="pt-4 flex gap-4">
                <button className="px-4 py-2 bg-soft-graphite text-white rounded-xl font-body-sm hover:opacity-90 transition-all">Generate Discharge Summary</button>
                <button className="px-4 py-2 border border-outline-variant text-soft-graphite rounded-xl font-body-sm hover:bg-surface-container-low transition-all">Flag for Nursing</button>
              </div>
            </div>
          </div>
        </div>
      </div>
      {/* User Query 2 (Divider style) */}
      <div className="flex flex-col items-center">
        <div className="w-full max-w-3xl border-b border-outline-variant/20 pb-8">
          <div className="flex gap-4 items-start">
            <div className="w-8 h-8 rounded-lg bg-surface-container-high flex items-center justify-center text-muted-slate mt-1 shrink-0">
              <User className="text-[18px]" />
            </div>
            <div className="space-y-1">
              <h3 className="font-body-lg text-body-lg text-soft-graphite">Are there any contraindications with his current antihypertensive regimen and the prescribed analgesic?</h3>
              <span className="font-body-sm text-body-sm text-muted-slate/60">Today at 10:52 AM</span>
            </div>
          </div>
        </div>
      </div>
      {/* AI Response 2 (Ongoing) */}
      <div className="flex flex-col items-center mb-24">
        <div className="w-full max-w-3xl bg-white soft-touch-shadow rounded-[24px] p-8 space-y-4 border border-white animate-pulse">
          <div className="flex gap-4 items-start">
            <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center text-on-primary mt-1 shrink-0">
              <Brain className="text-[18px]" />
            </div>
            <div className="space-y-2 flex-1">
              <h3 className="font-body-lg text-body-lg text-soft-graphite">Analyzing medication safety profiles...</h3>
              <div className="w-1/2 h-2 bg-surface-container rounded-full overflow-hidden">
                <div className="h-full bg-primary/20 w-1/3" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    {/* Reasoning Context Panel (Docked Right) */}
    <div className="absolute right-8 top-24 w-64 flex flex-col gap-4 z-20 pointer-events-none">
      <div className="pointer-events-auto bg-smoke-glass backdrop-blur-md rounded-2xl p-4 border border-outline-variant/20 soft-touch-shadow transition-all hover:translate-y-[-2px]">
        <div className="flex items-center gap-2 mb-3">
          <Eye className="text-muted-slate text-[16px]" />
          <span className="font-label-caps text-label-caps text-muted-slate uppercase">Reasoning Context</span>
        </div>
        <div className="space-y-3">
          <div className="flex flex-col">
            <span className="text-[10px] text-muted-slate mb-1">DATA NODES ANALYZED</span>
            <div className="flex -space-x-2">
              <div className="w-6 h-6 rounded-full bg-blue-50 border-2 border-white flex items-center justify-center">
                <FlaskConical className="text-[12px] text-blue-500" />
              </div>
              <div className="w-6 h-6 rounded-full bg-green-50 border-2 border-white flex items-center justify-center">
                <Heart className="text-[12px] text-green-500" />
              </div>
              <div className="w-6 h-6 rounded-full bg-purple-50 border-2 border-white flex items-center justify-center">
                <FileText className="text-[12px] text-purple-500" />
              </div>
            </div>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] text-muted-slate mb-1">ONTOLOGY MATCH</span>
            <span className="text-[12px] font-medium text-soft-graphite">ICD-10-CM Z47.1</span>
          </div>
        </div>
      </div>
    </div>
    {/* Input Area (Linear Style) */}
    <footer className="absolute bottom-0 left-0 right-0 p-8 flex justify-center pointer-events-none">
      <div className="w-full max-w-3xl pointer-events-auto">
        <div className="relative group">
          <div className="absolute inset-0 bg-white/20 blur-xl group-focus-within:bg-white/40 transition-all rounded-2xl" />
          <div className="relative bg-white soft-touch-shadow border border-outline-variant/40 rounded-2xl p-2 flex items-center gap-2">
            <input className="flex-1 bg-transparent border-none focus:ring-0 font-body-lg text-body-lg px-4 text-soft-graphite placeholder-muted-slate/50" placeholder="Ask Copilot about P-882..." type="text" />
            <div className="flex items-center gap-1 px-2">
              <button className="p-2 text-muted-slate hover:text-soft-graphite hover:bg-surface-container-low rounded-lg transition-all">
                <Paperclip />
              </button>
              <button className="w-10 h-10 bg-soft-graphite text-white rounded-xl flex items-center justify-center hover:bg-black transition-all">
                <ArrowUp />
              </button>
            </div>
          </div>
        </div>
        <div className="flex justify-center mt-3 gap-6">
          <div className="flex items-center gap-1.5 opacity-40 hover:opacity-100 transition-opacity cursor-help">
            <Shield className="text-[14px]" />
            <span className="font-label-caps text-label-caps uppercase">HIPAA Encrypted</span>
          </div>
          <div className="flex items-center gap-1.5 opacity-40 hover:opacity-100 transition-opacity cursor-help">
            <Zap className="text-[14px]" />
            <span className="font-label-caps text-label-caps uppercase">GPT-4 Medical Tuned</span>
          </div>
        </div>
      </div>
    </footer>
  </main>
  {/* Micro-interaction Script */}
</div>

    </div>
  );
};
