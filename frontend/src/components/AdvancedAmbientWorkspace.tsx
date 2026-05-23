import React from 'react';
import { FileText, HelpCircle, History, Search, User } from 'lucide-react';


export const AdvancedAmbientWorkspace: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#FAFAF9]">
<div>
  {/* Top Navigation Shell (Shared Components Hierarchy) */}
  <header className="fixed top-0 w-full z-40 bg-transparent flex justify-between items-center px-panel-margin py-gutter w-full max-w-max-width mx-auto">
    <h1 className="text-headline-sm font-headline-sm text-on-surface tracking-tight">Ambient Clinical Intelligence</h1>
    <div className="flex items-center gap-6">
      <nav className="hidden md:flex items-center gap-8 text-label-caps font-label-caps">
        <a className="text-primary font-bold hover:text-primary transition-colors duration-200" href="#">WORKSPACE</a>
        <a className="text-muted-slate hover:text-primary transition-colors duration-200" href="#">PATIENTS</a>
        <a className="text-muted-slate hover:text-primary transition-colors duration-200" href="#">PROTOCOLS</a>
      </nav>
      <User className="text-primary" />
    </div>
  </header>
  {/* Main Fluid Canvas */}
  <main className="relative pt-32 pb-48 px-panel-margin max-w-max-width mx-auto space-y-32">
    {/* 1. Stable State: Patient Summary */}
    <section className="reveal-text" id="patient-summary">
      <div className="space-y-6 max-w-2xl">
        <div className="flex items-baseline gap-4">
          <h2 className="font-display-lg text-display-lg text-on-background">Sarah Jenkins</h2>
          <span className="font-body-lg text-body-lg text-muted-slate">42F, Post-Op Day 2</span>
        </div>
        <p className="font-body-lg text-body-lg leading-relaxed text-soft-graphite">
          Patient is recovering from an uncomplicated laparoscopic cholecystectomy. Currently resting comfortably. Vitals are stable with a heart rate of 
          <span className="font-mono font-bold text-on-background">72</span>
          <svg className="sparkline" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 100 20"><path d="M0 15 Q 10 5, 20 12 T 40 10 T 60 14 T 80 8 T 100 12" /></svg>
          bpm and SpO2 at 
          <span className="font-mono font-bold text-on-background">98%</span>
          <svg className="sparkline" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 100 20"><path d="M0 10 Q 15 8, 30 12 T 50 10 T 70 11 T 85 9 T 100 10" /></svg>
          on room air. Pain is managed with PRN analgesics, reported at 2/10.
        </p>
      </div>
    </section>
    {/* 2. AI Reasoning In Progress: Intelligence Stream */}
    <section className="relative" id="ai-reasoning">
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <div className="relative w-2 h-2 rounded-full bg-soft-graphite animate-pulse" />
          <span className="font-label-caps text-label-caps text-muted-slate tracking-widest uppercase">Analyzing Laboratory Data</span>
        </div>
        <div className="space-y-3">
          <div className="h-4 w-3/4 shimmer rounded-full opacity-20" />
          <div className="h-4 w-1/2 shimmer rounded-full opacity-10" />
        </div>
      </div>
    </section>
    {/* 3. Streaming Insight & Evidence Provenance Overlay */}
    <section className="relative group" id="streaming-insight">
      <div className="flex flex-col md:flex-row gap-12 items-start">
        <div className="flex-1 space-y-6">
          <div className="flex items-center gap-4">
            <HelpCircle className="text-primary" />
            <span className="font-label-caps text-label-caps text-on-surface">CLINICAL INFERENCE</span>
            <div className="px-2 py-0.5 rounded bg-surface-container text-muted-slate text-[10px] font-bold">94% CONFIDENCE</div>
          </div>
          <div className="space-y-4">
            <p className="font-headline-md text-headline-md text-soft-graphite leading-snug reveal-text" style={{animationDelay: '0.2s'}}>
              Based on current trends, the patient’s renal function is showing signs of early <span className="text-on-background font-semibold">hypokalemic shift</span>. 
              The recent lab panels from 06:00 show a gradual decline in Serum Potassium.
            </p>
            <p className="font-body-lg text-body-lg text-muted-slate reveal-text" style={{animationDelay: '0.4s'}}>
              Recommendation: Consider oral potassium supplementation and repeat BMP in 4 hours. Ensure hydration remains at 125ml/hr.
            </p>
          </div>
        </div>
        {/* Evidence Provenance (Smoke-glass overlay appearance) */}
        <aside className="w-full md:w-80 smoke-glass-effect p-6 rounded-xl shadow-[0_20px_40px_rgba(0,0,0,0.04)] reveal-text" style={{animationDelay: '0.6s'}}>
          <div className="flex items-center gap-2 mb-4">
            <HelpCircle className="text-[16px] text-muted-slate" />
            <span className="font-label-caps text-label-caps text-muted-slate">SOURCE PROVENANCE</span>
          </div>
          <div className="space-y-4">
            <div className="p-3 bg-white/50 rounded-lg">
              <p className="font-body-sm text-body-sm text-on-surface mb-1">Standard Post-Op Protocol #402</p>
              <p className="text-[12px] text-muted-slate italic">"Early detection of postoperative electrolyte imbalance..."</p>
            </div>
            <div className="p-3 bg-white/50 rounded-lg">
              <p className="font-body-sm text-body-sm text-on-surface mb-1">EHR: Lab History</p>
              <p className="text-[12px] text-muted-slate italic">K+ 4.1 (Pre-op) → 3.6 (06:00)</p>
            </div>
          </div>
        </aside>
      </div>
    </section>
    {/* 4. Escalating Anomaly */}
    <section className="relative py-12 px-8 -mx-8 bg-on-background text-white rounded-3xl overflow-hidden transition-all duration-500 hover:scale-[1.01]" id="anomaly">
      <div className="absolute inset-0 bg-gradient-to-br from-soft-graphite/20 to-transparent" />
      <div className="relative z-10 flex flex-col md:flex-row justify-between items-end gap-8">
        <div className="space-y-4 max-w-xl">
          <div className="flex items-center gap-2">
            <HelpCircle className="text-clinical-error animate-pulse" />
            <span className="font-label-caps text-label-caps text-white/60 tracking-widest uppercase">Urgent Contextual Shift</span>
          </div>
          <h3 className="font-display-lg text-display-lg-mobile md:text-headline-md font-bold leading-tight">
            Potassium levels have dropped to <span className="text-error-container underline underline-offset-8">3.2 mEq/L</span>
          </h3>
          <p className="font-body-lg text-body-lg text-white/80">
            This reflects a 12% decrease in the last 2 hours. Compensatory tachycardia may follow if unaddressed.
          </p>
        </div>
        {/* 6. Clinician Override */}
        <div className="flex gap-4">
          <button className="bg-white text-on-background px-6 py-3 rounded-xl font-label-caps text-label-caps hover:bg-white/90 transition-all active:scale-95 shadow-xl">
            ADJUST PLAN
          </button>
          <button className="bg-white/10 backdrop-blur-md text-white border border-white/20 px-6 py-3 rounded-xl font-label-caps text-label-caps hover:bg-white/20 transition-all active:scale-95">
            DISMISS
          </button>
        </div>
      </div>
    </section>
    {/* 5. Future State Simulation / Ambient Workspace Expansion */}
    <section className="max-w-3xl opacity-40 hover:opacity-100 transition-opacity duration-700">
      <p className="font-body-lg text-body-lg text-soft-graphite">
        Additional observations scheduled for <span className="font-mono text-on-background">14:00</span>. Physical therapy consult pending verification of mobility status.
      </p>
    </section>
  </main>
  {/* Bottom Navigation Bar (Shared Components Execution) */}
  <nav className="fixed bottom-12 left-1/2 -translate-x-1/2 z-50 flex items-center justify-center space-x-2 bg-smoke-glass backdrop-blur-xl rounded-full px-6 py-3 w-auto min-w-[320px] shadow-[0_20px_40px_rgba(0,0,0,0.06)]" id="main-nav">
    {/* Active: Workspace */}
    <a className="flex items-center gap-2 bg-on-background dark:bg-on-primary text-white dark:text-primary rounded-full px-4 py-2 scale-95 transition-transform" href="#">
      <FileText className="text-[20px]" />
      <span className="font-label-caps text-label-caps">Workspace</span>
    </a>
    <a className="flex items-center gap-2 text-soft-graphite dark:text-on-primary-fixed-variant px-4 py-2 hover:bg-surface-variant/50 transition-all duration-300" href="#">
      <History className="text-[20px]" />
      <span className="font-label-caps text-label-caps">Timeline</span>
    </a>
    <a className="flex items-center gap-2 text-soft-graphite dark:text-on-primary-fixed-variant px-4 py-2 hover:bg-surface-variant/50 transition-all duration-300" href="#">
      <User className="text-[20px]" />
      <span className="font-label-caps text-label-caps">Patient</span>
    </a>
    <a className="flex items-center gap-2 text-soft-graphite dark:text-on-primary-fixed-variant px-4 py-2 hover:bg-surface-variant/50 transition-all duration-300" href="#">
      <HelpCircle className="text-[20px]" />
      <span className="font-label-caps text-label-caps">Insights</span>
    </a>
    <a className="flex items-center gap-2 text-soft-graphite dark:text-on-primary-fixed-variant px-4 py-2 hover:bg-surface-variant/50 transition-all duration-300" href="#">
      <Search className="text-[20px]" />
      <span className="font-label-caps text-label-caps">Search</span>
    </a>
  </nav>
  {/* Micro-interactions Script */}
</div>

    </div>
  );
};
