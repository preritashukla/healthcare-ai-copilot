import React from 'react';
import { HelpCircle, Microscope, Search, Users } from 'lucide-react';


export const AmbientWorkspace: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#FAFAF9]">
<div>
  {/* Ambient Canvas */}
  <main className="max-w-[1200px] mx-auto px-gutter pt-24 pb-48 canvas-entrance">
    {/* Header Section */}
    <header className="mb-panel-margin">
      <h1 className="font-display-lg text-display-lg font-light tracking-tight text-on-surface opacity-90">
        Sarah Jenkins, 42F — <span className="text-muted-slate">Post-op Day 2</span>
      </h1>
    </header>
    {/* Section 1: Clinical Trajectory */}
    <section className="mb-panel-margin space-y-4 max-w-3xl">
      <h2 className="font-label-caps text-label-caps text-muted-slate uppercase">Clinical Trajectory</h2>
      <p className="font-headline-md text-headline-md text-soft-graphite leading-relaxed">
        The patient is transitioning effectively into the sub-acute phase following a successful robotic-assisted cholecystectomy. Her mobility has increased by 40% since yesterday evening, and she reports manageable pain levels without the need for intensive opioid intervention. Post-operative inflammation markers are trending toward baseline, suggesting a low likelihood of immediate complications.
      </p>
    </section>
    {/* Section 2: Vital Intelligence */}
    <section className="mb-panel-margin space-y-4">
      <h2 className="font-label-caps text-label-caps text-muted-slate uppercase">Vital Intelligence</h2>
      <div className="flex flex-wrap gap-x-12 gap-y-6 items-baseline">
        <div className="flex items-center gap-3">
          <span className="font-body-lg text-body-lg text-soft-graphite">Heart Rate: <span className="font-bold text-on-surface">72 bpm</span></span>
          <div aria-hidden="true" className="sparkline-container">
            <div className="spark-bar h-2" />
            <div className="spark-bar h-3" />
            <div className="spark-bar h-4" />
            <div className="spark-bar h-2" />
            <div className="spark-bar h-5 active" />
            <div className="spark-bar h-4" />
            <div className="spark-bar h-3" />
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="font-body-lg text-body-lg text-soft-graphite">SpO2: <span className="font-bold text-on-surface">98%</span></span>
          <div aria-hidden="true" className="sparkline-container">
            <div className="spark-bar h-4" />
            <div className="spark-bar h-4 active" />
            <div className="spark-bar h-4" />
            <div className="spark-bar h-4" />
            <div className="spark-bar h-4" />
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="font-body-lg text-body-lg text-soft-graphite">Mean BP: <span className="font-bold text-on-surface">94 mmHg</span></span>
        </div>
      </div>
    </section>
    {/* Section 3: AI Risk Assessment */}
    <section className="mb-panel-margin space-y-4 max-w-3xl">
      <h2 className="font-label-caps text-label-caps text-muted-slate uppercase">AI Risk Assessment</h2>
      <p className="font-body-lg text-body-lg text-soft-graphite leading-relaxed">
        Predictive analytics indicate a <span className="text-primary font-medium">92% probability</span> of discharge within the next 24 hours. Electrolyte stability remains optimal, with Potassium (4.1 mEq/L) and Magnesium (2.0 mg/dL) both within therapeutic ranges. There are no indications of pulmonary edema or localized site infection. The AI agent recommends transitioning to oral medications for discharge planning.
      </p>
    </section>
    {/* Atmospheric Visual Element (Asymmetric) */}
    <div className="mt-panel-margin opacity-40">
      <div className="w-64 h-[1px] bg-outline-variant" />
      <p className="mt-4 font-body-sm text-body-sm text-muted-slate italic">
        Last updated: 4 minutes ago via Telemetry Cluster 4
      </p>
    </div>
  </main>
  {/* Floating Command Palette (Shared Component Logic) */}
  <nav className="fixed bottom-0 left-0 right-0 z-50 flex justify-center pb-8 px-gutter">
    <div className="smoke-glass backdrop-blur-xl shadow-2xl shadow-black/5 rounded-full px-6 py-3 flex items-center gap-6 border border-white/20">
      {/* Search/Command Trigger */}
      <button className="flex items-center gap-3 px-4 py-2 text-soft-graphite hover:bg-surface-container-highest rounded-full transition-all duration-300 active:scale-95">
        <Search />
        <span className="font-label-caps text-label-caps">Search Command</span>
      </button>
      <div className="w-[1px] h-6 bg-outline-variant/30" />
      {/* Contextual Actions */}
      <div className="flex items-center gap-2">
        <button className="flex items-center gap-2 bg-primary text-on-primary rounded-full px-5 py-2.5 font-label-caps text-label-caps hover:opacity-90 transition-opacity active:scale-95">
          <HelpCircle className="text-[18px]" />
          New Note
        </button>
        <button className="flex items-center gap-2 text-soft-graphite hover:bg-surface-container-highest rounded-full px-5 py-2.5 transition-all duration-300 font-label-caps text-label-caps active:scale-95">
          <Microscope className="text-[18px]" />
          Order Labs
        </button>
        <button className="flex items-center gap-2 text-soft-graphite hover:bg-surface-container-highest rounded-full px-5 py-2.5 transition-all duration-300 font-label-caps text-label-caps active:scale-95">
          <Users className="text-[18px]" />
          Consult Team
        </button>
      </div>
    </div>
  </nav>
  {/* Subtle Background Detail */}
  <div className="fixed top-0 right-0 p-gutter pointer-events-none opacity-20">
    <div className="text-right">
      <span className="font-label-caps text-label-caps block text-muted-slate">Aether System v4.0</span>
      <span className="font-label-caps text-label-caps block text-muted-slate">St. Jude Medical Center</span>
    </div>
  </div>
</div>

    </div>
  );
};
