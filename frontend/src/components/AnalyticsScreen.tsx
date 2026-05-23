import React from 'react';
import { Bell, Heart, HelpCircle, Microscope, Plus, Settings, Wind } from 'lucide-react';


export const AnalyticsScreen: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#FAFAF9]">
<div>
  {/* SideNavBar */}
  <aside className="docked full-height left-0 w-72 flex flex-col bg-surface border-r border-border-subtle z-50">
    <div className="flex flex-col h-full py-12 px-4 gap-6">
      {/* Brand Header */}
      <div className="px-4 mb-4">
        <h2 className="font-headline-sm text-headline-sm font-semibold text-soft-graphite">Copilot OS</h2>
        <p className="font-label-caps text-label-caps text-muted-slate uppercase tracking-wider">Intelligence Workspace</p>
      </div>
      {/* CTA */}
      <button className="mx-4 bg-soft-graphite text-white font-medium py-3 rounded-xl hover:opacity-90 transition-opacity flex items-center justify-center gap-2 mb-4">
        <Plus className="text-sm" />
        <span>New Encounter</span>
      </button>
      {/* Nav Items */}
      <nav className="flex-1 space-y-1">
        <a className="flex items-center gap-3 px-4 py-3 bg-surface-container-high text-primary font-semibold rounded-xl translate-x-1 transition-transform" href="#">
          <HelpCircle />
          <span className="font-label-caps text-label-caps">CLINICAL NODES</span>
        </a>
        <a className="flex items-center gap-3 px-4 py-3 text-muted-slate hover:text-soft-graphite hover:bg-surface-container-low rounded-xl transition-colors" href="#">
          <HelpCircle />
          <span className="font-label-caps text-label-caps">PATIENT LISTS</span>
        </a>
        <a className="flex items-center gap-3 px-4 py-3 text-muted-slate hover:text-soft-graphite hover:bg-surface-container-low rounded-xl transition-colors" href="#">
          <HelpCircle />
          <span className="font-label-caps text-label-caps">SYSTEM STATUS</span>
        </a>
        <a className="flex items-center gap-3 px-4 py-3 text-muted-slate hover:text-soft-graphite hover:bg-surface-container-low rounded-xl transition-colors" href="#">
          <HelpCircle />
          <span className="font-label-caps text-label-caps">ANALYTICS</span>
        </a>
      </nav>
      {/* Footer Tabs */}
      <div className="pt-6 border-t border-border-subtle space-y-1">
        <a className="flex items-center gap-3 px-4 py-2 text-muted-slate hover:text-soft-graphite transition-colors" href="#">
          <HelpCircle className="text-xl" />
          <span className="font-body-sm text-body-sm">Support</span>
        </a>
        <a className="flex items-center gap-3 px-4 py-2 text-muted-slate hover:text-soft-graphite transition-colors" href="#">
          <HelpCircle className="text-xl" />
          <span className="font-body-sm text-body-sm">Settings</span>
        </a>
      </div>
    </div>
  </aside>
  {/* Main Workspace */}
  <div className="flex-1 flex flex-col relative overflow-hidden">
    {/* TopAppBar */}
    <header className="docked full-width top-0 z-40 backdrop-blur-md bg-smoke-glass border-b border-border-subtle shadow-sm flex justify-between items-center px-8 w-full h-16">
      <div className="flex items-center gap-4">
        <h1 className="font-headline-md text-headline-md font-bold text-soft-graphite">Copilot Clinical</h1>
        <div className="h-6 w-px bg-border-subtle mx-2" />
        <div className="flex items-center gap-2 text-muted-slate">
          <span className="font-body-sm text-body-sm font-medium">Patient Analytics // P-882</span>
        </div>
      </div>
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-4">
          <Bell className="text-soft-graphite hover:bg-surface-container-low p-2 rounded-full cursor-pointer transition-colors" />
          <Settings className="text-soft-graphite hover:bg-surface-container-low p-2 rounded-full cursor-pointer transition-colors" />
          <HelpCircle className="text-soft-graphite hover:bg-surface-container-low p-2 rounded-full cursor-pointer transition-colors" />
        </div>
        <div className="flex items-center gap-3 pl-4 border-l border-border-subtle">
          <img alt="Medical provider profile" className="w-8 h-8 rounded-full object-cover grayscale brightness-110" data-alt="A professional portrait of a medical provider wearing a white lab coat, presented in a clean, high-key studio setting. The lighting is soft and even, highlighting a calm and trustworthy expression. The background is a minimalist light gray, maintaining a premium medical aesthetic consistent with the clinical software environment." src="https://lh3.googleusercontent.com/aida-public/AB6AXuA8fRe7ibDi6zpXdUsGhvlDWefCY1ctNZ5kjDcW-tuO4Pdvg7IMTPsem1qJq9MJLfQEMas4598VZlk4Hpc4_lsWMtUIpqj4lbJh-G5HsUTc4wYiFdskWVZxu1OKHi9adIpvhwQbeDWxFl3W00H8AiE2xV-EFXvlK87wKaIUqXWBJE6URaHvBI8C3YJXQPYi4UbOAUwvtajbJAoBDkH94AjRgl-ZiaHrY5wioZyUsD7B6yznIlpBDEMC1mG0fttUiZS0b71jHHDP3lEA" />
        </div>
      </div>
    </header>
    {/* Content Canvas */}
    <main className="flex-1 overflow-y-auto p-gutter relative">
      <div className="canvas-container space-y-8 pb-12">
        {/* Patient Status Summary (Hero) */}
        <section className="animate-in fade-in slide-in-from-bottom-4 duration-700">
          <div className="bg-white rounded-xl p-8 soft-shadow border border-border-subtle flex flex-col md:flex-row gap-8 items-start">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-4">
                <span className="px-2 py-1 bg-surface-container-high rounded text-[10px] font-bold text-soft-graphite uppercase tracking-widest">Active Insight</span>
                <span className="text-muted-slate font-label-caps text-label-caps">Last updated 2m ago</span>
              </div>
              <h2 className="font-headline-md text-headline-md text-soft-graphite mb-4">Clinical Reasoning Summary</h2>
              <p className="font-body-lg text-body-lg text-soft-graphite leading-relaxed max-w-2xl">
                Patient exhibits stable post-operative recovery. Hemodynamic markers remain within nominal range with a slight downward trend in potassium. Respiratory effort is unlabored on 2L NC. Recommended focus: monitor electrolyte balance for potential replacement during the next 4-hour window.
              </p>
            </div>
            <div className="w-full md:w-64 bg-surface-container-low rounded-xl p-6 border border-border-subtle/50">
              <div className="space-y-4">
                <div>
                  <p className="font-label-caps text-label-caps text-muted-slate mb-1">PATIENT ID</p>
                  <p className="font-headline-sm text-headline-sm font-semibold">P-882 (AJ)</p>
                </div>
                <div>
                  <p className="font-label-caps text-label-caps text-muted-slate mb-1">ADMISSION</p>
                  <p className="font-body-sm text-body-sm">Post-Op Day 2</p>
                </div>
                <div>
                  <p className="font-label-caps text-label-caps text-muted-slate mb-1">CARE TEAM</p>
                  <p className="font-body-sm text-body-sm text-primary">Dr. Sarah Jenkins</p>
                </div>
              </div>
            </div>
          </div>
        </section>
        {/* Telemetry Vitals */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Heart Rate */}
          <div className="bg-white p-6 rounded-xl soft-shadow border border-border-subtle group hover:border-primary/20 transition-all">
            <div className="flex justify-between items-start mb-6">
              <div>
                <p className="font-label-caps text-label-caps text-muted-slate mb-1">HEART RATE</p>
                <div className="flex items-baseline gap-2">
                  <span className="font-headline-md text-headline-md font-bold">72</span>
                  <span className="font-body-sm text-body-sm text-muted-slate">bpm</span>
                </div>
              </div>
              <Heart className="text-clinical-error animate-pulse" />
            </div>
            <div className="h-12 w-full">
              <svg className="w-full h-full" viewBox="0 0 100 20">
                <path className="text-soft-graphite/30 sparkline-svg" d="M0 10 Q 10 5, 20 12 T 40 8 T 60 15 T 80 5 T 100 10" fill="none" stroke="currentColor" strokeWidth="1.5" />
              </svg>
            </div>
          </div>
          {/* SpO2 */}
          <div className="bg-white p-6 rounded-xl soft-shadow border border-border-subtle group hover:border-primary/20 transition-all">
            <div className="flex justify-between items-start mb-6">
              <div>
                <p className="font-label-caps text-label-caps text-muted-slate mb-1">SPO2</p>
                <div className="flex items-baseline gap-2">
                  <span className="font-headline-md text-headline-md font-bold">98</span>
                  <span className="font-body-sm text-body-sm text-muted-slate">%</span>
                </div>
              </div>
              <Wind className="text-muted-slate" />
            </div>
            <div className="h-12 w-full">
              <svg className="w-full h-full" viewBox="0 0 100 20">
                <path className="text-soft-graphite/30 sparkline-svg" d="M0 10 L 10 9 L 20 10 L 30 10 L 40 10 L 50 9.5 L 60 10 L 70 10 L 80 10 L 90 9.5 L 100 10" fill="none" stroke="currentColor" strokeWidth="1.5" />
              </svg>
            </div>
          </div>
          {/* Blood Pressure */}
          <div className="bg-white p-6 rounded-xl soft-shadow border border-border-subtle group hover:border-primary/20 transition-all">
            <div className="flex justify-between items-start mb-6">
              <div>
                <p className="font-label-caps text-label-caps text-muted-slate mb-1">BLOOD PRESSURE</p>
                <div className="flex items-baseline gap-2">
                  <span className="font-headline-md text-headline-md font-bold">118/74</span>
                  <span className="font-body-sm text-body-sm text-muted-slate">mmHg</span>
                </div>
              </div>
              <HelpCircle className="text-muted-slate" />
            </div>
            <div className="h-12 w-full">
              <svg className="w-full h-full" viewBox="0 0 100 20">
                <path className="text-soft-graphite/30 sparkline-svg" d="M0 12 Q 25 2, 50 12 T 100 12" fill="none" stroke="currentColor" strokeWidth="1.5" />
              </svg>
            </div>
          </div>
        </section>
        {/* Patient Visualization Placeholder */}
        <section className="h-80 bg-white rounded-xl soft-shadow border border-border-subtle overflow-hidden relative group">
          <img alt="Abstract medical data visualization" className="w-full h-full object-cover opacity-60 mix-blend-multiply" data-alt="An abstract medical data visualization featuring soft, organic 3D fluid forms in muted shades of slate, charcoal, and warm off-white. The shapes intersect gracefully, suggesting complex biological data structures. The lighting is diffused and professional, creating a sense of calm intelligence and high-fidelity technological precision." src="https://lh3.googleusercontent.com/aida-public/AB6AXuCcBuV7crP7RiuxDmlfCCs_UHoY5Gn59T_VKLf57dtUGgs-VGhbkQ-TfFZiWpcDTMYy7La1FBbshjmtgEZwIYWWO4XsvqaI9xqGt2v56igwR_ekkHIkRU_1eWIUJaCg24ET3nG8SUssm9dOf0wPawLmzhxRm-n-5WztYd29x7XuBqgRDokjLZMaC1ArI_79Q1ydOCtWjnD5EQXxkmlP5rvfJgjzEklbE_Nhj2zMOT9evLpvLklZy-tLCzhsWtICpmfS15dgnTgxi0q0" />
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-12 bg-gradient-to-t from-white via-transparent to-transparent">
            <Microscope className="text-4xl mb-4 text-muted-slate" />
            <h3 className="font-headline-sm text-headline-sm text-soft-graphite">Longitudinal Recovery Trend</h3>
            <p className="text-muted-slate font-body-sm text-body-sm mt-2 max-w-sm">Aggregated telemetry data indicates a consistent 12% improvement in mobility markers over the last 24 hours.</p>
          </div>
        </section>
      </div>
    </main>
    {/* Clinical Events Log (Smoke Glass Floating Container) */}
    <aside className="fixed right-8 top-24 w-80 glass-panel rounded-2xl soft-shadow border border-border-subtle/50 p-6 z-30 transition-all hover:shadow-xl">
      <div className="flex justify-between items-center mb-6">
        <h3 className="font-headline-sm text-headline-sm font-semibold text-soft-graphite">Events Log</h3>
        <span className="flex h-2 w-2 rounded-full bg-clinical-error" />
      </div>
      <div className="space-y-6 relative">
        {/* Vertical Timeline Line */}
        <div className="absolute left-[11px] top-2 bottom-2 w-px bg-border-subtle" />
        <div className="flex gap-4 relative">
          <div className="mt-1 h-6 w-6 rounded-full bg-white border border-border-subtle flex items-center justify-center z-10">
            <div className="h-2 w-2 rounded-full bg-soft-graphite" />
          </div>
          <div>
            <p className="font-label-caps text-label-caps text-muted-slate">09:42</p>
            <p className="font-body-sm text-body-sm text-soft-graphite">Vitals synchronized with central telemetry</p>
          </div>
        </div>
        <div className="flex gap-4 relative">
          <div className="mt-1 h-6 w-6 rounded-full bg-white border border-border-subtle flex items-center justify-center z-10">
            <div className="h-2 w-2 rounded-full bg-muted-slate/40" />
          </div>
          <div>
            <p className="font-label-caps text-label-caps text-muted-slate">08:15</p>
            <p className="font-body-sm text-body-sm text-soft-graphite">Potassium lab results updated (3.4 mmol/L)</p>
          </div>
        </div>
        <div className="flex gap-4 relative opacity-60">
          <div className="mt-1 h-6 w-6 rounded-full bg-white border border-border-subtle flex items-center justify-center z-10">
            <div className="h-2 w-2 rounded-full bg-muted-slate/40" />
          </div>
          <div>
            <p className="font-label-caps text-label-caps text-muted-slate">07:00</p>
            <p className="font-body-sm text-body-sm text-soft-graphite">Shift handover complete: Nurse R. Miller</p>
          </div>
        </div>
        <div className="flex gap-4 relative opacity-60">
          <div className="mt-1 h-6 w-6 rounded-full bg-white border border-border-subtle flex items-center justify-center z-10">
            <div className="h-2 w-2 rounded-full bg-muted-slate/40" />
          </div>
          <div>
            <p className="font-label-caps text-label-caps text-muted-slate">06:45</p>
            <p className="font-body-sm text-body-sm text-soft-graphite">Wound dressing inspected; dry and intact</p>
          </div>
        </div>
      </div>
      <button className="w-full mt-6 py-2 text-center text-muted-slate font-label-caps text-label-caps hover:text-soft-graphite transition-colors">
        VIEW FULL FEED
      </button>
    </aside>
    {/* Context Bar Footer */}
    <footer className="h-12 glass-panel border-t border-border-subtle px-8 flex items-center justify-between z-40">
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green-500" />
          <span className="font-label-caps text-label-caps text-muted-slate">SYSTEM ONLINE</span>
        </div>
        <div className="h-4 w-px bg-border-subtle" />
        <span className="font-body-sm text-body-sm text-muted-slate italic">AI Copilot: Monitoring electrolyte drift...</span>
      </div>
      <div className="flex gap-4">
        <span className="font-label-caps text-label-caps text-muted-slate">LATENCY: 14MS</span>
        <span className="font-label-caps text-label-caps text-muted-slate">ENCRYPTION: AES-256</span>
      </div>
    </footer>
  </div>
</div>

    </div>
  );
};
