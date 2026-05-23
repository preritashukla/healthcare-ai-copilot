import React from 'react';
import { ArrowRight, ArrowUp, BarChart2, Bell, Brain, Clock, Heart, Pill, Search, TrendingUp, User, Users } from 'lucide-react';


export const UltraMinimalWorkspace: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#FAFAF9]">
<div>
  {/* TopAppBar */}
  <header className="fixed top-0 left-0 right-0 z-50 bg-transparent pt-8">
    <nav className="flex justify-between items-center max-w-[1200px] mx-auto px-gutter w-full">
      <div className="flex items-center gap-8">
        <span className="font-headline-sm text-headline-sm font-medium tracking-tight text-soft-graphite">Aether AI</span>
        <div className="hidden md:flex gap-6">
          <a className="text-primary font-medium font-label-caps text-label-caps hover:text-primary transition-colors" href="#">Patients</a>
          <a className="text-muted-slate font-label-caps text-label-caps hover:text-primary transition-colors" href="#">Timeline</a>
          <a className="text-muted-slate font-label-caps text-label-caps hover:text-primary transition-colors" href="#">Insights</a>
        </div>
      </div>
      <div className="flex items-center gap-4 text-soft-graphite">
        <Bell className="cursor-pointer hover:opacity-70 transition-opacity" />
        <User className="cursor-pointer hover:opacity-70 transition-opacity" />
      </div>
    </nav>
  </header>
  <main className="main-canvas pt-32">
    {/* Header Section */}
    <section className="mb-16">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="font-display-lg text-display-lg text-primary tracking-tight mb-2">Sarah Jenkins</h1>
          <div className="flex items-center gap-4 text-muted-gray">
            <span className="font-body-lg text-body-lg">42F</span>
            <span className="w-1 h-1 rounded-full bg-outline" />
            <span className="font-body-lg text-body-lg">Post-op Day 2</span>
          </div>
        </div>
        <div className="pb-2">
          <span className="inline-flex items-center px-3 py-1 rounded-full bg-warm-beige text-secondary font-label-caps text-label-caps">
            Active Recovery
          </span>
        </div>
      </div>
    </section>
    {/* Minimal Telemetry */}
    <section className="mb-12 py-4 border-y border-outline-variant/30">
      <div className="flex flex-wrap justify-between gap-8 opacity-70">
        <div className="flex items-center gap-2">
          <Heart className="text-[18px]" />
          <span className="font-label-caps text-label-caps">Vitals: Stable (BP 118/76)</span>
        </div>
        <div className="flex items-center gap-2">
          <TrendingUp className="text-[18px]" />
          <span className="font-label-caps text-label-caps">Risk: 92% Discharge Prob.</span>
        </div>
        <div className="flex items-center gap-2">
          <Pill className="text-[18px]" />
          <span className="font-label-caps text-label-caps">Meds: On Schedule</span>
        </div>
        <div className="flex items-center gap-2">
          <Clock className="text-[18px]" />
          <span className="font-label-caps text-label-caps">Next: Neuro Check 14:00</span>
        </div>
      </div>
    </section>
    {/* Clinical Summary */}
    <article className="mb-16">
      <div className="max-w-[640px]">
        <p className="font-body-lg text-body-lg text-on-surface leading-relaxed mb-6">
          Sarah is recovering exceptionally well following Tuesday's laparoscopic cholecystectomy. Her pain levels have stabilized at a self-reported 2/10 with oral management only.
        </p>
        <p className="font-body-lg text-body-lg text-on-surface leading-relaxed">
          Mobility is increasing; she completed three laps of the ward this morning without assistance. Primary focus for today is monitoring fluid tolerance and preparing discharge education for her family.
        </p>
      </div>
    </article>
    {/* Primary Focus: Copilot Interface */}
    <section className="relative">
      <div className="bg-white rounded-2xl shadow-sm border border-outline-variant/20 p-card-padding">
        <div className="flex items-start gap-4 mb-8">
          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-on-primary">
            <Brain className="text-[20px]" />
          </div>
          <div className="flex-1">
            <div className="font-headline-sm text-headline-sm mb-1">Aether Copilot</div>
            <p className="text-muted-gray font-body-sm text-body-sm">Deep clinical reasoning enabled</p>
          </div>
        </div>
        <div className="space-y-4 mb-8">
          <button className="w-full text-left p-4 rounded-xl hover:bg-warm-beige transition-colors group flex items-center justify-between border border-transparent hover:border-outline-variant/30">
            <span className="font-body-sm text-body-sm text-secondary">"Compare today's lab results with admission baseline."</span>
            <ArrowRight className="opacity-0 group-hover:opacity-100 transition-opacity text-primary" />
          </button>
          <button className="w-full text-left p-4 rounded-xl hover:bg-warm-beige transition-colors group flex items-center justify-between border border-transparent hover:border-outline-variant/30">
            <span className="font-body-sm text-body-sm text-secondary">"Draft a discharge summary for Dr. Aris."</span>
            <ArrowRight className="opacity-0 group-hover:opacity-100 transition-opacity text-primary" />
          </button>
        </div>
        <div className="relative">
          <input className="w-full bg-warm-beige border-none rounded-xl py-4 pl-6 pr-12 focus:ring-1 focus:ring-primary/10 font-body-lg text-body-lg placeholder:text-muted-slate transition-all" placeholder="Ask Copilot about Sarah..." type="text" />
          <button className="absolute right-4 top-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-primary text-on-primary flex items-center justify-center hover:scale-105 active:scale-95 transition-transform">
            <ArrowUp className="text-[20px]" />
          </button>
        </div>
      </div>
    </section>
  </main>
  {/* BottomNavBar (Floating Contextual) */}
  <nav className="fixed bottom-0 left-0 right-0 z-50 flex justify-center pb-safe">
    <div className="bg-smoke-glass backdrop-blur-xl rounded-full mb-8 mx-auto w-max px-6 py-3 shadow-2xl shadow-black/5 flex items-center gap-2">
      <a className="flex flex-row items-center gap-2 text-soft-graphite px-4 py-2 hover:bg-surface-container-highest transition-all duration-300 rounded-full" href="#">
        <Search />
        <span className="font-label-caps text-label-caps">Command</span>
      </a>
      <a className="flex flex-row items-center gap-2 text-soft-graphite px-4 py-2 hover:bg-surface-container-highest transition-all duration-300 rounded-full" href="#">
        <BarChart2 />
        <span className="font-label-caps text-label-caps">Timeline</span>
      </a>
      <a className="flex flex-row items-center gap-2 bg-primary text-on-primary rounded-full px-4 py-2 transition-all duration-300 scale-95 duration-150" href="#">
        <Users />
        <span className="font-label-caps text-label-caps">Patients</span>
      </a>
      <a className="flex flex-row items-center gap-2 text-soft-graphite px-4 py-2 hover:bg-surface-container-highest transition-all duration-300 rounded-full" href="#">
        <Brain />
        <span className="font-label-caps text-label-caps">Insights</span>
      </a>
    </div>
  </nav>
  {/* Background Decorative Element (Subtle Gradient) */}
  <div className="fixed top-0 right-0 w-1/3 h-1/2 bg-gradient-to-bl from-warm-beige/30 to-transparent pointer-events-none -z-10" />
</div>

    </div>
  );
};
