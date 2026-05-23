import React from 'react';
import { BarChart2, Bell, HelpCircle, Plus, Printer, Search, Share2 } from 'lucide-react';


export const RefinedTypographyWorkspace: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#FAFAF9]">
<div>
  {/* Global Navigation */}
  <header className="w-full pt-12 pb-12 px-margin-page">
    <div className="max-w-content-max-width mx-auto flex justify-between items-center">
      <div className="flex items-center gap-12">
        <span className="text-headline-sm font-medium text-soft-slate">ClinicalOS</span>
        <nav className="hidden md:flex items-center gap-10">
          <a className="text-body-sm font-medium text-on-surface-variant hover:text-soft-slate transition-colors" href="#">Dashboard</a>
          <a className="text-body-sm font-medium text-soft-slate border-b border-soft-slate pb-1" href="#">Patients</a>
          <a className="text-body-sm font-medium text-on-surface-variant hover:text-soft-slate transition-colors" href="#">Schedules</a>
        </nav>
      </div>
      <div className="flex items-center gap-8">
        <Bell className="text-on-surface-variant cursor-pointer hover:text-soft-slate" />
        <div className="w-10 h-10 rounded-full overflow-hidden border border-outline-soft">
          <img alt="Provider" className="w-full h-full object-cover" src="https://lh3.googleusercontent.com/aida-public/AB6AXuC-q3IvDnbn8SxmAEHdyjdoR0A6v_JuUGRVDwkH7HwlnwyrHl2XLQcHg10oV3YmELdhnmQFAM9DkRuIKku_Y96C1hKYGuYsMOANnUsqpFkdddPWF0kCp36O79JTT3a03ifuDiu10_ufrmlmbv2sWqiDqBsqqQCXbiIYNcIQIs9phlUhNeRr3dFpHFADnPZ9zsYgIO3Pwfpjv6syoLDXvJ_mPTDIULIft6DvNqkVZSKop9vKaFgvMfcMZbBfNRARc1DYyuS-AvS935UP" />
        </div>
      </div>
    </div>
  </header>
  <main className="max-w-content-max-width mx-auto px-margin-page pb-24 fade-in">
    {/* Patient Header */}
    <section className="mb-20 flex flex-col md:flex-row md:items-end justify-between gap-8">
      <div className="flex items-center gap-8">
        <div className="w-24 h-24 rounded-2xl overflow-hidden border border-outline-soft ultra-soft-shadow">
          <img alt="Sarah Jenkins" className="w-full h-full object-cover" src="https://lh3.googleusercontent.com/aida-public/AB6AXuA56kuiIQWRr_-wchEdLJ29lW22XbyVWcYgtRRK7d3f3vmv1zT5luCE1IM3HzXnHn2JLQXXYOUY9AW-nWt_zutsUo_eQ61oZlLnj7OpBru3jjm1bH7Cx7QmtjJVm4D6CIMTbI3h87dv1nxws_GKZSE4mm5sKsKWVsGC3sFOM1hEbEaiiJrUaoqmzM8YDVbOU4CdnE1sXJdXoU9VO5Y_RJgTA6ngnAwi2Ajs70nh6KqLMNZIZ5m42r4U5m4tyPSF49lWtBIJdSpLW30r" />
        </div>
        <div>
          <nav className="text-label-caps text-on-surface-variant mb-3 uppercase">Patient • Active Care</nav>
          <h1 className="text-display-lg font-medium text-soft-slate">Sarah Jenkins, 42F</h1>
          <div className="flex gap-6 mt-3">
            <span className="text-body-sm text-on-surface-variant">MRN: 884-219-00</span>
            <span className="text-body-sm font-medium text-muted-sage">Room 402, North Wing</span>
          </div>
        </div>
      </div>
      <button className="bg-muted-sage text-white px-8 py-4 rounded-full text-body-sm font-medium hover:opacity-90 transition-opacity flex items-center gap-2 ultra-soft-shadow">
        <Plus className="!text-[20px]" /> New Analysis
      </button>
    </section>
    {/* Central Workspace Content */}
    <div className="space-y-card-gap">
      {/* Floating Command Palette Simulation (Integrated Search) */}
      <div className="relative max-w-[500px] mx-auto mb-16">
        <div className="flex items-center gap-4 px-6 py-4 bg-surface-ivory rounded-full border border-outline-soft ultra-soft-shadow">
          <Search className="text-on-surface-variant" />
          <input className="bg-transparent border-none focus:ring-0 text-body-sm w-full placeholder:text-on-surface-variant/50" placeholder="Search clinical history or ask a question..." type="text" />
          <span className="text-[10px] text-on-surface-variant/50 border border-outline-soft rounded px-1.5 py-0.5">⌘K</span>
        </div>
      </div>
      {/* Conversational Intelligence Area */}
      <section className="bg-surface-ivory rounded-2xl p-12 ultra-soft-shadow border border-outline-soft/40">
        <div className="flex items-center gap-4 mb-10">
          <HelpCircle className="text-desaturated-blue" />
          <h3 className="text-headline-md font-medium text-soft-slate">Clinical Intelligence Summary</h3>
        </div>
        <div className="space-y-8 max-w-[700px]">
          <p className="text-body-lg text-soft-slate">
            The patient is showing an exceptionally stable recovery trajectory following her successful <span className="font-medium text-desaturated-blue decoration-desaturated-blue/20 underline underline-offset-4">robotic-assisted cholecystectomy</span> performed 48 hours ago. Surgical sites appear clean, dry, and intact with no signs of erythema or drainage.
          </p>
          <p className="text-body-lg text-soft-slate">
            Pain management has been effectively transitioned to oral analgesics, with the patient reporting a current pain score of <span className="text-muted-sage font-medium">2/10</span>. Normal gastrointestinal motility has returned.
          </p>
          <div className="mt-16 pt-10 border-t border-outline-soft/50">
            <span className="text-label-caps text-on-surface-variant uppercase">Recommended Next Steps</span>
            <ul className="mt-6 space-y-4">
              <li className="flex items-start gap-4 text-body-sm text-soft-slate">
                <span className="mt-2.5 w-1 h-1 rounded-full bg-muted-sage flex-shrink-0" />
                Finalize electrolyte panel review for discharge clearance.
              </li>
              <li className="flex items-start gap-4 text-body-sm text-soft-slate">
                <span className="mt-2.5 w-1 h-1 rounded-full bg-muted-sage flex-shrink-0" />
                Consult physical therapy for final mobility assessment.
              </li>
            </ul>
          </div>
        </div>
      </section>
      {/* Focused Insights Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-gutter">
        {/* Telemetry */}
        <section className="bg-surface-ivory/60 rounded-2xl p-10 border border-outline-soft/40 ultra-soft-shadow">
          <div className="flex justify-between items-center mb-10">
            <h4 className="text-label-caps text-on-surface-variant uppercase">Key Telemetry</h4>
            <span className="text-[11px] text-on-surface-variant/60">Updated 2m ago</span>
          </div>
          <div className="space-y-8">
            <div className="flex justify-between items-center">
              <span className="text-body-sm text-on-surface-variant">Heart Rate</span>
              <div className="text-right">
                <span className="text-headline-sm text-soft-slate">72</span> <span className="text-[12px] text-on-surface-variant">bpm</span>
              </div>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-body-sm text-on-surface-variant">SpO2</span>
              <div className="text-right">
                <span className="text-headline-sm text-soft-slate">98</span> <span className="text-[12px] text-on-surface-variant">%</span>
              </div>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-body-sm text-on-surface-variant">Mean BP</span>
              <div className="text-right">
                <span className="text-headline-sm text-soft-slate">94</span> <span className="text-[12px] text-on-surface-variant">mmHg</span>
              </div>
            </div>
          </div>
        </section>
        {/* Risk Profile */}
        <section className="bg-surface-ivory/60 rounded-2xl p-10 border border-outline-soft/40 ultra-soft-shadow">
          <div className="flex justify-between items-center mb-10">
            <h4 className="text-label-caps text-on-surface-variant uppercase">Risk Profile</h4>
            <BarChart2 className="text-desaturated-blue" />
          </div>
          <div className="space-y-10">
            <div>
              <div className="flex justify-between items-center mb-4">
                <span className="text-body-sm text-on-surface-variant">Discharge Probability</span>
                <span className="text-body-sm font-medium text-muted-sage">92%</span>
              </div>
              <div className="w-full bg-outline-soft/40 h-1 rounded-full">
                <div className="bg-muted-sage h-full rounded-full" style={{width: '92%'}} />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <span className="block text-[11px] text-on-surface-variant uppercase mb-1">Infection</span>
                <span className="text-body-sm font-medium">Low (3%)</span>
              </div>
              <div>
                <span className="block text-[11px] text-on-surface-variant uppercase mb-1">Stability</span>
                <span className="text-body-sm font-medium">Optimal</span>
              </div>
            </div>
          </div>
        </section>
      </div>
      {/* Footer Metadata */}
      <section className="pt-16 grid grid-cols-1 md:grid-cols-3 gap-12 border-t border-outline-soft/30">
        <div>
          <span className="text-label-caps text-on-surface-variant uppercase">Admitting Team</span>
          <p className="mt-3 text-body-sm font-medium text-soft-slate">Dr. Aris Thorne, MD</p>
          <p className="text-[13px] text-on-surface-variant">Surgical Oncology</p>
        </div>
        <div>
          <span className="text-label-caps text-on-surface-variant uppercase">Critical Alerts</span>
          <p className="mt-3 text-body-sm font-medium text-error-muted">Penicillin Allergy (Severe)</p>
        </div>
        <div className="flex md:justify-end items-center gap-8">
          <button className="text-desaturated-blue hover:text-soft-slate text-body-sm font-medium transition-colors flex items-center gap-2">
            <Printer className="!text-[18px]" /> Print
          </button>
          <button className="text-desaturated-blue hover:text-soft-slate text-body-sm font-medium transition-colors flex items-center gap-2">
            <Share2 className="!text-[18px]" /> Share
          </button>
        </div>
      </section>
    </div>
  </main>
  <footer className="w-full py-16 px-margin-page border-t border-outline-soft/20">
    <div className="max-w-content-max-width mx-auto flex flex-col md:flex-row justify-between items-center gap-8">
      <span className="text-[12px] text-on-surface-variant/60">
        © 2024 ClinicalOS • High Fidelity Clinical Intelligence
      </span>
      <div className="flex gap-10">
        <a className="text-[12px] text-on-surface-variant/60 hover:text-soft-slate" href="#">Privacy</a>
        <a className="text-[12px] text-on-surface-variant/60 hover:text-soft-slate" href="#">HIPAA</a>
        <a className="text-[12px] text-on-surface-variant/60 hover:text-soft-slate" href="#">Support</a>
      </div>
    </div>
  </footer>
</div>

    </div>
  );
};
