import React from 'react';
import { BarChart2, Bell, HelpCircle, Plus, Printer, Share2 } from 'lucide-react';


export const SoftPremiumWorkspace: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#FAFAF9]">
<div>
  {/* Integrated Navigation */}
  <header className="w-full pt-12 pb-6 px-margin-page">
    <div className="max-w-content-max-width mx-auto flex justify-between items-center">
      <div className="flex items-center gap-12">
        <span className="text-headline-sm font-bold tracking-tight text-soft-slate">ClinicalOS</span>
        <nav className="hidden md:flex items-center gap-8">
          <a className="text-body-sm font-medium text-on-surface-variant hover:text-soft-slate transition-colors" href="#">Dashboard</a>
          <a className="text-body-sm font-semibold text-soft-slate border-b-2 border-soft-slate pb-0.5" href="#">Patients</a>
          <a className="text-body-sm font-medium text-on-surface-variant hover:text-soft-slate transition-colors" href="#">Schedules</a>
        </nav>
      </div>
      <div className="flex items-center gap-6">
        <Bell className="text-on-surface-variant cursor-pointer hover:text-soft-slate" />
        <div className="w-9 h-9 rounded-full overflow-hidden border border-outline-soft">
          <img alt="Provider" className="w-full h-full object-cover" src="https://lh3.googleusercontent.com/aida-public/AB6AXuC-q3IvDnbn8SxmAEHdyjdoR0A6v_JuUGRVDwkH7HwlnwyrHl2XLQcHg10oV3YmELdhnmQFAM9DkRuIKku_Y96C1hKYGuYsMOANnUsqpFkdddPWF0kCp36O79JTT3a03ifuDiu10_ufrmlmbv2sWqiDqBsqqQCXbiIYNcIQIs9phlUhNeRr3dFpHFADnPZ9zsYgIO3Pwfpjv6syoLDXvJ_mPTDIULIft6DvNqkVZSKop9vKaFgvMfcMZbBfNRARc1DYyuS-AvS935UP" />
        </div>
      </div>
    </div>
  </header>
  <main className="max-w-content-max-width mx-auto px-6 pb-24 fade-in">
    {/* Patient Header */}
    <section className="mb-16 mt-8 flex flex-col md:flex-row md:items-end justify-between gap-6">
      <div className="flex items-center gap-6">
        <div className="w-20 h-20 rounded-2xl overflow-hidden border border-outline-soft">
          <img alt="Sarah Jenkins" className="w-full h-full object-cover" src="https://lh3.googleusercontent.com/aida-public/AB6AXuA56kuiIQWRr_-wchEdLJ29lW22XbyVWcYgtRRK7d3f3vmv1zT5luCE1IM3HzXnHn2JLQXXYOUY9AW-nWt_zutsUo_eQ61oZlLnj7OpBru3jjm1bH7Cx7QmtjJVm4D6CIMTbI3h87dv1nxws_GKZSE4mm5sKsKWVsGC3sFOM1hEbEaiiJrUaoqmzM8YDVbOU4CdnE1sXJdXoU9VO5Y_RJgTA6ngnAwi2Ajs70nh6KqLMNZIZ5m42r4U5m4tyPSF49lWtBIJdSpLW30r" />
        </div>
        <div>
          <nav className="text-label-caps text-on-surface-variant uppercase mb-2">Patient • Active Care</nav>
          <h1 className="text-display-lg font-bold text-soft-slate">Sarah Jenkins, 42F</h1>
          <div className="flex gap-4 mt-2">
            <span className="text-body-sm text-on-surface-variant">MRN: 884-219-00</span>
            <span className="text-body-sm font-semibold text-muted-sage">• Room 402, North Wing</span>
          </div>
        </div>
      </div>
      <button className="bg-muted-sage text-white px-6 py-3 rounded-full text-body-sm font-semibold hover:opacity-95 transition-opacity flex items-center gap-2 soft-shadow">
        <Plus className="!text-[20px]" /> New Analysis
      </button>
    </section>
    {/* Central Workspace */}
    <div className="space-y-card-gap">
      {/* Conversational Area */}
      <section className="bg-surface-ivory rounded-2xl p-10 soft-shadow border border-outline-soft">
        <div className="flex items-center gap-3 mb-8">
          <HelpCircle className="text-desaturated-blue" />
          <h3 className="text-headline-sm font-semibold text-soft-slate">Clinical Intelligence Summary</h3>
        </div>
        <div className="space-y-6 max-w-[720px]">
          <p className="text-body-lg text-soft-slate leading-relaxed">
            The patient is showing an exceptionally stable recovery trajectory following her successful <span className="font-semibold border-b border-desaturated-blue/30">robotic-assisted cholecystectomy</span> performed 48 hours ago. Surgical sites appear clean, dry, and intact with no signs of erythema or drainage.
          </p>
          <p className="text-body-lg text-soft-slate leading-relaxed">
            Pain management has been effectively transitioned to oral analgesics, with the patient reporting a current pain score of <span className="text-muted-sage font-bold">2/10</span>. Normal gastrointestinal motility has returned.
          </p>
          <div className="mt-12 pt-8 border-t border-outline-soft">
            <span className="text-label-caps text-on-surface-variant uppercase">Recommended Next Steps</span>
            <ul className="mt-4 space-y-3">
              <li className="flex items-start gap-3 text-body-sm text-soft-slate">
                <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-muted-sage flex-shrink-0" />
                Finalize electrolyte panel review for discharge clearance.
              </li>
              <li className="flex items-start gap-3 text-body-sm text-soft-slate">
                <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-muted-sage flex-shrink-0" />
                Consult physical therapy for final mobility assessment.
              </li>
            </ul>
          </div>
        </div>
      </section>
      {/* Minimal Insights Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-card-gap">
        {/* Textual Telemetry */}
        <section className="bg-surface-ivory/50 rounded-2xl p-8 border border-outline-soft/60">
          <div className="flex justify-between items-center mb-6">
            <h4 className="text-label-caps text-on-surface-variant uppercase">Key Telemetry</h4>
            <span className="text-[11px] text-on-surface-variant/70 italic">Updated 2m ago</span>
          </div>
          <div className="space-y-5">
            <div className="flex justify-between items-end">
              <span className="text-body-sm">Heart Rate</span>
              <div className="text-right">
                <span className="text-headline-sm font-bold">72</span> <span className="text-body-sm text-on-surface-variant">bpm</span>
                <div className="text-[11px] text-muted-sage font-medium">Stable</div>
              </div>
            </div>
            <div className="flex justify-between items-end">
              <span className="text-body-sm">SpO2</span>
              <div className="text-right">
                <span className="text-headline-sm font-bold">98</span> <span className="text-body-sm text-on-surface-variant">%</span>
                <div className="text-[11px] text-muted-sage font-medium">Optimal</div>
              </div>
            </div>
            <div className="flex justify-between items-end">
              <span className="text-body-sm">Mean BP</span>
              <div className="text-right">
                <span className="text-headline-sm font-bold">94</span> <span className="text-body-sm text-on-surface-variant">mmHg</span>
                <div className="text-[11px] text-muted-sage font-medium">Normal</div>
              </div>
            </div>
          </div>
        </section>
        {/* Risk Summary */}
        <section className="bg-surface-ivory/50 rounded-2xl p-8 border border-outline-soft/60">
          <div className="flex justify-between items-center mb-6">
            <h4 className="text-label-caps text-on-surface-variant uppercase">Risk Analysis</h4>
            <BarChart2 className="text-desaturated-blue" />
          </div>
          <div className="space-y-6">
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-body-sm">Discharge Probability</span>
                <span className="text-body-sm font-bold text-muted-sage">92%</span>
              </div>
              <div className="w-full bg-outline-soft/40 h-1 rounded-full">
                <div className="bg-muted-sage h-full rounded-full" style={{width: '92%'}} />
              </div>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-body-sm">Infection Risk</span>
                <span className="text-body-sm font-medium">Low (3%)</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-body-sm">Electrolyte Stability</span>
                <span className="text-body-sm font-medium">Stable</span>
              </div>
            </div>
          </div>
        </section>
      </div>
      {/* Metadata / Footer section */}
      <section className="pt-12 grid grid-cols-1 md:grid-cols-3 gap-8 border-t border-outline-soft/30">
        <div>
          <span className="text-label-caps text-on-surface-variant uppercase">Admitting Team</span>
          <p className="mt-2 text-body-sm font-medium">Dr. Aris Thorne, MD</p>
          <p className="text-[13px] text-on-surface-variant">Surgical Oncology</p>
        </div>
        <div>
          <span className="text-label-caps text-on-surface-variant uppercase">Critical Alerts</span>
          <p className="mt-2 text-body-sm font-bold text-error-muted">Penicillin Allergy (Severe)</p>
        </div>
        <div className="flex md:justify-end items-center gap-4">
          <button className="text-desaturated-blue hover:text-soft-slate text-body-sm font-medium transition-colors flex items-center gap-2">
            <Printer /> Print
          </button>
          <button className="text-desaturated-blue hover:text-soft-slate text-body-sm font-medium transition-colors flex items-center gap-2">
            <Share2 /> Share
          </button>
        </div>
      </section>
    </div>
  </main>
  <footer className="w-full border-t border-outline-soft/30 py-12 px-margin-page">
    <div className="max-w-content-max-width mx-auto flex flex-col md:flex-row justify-between items-center gap-6">
      <span className="text-[13px] text-on-surface-variant">
        © 2024 ClinicalOS • High Fidelity Clinical Intelligence
      </span>
      <div className="flex gap-8">
        <a className="text-[13px] text-on-surface-variant hover:text-soft-slate" href="#">Privacy</a>
        <a className="text-[13px] text-on-surface-variant hover:text-soft-slate" href="#">HIPAA Compliance</a>
        <a className="text-[13px] text-on-surface-variant hover:text-soft-slate" href="#">Support</a>
      </div>
    </div>
  </footer>
</div>

    </div>
  );
};
