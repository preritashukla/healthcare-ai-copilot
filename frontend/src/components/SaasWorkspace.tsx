import React from 'react';
import { ArrowRight, ArrowUp, BarChart2, Bell, ChevronRight, Heart, HelpCircle, Printer, Share2, Wind } from 'lucide-react';


export const SaasWorkspace: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#FAFAF9]">
<div>
  {/* Top Navigation Bar */}
  <header className="w-full h-16 bg-soft-off-white dark:bg-background sticky top-0 z-50">
    <div className="flex justify-between items-center max-w-[1200px] mx-auto px-margin-desktop w-full h-full">
      <div className="flex items-center gap-8">
        <span className="text-headline-md font-headline-md text-primary dark:text-on-primary-fixed">ClinicalOS</span>
        <nav className="hidden md:flex items-center gap-6">
          <a className="text-muted-gray dark:text-on-surface-variant font-body-lg text-body-lg hover:text-primary transition-colors duration-200 cursor-pointer active:opacity-70" href="#">Dashboard</a>
          <a className="text-primary dark:text-on-primary-fixed font-bold border-b-2 border-primary dark:border-on-primary-fixed pb-1 font-body-lg text-body-lg cursor-pointer active:opacity-70" href="#">Patients</a>
          <a className="text-muted-gray dark:text-on-surface-variant font-body-lg text-body-lg hover:text-primary transition-colors duration-200 cursor-pointer active:opacity-70" href="#">Schedules</a>
          <a className="text-muted-gray dark:text-on-surface-variant font-body-lg text-body-lg hover:text-primary transition-colors duration-200 cursor-pointer active:opacity-70" href="#">Reports</a>
        </nav>
      </div>
      <div className="flex items-center gap-4">
        <div className="hidden md:flex items-center gap-2">
          <Bell className="text-muted-gray cursor-pointer hover:text-primary" />
          <HelpCircle className="text-muted-gray cursor-pointer hover:text-primary" />
        </div>
        <button className="bg-primary text-on-primary px-4 py-2 rounded-lg font-body-sm text-body-sm font-semibold hover:opacity-90 transition-opacity">New Consultation</button>
        <div className="w-8 h-8 rounded-full overflow-hidden border border-outline">
          <img alt="Provider Profile" className="w-full h-full object-cover" data-alt="A professional medical provider headshot in a bright, clean clinic environment. The lighting is soft and natural, emphasizing a trustworthy and approachable persona. The background is slightly blurred with soft warm tones and clinical white surfaces, matching a high-end minimalist SaaS aesthetic." src="https://lh3.googleusercontent.com/aida-public/AB6AXuC-q3IvDnbn8SxmAEHdyjdoR0A6v_JuUGRVDwkH7HwlnwyrHl2XLQcHg10oV3YmELdhnmQFAM9DkRuIKku_Y96C1hKYGuYsMOANnUsqpFkdddPWF0kCp36O79JTT3a03ifuDiu10_ufrmlmbv2sWqiDqBsqqQCXbiIYNcIQIs9phlUhNeRr3dFpHFADnPZ9zsYgIO3Pwfpjv6syoLDXvJ_mPTDIULIft6DvNqkVZSKop9vKaFgvMfcMZbBfNRARc1DYyuS-AvS935UP" />
        </div>
      </div>
    </div>
  </header>
  <main className="max-w-[1200px] mx-auto px-margin-desktop py-margin-desktop">
    {/* Breadcrumb / Header Actions */}
    <div className="flex justify-between items-end mb-8">
      <div>
        <nav className="flex items-center gap-2 mb-2 text-muted-gray font-label-caps text-label-caps uppercase tracking-wider">
          <span>Patients</span>
          <ChevronRight className="text-[14px]" />
          <span className="text-primary font-bold">Active Care</span>
        </nav>
        <h1 className="font-display-lg text-display-lg text-primary">Patient Workspace</h1>
      </div>
      <div className="flex gap-3">
        <button className="bg-warm-beige text-secondary px-4 py-2 rounded-xl font-body-sm text-body-sm font-semibold hover:bg-surface-container-highest transition-colors flex items-center gap-2">
          <Printer className="text-[20px]" /> Print Summary
        </button>
        <button className="bg-primary text-on-primary px-4 py-2 rounded-xl font-body-sm text-body-sm font-semibold hover:opacity-90 transition-opacity flex items-center gap-2">
          <Share2 className="text-[20px]" /> Share Rounds
        </button>
      </div>
    </div>
    {/* Dashboard Grid Layout */}
    <div className="grid grid-cols-12 gap-gutter">
      {/* 1. Patient Profile Card */}
      <div className="col-span-12 lg:col-span-4 bg-white rounded-2xl p-card-padding soft-elevation card-transition border border-outline-variant/30">
        <div className="flex items-start gap-4 mb-6">
          <div className="w-20 h-20 rounded-2xl overflow-hidden shadow-sm">
            <img alt="Sarah Jenkins" className="w-full h-full object-cover" data-alt="A warm and clear portrait of a 42-year-old woman with a friendly expression. The photograph uses soft, natural window lighting in a high-key clinical environment. The style is professional and empathetic, with a color palette of soft whites, warm beiges, and subtle charcoal accents that align with a premium healthcare application." src="https://lh3.googleusercontent.com/aida-public/AB6AXuA56kuiIQWRr_-wchEdLJ29lW22XbyVWcYgtRRK7d3f3vmv1zT5luCE1IM3HzXnHn2JLQXXYOUY9AW-nWt_zutsUo_eQ61oZlLnj7OpBru3jjm1bH7Cx7QmtjJVm4D6CIMTbI3h87dv1nxws_GKZSE4mm5sKsKWVsGC3sFOM1hEbEaiiJrUaoqmzM8YDVbOU4CdnE1sXJdXoU9VO5Y_RJgTA6ngnAwi2Ajs70nh6KqLMNZIZ5m42r4U5m4tyPSF49lWtBIJdSpLW30r" />
          </div>
          <div>
            <h2 className="font-headline-md text-headline-md text-primary mb-1">Sarah Jenkins, 42F</h2>
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-label-caps font-label-caps bg-surface-container-low text-secondary border border-outline-variant">MRN: 884-219-00</span>
          </div>
        </div>
        <div className="space-y-4">
          <div className="flex justify-between items-center py-3 border-b border-outline-variant/20">
            <span className="font-body-sm text-body-sm text-muted-gray">Current Status</span>
            <span className="font-headline-sm text-headline-sm text-primary">Post-Op Day 2</span>
          </div>
          <div className="flex justify-between items-center py-3 border-b border-outline-variant/20">
            <span className="font-body-sm text-body-sm text-muted-gray">Location</span>
            <span className="font-headline-sm text-headline-sm text-primary">Room 402, North Wing</span>
          </div>
          <div className="flex justify-between items-center py-3 border-b border-outline-variant/20">
            <span className="font-body-sm text-body-sm text-muted-gray">Admitting Physician</span>
            <span className="font-headline-sm text-headline-sm text-primary">Dr. Aris Thorne</span>
          </div>
          <div className="flex justify-between items-center py-3">
            <span className="font-body-sm text-body-sm text-muted-gray">Allergies</span>
            <span className="font-headline-sm text-headline-sm text-error">Penicillin (Severe)</span>
          </div>
        </div>
      </div>
      {/* 2. AI Clinical Summary Card */}
      <div className="col-span-12 lg:col-span-8 bg-white rounded-2xl p-card-padding soft-elevation card-transition border border-outline-variant/30 flex flex-col">
        <div className="flex items-center gap-2 mb-6">
          <HelpCircle className="text-primary" />
          <h3 className="font-headline-md text-headline-md text-primary">Clinical Intelligence Summary</h3>
        </div>
        <div className="flex-grow">
          <p className="font-body-lg text-body-lg text-secondary leading-relaxed mb-6">
            The patient is showing a exceptionally stable recovery trajectory following her successful <span className="font-semibold text-primary">robotic-assisted cholecystectomy</span> performed 48 hours ago. Surgical sites appear clean, dry, and intact with no signs of erythema or drainage. 
          </p>
          <p className="font-body-lg text-body-lg text-secondary leading-relaxed mb-6">
            Pain management has been effectively transitioned to oral analgesics, with the patient reporting a current pain score of 2/10. Normal gastrointestinal motility has returned, and she is tolerating a soft diet well. No respiratory complications or febrile episodes noted during the overnight shift.
          </p>
          <div className="bg-warm-beige/50 rounded-xl p-4 border border-outline-variant/20">
            <h4 className="font-label-caps text-label-caps text-muted-gray mb-2 uppercase tracking-widest">Recommended Next Steps</h4>
            <ul className="space-y-2">
              <li className="flex items-center gap-2 font-body-sm text-body-sm text-secondary">
                <span className="w-1.5 h-1.5 rounded-full bg-primary" /> Finalize electrolyte panel review for discharge clearance.
              </li>
              <li className="flex items-center gap-2 font-body-sm text-body-sm text-secondary">
                <span className="w-1.5 h-1.5 rounded-full bg-primary" /> Consult physical therapy for final mobility assessment.
              </li>
            </ul>
          </div>
        </div>
      </div>
      {/* 3. Telemetry Metrics Card */}
      <div className="col-span-12 lg:col-span-7 bg-white rounded-2xl p-card-padding soft-elevation card-transition border border-outline-variant/30">
        <div className="flex justify-between items-center mb-6">
          <h3 className="font-headline-md text-headline-md text-primary">Telemetry Metrics</h3>
          <span className="text-body-sm text-muted-gray">Live Update: 2m ago</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Metric Block: Heart Rate */}
          <div className="bg-warm-beige p-5 rounded-2xl border border-outline-variant/10">
            <div className="flex justify-between items-start mb-3">
              <span className="font-label-caps text-label-caps text-muted-gray uppercase">Heart Rate</span>
              <Heart className="text-secondary text-[20px]" />
            </div>
            <div className="flex items-baseline gap-1">
              <span className="font-display-lg text-[36px] font-bold text-primary leading-none">72</span>
              <span className="font-body-sm text-body-sm text-muted-gray">bpm</span>
            </div>
            <div className="mt-3 flex items-center gap-1 text-[12px] text-green-600 font-medium">
              <ArrowRight className="text-[16px]" /> Stable
            </div>
          </div>
          {/* Metric Block: SpO2 */}
          <div className="bg-warm-beige p-5 rounded-2xl border border-outline-variant/10">
            <div className="flex justify-between items-start mb-3">
              <span className="font-label-caps text-label-caps text-muted-gray uppercase">SpO2</span>
              <Wind className="text-secondary text-[20px]" />
            </div>
            <div className="flex items-baseline gap-1">
              <span className="font-display-lg text-[36px] font-bold text-primary leading-none">98</span>
              <span className="font-body-sm text-body-sm text-muted-gray">%</span>
            </div>
            <div className="mt-3 flex items-center gap-1 text-[12px] text-green-600 font-medium">
              <ArrowUp className="text-[16px]" /> Normal Range
            </div>
          </div>
          {/* Metric Block: Mean BP */}
          <div className="bg-warm-beige p-5 rounded-2xl border border-outline-variant/10">
            <div className="flex justify-between items-start mb-3">
              <span className="font-label-caps text-label-caps text-muted-gray uppercase">Mean BP</span>
              <HelpCircle className="text-secondary text-[20px]" />
            </div>
            <div className="flex items-baseline gap-1">
              <span className="font-display-lg text-[36px] font-bold text-primary leading-none">94</span>
              <span className="font-body-sm text-body-sm text-muted-gray">mmHg</span>
            </div>
            <div className="mt-3 flex items-center gap-1 text-[12px] text-green-600 font-medium">
              <ArrowRight className="text-[16px]" /> Optimal
            </div>
          </div>
        </div>
      </div>
      {/* 4. Risk Assessment Card */}
      <div className="col-span-12 lg:col-span-5 bg-white rounded-2xl p-card-padding soft-elevation card-transition border border-outline-variant/30">
        <div className="flex items-center gap-2 mb-6">
          <BarChart2 className="text-primary" />
          <h3 className="font-headline-md text-headline-md text-primary">AI Risk Insights</h3>
        </div>
        <div className="space-y-6">
          <div>
            <div className="flex justify-between items-end mb-2">
              <span className="font-body-sm text-body-sm text-secondary">Discharge Probability</span>
              <span className="font-headline-sm text-headline-sm text-primary">92%</span>
            </div>
            <div className="w-full bg-surface-container-low rounded-full h-2 overflow-hidden">
              <div className="bg-primary h-full rounded-full" style={{width: '92%'}} />
            </div>
            <p className="mt-2 font-body-sm text-body-sm text-muted-gray">High confidence for discharge within 24 hours based on mobility and pain scores.</p>
          </div>
          <div className="p-4 bg-warm-beige/30 rounded-xl border border-outline-variant/20">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 rounded-full bg-green-500" />
                <span className="font-body-sm text-body-sm text-secondary">Electrolyte stability</span>
              </div>
              <span className="font-headline-sm text-headline-sm text-primary">Optimal</span>
            </div>
          </div>
          <div className="p-4 bg-warm-beige/30 rounded-xl border border-outline-variant/20">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 rounded-full bg-green-500" />
                <span className="font-body-sm text-body-sm text-secondary">Post-Op Infection Risk</span>
              </div>
              <span className="font-headline-sm text-headline-sm text-primary">Low (3%)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </main>
  {/* Footer */}
  <footer className="w-full mt-12 bg-warm-beige dark:bg-surface-container">
    <div className="flex flex-col md:flex-row justify-between items-center max-w-[1200px] mx-auto px-margin-desktop py-8 w-full">
      <span className="font-body-sm text-body-sm text-secondary dark:text-on-secondary-container mb-4 md:mb-0">
        © 2024 ClinicalOS. All rights reserved. System Status: Operational
      </span>
      <div className="flex gap-6">
        <a className="font-body-sm text-body-sm text-secondary dark:text-on-secondary-container hover:underline transition-all cursor-pointer" href="#">Privacy Policy</a>
        <a className="font-body-sm text-body-sm text-secondary dark:text-on-secondary-container hover:underline transition-all cursor-pointer" href="#">Terms of Service</a>
        <a className="font-body-sm text-body-sm text-secondary dark:text-on-secondary-container hover:underline transition-all cursor-pointer" href="#">HIPAA Compliance</a>
        <a className="font-body-sm text-body-sm text-secondary dark:text-on-secondary-container hover:underline transition-all cursor-pointer" href="#">Contact Support</a>
      </div>
    </div>
  </footer>
</div>

    </div>
  );
};
