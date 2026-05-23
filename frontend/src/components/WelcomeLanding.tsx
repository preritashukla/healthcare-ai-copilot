import React, { useRef, useEffect, type KeyboardEvent } from 'react';
import { Sun, Moon, Settings, UserSearch, Sparkles, FileText, ArrowRight, Send, Loader2 } from 'lucide-react';
import { useBackendStatus } from '../hooks/useBackendStatus';
import { useRagQuery } from '../hooks/useRagQuery';

interface WelcomeLandingProps {
  isDark?: boolean;
  toggleTheme?: () => void;
}

const SUGGESTED_PROMPTS = [
  'Does the patient have any known drug allergies?',
  'Summarize recent vital sign changes.',
  'What medications are currently prescribed?',
  'Highlight any active clinical risks.',
];

export const WelcomeLanding: React.FC<WelcomeLandingProps> = ({ isDark, toggleTheme }) => {
  const { isOnline } = useBackendStatus();
  const { query, setQuery, submit, response, isLoading, isMock } = useRagQuery();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const responseRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
  }, [query]);

  useEffect(() => {
    if (response && responseRef.current) {
      responseRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [response]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const handleSuggestedPrompt = (prompt: string) => {
    setQuery(prompt);
    textareaRef.current?.focus();
  };

  return (
    <div className="min-h-screen bg-theme-ambient transition-colors duration-300">
      <div className="flex flex-col min-h-screen">

        {/* Header */}
        <header className="w-full top-0 bg-theme-ambient transition-colors duration-300 ease-in-out border-b border-theme-outline/50">
          <div className="flex justify-between items-center w-full px-8 md:px-12 py-6 max-w-7xl mx-auto">
            <div className="flex items-center gap-3">
              <span className="font-headline-md text-headline-md text-theme-text-primary tracking-tight">Copilot OS</span>
              <div
                className={`w-2 h-2 rounded-full transition-colors duration-700 ${
                  isOnline === null
                    ? 'bg-theme-outline animate-pulse'
                    : isOnline
                    ? 'bg-emerald-400'
                    : 'bg-theme-outline'
                }`}
                title={isOnline === null ? 'Checking backend\u2026' : isOnline ? 'Backend online' : 'Backend offline \u2014 using demo data'}
              />
            </div>
            <div className="flex items-center gap-6">
              <button
                onClick={toggleTheme}
                className="text-theme-text-secondary hover:text-theme-text-primary transition-colors cursor-pointer active:scale-95"
                aria-label="Toggle theme"
              >
                {isDark ? <Sun className="w-6 h-6" /> : <Moon className="w-6 h-6" />}
              </button>
              <button
                className="text-theme-text-secondary hover:text-theme-text-primary transition-colors cursor-pointer active:scale-95"
                aria-label="Settings"
              >
                <Settings className="w-6 h-6" />
              </button>
              <img
                alt="Doctor's profile"
                className="w-10 h-10 rounded-full object-cover border-2 border-theme-outline shadow-sm"
                src="https://lh3.googleusercontent.com/aida-public/AB6AXuCv-k7Azq5iDo172MfYfPTVafjHyQpnxWpxW-Q8uh6WNrglS1xJOQCBzx0ibs8Av_8LPTFjdjZHSpWbMFuSM9UcYnun96_pAwMBv9E--FoKnbwOBHD9v85Va-7CH3UhzhlrwKfViE0qNa9qcvwncdwj7jYoonYgZ8IxfIK7-Z1l3svFDBBSjuYUJP7vFv7aymNS0o2mxY7nZ_jpaowo6Nf0J0BY70bsBKB3vKsHWe8X7Z8wPJnmlJ7IxYb9pYGgDDw-Cvyo88GCv1Nm"
              />
            </div>
          </div>
        </header>

        <main className="flex-grow">

          {/* 1. Greeting */}
          <section className="max-w-7xl 2xl:max-w-[96rem] mx-auto px-8 md:px-12 pt-16 pb-8 2xl:pt-20 2xl:pb-10 text-center">
            <h1 className="font-display-lg text-4xl md:text-5xl lg:text-6xl 2xl:text-7xl text-theme-text-primary mb-4 tracking-tight">
              Good morning, Dr. Smith
            </h1>
            <p className="font-body-lg text-lg md:text-xl 2xl:text-2xl text-theme-text-secondary max-w-2xl 2xl:max-w-4xl mx-auto leading-relaxed">
              Ask the copilot anything about your patient.
            </p>
          </section>

          {/* 2. AI Workspace — dominant primary element */}
          <section className="max-w-4xl 2xl:max-w-5xl mx-auto px-6 sm:px-8 md:px-12 pb-8">
            <div className="bg-theme-surface border border-theme-outline rounded-3xl shadow-sm overflow-hidden">

              {/* Response region */}
              <div
                ref={responseRef}
                className={response || isLoading ? 'block' : 'hidden'}
              >
                <div className="px-6 sm:px-8 pt-8 pb-4">
                  {isLoading && (
                    <div className="flex items-center gap-3 text-theme-text-secondary py-4">
                      <Loader2 className="w-4 h-4 animate-spin flex-shrink-0 text-theme-accent" />
                      <span className="text-sm font-medium tracking-wide">Retrieving clinical context&hellip;</span>
                    </div>
                  )}
                  {response && !isLoading && (
                    <div className="space-y-5">
                      <div className="flex items-start gap-3">
                        <div className="flex-shrink-0 w-7 h-7 rounded-full bg-theme-outline/50 flex items-center justify-center mt-0.5">
                          <span className="text-xs font-medium text-theme-text-secondary">You</span>
                        </div>
                        <p className="text-sm text-theme-text-secondary leading-relaxed pt-1 italic">{response.query}</p>
                      </div>
                      <div className="w-full h-px bg-theme-outline/40" />
                      <div className="flex items-start gap-3">
                        <div className="flex-shrink-0 w-7 h-7 rounded-full bg-theme-accent/15 border border-theme-accent/20 flex items-center justify-center mt-0.5">
                          <Sparkles className="w-3.5 h-3.5 text-theme-accent" />
                        </div>
                        <div className="flex-grow min-w-0">
                          <p className="text-sm text-theme-text-primary leading-7 whitespace-pre-wrap">{response.response}</p>
                          {isMock && (
                            <p className="mt-3 text-xs text-theme-text-secondary/60 italic">
                              Demo mode &mdash; connect backend for live clinical data.
                            </p>
                          )}
                          {response.sources.length > 0 && (
                            <div className="flex flex-wrap gap-2 mt-4">
                              {response.sources.map((src, i) => (
                                <span
                                  key={i}
                                  className="inline-flex items-center gap-1 text-xs px-3 py-1 rounded-full bg-theme-ambient border border-theme-outline text-theme-text-secondary"
                                  title={src.text.slice(0, 120)}
                                >
                                  {src.source}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
                <div className="h-px bg-theme-outline/50 mx-6 sm:mx-8" />
              </div>

              {/* Suggested prompts — empty state */}
              {!response && !isLoading && (
                <div className="px-6 sm:px-8 pt-7 pb-3">
                  <p className="text-xs text-theme-text-secondary/60 uppercase tracking-[0.18em] font-medium mb-4">
                    Suggested queries
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                    {SUGGESTED_PROMPTS.map((prompt) => (
                      <button
                        key={prompt}
                        onClick={() => handleSuggestedPrompt(prompt)}
                        className="text-left text-sm text-theme-text-secondary hover:text-theme-text-primary px-4 py-3 rounded-2xl bg-theme-ambient hover:bg-theme-accent/5 border border-theme-outline/60 hover:border-theme-accent/20 transition-all duration-200 leading-snug"
                      >
                        {prompt}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Input area */}
              <div className="px-6 sm:px-8 py-5 flex items-end gap-4">
                <textarea
                  ref={textareaRef}
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={handleKeyDown}
                  rows={1}
                  disabled={isLoading}
                  placeholder="Ask Copilot about the patient\u2026"
                  className="flex-grow resize-none bg-transparent text-sm text-theme-text-primary placeholder:text-theme-text-secondary/50 focus:outline-none leading-relaxed disabled:opacity-50 py-1"
                  style={{ maxHeight: '200px' }}
                  aria-label="Clinical query input"
                />
                <button
                  onClick={submit}
                  disabled={isLoading || !query.trim()}
                  className="flex-shrink-0 w-10 h-10 rounded-2xl bg-theme-accent/10 hover:bg-theme-accent/20 border border-theme-accent/20 hover:border-theme-accent/40 flex items-center justify-center transition-all duration-200 disabled:opacity-30 disabled:cursor-not-allowed active:scale-95"
                  aria-label="Submit query"
                >
                  {isLoading
                    ? <Loader2 className="w-4 h-4 animate-spin text-theme-accent" />
                    : <Send className="w-4 h-4 text-theme-accent" />
                  }
                </button>
              </div>

              {/* Input hint */}
              <div className="px-6 sm:px-8 pb-4">
                <p className="text-xs text-theme-text-secondary/40">
                  Press Enter to send &middot; Shift+Enter for new line
                </p>
              </div>

            </div>
          </section>

          {/* 3. Secondary: Workflow Utility Cards */}
          <section className="max-w-7xl 2xl:max-w-[96rem] mx-auto px-6 sm:px-8 md:px-12 pb-10">
            <p className="text-xs text-theme-text-secondary/50 uppercase tracking-[0.18em] font-medium mb-4 px-1">
              Workflow tools
            </p>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6 lg:gap-8">
              <button className="w-full text-left bg-theme-surface border border-theme-outline rounded-2xl sm:rounded-3xl p-6 sm:p-8 2xl:p-12 shadow-sm hover:shadow-md hover:-translate-y-2 transition-all duration-300 flex flex-row lg:flex-col items-center lg:items-start gap-4 sm:gap-6 lg:gap-8 group cursor-pointer overflow-hidden">
                <div className="flex-shrink-0 w-12 h-12 sm:w-16 sm:h-16 2xl:w-20 2xl:h-20 rounded-xl sm:rounded-2xl bg-theme-ambient border border-theme-outline flex items-center justify-center group-hover:bg-theme-accent/10 transition-colors">
                  <UserSearch className="w-6 h-6 sm:w-8 sm:h-8 2xl:w-10 2xl:h-10 text-theme-accent" />
                </div>
                <div className="flex-grow min-w-0 w-full lg:mt-2">
                  <h3 className="font-headline-md text-base sm:text-xl 2xl:text-2xl font-medium text-theme-text-primary mb-1 sm:mb-2 2xl:mb-4 group-hover:text-theme-accent transition-colors truncate lg:whitespace-normal lg:overflow-visible">Review Patient Nodes</h3>
                  <p className="font-body-lg text-sm sm:text-base 2xl:text-lg text-theme-text-secondary leading-relaxed line-clamp-2 lg:line-clamp-none">Start by selecting a patient from your clinical list to begin analysis.</p>
                </div>
                <div className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-all transform -translate-x-2 group-hover:translate-x-0 hidden sm:block lg:mt-auto lg:self-end">
                  <ArrowRight className="w-5 h-5 sm:w-6 sm:h-6 2xl:w-8 2xl:h-8 text-theme-text-secondary" />
                </div>
              </button>
              <button className="w-full text-left bg-theme-surface border border-theme-outline rounded-2xl sm:rounded-3xl p-6 sm:p-8 2xl:p-12 shadow-sm hover:shadow-md hover:-translate-y-2 transition-all duration-300 flex flex-row lg:flex-col items-center lg:items-start gap-4 sm:gap-6 lg:gap-8 group cursor-pointer overflow-hidden">
                <div className="flex-shrink-0 w-12 h-12 sm:w-16 sm:h-16 2xl:w-20 2xl:h-20 rounded-xl sm:rounded-2xl bg-theme-ambient border border-theme-outline flex items-center justify-center group-hover:bg-theme-accent/10 transition-colors">
                  <Sparkles className="w-6 h-6 sm:w-8 sm:h-8 2xl:w-10 2xl:h-10 text-theme-accent" />
                </div>
                <div className="flex-grow min-w-0 w-full lg:mt-2">
                  <h3 className="font-headline-md text-base sm:text-xl 2xl:text-2xl font-medium text-theme-text-primary mb-1 sm:mb-2 2xl:mb-4 group-hover:text-theme-accent transition-colors truncate lg:whitespace-normal lg:overflow-visible">Review AI Insights</h3>
                  <p className="font-body-lg text-sm sm:text-base 2xl:text-lg text-theme-text-secondary leading-relaxed line-clamp-2 lg:line-clamp-none">Copilot highlights critical telemetry changes and risks automatically.</p>
                </div>
                <div className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-all transform -translate-x-2 group-hover:translate-x-0 hidden sm:block lg:mt-auto lg:self-end">
                  <ArrowRight className="w-5 h-5 sm:w-6 sm:h-6 2xl:w-8 2xl:h-8 text-theme-text-secondary" />
                </div>
              </button>
              <button className="w-full text-left bg-theme-surface border border-theme-outline rounded-2xl sm:rounded-3xl p-6 sm:p-8 2xl:p-12 shadow-sm hover:shadow-md hover:-translate-y-2 transition-all duration-300 flex flex-row lg:flex-col items-center lg:items-start gap-4 sm:gap-6 lg:gap-8 group cursor-pointer overflow-hidden">
                <div className="flex-shrink-0 w-12 h-12 sm:w-16 sm:h-16 2xl:w-20 2xl:h-20 rounded-xl sm:rounded-2xl bg-theme-ambient border border-theme-outline flex items-center justify-center group-hover:bg-theme-accent/10 transition-colors">
                  <FileText className="w-6 h-6 sm:w-8 sm:h-8 2xl:w-10 2xl:h-10 text-theme-accent" />
                </div>
                <div className="flex-grow min-w-0 w-full lg:mt-2">
                  <h3 className="font-headline-md text-base sm:text-xl 2xl:text-2xl font-medium text-theme-text-primary mb-1 sm:mb-2 2xl:mb-4 group-hover:text-theme-accent transition-colors truncate lg:whitespace-normal lg:overflow-visible">Draft Summaries</h3>
                  <p className="font-body-lg text-sm sm:text-base 2xl:text-lg text-theme-text-secondary leading-relaxed line-clamp-2 lg:line-clamp-none">Generate discharge notes and patient summaries with a single command.</p>
                </div>
                <div className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-all transform -translate-x-2 group-hover:translate-x-0 hidden sm:block lg:mt-auto lg:self-end">
                  <ArrowRight className="w-5 h-5 sm:w-6 sm:h-6 2xl:w-8 2xl:h-8 text-theme-text-secondary" />
                </div>
              </button>
            </div>
          </section>

        </main>

        {/* Footer */}
        <footer className="w-full border-t border-theme-outline bg-theme-ambient transition-colors duration-300 mt-auto">
          <div className="flex flex-col md:flex-row justify-between items-center w-full px-8 md:px-12 py-8 max-w-7xl mx-auto gap-6">
            <span className="text-xs text-theme-text-secondary/80 tracking-wider">&copy; 2024 COPILOT OS. CLINICAL PRECISION, AMBIENT DESIGN.</span>
            <nav className="flex flex-wrap justify-center gap-6 md:gap-8">
              <a className="text-sm text-theme-text-secondary hover:text-theme-text-primary transition-colors" href="#">Privacy Policy</a>
              <a className="text-sm text-theme-text-secondary hover:text-theme-text-primary transition-colors" href="#">Terms of Service</a>
              <a className="text-sm text-theme-text-secondary hover:text-theme-text-primary transition-colors" href="#">Security</a>
              <a className="text-sm text-theme-text-secondary hover:text-theme-text-primary transition-colors" href="#">Support</a>
            </nav>
          </div>
        </footer>

      </div>
    </div>
  );
};
