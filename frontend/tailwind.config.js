/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: "#F8FAFC",
        surface: "#FFFFFF",
        ink: "#0F172A",
        steel: "#64748B",
        borderwhisper: "#E2E8F0",
        pine: {
          light: "#115E59", // Teal-800 for high readability
          DEFAULT: "#0F766E", // Teal-700
          dark: "#115E59",
        },
        stable: "#047857",
        critical: "#BE123C",
        theme: {
          ambient: "var(--theme-ambient)",
          surface: "var(--theme-surface)",
          text: {
            primary: "var(--theme-text-primary)",
            secondary: "var(--theme-text-secondary)",
          },
          accent: "var(--theme-accent)",
          outline: "var(--theme-outline)",
        }
      },
      fontFamily: {
        sans: ["Geist Sans", "Satoshi", "system-ui", "sans-serif"],
        mono: ["Geist Mono", "JetBrains Mono", "monospace"],
      },
    },
  },
  plugins: [],
}
