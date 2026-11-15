import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        dark: { 900: "#0a0e1a", 800: "#111827", 700: "#1a2236", 600: "#1e293b" },
        brand: { DEFAULT: "#6366f1", light: "#818cf8", glow: "rgba(99,102,241,0.3)" },
      },
      fontFamily: { sans: ["Inter", "system-ui", "sans-serif"] },
    },
  },
  plugins: [],
};
export default config;
