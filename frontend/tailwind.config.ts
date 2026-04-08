import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        canvas: "#f4f0e8",
        ink: "#13202f",
        ember: "#c2542f",
        moss: "#5f7763",
        sand: "#f7f2ec",
        dusk: "#22415d"
      },
      boxShadow: {
        panel: "0 24px 80px rgba(17, 29, 43, 0.14)"
      },
      backgroundImage: {
        "hero-grid":
          "linear-gradient(rgba(19,32,47,0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(19,32,47,0.08) 1px, transparent 1px)"
      }
    }
  },
  plugins: []
};

export default config;
