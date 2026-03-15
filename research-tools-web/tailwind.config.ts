import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0f172a",
        mist: "#f8fafc",
        signal: "#0ea5e9",
        stress: "#f97316",
      },
    },
  },
  plugins: [],
};

export default config;
