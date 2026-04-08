"use client";

import { motion } from "framer-motion";

import { useLanguage } from "./language-provider";

export function LanguageToggle() {
  const { language, setLanguage } = useLanguage();

  return (
    <div className="flex items-center gap-1 rounded-full border border-white/60 bg-white/80 p-1">
      {[
        { value: "en" as const, label: "EN" },
        { value: "zh" as const, label: "中文" }
      ].map((item) => {
        const active = item.value === language;
        return (
          <button
            key={item.value}
            type="button"
            onClick={() => setLanguage(item.value)}
            className={`relative rounded-full px-3 py-2 text-xs font-medium transition ${
              active ? "text-white" : "text-ink/70 hover:text-ink"
            }`}
          >
            {active ? (
              <motion.span
                layoutId="language-pill"
                className="absolute inset-0 rounded-full bg-dusk shadow-panel"
              />
            ) : null}
            <span className="relative">{item.label}</span>
          </button>
        );
      })}
    </div>
  );
}
