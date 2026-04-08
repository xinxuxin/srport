"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { LanguageToggle } from "./language-toggle";
import { useLanguage } from "./language-provider";

const links = [
  { href: "/", en: "Demo", zh: "演示" },
  { href: "/model", en: "Model", zh: "模型解释" }
];

export function SiteNav() {
  const pathname = usePathname();
  const { language } = useLanguage();

  return (
    <header className="sticky top-0 z-40 border-b border-white/40 bg-sand/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 md:px-8">
        <Link href="/" className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-dusk text-sm font-semibold text-white shadow-panel">
            EP
          </div>
          <div>
            <p className="mono text-[11px] uppercase tracking-[0.32em] text-ember">EPNet</p>
            <p className="text-sm text-ink/70">
              {language === "zh" ? "AI 超分辨率解决方案演示" : "AI super-resolution solution demo"}
            </p>
          </div>
        </Link>

        <div className="flex items-center gap-3">
          <nav className="flex items-center gap-2 rounded-full border border-white/60 bg-white/80 p-1">
            {links.map((link) => {
              const active = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`rounded-full px-4 py-2 text-sm transition ${
                    active ? "bg-dusk text-white shadow-panel" : "text-ink/70 hover:bg-dusk/8"
                  }`}
                >
                  {language === "zh" ? link.zh : link.en}
                </Link>
              );
            })}
          </nav>
          <LanguageToggle />
        </div>
      </div>
    </header>
  );
}
