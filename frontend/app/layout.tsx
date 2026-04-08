import type { Metadata } from "next";

import { LanguageProvider } from "../components/language-provider";
import { SiteNav } from "../components/site-nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "EPNet Super-Resolution Demo",
  description:
    "Single-image super-resolution demo for EPNet with model telemetry, latency, and analytics."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-canvas text-ink antialiased">
        <LanguageProvider>
          <SiteNav />
          {children}
        </LanguageProvider>
      </body>
    </html>
  );
}
