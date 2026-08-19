import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CreditGuard AI — Default Risk Predictor",
  description: "ML-powered credit card default prediction",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="ambient min-h-screen bg-dark-900 text-slate-100 antialiased">
        {children}
      </body>
    </html>
  );
}
