import type { Metadata, Viewport } from "next";
import { Inter, Calistoga, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const calistoga = Calistoga({
  variable: "--font-calistoga",
  subsets: ["latin"],
  weight: "400",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Tomorrow's Paper | AI-Powered Market Simulation & News",
  description: "Query anything. Get tomorrow's news today through AI-powered market simulation and multi-agent orchestration.",
  keywords: ["AI", "market simulation", "news generation", "agents", "prediction"],
  authors: [{ name: "Tomorrow's Paper" }],
  openGraph: {
    title: "Tomorrow's Paper",
    description: "AI-Powered Market Simulation & News Generation",
    type: "website",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#0052FF",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${calistoga.variable} ${jetbrainsMono.variable}`}>
      <body className="min-h-screen bg-background text-foreground antialiased">
        {children}
      </body>
    </html>
  );
}
