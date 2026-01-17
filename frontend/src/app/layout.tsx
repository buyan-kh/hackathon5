import type { Metadata, Viewport } from "next";
import Script from "next/script";
import { Inter, Calistoga, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { ForceLightMode } from "@/components/ForceLightMode";

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
  colorScheme: "light",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${calistoga.variable} ${jetbrainsMono.variable}`}
      style={{ colorScheme: 'light' }}
      suppressHydrationWarning
    >
      <body className="min-h-screen bg-background text-foreground antialiased" suppressHydrationWarning>
        <Script
          id="force-light-mode"
          strategy="beforeInteractive"
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                if (typeof document !== 'undefined') {
                  document.documentElement.classList.remove('dark');
                  document.documentElement.removeAttribute('data-theme');
                  document.documentElement.style.colorScheme = 'light';
                }
              })();
            `,
          }}
        />
        <ForceLightMode />
        {children}
      </body>
    </html>
  );
}
