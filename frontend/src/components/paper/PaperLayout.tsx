"use client";

import { motion } from "framer-motion";
import {
    IconTrendingUp,
    IconTrendingDown,
    IconExternalLink,
    IconClock,
} from "@tabler/icons-react";
import { cn } from "@/lib/utils";
import { SimulationChart } from "@/components/simulation/SimulationChart";

interface NewsSource {
    title: string;
    source: string;
    content?: string;
    agent?: string;
}

interface Article {
    id: string;
    title: string;
    content: string;
    summary?: string;
    category: string;
    importance: number;
}

interface MarketItem {
    asset: string;
    value: number;
    change: number;
    changePercent: number;
}

interface TrendingTopic {
    topic: string;
    sentiment: number;
    mentions: number;
}

interface PaperLayoutProps {
    date: string;
    headline: string;
    subheadline?: string;
    coverImageUrl?: string;
    secondaryImageUrl?: string;
    tertiaryImageUrl?: string;
    articles: Article[];
    marketSnapshot: MarketItem[];
    trendingTopics: TrendingTopic[];
    newsContext?: NewsSource[];
    query: string;
    className?: string;
}

export function PaperLayout({
    date,
    headline,
    subheadline,
    coverImageUrl,
    secondaryImageUrl,
    tertiaryImageUrl,
    articles,
    marketSnapshot,
    trendingTopics,
    newsContext = [],
    query,
    className,
}: PaperLayoutProps) {
    // Get news by type for different sections
    const featuredNews = newsContext.slice(0, 2);
    const sidebarNews = newsContext.slice(2, 6);
    const bottomNews = newsContext.slice(6, 10);

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4 }}
            className={cn("w-full max-w-6xl mx-auto bg-white", className)}
        >
            {/* ═══════ MARKET TICKER BAR ═══════ */}
            <div className="bg-[#f5f5f5] border-b border-[#e0e0e0] px-4 py-2 overflow-x-auto">
                <div className="flex items-center gap-6 text-xs whitespace-nowrap">
                    {marketSnapshot.slice(0, 8).map((item, i) => (
                        <div key={item.asset} className="flex items-center gap-2">
                            <span className="font-semibold text-[#333]">{item.asset}</span>
                            <span className="text-[#666]">
                                {item.asset === "VIX"
                                    ? (item.value || 0).toFixed(2)
                                    : (item.value || 0).toLocaleString()}
                            </span>
                            <span
                                className={cn(
                                    "flex items-center gap-0.5",
                                    (item.changePercent || 0) >= 0 ? "text-green-600" : "text-red-600"
                                )}
                            >
                                {(item.changePercent || 0) >= 0 ? <IconTrendingUp size={12} /> : <IconTrendingDown size={12} />}
                                {(item.changePercent || 0) >= 0 ? "+" : ""}{(item.changePercent || 0).toFixed(2)}%
                            </span>
                            {i < marketSnapshot.length - 1 && <span className="text-[#ccc] mx-2">│</span>}
                        </div>
                    ))}
                </div>
            </div>

            {/* ═══════ MASTHEAD ═══════ */}
            <header className="text-center py-6 border-b border-[#1a1a1a]">
                <h1
                    className="text-5xl md:text-7xl tracking-tighter text-[#1a1a1a]"
                    style={{ fontFamily: "'Playfair Display', Georgia, serif", fontWeight: 700 }}
                >
                    TOMORROW&apos;S PAPER
                </h1>
                <div className="flex items-center justify-center gap-4 mt-2 text-[11px] text-[#666] font-sans uppercase tracking-widest">
                    <span>{date}</span>
                    <span>•</span>
                    <span className="font-bold">VOL. 1 • NO. 1</span>
                    <span>•</span>
                    <span>AI-GENERATED EDITION</span>
                </div>
            </header>

            {/* ═══════ NAV BAR ═══════ */}
            <nav className="border-b double-border px-4 py-2 mb-4 border-[#e0e0e0]">
                <div className="flex justify-center gap-8 text-[10px] font-bold uppercase tracking-widest text-[#333]">
                    <span className="hover:text-[#c4161c] cursor-pointer">World News</span>
                    <span className="hover:text-[#c4161c] cursor-pointer">U.S. Politics</span>
                    <span className="hover:text-[#c4161c] cursor-pointer">Economy</span>
                    <span className="hover:text-[#c4161c] cursor-pointer">Markets</span>
                    <span className="hover:text-[#c4161c] cursor-pointer">Tech</span>
                    <span className="hover:text-[#c4161c] cursor-pointer">Opinion</span>
                </div>
            </nav>

            {/* ═══════ MAIN CONTENT GRID ═══════ */}
            <div className="grid grid-cols-12 gap-6 px-6 pb-12">

                {/* LEFT COLUMN - Featured Stories */}
                <div className="col-span-12 md:col-span-3 border-r border-[#e0e0e0] pr-6">
                    {/* Main Headline */}
                    <article className="mb-6">
                        <span className="text-[10px] font-bold text-[#c4161c] uppercase tracking-wider mb-2 block">
                            Lead Story
                        </span>
                        <h2
                            className="text-2xl md:text-3xl font-bold leading-[1.1] text-[#1a1a1a] mb-3"
                            style={{ fontFamily: "'Playfair Display', Georgia, serif" }}
                        >
                            {headline}
                        </h2>
                        <p className="text-sm text-[#444] leading-relaxed mb-3 font-serif">
                            {subheadline || articles[0]?.content?.slice(0, 150)}...
                        </p>
                        <div className="flex items-center gap-2 text-[10px] text-[#999] uppercase">
                            <IconClock size={12} />
                            <span>Analysis Agent</span>
                        </div>

                        {/* Sub-headlines */}
                        <ul className="mt-6 space-y-4 pt-4 border-t border-[#e0e0e0]">
                            {featuredNews.slice(0, 3).map((news, i) => (
                                <li key={i} className="group cursor-pointer">
                                    <h4 className="font-bold text-sm leading-tight mb-1 group-hover:text-[#c4161c] transition-colors font-serif">
                                        {news.title}
                                    </h4>
                                    <p className="text-xs text-[#666] line-clamp-2">
                                        {news.content || "Breaking news analysis..."}
                                    </p>
                                </li>
                            ))}
                        </ul>
                    </article>
                </div>

                {/* CENTER COLUMN - Main Visual */}
                <div className="col-span-12 md:col-span-6 border-r border-[#e0e0e0] pr-6">
                    {/* Cover Image */}
                    <div className="relative mb-4 group cursor-pointer">
                        {coverImageUrl ? (
                            <div className="overflow-hidden">
                                <img
                                    src={coverImageUrl}
                                    alt="Cover"
                                    className="w-full h-auto transform transition-transform duration-700 group-hover:scale-105"
                                />
                            </div>
                        ) : (
                            <div className="bg-[#f0f0f0] aspect-[3/2] flex items-center justify-center">
                                <span className="text-xs text-muted-foreground">Generating visual...</span>
                            </div>
                        )}
                        <p className="text-[10px] text-[#666] mt-2 italic text-right">
                            Figure 1: AI-generated visualization of the current geopolitical climate.
                        </p>
                    </div>

                    <article className="mb-8">
                        <h3
                            className="text-xl font-bold text-[#1a1a1a] mb-2"
                            style={{ fontFamily: "'Playfair Display', Georgia, serif" }}
                        >
                            Global Implications and Market Fallout
                        </h3>
                        <div className="columns-2 gap-6 text-sm text-[#333] font-serif leading-relaxed text-justify">
                            <p className="mb-4 first-letter:text-3xl first-letter:font-bold first-letter:float-left first-letter:mr-1">
                                {articles[0]?.content || "The markets are reacting swiftly to the latest developments..."}
                            </p>
                            <p>
                                Analysts predict significant volatility in the coming sessions as traders digest the news.
                                Key sectors likely to be affected include energy, technology, and financials.
                            </p>
                        </div>
                    </article>

                    {/* Secondary Visual Section */}
                    <div className="grid grid-cols-2 gap-4 pt-4 border-t border-[#e0e0e0]">
                        <article>
                            <div className="aspect-video bg-[#f5f5f5] mb-2 overflow-hidden">
                                {secondaryImageUrl && (
                                    <img src={secondaryImageUrl} className="w-full h-full object-cover" alt="Secondary" />
                                )}
                            </div>
                            <h4 className="font-bold text-sm font-serif">Economic Impact</h4>
                            <p className="text-xs text-[#666]">Market data suggests a rotation into safe-haven assets.</p>
                        </article>
                        <article>
                            <div className="aspect-video bg-[#f5f5f5] mb-2 overflow-hidden">
                                {tertiaryImageUrl && (
                                    <img src={tertiaryImageUrl} className="w-full h-full object-cover" alt="Tertiary" />
                                )}
                            </div>
                            <h4 className="font-bold text-sm font-serif">Global Reaction</h4>
                            <p className="text-xs text-[#666]">International leaders convene to discuss strategy.</p>
                        </article>
                    </div>
                </div>

                {/* RIGHT COLUMN - Analysis & Charts */}
                <div className="col-span-12 md:col-span-3">

                    {/* Market Simulation Chart */}
                    <div className="mb-6">
                        <div className="border-b-2 border-black mb-3 pb-1">
                            <h3 className="font-bold text-xs uppercase tracking-widest text-[#c4161c]">Market Forecast</h3>
                        </div>
                        {marketSnapshot[0] ? (
                            <SimulationChart
                                asset={marketSnapshot[0]}
                                height={220}
                                className="border-none shadow-none bg-transparent p-0"
                            />
                        ) : (
                            <div className="h-40 bg-zinc-100 flex items-center justify-center text-xs text-muted-foreground">
                                Running simulation...
                            </div>
                        )}
                        <p className="text-[10px] text-[#666] mt-2 leading-tight">
                            AI-driven projection based on historical volatility and current news sentiment.
                        </p>
                    </div>

                    <div className="bg-[#f7f7f7] p-4">
                        <h3 className="text-[#c4161c] text-xs font-bold mb-4 uppercase tracking-widest border-b border-[#ddd] pb-2">
                            Opinion & Analysis
                        </h3>

                        <div className="space-y-6">
                            {/* Expand to show more news items (Total 9 requested) */}
                            {/* Featured (3) are in left col, so show remaining here */}
                            {newsContext.length > 3 && newsContext.slice(3, 9).map((news, i) => (
                                <article key={i}>
                                    <p className="text-[10px] text-[#666] uppercase mb-1">
                                        {news.agent?.replace("yutori_", "") || "Global Wire"}
                                    </p>
                                    <h4 className="text-sm font-bold text-[#1a1a1a] leading-tight mb-2 hover:text-[#c4161c] cursor-pointer font-serif">
                                        {news.title}
                                    </h4>
                                    <p className="text-xs text-[#555]">
                                        {news.content ? news.content.slice(0, 80) + "..." : "Updates on the developing situation..."}
                                    </p>
                                </article>
                            ))}
                        </div>

                        {/* Trending Topics */}
                        <div className="mt-8 pt-4 border-t border-[#ddd]">
                            <h4 className="text-[10px] font-bold text-[#666] uppercase tracking-wider mb-3">
                                Trending Now
                            </h4>
                            <div className="flex flex-wrap gap-2">
                                {trendingTopics.map((topic) => (
                                    <span
                                        key={topic.topic}
                                        className="text-[10px] uppercase font-bold px-2 py-1 bg-white border border-[#ddd] text-[#333]"
                                    >
                                        {topic.topic}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* ═══════ FOOTER ═══════ */}
            <footer className="border-t-4 border-[#1a1a1a] bg-white pt-2 pb-6 px-6">
                <div className="flex items-center justify-between text-[10px] uppercase tracking-wider font-bold text-[#333]">
                    <span>Copyright © 2026 Tomorrow's Paper</span>
                    <span>Powered by Yutori Agents</span>
                </div>
            </footer>
        </motion.div>
    );
}
