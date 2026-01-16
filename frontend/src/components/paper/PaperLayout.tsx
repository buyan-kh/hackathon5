"use client";

import { motion } from "framer-motion";
import { IconTrendingUp, IconTrendingDown } from "@tabler/icons-react";
import { SimulationChart } from "@/components/simulation";
import { cn } from "@/lib/utils";
import { WorldMapView, normalizeCountryName } from "./WorldMapView";

interface Article {
    id: string;
    title: string;
    content: string;
    category: string;
    importance?: number;
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

interface NewsSource {
    title: string;
    source: string;
    content?: string;
    agent?: string;
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
    onTopicClick?: (topic: string) => void;
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
    onTopicClick,
}: PaperLayoutProps) {
    // Get news by type for different sections
    const featuredNews = newsContext.slice(0, 2);

    // Extract potential country names from query and headlines for the map
    const affectedCountries: { [key: string]: number } = {};
    const textToScan = (query + " " + headline).toLowerCase();

    // Simple client-side country extraction (backend ideally does this)
    const commonCountries = ["China", "Japan", "USA", "United States", "Russia", "Ukraine", "Taiwan", "Korea", "Iran", "Israel", "UK", "Germany", "France"];

    // Determine severity color based on keywords
    const isCrisis = textToScan.includes("war") || textToScan.includes("nuke") || textToScan.includes("attack") || textToScan.includes("crisis") || textToScan.includes("crash");
    const intensity = isCrisis ? 1.0 : 0.6; // Red for crisis, lighter for news

    commonCountries.forEach(country => {
        if (textToScan.includes(country.toLowerCase())) {
            affectedCountries[normalizeCountryName(country)] = intensity;
        }
    });

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4 }}
            className={cn("w-full bg-[#FAFAFA] min-h-screen", className)}
        >
            <div className="max-w-[1600px] mx-auto bg-white shadow-2xl my-4"> {/* Paper Container */}

                {/* ═══════ MARKET TICKER BAR ═══════ */}
                <div className="bg-[#f5f5f5] border-b border-[#e0e0e0] px-4 py-2 overflow-x-auto sticky top-0 z-10">
                    <div className="flex items-center justify-between w-full">
                        <div className="flex items-center gap-6 text-xs whitespace-nowrap">
                            <span className="font-bold text-[#c4161c] animate-pulse uppercase tracking-wider">
                                {isCrisis ? "🔴 GLOBAL ALERT" : "LIVE MARKETS"}
                            </span>
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
                                    <span className="text-[#ccc] mx-2">│</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* ═══════ MASTHEAD ═══════ */}
                <header className="text-center py-8 border-b-4 border-black px-6">
                    <h1
                        className="text-6xl md:text-8xl tracking-tighter text-[#1a1a1a] leading-none mb-2"
                        style={{ fontFamily: "'Playfair Display', Georgia, serif", fontWeight: 900 }}
                    >
                        TOMORROW&apos;S PAPER
                    </h1>
                    <div className="flex items-center justify-center gap-6 mt-4 text-xs text-[#333] font-bold uppercase tracking-[0.2em] border-t border-b border-[#e0e0e0] py-2 max-w-4xl mx-auto">
                        <span>{date}</span>
                        <span>•</span>
                        <span>Simulated Futures Edition</span>
                        <span>•</span>
                        <span>Global Coverage</span>
                    </div>
                </header>

                {/* ═══════ MAIN CONTENT GRID (4-COLUMN) ═══════ */}
                <div className="grid grid-cols-12 min-h-screen">

                    {/* COL 1: LEFT SIDEBAR (News List) - 2 Cols (Hidden on small screens) */}
                    <div className="col-span-12 md:col-span-2 border-r border-[#e0e0e0] p-4 bg-[#fcfcfc] hidden md:block">
                        <div className="mb-6">
                            <h4 className="font-bold text-xs uppercase tracking-widest text-[#c4161c] mb-3 border-b border-[#c4161c] pb-1">
                                Latest Wires
                            </h4>
                            <div className="space-y-4">
                                {newsContext.slice(0, 5).map((news, i) => (
                                    <article key={i} className="group cursor-pointer hover:bg-white p-2 rounded transition-colors" onClick={() => onTopicClick?.(news.title)}>
                                        <p className="text-[9px] text-[#999] uppercase mb-1 flex items-center gap-1">
                                            <span className="w-1.5 h-1.5 bg-[#c4161c] rounded-full inline-block"></span>
                                            {news.agent?.replace("yutori_", "") || "Wire"}
                                        </p>
                                        <h5 className="font-bold text-xs leading-tight text-[#222] group-hover:text-[#c4161c] font-serif mb-1">
                                            {news.title}
                                        </h5>
                                    </article>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* COL 2: MAIN STORY (Center) - 5 Cols */}
                    <div className="col-span-12 md:col-span-5 border-r border-[#e0e0e0] p-8">
                        {/* Main Headline */}
                        <article className="text-center mb-8">
                            <h2
                                className="text-4xl md:text-5xl font-bold leading-[1.1] text-[#1a1a1a] mb-4"
                                style={{ fontFamily: "'Playfair Display', Georgia, serif" }}
                            >
                                {headline}
                            </h2>
                            <p className="text-lg text-[#555] font-serif italic max-w-2xl mx-auto leading-relaxed">
                                {subheadline || articles[0]?.content?.slice(0, 150)}...
                            </p>
                        </article>

                        {/* Main Visual */}
                        <div className="mb-8 relative group cursor-pointer shadow-xl">
                            {coverImageUrl ? (
                                <div className="overflow-hidden border border-[#333]">
                                    <img
                                        src={coverImageUrl}
                                        alt="Cover"
                                        className="w-full h-auto transform transition-transform duration-1000 group-hover:scale-[1.01]"
                                    />
                                    {/* Vignette Overlay */}
                                    <div className="absolute inset-0 ring-1 ring-inset ring-black/10 pointer-events-none"></div>
                                </div>
                            ) : (
                                <div className="bg-[#f0f0f0] aspect-[16/9] flex items-center justify-center border border-[#ddd]">
                                    <span className="text-xs text-muted-foreground animate-pulse">Generating visualization...</span>
                                </div>
                            )}
                            <div className="flex justify-between items-start mt-2 px-1">
                                <p className="text-[10px] text-[#666] font-sans uppercase tracking-wider">
                                    Fig. 1 — Scenario Visualization
                                </p>
                                <p className="text-[10px] text-[#999] font-sans">
                                    Source: AI Generative Model
                                </p>
                            </div>
                        </div>

                        {/* Main Content Text */}
                        <div className="font-serif text-[#111] leading-relaxed text-base space-y-4">
                            {articles[0]?.content?.split('\n').map((para, i) => (
                                <p key={i} className="mb-4 first-letter:float-left first-letter:text-5xl first-letter:pr-2 first-letter:font-bold first-letter:text-[#111]">
                                    {para}
                                </p>
                            ))}
                        </div>
                    </div>

                    {/* COL 3: ANALYSIS (Right) - 2 Cols */}
                    <div className="col-span-12 md:col-span-2 border-r border-[#e0e0e0] p-4 bg-[#fcfcfc] hidden lg:block">
                        <h4 className="font-bold text-xs uppercase tracking-widest text-[#333] mb-4 border-b border-[#333] pb-1">
                            Market Analysis
                        </h4>
                        <div className="space-y-6">
                            {articles.slice(1, 4).map((news) => (
                                <article key={news.id} className="border-b border-[#eee] pb-4 last:border-0">
                                    <h5 className="font-bold text-sm leading-tight text-[#1a1a1a] font-serif mb-2">
                                        {news.title}
                                    </h5>
                                    <p className="text-xs text-[#555] leading-snug line-clamp-3">
                                        {news.summary}
                                    </p>
                                </article>
                            ))}
                        </div>

                        {/* Secondary Visual */}
                        <div className="mt-8 relative group cursor-pointer">
                            {secondaryImageUrl && (
                                <div className="overflow-hidden border border-[#ddd] mb-2">
                                    <img src={secondaryImageUrl} className="w-full h-32 object-cover transition-transform duration-500 group-hover:scale-105" alt="Secondary" />
                                </div>
                            )}
                            <h5 className="font-bold text-xs font-serif">Economic Fallout</h5>
                        </div>
                    </div>

                    {/* COL 4: DATA & MAPS (Far Right) - 3 Cols */}
                    <div className="col-span-12 md:col-span-3 p-4 bg-[#f4f4f5]">
                        <h4 className="font-bold text-xs uppercase tracking-widest text-[#c4161c] mb-4 border-b border-[#c4161c] pb-1">
                            Situation Room
                        </h4>

                        {/* WORLD MAP WIDGET */}
                        <div className="bg-[#fff] border border-[#dce] rounded-lg overflow-hidden mb-6 shadow-sm">
                            <div className="p-2 bg-[#fcfcfc] border-b border-[#eec] text-[10px] font-bold uppercase tracking-wider text-[#666]">
                                Geopolitical Impact Zone
                            </div>
                            <WorldMapView
                                className="w-full h-40 bg-[#eef]"
                                highlightedCountries={affectedCountries}
                                onCountryClick={(c) => onTopicClick?.(c)}
                            />
                        </div>

                        {/* MARKET CHART WIDGET */}
                        <div className="bg-[#fff] border border-[#dce] rounded-lg overflow-hidden mb-6 shadow-sm p-0">
                            <div className="p-2 bg-[#fcfcfc] border-b border-[#eec] flex justify-between items-center">
                                <span className="text-[10px] font-bold uppercase tracking-wider text-[#666]">
                                    {marketSnapshot[0]?.asset || "Market"} {marketSnapshot[0]?.asset?.includes("(Simulation)") ? "(Simulated)" : "Forecast"}
                                </span>
                                <span className="text-[9px] px-1.5 py-0.5 bg-[#c4161c] text-white font-bold rounded">AI SIM</span>
                            </div>

                            <div className="p-3">
                                {marketSnapshot[0] ? (() => {
                                    const asset = marketSnapshot[0];
                                    // Transform data to match SimulationChart interface
                                    // Handle both camelCase and snake_case formats
                                    const currentValue = asset.current_value ?? (asset.value && asset.change ? asset.value - asset.change : asset.value) ?? 1000;
                                    const projectedValue = asset.projected_value ?? asset.value ?? currentValue;
                                    const changePercent = asset.change_percent ?? asset.changePercent ?? 0;
                                    
                                    // Ensure we have valid values (non-zero)
                                    const safeCurrentValue = currentValue > 0 ? currentValue : 1000;
                                    const safeProjectedValue = projectedValue > 0 ? projectedValue : safeCurrentValue * (1 + changePercent / 100);
                                    
                                    const chartAsset = {
                                        asset: asset.asset || "Market",
                                        current_value: safeCurrentValue,
                                        projected_value: safeProjectedValue,
                                        change_percent: changePercent,
                                        history: asset.history || [],
                                    };
                                    
                                    return (
                                        <>
                                            <div className="flex items-end justify-between mb-2">
                                                <div className="text-2xl font-bold font-serif">
                                                    {safeCurrentValue.toFixed(2)}
                                                </div>
                                                <div className={`text-xs font-bold ${changePercent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                                    {changePercent >= 0 ? '▲' : '▼'} {Math.abs(changePercent).toFixed(2)}%
                                                </div>
                                            </div>
                                            <SimulationChart
                                                asset={chartAsset}
                                                height={150}
                                                className="border-none shadow-none bg-transparent p-0"
                                            />
                                        </>
                                    );
                                })() : (
                                    <div className="h-32 bg-zinc-50 flex items-center justify-center text-xs text-muted-foreground">
                                        Waiting for simulation data...
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* TRENDING TOPICS */}
                        <div className="bg-[#fff] border border-[#dce] rounded-lg p-4 shadow-sm">
                            <h5 className="font-bold text-[10px] uppercase tracking-wider text-[#333] mb-3">Trending Vectors</h5>
                            <div className="flex flex-wrap gap-2">
                                {trendingTopics?.map((topic, i) => (
                                    <span key={i} className="px-2 py-1 bg-[#f0f0f0] text-[#333] text-[10px] font-bold rounded uppercase border border-[#ddd]">
                                        #{topic.topic}
                                    </span>
                                ))}
                            </div>
                        </div>

                    </div>
                </div>

                {/* FOOTER */}
                <div className="bg-[#111] text-[#fff] py-3 text-center text-[10px] uppercase tracking-widest">
                    Vol. 1 | Generated by Agent Orchestrator | {new Date().getFullYear()}
                </div>
            </div>
        </motion.div>
    );
}
