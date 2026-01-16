"use client";

import { motion } from "framer-motion";
import { IconCalendar, IconShare, IconDownload } from "@tabler/icons-react";
import { Button } from "@/components/ui";
import { cn } from "@/lib/utils";

interface Article {
    id: string;
    title: string;
    content: string;
    summary: string;
    category: string;
    importance: number;
    imageUrl?: string;
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
    articles: Article[];
    marketSnapshot: MarketItem[];
    trendingTopics: TrendingTopic[];
    query: string;
    className?: string;
}

export function PaperLayout({
    date,
    headline,
    subheadline,
    coverImageUrl,
    articles,
    marketSnapshot,
    trendingTopics,
    query,
    className,
}: PaperLayoutProps) {
    const leadArticle = articles.find(a => a.category === "headline") || articles[0];
    const sideArticles = articles.filter(a => a.id !== leadArticle?.id).slice(0, 2);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className={cn(
                "w-full max-w-4xl mx-auto rounded-xl border-2 border-border bg-card overflow-hidden",
                "shadow-xl",
                className
            )}
        >
            {/* Masthead */}
            <header className="border-b-2 border-border bg-foreground text-background px-6 py-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="h-8 w-1 rounded-full bg-gradient-to-b from-accent to-accent-secondary" />
                        <h1 className="font-display text-2xl md:text-3xl tracking-tight">
                            Tomorrow&apos;s Paper
                        </h1>
                    </div>
                    <div className="flex items-center gap-2 text-sm opacity-80">
                        <IconCalendar size={16} />
                        <span>{date}</span>
                    </div>
                </div>
                <p className="mt-1 text-xs opacity-60 font-mono">
                    AI-Generated Simulation • Based on: &quot;{query}&quot;
                </p>
            </header>

            {/* Main Content */}
            <div className="p-6">
                {/* Headline */}
                <div className="mb-6">
                    <h2 className="font-display text-3xl md:text-4xl lg:text-5xl font-normal leading-tight mb-2">
                        {headline}
                    </h2>
                    {subheadline && (
                        <p className="text-lg text-muted-foreground">
                            {subheadline}
                        </p>
                    )}
                </div>

                {/* Cover Image */}
                {coverImageUrl ? (
                    <div className="mb-6 overflow-hidden rounded-lg bg-muted aspect-[16/9]">
                        <img
                            src={coverImageUrl}
                            alt="Cover"
                            className="w-full h-full object-cover"
                        />
                    </div>
                ) : (
                    <div className="mb-6 overflow-hidden rounded-lg bg-gradient-to-br from-accent/10 via-muted to-accent-secondary/10 aspect-[16/9] flex items-center justify-center">
                        <div className="text-center text-muted-foreground">
                            <div className="text-6xl mb-2">📰</div>
                            <p className="text-sm">AI-generated cover image</p>
                        </div>
                    </div>
                )}

                {/* Two Column Layout */}
                <div className="grid md:grid-cols-3 gap-6">
                    {/* Main Article */}
                    <div className="md:col-span-2">
                        {leadArticle && (
                            <article>
                                <h3 className="text-xl font-semibold mb-3">{leadArticle.title}</h3>
                                <p className="text-muted-foreground leading-relaxed">
                                    {leadArticle.content}
                                </p>
                            </article>
                        )}

                        {/* Secondary Articles */}
                        {sideArticles.length > 0 && (
                            <div className="mt-6 pt-6 border-t border-border grid gap-4 md:grid-cols-2">
                                {sideArticles.map((article) => (
                                    <article key={article.id}>
                                        <span className="text-xs font-mono uppercase tracking-wider text-accent">
                                            {article.category}
                                        </span>
                                        <h4 className="text-base font-semibold mt-1 mb-2">
                                            {article.title}
                                        </h4>
                                        <p className="text-sm text-muted-foreground line-clamp-3">
                                            {article.summary}
                                        </p>
                                    </article>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Sidebar */}
                    <aside className="space-y-6">
                        {/* Market Snapshot */}
                        <div className="rounded-lg border border-border p-4">
                            <h4 className="text-xs font-mono uppercase tracking-wider text-muted-foreground mb-3">
                                Markets at a Glance
                            </h4>
                            <div className="space-y-3">
                                {marketSnapshot.map((item) => (
                                    <div key={item.asset} className="flex items-center justify-between">
                                        <span className="text-sm font-medium">{item.asset}</span>
                                        <div className="text-right">
                                            <div className="text-sm">${item.value.toLocaleString()}</div>
                                            <div className={cn(
                                                "text-xs",
                                                item.changePercent >= 0 ? "text-green-600" : "text-red-600"
                                            )}>
                                                {item.changePercent >= 0 ? "+" : ""}{item.changePercent.toFixed(1)}%
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Trending Topics */}
                        <div className="rounded-lg border border-border p-4">
                            <h4 className="text-xs font-mono uppercase tracking-wider text-muted-foreground mb-3">
                                Trending Topics
                            </h4>
                            <div className="flex flex-wrap gap-2">
                                {trendingTopics.map((topic) => (
                                    <span
                                        key={topic.topic}
                                        className={cn(
                                            "px-2 py-1 rounded-full text-xs font-medium",
                                            topic.sentiment > 0.3 && "bg-green-500/10 text-green-700",
                                            topic.sentiment < -0.3 && "bg-red-500/10 text-red-700",
                                            topic.sentiment >= -0.3 && topic.sentiment <= 0.3 && "bg-muted text-muted-foreground"
                                        )}
                                    >
                                        {topic.topic}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </aside>
                </div>
            </div>

            {/* Footer */}
            <footer className="border-t border-border bg-muted/30 px-6 py-4 flex items-center justify-between">
                <p className="text-xs text-muted-foreground">
                    Generated by AI agents: Yutori • Tonic Fabricate • Freepik
                </p>
                <div className="flex items-center gap-2">
                    <Button variant="ghost" size="icon-sm">
                        <IconShare size={16} />
                    </Button>
                    <Button variant="ghost" size="icon-sm">
                        <IconDownload size={16} />
                    </Button>
                </div>
            </footer>
        </motion.div>
    );
}
