"use client";

import { motion } from "framer-motion";
import {
    IconNews,
    IconMoodSmile,
    IconReportAnalytics,
    IconChartLine,
    IconCheck,
    IconX,
    IconLoader2,
} from "@tabler/icons-react";
import { useChatStore, AgentStatus as AgentStatusType } from "@/stores/chat.store";
import { cn } from "@/lib/utils";

const agentConfig: Record<string, {
    icon: typeof IconNews;
    color: string;
    bgColor: string;
    borderColor: string;
    label: string;
}> = {
    yutori_news: {
        icon: IconNews,
        color: "text-blue-500",
        bgColor: "bg-blue-500/10",
        borderColor: "border-blue-500/20",
        label: "Breaking News",
    },
    yutori_sentiment: {
        icon: IconMoodSmile,
        color: "text-orange-500",
        bgColor: "bg-orange-500/10",
        borderColor: "border-orange-500/20",
        label: "Social Sentiment",
    },
    yutori_analysis: {
        icon: IconReportAnalytics,
        color: "text-cyan-500",
        bgColor: "bg-cyan-500/10",
        borderColor: "border-cyan-500/20",
        label: "Expert Analysis",
    },
    yutori_target: {
        icon: IconNews,
        color: "text-red-500",
        bgColor: "bg-red-500/10",
        borderColor: "border-red-500/20",
        label: "Local Impact",
    },
    yutori_global: {
        icon: IconNews,
        color: "text-indigo-500",
        bgColor: "bg-indigo-500/10",
        borderColor: "border-indigo-500/20",
        label: "Global Reaction",
    },
    yutori_econ: {
        icon: IconChartLine,
        color: "text-emerald-500",
        bgColor: "bg-emerald-500/10",
        borderColor: "border-emerald-500/20",
        label: "Economic Outlook",
    },
    fabricate: {
        icon: IconChartLine,
        color: "text-purple-500",
        bgColor: "bg-purple-500/10",
        borderColor: "border-purple-500/20",
        label: "Market Simulation",
    },
};

function AgentCard({ agent }: { agent: AgentStatusType }) {
    const config = agentConfig[agent.type];
    if (!config) return null;

    const Icon = config.icon;

    const statusIcon = {
        idle: null,
        running: <IconLoader2 size={14} className="animate-spin text-accent" />,
        completed: <IconCheck size={14} className="text-green-500" />,
        error: <IconX size={14} className="text-red-500" />,
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={cn(
                "rounded-lg border p-3 transition-all",
                config.borderColor,
                config.bgColor
            )}
        >
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                    <div className={cn("rounded-md p-1.5", config.bgColor)}>
                        <Icon size={16} className={config.color} />
                    </div>
                    <div>
                        <p className="text-sm font-medium">{agent.name}</p>
                        <p className="text-xs text-muted-foreground">{config.label}</p>
                    </div>
                </div>
                {statusIcon[agent.status]}
            </div>

            {/* Progress Bar */}
            <div className="h-1.5 w-full rounded-full bg-background/50 overflow-hidden">
                <motion.div
                    className={cn(
                        "h-full rounded-full",
                        agent.status === "completed"
                            ? "bg-green-500"
                            : agent.status === "error"
                                ? "bg-red-500"
                                : "bg-gradient-to-r from-accent to-accent-secondary"
                    )}
                    initial={{ width: 0 }}
                    animate={{ width: `${agent.progress}%` }}
                    transition={{ duration: 0.3, ease: "easeOut" }}
                />
            </div>

            {/* Current Task */}
            {agent.currentTask && (
                <p className="mt-2 text-xs text-muted-foreground truncate">
                    {agent.currentTask}
                </p>
            )}

            {/* Sources Count */}
            {agent.sources !== undefined && agent.sources > 0 && (
                <p className="mt-1 text-xs text-muted-foreground">
                    {agent.sources} sources found
                </p>
            )}
        </motion.div>
    );
}

export function AgentStatus() {
    const agents = useChatStore((state) => state.agents);
    const isGenerating = useChatStore((state) => state.isGenerating);

    if (!isGenerating && agents.every((a) => a.status === "idle")) {
        return null;
    }

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="w-full max-w-3xl mx-auto px-4"
        >
            <div className="rounded-xl border border-border/50 bg-card p-4 shadow-lg">
                <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                        <div className="section-label-dot animate-pulse-live" />
                        <span className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
                            Agents Working
                        </span>
                    </div>
                    <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                        <span className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
                        Live
                    </span>
                </div>

                {/* 2x2 Grid for 4 agents */}
                <div className="grid gap-3 grid-cols-2">
                    {agents.map((agent) => (
                        <AgentCard key={agent.id} agent={agent} />
                    ))}
                </div>
            </div>
        </motion.div>
    );
}
