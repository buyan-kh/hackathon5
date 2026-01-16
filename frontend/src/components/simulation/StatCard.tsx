"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import {
    IconTrendingUp,
    IconTrendingDown,
    IconMinus
} from "@tabler/icons-react";

interface StatCardProps {
    label: string;
    value: string | number;
    change?: number;
    changeLabel?: string;
    icon?: React.ReactNode;
    className?: string;
}

export function StatCard({
    label,
    value,
    change,
    changeLabel,
    icon,
    className,
}: StatCardProps) {
    const isPositive = change !== undefined && change >= 0;
    const isNeutral = change === 0;

    const TrendIcon = isNeutral
        ? IconMinus
        : isPositive
            ? IconTrendingUp
            : IconTrendingDown;

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={cn(
                "rounded-xl border border-border bg-card p-4",
                "transition-shadow duration-200 hover:shadow-md",
                className
            )}
        >
            <div className="flex items-start justify-between">
                <div>
                    <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                        {label}
                    </p>
                    <p className="mt-1 text-2xl font-bold">
                        {typeof value === "number" ? value.toLocaleString() : value}
                    </p>
                </div>
                {icon && (
                    <div className="rounded-lg bg-muted/50 p-2">
                        {icon}
                    </div>
                )}
            </div>

            {change !== undefined && (
                <div className="mt-3 flex items-center gap-1">
                    <div className={cn(
                        "flex items-center gap-0.5 rounded px-1.5 py-0.5 text-xs font-medium",
                        isNeutral && "bg-muted text-muted-foreground",
                        isPositive && !isNeutral && "bg-green-500/10 text-green-600",
                        !isPositive && !isNeutral && "bg-red-500/10 text-red-600",
                    )}>
                        <TrendIcon size={12} />
                        <span>{isPositive && !isNeutral ? "+" : ""}{change.toFixed(1)}%</span>
                    </div>
                    {changeLabel && (
                        <span className="text-xs text-muted-foreground">{changeLabel}</span>
                    )}
                </div>
            )}
        </motion.div>
    );
}

interface StatGridProps {
    stats: Array<{
        label: string;
        value: string | number;
        change?: number;
        changeLabel?: string;
    }>;
    className?: string;
}

export function StatGrid({ stats, className }: StatGridProps) {
    return (
        <div className={cn("grid grid-cols-2 gap-3 md:grid-cols-4", className)}>
            {stats.map((stat, index) => (
                <motion.div
                    key={stat.label}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                >
                    <StatCard {...stat} />
                </motion.div>
            ))}
        </div>
    );
}
