"use client";

import { useMemo } from "react";
import {
    AnimatedAxis,
    AnimatedGrid,
    AnimatedLineSeries,
    AnimatedAreaSeries,
    XYChart,
    Tooltip,
    buildChartTheme,
} from "@visx/xychart";
import { curveMonotoneX } from "@visx/curve";
import { LinearGradient } from "@visx/gradient";
import { cn } from "@/lib/utils";

export interface DataPoint {
    date: string;
    value: number;
    projected?: boolean;
}

interface SimulationChartProps {
    data: DataPoint[];
    title?: string;
    subtitle?: string;
    currentValue?: number;
    projectedValue?: number;
    changePercent?: number;
    height?: number;
    className?: string;
}

// Custom theme matching our design system
const chartTheme = buildChartTheme({
    backgroundColor: "transparent",
    colors: ["#0052FF", "#4D7CFF"],
    tickLength: 4,
    gridColor: "#E2E8F0",
    gridColorDark: "#334155",
});

const accessors = {
    xAccessor: (d: DataPoint) => d.date,
    yAccessor: (d: DataPoint) => d.value,
};

export function SimulationChart({
    data,
    title,
    subtitle,
    currentValue,
    projectedValue,
    changePercent,
    height = 300,
    className,
}: SimulationChartProps) {
    // Split data into current and projected
    const { currentData, projectedData } = useMemo(() => {
        const current: DataPoint[] = [];
        const projected: DataPoint[] = [];

        data.forEach((point, index) => {
            if (point.projected) {
                // Include the last current point for continuity
                if (projected.length === 0 && current.length > 0) {
                    projected.push(current[current.length - 1]);
                }
                projected.push(point);
            } else {
                current.push(point);
            }
        });

        return { currentData: current, projectedData: projected };
    }, [data]);

    const isPositiveChange = (changePercent ?? 0) >= 0;

    return (
        <div className={cn("w-full rounded-xl border border-border bg-card p-4", className)}>
            {/* Header */}
            {(title || subtitle) && (
                <div className="mb-4 flex items-start justify-between">
                    <div>
                        {title && (
                            <h3 className="text-lg font-semibold">{title}</h3>
                        )}
                        {subtitle && (
                            <p className="text-sm text-muted-foreground">{subtitle}</p>
                        )}
                    </div>

                    {/* Value Display */}
                    {(currentValue !== undefined || projectedValue !== undefined) && (
                        <div className="text-right">
                            {projectedValue !== undefined && (
                                <div className="text-2xl font-bold">
                                    ${projectedValue.toLocaleString()}
                                </div>
                            )}
                            {changePercent !== undefined && (
                                <div className={cn(
                                    "text-sm font-medium",
                                    isPositiveChange ? "text-green-600" : "text-red-600"
                                )}>
                                    {isPositiveChange ? "+" : ""}{changePercent.toFixed(1)}%
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}

            {/* Chart */}
            <div style={{ height }}>
                <XYChart
                    height={height}
                    xScale={{ type: "band", paddingInner: 0.3 }}
                    yScale={{ type: "linear", nice: true }}
                    theme={chartTheme}
                >
                    {/* Gradient Definition */}
                    <LinearGradient
                        id="area-gradient"
                        from="#0052FF"
                        to="#0052FF"
                        fromOpacity={0.3}
                        toOpacity={0.05}
                    />
                    <LinearGradient
                        id="area-gradient-projected"
                        from="#4D7CFF"
                        to="#4D7CFF"
                        fromOpacity={0.2}
                        toOpacity={0.02}
                    />

                    {/* Grid */}
                    <AnimatedGrid
                        columns={false}
                        numTicks={4}
                        lineStyle={{
                            stroke: "#E2E8F0",
                            strokeOpacity: 0.5,
                            strokeDasharray: "4,4",
                        }}
                    />

                    {/* Current Data Area */}
                    {currentData.length > 0 && (
                        <AnimatedAreaSeries
                            dataKey="Current"
                            data={currentData}
                            {...accessors}
                            curve={curveMonotoneX}
                            fill="url(#area-gradient)"
                            lineProps={{ stroke: "#0052FF", strokeWidth: 2 }}
                        />
                    )}

                    {/* Projected Data Area */}
                    {projectedData.length > 0 && (
                        <AnimatedAreaSeries
                            dataKey="Projected"
                            data={projectedData}
                            {...accessors}
                            curve={curveMonotoneX}
                            fill="url(#area-gradient-projected)"
                            lineProps={{
                                stroke: "#4D7CFF",
                                strokeWidth: 2,
                                strokeDasharray: "6,4"
                            }}
                        />
                    )}

                    {/* Current Data Line */}
                    {currentData.length > 0 && (
                        <AnimatedLineSeries
                            dataKey="Current Line"
                            data={currentData}
                            {...accessors}
                            curve={curveMonotoneX}
                            stroke="#0052FF"
                            strokeWidth={2.5}
                        />
                    )}

                    {/* Projected Data Line */}
                    {projectedData.length > 0 && (
                        <AnimatedLineSeries
                            dataKey="Projected Line"
                            data={projectedData}
                            {...accessors}
                            curve={curveMonotoneX}
                            stroke="#4D7CFF"
                            strokeWidth={2.5}
                            strokeDasharray="6,4"
                        />
                    )}

                    {/* Axes */}
                    <AnimatedAxis
                        orientation="bottom"
                        tickLabelProps={{
                            fill: "#64748B",
                            fontSize: 11,
                            fontFamily: "var(--font-inter)",
                        }}
                        hideAxisLine
                        hideTicks
                    />
                    <AnimatedAxis
                        orientation="left"
                        numTicks={4}
                        tickFormat={(v) => `$${Number(v).toLocaleString()}`}
                        tickLabelProps={{
                            fill: "#64748B",
                            fontSize: 11,
                            fontFamily: "var(--font-inter)",
                            dx: -8,
                        }}
                        hideAxisLine
                        hideTicks
                    />

                    {/* Tooltip */}
                    <Tooltip
                        snapTooltipToDatumX
                        snapTooltipToDatumY
                        showVerticalCrosshair
                        showSeriesGlyphs
                        glyphStyle={{
                            fill: "#0052FF",
                            strokeWidth: 2,
                            stroke: "#FFFFFF",
                        }}
                        renderTooltip={({ tooltipData }) => {
                            const datum = tooltipData?.nearestDatum?.datum as DataPoint | undefined;
                            if (!datum) return null;

                            return (
                                <div className="rounded-lg border border-border bg-card p-2 shadow-lg">
                                    <div className="text-xs text-muted-foreground">{datum.date}</div>
                                    <div className="text-sm font-semibold">
                                        ${datum.value.toLocaleString()}
                                    </div>
                                    {datum.projected && (
                                        <div className="text-xs text-accent">Projected</div>
                                    )}
                                </div>
                            );
                        }}
                    />
                </XYChart>
            </div>

            {/* Legend */}
            <div className="mt-4 flex items-center gap-4 text-xs text-muted-foreground">
                <div className="flex items-center gap-1.5">
                    <div className="h-0.5 w-4 rounded bg-accent" />
                    <span>Current</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <div className="h-0.5 w-4 rounded bg-accent-secondary" style={{ background: "repeating-linear-gradient(90deg, #4D7CFF, #4D7CFF 4px, transparent 4px, transparent 8px)" }} />
                    <span>Projected</span>
                </div>
            </div>
        </div>
    );
}
