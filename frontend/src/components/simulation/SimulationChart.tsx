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

interface MarketAsset {
    asset: string;
    current_value: number;
    projected_value: number;
    change_percent: number;
    history?: { date: string; value: number }[];
}

interface SimulationChartProps {
    asset: MarketAsset;
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

interface ChartPoint {
    date: string;
    value: number;
    type: "history" | "projected";
}

const accessors = {
    xAccessor: (d: ChartPoint) => d.date,
    yAccessor: (d: ChartPoint) => d.value,
};

export function SimulationChart({
    asset,
    height = 250,
    className,
}: SimulationChartProps) {
    const { historyData, projectedData, fullData } = useMemo(() => {
        const history: ChartPoint[] = (asset.history || []).map(h => ({
            ...h,
            type: "history"
        }));

        // If no history, mock some for visualization
        if (history.length === 0) {
            const base = asset.current_value;
            for (let i = 10; i > 0; i--) {
                history.push({
                    date: `T-${i}`,
                    value: base * (1 + (Math.random() * 0.02 - 0.01)),
                    type: "history"
                });
            }
        }

        // Add current point as the bridge
        const lastDate = history[history.length - 1]?.date || "Today";
        // Ensure strictly historical sequence if using real dates
        const currentPoint: ChartPoint = {
            date: "Today",
            value: asset.current_value,
            type: "history"
        };

        if (history[history.length - 1]?.date !== "Today") {
            history.push(currentPoint);
        }

        const projectDate = "Future";
        const projected: ChartPoint[] = [
            currentPoint, // Start projection from current
            {
                date: projectDate,
                value: asset.projected_value,
                type: "projected"
            }
        ];

        return {
            historyData: history,
            projectedData: projected,
            fullData: [...history, projected[1]]
        };
    }, [asset]);

    const isPositive = asset.change_percent >= 0;

    return (
        <div className={cn("w-full rounded-xl border border-border bg-white shadow-sm p-4", className)}>
            <div className="mb-4 flex items-start justify-between">
                <div>
                    <h3 className="text-lg font-bold font-serif text-[#1a1a1a]">{asset.asset}</h3>
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">Market Forecast</p>
                </div>
                <div className="text-right">
                    <div className="text-2xl font-bold tabular-nums">
                        {asset.projected_value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                    </div>
                    <div className={cn(
                        "text-xs font-bold",
                        isPositive ? "text-green-600" : "text-red-600"
                    )}>
                        {isPositive ? "▲" : "▼"} {Math.abs(asset.change_percent).toFixed(2)}%
                    </div>
                </div>
            </div>

            <div style={{ height }}>
                <XYChart
                    height={height}
                    xScale={{ type: "band", paddingInner: 0.3 }}
                    yScale={{ type: "linear", nice: true }}
                    theme={chartTheme}
                >
                    <LinearGradient
                        id="hist-gradient"
                        from="#1a1a1a"
                        to="#1a1a1a"
                        fromOpacity={0.1}
                        toOpacity={0.01}
                    />
                    <LinearGradient
                        id="proj-gradient"
                        from={isPositive ? "#16a34a" : "#dc2626"}
                        to={isPositive ? "#16a34a" : "#dc2626"}
                        fromOpacity={0.2}
                        toOpacity={0.02}
                    />

                    <AnimatedGrid
                        columns={false}
                        numTicks={4}
                        lineStyle={{
                            stroke: "#E2E8F0",
                            strokeOpacity: 0.8,
                            strokeDasharray: "4,4",
                        }}
                    />

                    {/* Historical Area */}
                    <AnimatedAreaSeries
                        dataKey="History"
                        data={historyData}
                        {...accessors}
                        curve={curveMonotoneX}
                        fill="url(#hist-gradient)"
                        lineProps={{ stroke: "#333", strokeWidth: 2 }}
                    />

                    {/* Projected Area */}
                    <AnimatedAreaSeries
                        dataKey="Projected"
                        data={projectedData}
                        {...accessors}
                        curve={curveMonotoneX}
                        fill="url(#proj-gradient)"
                        lineProps={{
                            stroke: isPositive ? "#16a34a" : "#dc2626",
                            strokeWidth: 2,
                            strokeDasharray: "4,4"
                        }}
                    />

                    <Tooltip
                        snapTooltipToDatumX
                        snapTooltipToDatumY
                        showVerticalCrosshair
                        showSeriesGlyphs
                        renderTooltip={({ tooltipData }) => {
                            const datum = tooltipData?.nearestDatum?.datum as ChartPoint | undefined;
                            if (!datum) return null;
                            return (
                                <div className="bg-white border border-[#e0e0e0] p-2 shadow-xl rounded text-xs">
                                    <div className="font-bold mb-0.5">{datum.date}</div>
                                    <div className="tabular-nums font-mono">
                                        {datum.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                                    </div>
                                    {datum.type === "projected" && (
                                        <div className="text-[10px] text-muted-foreground italic">Forecast</div>
                                    )}
                                </div>
                            );
                        }}
                    />
                </XYChart>
            </div>

            <div className="mt-3 flex items-center justify-center gap-6 text-[10px] uppercase text-[#666] tracking-wider font-semibold">
                <div className="flex items-center gap-2">
                    <div className="w-3 h-0.5 bg-[#333]"></div>
                    <span>Historical</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-3 h-0.5 border-t border-dashed border-CURRENT" style={{ borderColor: isPositive ? "#16a34a" : "#dc2626" }}></div>
                    <span>Projected</span>
                </div>
            </div>
        </div>
    );
}
