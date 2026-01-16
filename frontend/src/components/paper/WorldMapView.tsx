"use client";

import React, { useState } from "react";
import { ComposableMap, Geographies, Geography, ZoomableGroup } from "react-simple-maps";
import { scaleLinear } from "d3-scale";
import { Tooltip } from "@radix-ui/react-tooltip";

const geoUrl = "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json";

interface WorldMapProps {
    highlightedCountries?: { [key: string]: number }; // country name -> intensity (0-1)
    onCountryClick?: (countryName: string) => void;
    className?: string;
}

export function WorldMapView({
    highlightedCountries = {},
    onCountryClick,
    className
}: WorldMapProps) {
    const [tooltipContent, setTooltipContent] = useState("");

    const colorScale = scaleLinear<string>()
        .domain([0, 1])
        .range(["#ffedea", "#ff5233"]);

    return (
        <div className={className} data-tip="">
            <ComposableMap projection="geoMercator" projectionConfig={{ scale: 100 }}>
                <ZoomableGroup zoom={1}>
                    <Geographies geography={geoUrl}>
                        {({ geographies }) =>
                            geographies.map((geo) => {
                                const countryName = geo.properties.name;
                                const intensity = highlightedCountries[countryName] || 0;
                                const isHighlighted = intensity > 0;

                                return (
                                    <Geography
                                        key={geo.rsmKey}
                                        geography={geo}
                                        onMouseEnter={() => {
                                            setTooltipContent(`${countryName}`);
                                        }}
                                        onMouseLeave={() => {
                                            setTooltipContent("");
                                        }}
                                        onClick={() => onCountryClick && onCountryClick(countryName)}
                                        style={{
                                            default: {
                                                fill: isHighlighted ? colorScale(intensity) : "#F5F4F6",
                                                stroke: "#D6D6DA",
                                                strokeWidth: 0.5,
                                                outline: "none",
                                            },
                                            hover: {
                                                fill: isHighlighted ? "#c4161c" : "#999",
                                                outline: "none",
                                                cursor: "pointer",
                                            },
                                            pressed: {
                                                fill: "#E42",
                                                outline: "none",
                                            },
                                        }}
                                    />
                                );
                            })
                        }
                    </Geographies>
                </ZoomableGroup>
            </ComposableMap>
            {tooltipContent && (
                <div className="absolute top-2 left-2 bg-black text-white text-xs px-2 py-1 rounded pointer-events-none">
                    {tooltipContent}
                </div>
            )}
        </div>
    );
}

// Map of common country name variations to standard names in the geoJSON
export const normalizeCountryName = (name: string): string => {
    const map: { [key: string]: string } = {
        "United States": "United States of America",
        "USA": "United States of America",
        "US": "United States of America",
        "UK": "United Kingdom",
        "Great Britain": "United Kingdom",
        "Russia": "Russian Federation",
        "China": "China",
        "Japan": "Japan",
        "Germany": "Germany",
        "France": "France",
        // Add more as needed
    };
    return map[name] || name;
};
