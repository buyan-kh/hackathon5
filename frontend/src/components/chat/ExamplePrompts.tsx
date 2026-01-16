"use client";

import { motion } from "framer-motion";
import {
    IconTrendingUp,
    IconWorld,
    IconCpu,
    IconBuildingBank,
    IconBolt
} from "@tabler/icons-react";
import { Button } from "@/components/ui";

const promptCategories = [
    {
        id: "market",
        label: "Market",
        icon: IconTrendingUp,
        examples: [
            "What if oil prices spike 40% tomorrow?",
            "How would a Fed rate cut impact tech stocks?",
            "Simulate Bitcoin hitting $200k next month",
        ],
    },
    {
        id: "geopolitical",
        label: "Geopolitical",
        icon: IconWorld,
        examples: [
            "What if trade tensions escalate between US and China?",
            "Simulate impact of new EU regulations on AI companies",
            "How would a major currency crisis unfold?",
        ],
    },
    {
        id: "tech",
        label: "Tech",
        icon: IconCpu,
        examples: [
            "What if a major cloud provider goes down for 24 hours?",
            "Simulate the announcement of AGI breakthrough",
            "How would quantum computing affect crypto?",
        ],
    },
    {
        id: "economy",
        label: "Economy",
        icon: IconBuildingBank,
        examples: [
            "What if unemployment suddenly doubles?",
            "Simulate a housing market correction",
            "How would hyperinflation impact global markets?",
        ],
    },
    {
        id: "blackswan",
        label: "Black Swan",
        icon: IconBolt,
        examples: [
            "What if a major bank fails unexpectedly?",
            "Simulate a global internet outage",
            "How would discovery of alien life affect markets?",
        ],
    },
];

interface ExamplePromptsProps {
    onSelect?: (prompt: string) => void;
}

export function ExamplePrompts({ onSelect }: ExamplePromptsProps) {
    const handleCategoryClick = (categoryId: string) => {
        const category = promptCategories.find((c) => c.id === categoryId);
        if (category) {
            const randomPrompt = category.examples[Math.floor(Math.random() * category.examples.length)];
            onSelect?.(randomPrompt);
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="w-full px-6 pb-8"
        >
            <div className="flex flex-wrap justify-center gap-2">
                {promptCategories.map((category, index) => (
                    <motion.div
                        key={category.id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{
                            duration: 0.3,
                            delay: 0.4 + index * 0.1,
                            ease: [0.16, 1, 0.3, 1]
                        }}
                    >
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleCategoryClick(category.id)}
                            className="gap-1.5 rounded-full border-border/50 text-muted-foreground hover:text-foreground hover:border-accent/30 hover:bg-accent/5"
                        >
                            <category.icon size={14} className="opacity-70" />
                            {category.label}
                        </Button>
                    </motion.div>
                ))}
            </div>
        </motion.div>
    );
}
