"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const greetings = {
    morning: "Good morning",
    afternoon: "Good afternoon",
    evening: "Good evening",
};

function getGreeting(): keyof typeof greetings {
    const hour = new Date().getHours();
    if (hour >= 5 && hour < 12) return "morning";
    if (hour >= 12 && hour < 18) return "afternoon";
    return "evening";
}

export function AnimatedGreeting() {
    const [greeting, setGreeting] = useState<keyof typeof greetings>("morning");

    useEffect(() => {
        setGreeting(getGreeting());

        // Update every minute in case time changes
        const interval = setInterval(() => {
            const newGreeting = getGreeting();
            if (newGreeting !== greeting) {
                setGreeting(newGreeting);
            }
        }, 60000);

        return () => clearInterval(interval);
    }, [greeting]);

    return (
        <div className="relative h-[80px] w-full flex items-center justify-center overflow-hidden">
            <AnimatePresence mode="wait">
                <motion.div
                    key={greeting}
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 10 }}
                    transition={{
                        duration: 0.8,
                        ease: [0.16, 1, 0.3, 1],
                    }}
                    className="text-center"
                >
                    <h1 className="font-display text-4xl md:text-5xl font-normal tracking-tight">
                        <span className="bg-gradient-to-r from-muted-foreground/70 via-muted-foreground/50 to-muted-foreground/30 bg-clip-text text-transparent">
                            {greetings[greeting]}
                        </span>
                    </h1>
                    <p className="mt-2 text-sm text-muted-foreground/60">
                        What would you like to explore today?
                    </p>
                </motion.div>
            </AnimatePresence>
        </div>
    );
}
