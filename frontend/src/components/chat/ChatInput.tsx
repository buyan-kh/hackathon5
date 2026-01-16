"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    IconSend,
    IconSquare,
    IconWorld,
    IconChartLine,
    IconNews,
    IconSearch,
    IconSparkles,
    IconPlugConnected,
    IconPlugConnectedX,
} from "@tabler/icons-react";
import { ChatEditor, Editor } from "./ChatEditor";
import { Button } from "@/components/ui";
import { useChatStore } from "@/stores/chat.store";
import { cn } from "@/lib/utils";

interface ChatInputProps {
    onSubmit?: (message: string) => void;
    onPaperGenerated?: (paper: Record<string, unknown>) => void;
    showGreeting?: boolean;
    onEditorReady?: (editor: Editor) => void;
}

const chatModes = [
    { id: "scout" as const, label: "Scout", icon: IconSearch, description: "Web research" },
    { id: "simulate" as const, label: "Simulate", icon: IconChartLine, description: "Market simulation" },
    { id: "paper" as const, label: "Paper", icon: IconNews, description: "Tomorrow's news" },
];

export function ChatInput({ onSubmit, onPaperGenerated, showGreeting = true, onEditorReady }: ChatInputProps) {
    const editorRef = useRef<Editor | null>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const [hasContent, setHasContent] = useState(false);
    const [isConnected, setIsConnected] = useState(false);

    const {
        isGenerating,
        setIsGenerating,
        useWebSearch,
        setUseWebSearch,
        chatMode,
        setChatMode,
        createThread,
        currentThreadId,
        setCurrentThread,
        addMessage,
        updateAgentStatus,
        resetAgents,
        setPaperData,
    } = useChatStore();

    // WebSocket connection
    useEffect(() => {
        const connect = () => {
            const ws = new WebSocket("ws://localhost:8000/ws");

            ws.onopen = () => {
                console.log("✅ Connected to backend");
                setIsConnected(true);
            };

            ws.onclose = () => {
                console.log("❌ Disconnected from backend");
                setIsConnected(false);
                // Reconnect after 3 seconds
                setTimeout(connect, 3000);
            };

            ws.onerror = (error) => {
                console.error("WebSocket error:", error);
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    console.log("📨 Agent update:", data);

                    if (data.event_type === "agent_update") {
                        const agentId = data.agent;
                        if (agentId) {
                            updateAgentStatus(agentId, {
                                status: data.data.status,
                                progress: data.data.progress || 0,
                                currentTask: data.data.task,
                                sources: data.data.sources,
                            });
                        }
                    } else if (data.event_type === "complete") {
                        console.log("🎉 Generation complete:", data.data);
                        setIsGenerating(false);
                        // Mark any remaining running agents as completed
                        const state = useChatStore.getState();
                        state.agents.forEach((agent) => {
                            if (agent.status === "running") {
                                updateAgentStatus(agent.id, {
                                    status: "completed",
                                    progress: 100,
                                });
                            }
                        });
                        resetAgents();

                        // Pass paper to parent and save to thread
                        if (data.data?.paper) {
                            const paper = data.data.paper;
                            console.log("📰 Paper generated:", paper);
                            onPaperGenerated?.(paper);

                            // Save paper to current thread for history
                            const threadId = useChatStore.getState().currentThreadId;
                            if (threadId) {
                                // Only save URLs, not base64 images to prevent localStorage quota issues
                                const coverImage = paper.cover_image_url?.startsWith('data:') 
                                    ? undefined 
                                    : paper.cover_image_url;
                                
                                setPaperData(threadId, {
                                    id: paper.paper_id,
                                    headline: paper.headline,
                                    subheadline: paper.subheadline,
                                    date: paper.date,
                                    query: paper.query,
                                    articles: (paper.articles || []).map(article => ({
                                        ...article,
                                        // Remove base64 images from articles too
                                        image: article.image?.startsWith('data:') 
                                            ? undefined 
                                            : article.image,
                                    })),
                                    generatedAt: new Date(),
                                    coverImage,
                                    secondaryImageUrl: paper.secondary_image_url?.startsWith('data:') 
                                        ? undefined 
                                        : paper.secondary_image_url,
                                    tertiaryImageUrl: paper.tertiary_image_url?.startsWith('data:') 
                                        ? undefined 
                                        : paper.tertiary_image_url,
                                    marketSnapshot: paper.market_snapshot,
                                    trendingTopics: paper.trending_topics,
                                    newsContext: paper.news_context,
                                });
                            }
                        }
                    } else if (data.event_type === "error") {
                        const errorMessage = data.data?.message || data.data?.error || JSON.stringify(data.data) || "Unknown error";
                        console.error("❌ Error:", errorMessage, data);
                        setIsGenerating(false);
                        resetAgents();
                    } else if (data.event_type === "start") {
                        console.log("🚀 Started processing:", data.data);
                        resetAgents();
                    }
                } catch (e) {
                    console.error("Failed to parse message:", e);
                }
            };

            wsRef.current = ws;
        };

        connect();

        return () => {
            wsRef.current?.close();
        };
    }, [updateAgentStatus, setIsGenerating, resetAgents, addMessage]);

    const handleSubmit = useCallback(
        (content: string) => {
            if (!content.trim() || isGenerating) return;

            // Create a new thread
            const threadId = createThread(content.slice(0, 50) + (content.length > 50 ? "..." : ""));

            // Add user message
            addMessage(threadId, {
                role: "user",
                content,
            });

            // Trigger external handler
            onSubmit?.(content);

            // Reset agents and start generating
            resetAgents();
            setIsGenerating(true);

            // Send to backend via WebSocket
            if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(
                    JSON.stringify({
                        action: "query",
                        query: content,
                        mode: chatMode,
                        use_web_search: useWebSearch,
                    })
                );
                console.log("📤 Sent query to backend:", content);
            } else {
                console.error("WebSocket not connected!");
                setIsGenerating(false);
            }
        },
        [isGenerating, createThread, addMessage, onSubmit, setIsGenerating, chatMode, useWebSearch, resetAgents]
    );

    const handleEditorReady = useCallback((editor: Editor) => {
        editorRef.current = editor;
        editor.on("update", () => {
            setHasContent(!!editor.getText().trim());
        });
        onEditorReady?.(editor);
    }, [onEditorReady]);

    const handleStop = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ action: "stop" }));
        }
        setIsGenerating(false);
        resetAgents();
    }, [setIsGenerating, resetAgents]);

    return (
        <div className="w-full px-3">
            <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2, ease: "easeOut" }}
                className="w-full"
            >
                <div
                    className={cn(
                        "relative z-10 w-full rounded-xl border border-border/50 bg-card",
                        "shadow-lg transition-shadow duration-200",
                        "focus-within:shadow-xl focus-within:border-accent/20"
                    )}
                >
                    {/* Editor Area */}
                    <div className="px-4 pt-4 pb-2">
                        <ChatEditor
                            onSubmit={handleSubmit}
                            placeholder="What happens if oil prices spike 40% tomorrow?"
                            disabled={isGenerating}
                            onEditorReady={handleEditorReady}
                            className="min-h-[60px]"
                        />
                    </div>

                    {/* Action Bar */}
                    <div className="flex items-center justify-between gap-2 border-t border-dashed border-border px-3 py-2">
                        {/* Left Actions */}
                        <div className="flex items-center gap-1">
                            {/* Connection Status */}
                            <div
                                className={cn(
                                    "flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs",
                                    isConnected ? "text-green-600" : "text-red-500"
                                )}
                                title={isConnected ? "Connected to backend" : "Disconnected"}
                            >
                                {isConnected ? <IconPlugConnected size={14} /> : <IconPlugConnectedX size={14} />}
                            </div>

                            {/* Chat Mode Selector */}
                            <div className="flex items-center gap-0.5 rounded-lg bg-muted/50 p-0.5">
                                {chatModes.map((mode) => (
                                    <button
                                        key={mode.id}
                                        onClick={() => setChatMode(mode.id)}
                                        className={cn(
                                            "flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-all",
                                            chatMode === mode.id
                                                ? "bg-card text-foreground shadow-sm"
                                                : "text-muted-foreground hover:text-foreground"
                                        )}
                                    >
                                        <mode.icon size={14} />
                                        <span className="hidden sm:inline">{mode.label}</span>
                                    </button>
                                ))}
                            </div>

                            {/* Web Search Toggle */}
                            <button
                                onClick={() => setUseWebSearch(!useWebSearch)}
                                className={cn(
                                    "flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium transition-all",
                                    useWebSearch
                                        ? "bg-accent/10 text-accent"
                                        : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                                )}
                            >
                                <IconWorld size={14} />
                                <span className="hidden sm:inline">Web</span>
                            </button>
                        </div>

                        {/* Right Actions */}
                        <div className="flex items-center gap-2">
                            <AnimatePresence mode="wait">
                                {isGenerating ? (
                                    <motion.div
                                        key="generating"
                                        initial={{ opacity: 0, scale: 0.9 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        exit={{ opacity: 0, scale: 0.9 }}
                                        className="flex items-center gap-2"
                                    >
                                        <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                                            <IconSparkles size={14} className="animate-pulse text-accent" />
                                            Generating...
                                        </span>
                                        <Button size="icon-sm" variant="outline" onClick={handleStop} className="h-8 w-8">
                                            <IconSquare size={14} />
                                        </Button>
                                    </motion.div>
                                ) : (
                                    <motion.div
                                        key="send"
                                        initial={{ opacity: 0, scale: 0.9 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        exit={{ opacity: 0, scale: 0.9 }}
                                    >
                                        <Button
                                            size="icon-sm"
                                            onClick={() => {
                                                const content = editorRef.current?.getText().trim();
                                                if (content) {
                                                    handleSubmit(content);
                                                    editorRef.current?.commands.clearContent();
                                                }
                                            }}
                                            disabled={!hasContent || !isConnected}
                                            className={cn("h-8 w-8 transition-all", hasContent && isConnected && "shadow-accent")}
                                        >
                                            <IconSend size={14} />
                                        </Button>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>
                    </div>
                </div>
            </motion.div>
        </div>
    );
}
