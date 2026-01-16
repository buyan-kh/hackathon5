"use client";

import { useCallback, useRef, useEffect, useState } from "react";

interface AgentUpdate {
    event_type: string;
    agent: string | null;
    data: Record<string, unknown>;
    timestamp: string;
}

interface UseAgentStreamOptions {
    onAgentUpdate?: (update: AgentUpdate) => void;
    onComplete?: (data: Record<string, unknown>) => void;
    onError?: (error: string) => void;
}

export function useAgentStream(options: UseAgentStreamOptions = {}) {
    const wsRef = useRef<WebSocket | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);

    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) return;

        const ws = new WebSocket("ws://localhost:8000/ws");

        ws.onopen = () => {
            console.log("WebSocket connected");
            setIsConnected(true);
        };

        ws.onclose = () => {
            console.log("WebSocket disconnected");
            setIsConnected(false);
            // Reconnect after 3 seconds
            setTimeout(connect, 3000);
        };

        ws.onerror = (error) => {
            console.error("WebSocket error:", error);
            options.onError?.("Connection error");
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data) as AgentUpdate;
                console.log("Agent update:", data);

                if (data.event_type === "agent_update") {
                    options.onAgentUpdate?.(data);
                } else if (data.event_type === "complete") {
                    setIsProcessing(false);
                    options.onComplete?.(data.data);
                } else if (data.event_type === "error") {
                    setIsProcessing(false);
                    options.onError?.(String(data.data?.message || "Unknown error"));
                }
            } catch (e) {
                console.error("Failed to parse WebSocket message:", e);
            }
        };

        wsRef.current = ws;
    }, [options]);

    const disconnect = useCallback(() => {
        wsRef.current?.close();
        wsRef.current = null;
        setIsConnected(false);
    }, []);

    const sendQuery = useCallback(
        (query: string, mode: string = "paper", useWebSearch: boolean = true) => {
            if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
                console.error("WebSocket not connected");
                options.onError?.("Not connected to server");
                return;
            }

            setIsProcessing(true);

            wsRef.current.send(
                JSON.stringify({
                    action: "query",
                    query,
                    mode,
                    use_web_search: useWebSearch,
                })
            );
        },
        [options]
    );

    const stopProcessing = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ action: "stop" }));
        }
        setIsProcessing(false);
    }, []);

    // Auto-connect on mount
    useEffect(() => {
        connect();
        return () => disconnect();
    }, [connect, disconnect]);

    return {
        isConnected,
        isProcessing,
        sendQuery,
        stopProcessing,
        connect,
        disconnect,
    };
}
