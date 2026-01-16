import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface Message {
    id: string;
    role: "user" | "assistant" | "system";
    content: string;
    timestamp: Date;
    agentData?: {
        agentId: string;
        agentName: string;
        status: "pending" | "running" | "completed" | "error";
        progress?: number;
    };
}

export interface Thread {
    id: string;
    title: string;
    messages: Message[];
    createdAt: Date;
    updatedAt: Date;
    simulationData?: SimulationData;
    paperData?: PaperData;
}

export interface SimulationData {
    id: string;
    query: string;
    status: "pending" | "running" | "completed" | "error";
    results?: SimulationResult[];
}

export interface SimulationResult {
    asset: string;
    currentValue: number;
    projectedValue: number;
    change: number;
    changePercent: number;
    timeline: { date: string; value: number }[];
}

export interface PaperData {
    id: string;
    headline: string;
    articles: Article[];
    generatedAt: Date;
    coverImage?: string;
}

export interface Article {
    id: string;
    title: string;
    content: string;
    category: string;
    image?: string;
}

export interface AgentStatus {
    id: string;
    name: string;
    type: "yutori" | "fabricate" | "freepik";
    status: "idle" | "running" | "completed" | "error";
    progress: number;
    currentTask?: string;
    sources?: number;
}

interface ChatState {
    // Current thread
    currentThreadId: string | null;
    threads: Thread[];

    // Agent statuses
    agents: AgentStatus[];

    // UI state
    isGenerating: boolean;
    useWebSearch: boolean;
    chatMode: "scout" | "simulate" | "paper";

    // Actions
    createThread: (title: string) => string;
    setCurrentThread: (threadId: string | null) => void;
    addMessage: (threadId: string, message: Omit<Message, "id" | "timestamp">) => void;
    updateMessage: (threadId: string, messageId: string, updates: Partial<Message>) => void;

    setIsGenerating: (isGenerating: boolean) => void;
    setUseWebSearch: (useWebSearch: boolean) => void;
    setChatMode: (mode: "scout" | "simulate" | "paper") => void;

    updateAgentStatus: (agentId: string, updates: Partial<AgentStatus>) => void;
    resetAgents: () => void;

    setSimulationData: (threadId: string, data: SimulationData) => void;
    setPaperData: (threadId: string, data: PaperData) => void;

    clearThread: (threadId: string) => void;
}

const initialAgents: AgentStatus[] = [
    { id: "yutori", name: "Yutori", type: "yutori", status: "idle", progress: 0 },
    { id: "fabricate", name: "Tonic Fabricate", type: "fabricate", status: "idle", progress: 0 },
    { id: "freepik", name: "Freepik", type: "freepik", status: "idle", progress: 0 },
];

export const useChatStore = create<ChatState>()(
    persist(
        (set, get) => ({
            currentThreadId: null,
            threads: [],
            agents: initialAgents,
            isGenerating: false,
            useWebSearch: true,
            chatMode: "paper",

            createThread: (title: string) => {
                const id = crypto.randomUUID();
                const newThread: Thread = {
                    id,
                    title,
                    messages: [],
                    createdAt: new Date(),
                    updatedAt: new Date(),
                };
                set((state) => ({
                    threads: [newThread, ...state.threads],
                    currentThreadId: id,
                }));
                return id;
            },

            setCurrentThread: (threadId) => set({ currentThreadId: threadId }),

            addMessage: (threadId, message) => {
                const newMessage: Message = {
                    ...message,
                    id: crypto.randomUUID(),
                    timestamp: new Date(),
                };
                set((state) => ({
                    threads: state.threads.map((thread) =>
                        thread.id === threadId
                            ? {
                                ...thread,
                                messages: [...thread.messages, newMessage],
                                updatedAt: new Date(),
                            }
                            : thread
                    ),
                }));
            },

            updateMessage: (threadId, messageId, updates) => {
                set((state) => ({
                    threads: state.threads.map((thread) =>
                        thread.id === threadId
                            ? {
                                ...thread,
                                messages: thread.messages.map((msg) =>
                                    msg.id === messageId ? { ...msg, ...updates } : msg
                                ),
                                updatedAt: new Date(),
                            }
                            : thread
                    ),
                }));
            },

            setIsGenerating: (isGenerating) => set({ isGenerating }),
            setUseWebSearch: (useWebSearch) => set({ useWebSearch }),
            setChatMode: (chatMode) => set({ chatMode }),

            updateAgentStatus: (agentId, updates) => {
                set((state) => ({
                    agents: state.agents.map((agent) =>
                        agent.id === agentId ? { ...agent, ...updates } : agent
                    ),
                }));
            },

            resetAgents: () => set({ agents: initialAgents }),

            setSimulationData: (threadId, data) => {
                set((state) => ({
                    threads: state.threads.map((thread) =>
                        thread.id === threadId
                            ? { ...thread, simulationData: data, updatedAt: new Date() }
                            : thread
                    ),
                }));
            },

            setPaperData: (threadId, data) => {
                set((state) => ({
                    threads: state.threads.map((thread) =>
                        thread.id === threadId
                            ? { ...thread, paperData: data, updatedAt: new Date() }
                            : thread
                    ),
                }));
            },

            clearThread: (threadId) => {
                set((state) => ({
                    threads: state.threads.filter((thread) => thread.id !== threadId),
                    currentThreadId:
                        state.currentThreadId === threadId ? null : state.currentThreadId,
                }));
            },
        }),
        {
            name: "tomorrows-paper-chat",
            partialize: (state) => ({
                threads: state.threads,
                chatMode: state.chatMode,
                useWebSearch: state.useWebSearch,
            }),
        }
    )
);
