import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

// Helper function to safely get timestamp from Date object or date string
const getTimestamp = (date: Date | string | null | undefined): number => {
    if (!date) {
        return 0;
    }
    if (date instanceof Date) {
        const time = date.getTime();
        return isNaN(time) ? 0 : time;
    }
    if (typeof date === 'string') {
        const parsed = new Date(date);
        const time = parsed.getTime();
        return isNaN(time) ? 0 : time;
    }
    return 0;
};

// Custom storage wrapper with quota error handling
const createSafeStorage = () => {
    const storage = {
        getItem: (name: string): string | null => {
            try {
                return localStorage.getItem(name);
            } catch (error) {
                console.error('Error reading from localStorage:', error);
                return null;
            }
        },
        setItem: (name: string, value: string): void => {
            try {
                localStorage.setItem(name, value);
            } catch (error) {
                if (error instanceof DOMException && error.name === 'QuotaExceededError') {
                    console.warn('⚠️ localStorage quota exceeded. Attempting cleanup...');
                    // Try to clear old data and retry
                    try {
                        const current = localStorage.getItem(name);
                        if (current) {
                            const parsed = JSON.parse(current);
                            if (parsed?.state?.threads) {
                                // Keep only last 5 threads
                                const limitedThreads = parsed.state.threads
                                    .sort((a: any, b: any) => {
                                        const timeA = typeof a.updatedAt === 'string' 
                                            ? new Date(a.updatedAt).getTime() 
                                            : (a.updatedAt?.getTime?.() || 0);
                                        const timeB = typeof b.updatedAt === 'string' 
                                            ? new Date(b.updatedAt).getTime() 
                                            : (b.updatedAt?.getTime?.() || 0);
                                        return timeB - timeA;
                                    })
                                    .slice(0, 5)
                                    .map((thread: any) => ({
                                        ...thread,
                                        paperData: thread.paperData ? {
                                            ...thread.paperData,
                                            coverImage: undefined,
                                        } : undefined,
                                    }));
                                
                                parsed.state.threads = limitedThreads;
                                localStorage.setItem(name, JSON.stringify(parsed));
                                return;
                            }
                        }
                    } catch (cleanupError) {
                        console.error('Error during cleanup:', cleanupError);
                        // If cleanup fails, clear everything
                        localStorage.removeItem(name);
                    }
                } else {
                    console.error('Error writing to localStorage:', error);
                }
            }
        },
        removeItem: (name: string): void => {
            try {
                localStorage.removeItem(name);
            } catch (error) {
                console.error('Error removing from localStorage:', error);
            }
        },
    };
    return storage;
};

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
    subheadline?: string;
    date?: string;
    query?: string;
    articles: Article[];
    generatedAt: Date;
    coverImage?: string;
    secondaryImageUrl?: string;
    tertiaryImageUrl?: string;
    marketSnapshot?: Array<{
        asset: string;
        value: number;
        change: number;
        changePercent: number;
    }>;
    trendingTopics?: Array<{
        topic: string;
        sentiment: number;
        mentions: number;
    }>;
    newsContext?: Array<{
        title: string;
        source: string;
        content?: string;
        agent?: string;
    }>;
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
    type: "yutori_news" | "yutori_sentiment" | "yutori_analysis" | "yutori_target" | "yutori_global" | "yutori_econ" | "fabricate" | "freepik";
    status: "idle" | "running" | "completed" | "error";
    progress: number;
    currentTask?: string;
    sources?: number;
}

interface ChatState {
    currentThreadId: string | null;
    threads: Thread[];
    agents: AgentStatus[];
    isGenerating: boolean;
    useWebSearch: boolean;
    chatMode: "scout" | "simulate" | "paper";

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

// 6 Yutori agents + Fabricate
const initialAgents: AgentStatus[] = [
    { id: "yutori_news", name: "News Scout", type: "yutori_news", status: "idle", progress: 0 },
    { id: "yutori_sentiment", name: "Sentiment Analysis", type: "yutori_sentiment", status: "idle", progress: 0 },
    { id: "yutori_analysis", name: "Expert Commentary", type: "yutori_analysis", status: "idle", progress: 0 },
    { id: "yutori_target", name: "Target Impact", type: "yutori_target", status: "idle", progress: 0 },
    { id: "yutori_global", name: "Global Reaction", type: "yutori_global", status: "idle", progress: 0 },
    { id: "yutori_econ", name: "Economic Outlook", type: "yutori_econ", status: "idle", progress: 0 },
    { id: "fabricate", name: "Market Simulation", type: "fabricate", status: "idle", progress: 0 },
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
                try {
                    const id = crypto.randomUUID();
                    const newThread: Thread = {
                        id,
                        title,
                        messages: [],
                        createdAt: new Date(),
                        updatedAt: new Date(),
                    };
                    set((state) => {
                        // Limit to 20 threads max to prevent quota issues
                        const updatedThreads = [newThread, ...state.threads].slice(0, 20);
                        return {
                            threads: updatedThreads,
                            currentThreadId: id,
                        };
                    });
                    return id;
                } catch (error) {
                    if (error instanceof DOMException && error.name === 'QuotaExceededError') {
                        console.warn('⚠️ localStorage quota exceeded. Clearing old threads...');
                        const state = get();
                        const recentThreads = state.threads
                            .sort((a, b) => getTimestamp(b.updatedAt) - getTimestamp(a.updatedAt))
                            .slice(0, 10);
                        
                        const id = crypto.randomUUID();
                        const newThread: Thread = {
                            id,
                            title,
                            messages: [],
                            createdAt: new Date(),
                            updatedAt: new Date(),
                        };
                        
                        set({
                            threads: [newThread, ...recentThreads],
                            currentThreadId: id,
                        });
                        return id;
                    }
                    throw error;
                }
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
                try {
                    // Strip base64 images to prevent localStorage quota issues
                    const sanitizedData: PaperData = {
                        ...data,
                        coverImage: data.coverImage?.startsWith('data:') ? undefined : data.coverImage,
                    };
                    
                    set((state) => ({
                        threads: state.threads.map((thread) =>
                            thread.id === threadId
                                ? { ...thread, paperData: sanitizedData, updatedAt: new Date() }
                                : thread
                        ),
                    }));
                } catch (error) {
                    // Handle quota exceeded error
                    if (error instanceof DOMException && error.name === 'QuotaExceededError') {
                        console.warn('⚠️ localStorage quota exceeded. Clearing old threads...');
                        // Keep only the most recent 10 threads
                        const state = get();
                        const recentThreads = state.threads
                            .sort((a, b) => getTimestamp(b.updatedAt) - getTimestamp(a.updatedAt))
                            .slice(0, 10);
                        
                        set({
                            threads: recentThreads,
                            currentThreadId: state.currentThreadId,
                        });
                        
                        // Retry setting paper data
                        set((state) => ({
                            threads: state.threads.map((thread) =>
                                thread.id === threadId
                                    ? { ...thread, paperData: { ...data, coverImage: undefined }, updatedAt: new Date() }
                                    : thread
                            ),
                        }));
                    } else {
                        console.error('Error setting paper data:', error);
                    }
                }
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
            storage: createJSONStorage(() => createSafeStorage()),
            partialize: (state) => {
                try {
                    // Limit threads to prevent quota issues (keep last 20)
                    // Filter out any invalid threads and ensure updatedAt exists
                    const validThreads = (state.threads || [])
                        .map(thread => {
                            // Ensure updatedAt exists and is valid
                            if (!thread.updatedAt) {
                                // If missing, use createdAt or current time
                                return {
                                    ...thread,
                                    updatedAt: thread.createdAt || new Date(),
                                };
                            }
                            return thread;
                        })
                        .filter(thread => thread && thread.id && thread.updatedAt);
                    
                    const limitedThreads = validThreads
                        .sort((a, b) => {
                            try {
                                // Handle both Date objects and date strings
                                const timeA = getTimestamp(a.updatedAt);
                                const timeB = getTimestamp(b.updatedAt);
                                return timeB - timeA;
                            } catch (e) {
                                console.warn('Error sorting threads:', e);
                                // Fallback: use thread ID for consistent ordering
                                return a.id.localeCompare(b.id);
                            }
                        })
                        .slice(0, 20)
                        .map(thread => ({
                            ...thread,
                            // Remove base64 images from persisted data
                            paperData: thread.paperData ? {
                                ...thread.paperData,
                                coverImage: thread.paperData.coverImage?.startsWith('data:') 
                                    ? undefined 
                                    : thread.paperData.coverImage,
                            } : undefined,
                        }));
                    
                    return {
                        threads: limitedThreads,
                        chatMode: state.chatMode,
                        useWebSearch: state.useWebSearch,
                    };
                } catch (error) {
                    console.error('Error in partialize:', error);
                    // Return minimal safe state
                    return {
                        threads: [],
                        chatMode: state.chatMode || 'paper',
                        useWebSearch: state.useWebSearch ?? true,
                    };
                }
            },
            onRehydrateStorage: () => {
                return (state, error) => {
                    if (error) {
                        console.error('Error rehydrating store:', error);
                        // Clear corrupted data
                        if (error instanceof DOMException && error.name === 'QuotaExceededError') {
                            localStorage.removeItem('tomorrows-paper-chat');
                        }
                    } else if (state) {
                        // Convert date strings back to Date objects after rehydration
                        if (state.threads) {
                            state.threads = state.threads.map(thread => ({
                                ...thread,
                                createdAt: typeof thread.createdAt === 'string' 
                                    ? new Date(thread.createdAt) 
                                    : thread.createdAt,
                                updatedAt: typeof thread.updatedAt === 'string' 
                                    ? new Date(thread.updatedAt) 
                                    : thread.updatedAt,
                                messages: thread.messages?.map(msg => ({
                                    ...msg,
                                    timestamp: typeof msg.timestamp === 'string' 
                                        ? new Date(msg.timestamp) 
                                        : msg.timestamp,
                                })) || [],
                                paperData: thread.paperData ? {
                                    ...thread.paperData,
                                    generatedAt: typeof thread.paperData.generatedAt === 'string'
                                        ? new Date(thread.paperData.generatedAt)
                                        : thread.paperData.generatedAt,
                                } : undefined,
                            }));
                        }
                    }
                };
            },
        }
    )
);
