"use client";

import { motion, AnimatePresence } from "framer-motion";
import {
    IconMenu2,
    IconX,
    IconHistory,
    IconTrash,
    IconNews,
} from "@tabler/icons-react";
import { useChatStore, Thread } from "@/stores/chat.store";
import { cn } from "@/lib/utils";

interface ThreadSidebarProps {
    isOpen: boolean;
    onClose: () => void;
    onSelectThread?: (thread: Thread) => void;
}

export function ThreadSidebar({ isOpen, onClose, onSelectThread }: ThreadSidebarProps) {
    const threads = useChatStore((state) => state.threads);
    const currentThreadId = useChatStore((state) => state.currentThreadId);
    const clearThread = useChatStore((state) => state.clearThread);
    const setCurrentThread = useChatStore((state) => state.setCurrentThread);

    const handleSelectThread = (thread: Thread) => {
        setCurrentThread(thread.id);
        onSelectThread?.(thread);
        onClose();
    };

    return (
        <>
            {/* Backdrop */}
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={onClose}
                        className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40"
                    />
                )}
            </AnimatePresence>

            {/* Sidebar */}
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ x: "-100%" }}
                        animate={{ x: 0 }}
                        exit={{ x: "-100%" }}
                        transition={{ type: "spring", damping: 25, stiffness: 200 }}
                        className="fixed left-0 top-0 bottom-0 w-80 max-w-[85vw] bg-white border-r border-[#e0e0e0] shadow-2xl z-50 flex flex-col"
                    >
                        {/* Header */}
                        <div className="flex items-center justify-between p-4 border-b border-[#e0e0e0] bg-[#fafafa]">
                            <div className="flex items-center gap-3">
                                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#1a1a1a]">
                                    <IconNews size={18} className="text-white" />
                                </div>
                                <div>
                                    <h2 className="font-semibold text-[#1a1a1a]">Paper History</h2>
                                    <p className="text-xs text-[#888]">
                                        {threads.length} edition{threads.length !== 1 ? "s" : ""}
                                    </p>
                                </div>
                            </div>
                            <button
                                onClick={onClose}
                                className="p-2 rounded-lg hover:bg-[#e0e0e0] transition-colors"
                            >
                                <IconX size={20} className="text-[#666]" />
                            </button>
                        </div>

                        {/* Thread List */}
                        <div className="flex-1 overflow-y-auto p-3">
                            {threads.length === 0 ? (
                                <div className="flex flex-col items-center justify-center py-12 text-center">
                                    <IconHistory size={48} className="text-[#ccc] mb-4" />
                                    <p className="text-sm text-[#666]">No papers yet</p>
                                    <p className="text-xs text-[#999] mt-1">
                                        Start a query to generate your first paper
                                    </p>
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    {threads.map((thread) => (
                                        <motion.div
                                            key={thread.id}
                                            initial={{ opacity: 0, y: 10 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            className={cn(
                                                "group relative rounded-lg border p-3 cursor-pointer transition-all",
                                                thread.id === currentThreadId
                                                    ? "border-[#c4161c] bg-red-50"
                                                    : "border-[#e0e0e0] hover:border-[#999] hover:bg-[#fafafa]"
                                            )}
                                            onClick={() => handleSelectThread(thread)}
                                        >
                                            <div className="flex items-start justify-between gap-2">
                                                <div className="flex-1 min-w-0">
                                                    <p className="text-sm font-semibold text-[#1a1a1a] truncate">
                                                        {thread.paperData?.headline || thread.title || "Untitled"}
                                                    </p>
                                                    <p className="text-[11px] text-[#888] mt-0.5">
                                                        {new Date(thread.createdAt).toLocaleDateString(undefined, {
                                                            month: "short",
                                                            day: "numeric",
                                                            hour: "2-digit",
                                                            minute: "2-digit",
                                                        })}
                                                    </p>
                                                    {thread.messages.length > 0 && (
                                                        <p className="text-xs text-[#666] mt-1.5 line-clamp-2 italic">
                                                            &quot;{thread.messages[0]?.content.slice(0, 60)}...&quot;
                                                        </p>
                                                    )}
                                                </div>
                                                <button
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        clearThread(thread.id);
                                                    }}
                                                    className="p-1.5 rounded opacity-0 group-hover:opacity-100 hover:bg-red-100 hover:text-red-600 transition-all"
                                                >
                                                    <IconTrash size={14} />
                                                </button>
                                            </div>
                                        </motion.div>
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Footer */}
                        <div className="p-4 border-t border-[#e0e0e0] bg-[#fafafa]">
                            <p className="text-[10px] text-[#999] text-center">
                                Tomorrow&apos;s Paper • AI Market Intelligence
                            </p>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    );
}

export function ThreadToggle({ onClick }: { onClick: () => void }) {
    return (
        <button
            onClick={onClick}
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-[#e0e0e0] bg-white hover:bg-[#f5f5f5] transition-colors"
        >
            <IconMenu2 size={18} className="text-[#333]" />
        </button>
    );
}
