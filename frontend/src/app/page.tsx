"use client";

import { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { IconNews, IconArrowLeft } from "@tabler/icons-react";
import { AnimatedGreeting, ChatInput, ExamplePrompts, AgentStatus, ThreadSidebar, ThreadToggle } from "@/components/chat";
import { PaperLayout } from "@/components/paper";
import { useChatStore } from "@/stores/chat.store";
import { Editor } from "@tiptap/react";

interface Paper {
  paper_id: string;
  date: string;
  headline: string;
  subheadline: string;
  query: string;
  cover_image_url: string | null;
  secondary_image_url?: string | null;
  tertiary_image_url?: string | null;
  articles: Array<{
    id: string;
    title: string;
    content: string;
    summary: string;
    category: string;
    importance: number;
  }>;
  market_snapshot: Array<{
    asset: string;
    value: number;
    current_value?: number;
    projected_value?: number;
    change: number;
    changePercent: number;
    change_percent?: number;
    history?: Array<{ date: string; value: number }>;
  }>;
  trending_topics: Array<{
    topic: string;
    sentiment: number;
    mentions: number;
  }>;
  news_context: Array<{
    title: string;
    source: string;
    content?: string;
    agent?: string;
  }>;
}

export default function Home() {
  const editorRef = useRef<Editor | null>(null);
  const [generatedPaper, setGeneratedPaper] = useState<Paper | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const threads = useChatStore((state) => state.threads);
  const isGenerating = useChatStore((state) => state.isGenerating);
  const agents = useChatStore((state) => state.agents);

  const hasActiveAgents = agents.some((a) => a.status !== "idle");

  const handlePromptSelect = useCallback((prompt: string) => {
    if (editorRef.current) {
      editorRef.current.commands.clearContent();
      editorRef.current.commands.insertContent(prompt);
      editorRef.current.commands.focus("end");
    }
  }, []);

  const handleBackToChat = useCallback(() => {
    setGeneratedPaper(null);
  }, []);

  const handleTopicClick = useCallback((topic: string) => {
    // Navigate back to chat and set the topic as a new query
    setGeneratedPaper(null);
    // Small delay to ensure state updates, then trigger query
    setTimeout(() => {
      if (editorRef.current) {
        editorRef.current.commands.clearContent();
        editorRef.current.commands.insertContent(`What happens if ${topic}?`);
        editorRef.current.commands.focus("end");
      }
    }, 100);
  }, []);

  // Show the paper if generated
  if (generatedPaper) {
    return (
      <main className="relative min-h-screen flex flex-col bg-[#f5f0e8]">
        {/* Thread Sidebar */}
        <ThreadSidebar
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          onSelectThread={(thread) => {
            // If thread has paper data, reconstruct and display the paper
            if (thread.paperData) {
              const paperData = thread.paperData;
              const reconstructedPaper: Paper = {
                paper_id: paperData.id,
                date: paperData.date || new Date(paperData.generatedAt).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' }),
                headline: paperData.headline,
                subheadline: paperData.subheadline || `Comprehensive coverage: ${paperData.query || 'Market analysis'}`,
                query: paperData.query || thread.messages[0]?.content || '',
                cover_image_url: paperData.coverImage || null,
                secondary_image_url: paperData.secondaryImageUrl || null,
                tertiary_image_url: paperData.tertiaryImageUrl || null,
                articles: paperData.articles.map(article => ({
                  id: article.id,
                  title: article.title,
                  content: article.content,
                  summary: article.content.slice(0, 150) + '...',
                  category: article.category || 'general',
                  importance: 5,
                })),
                market_snapshot: paperData.marketSnapshot || [],
                trending_topics: paperData.trendingTopics || [],
                news_context: paperData.newsContext || [],
              };
              setGeneratedPaper(reconstructedPaper);
            }
          }}
        />

        {/* Header */}
        <header className="sticky top-0 z-20 flex items-center justify-between px-6 py-4 bg-white/90 backdrop-blur border-b border-[#e0d8cc]">
          <div className="flex items-center gap-3">
            <ThreadToggle onClick={() => setSidebarOpen(true)} />
            <button
              onClick={handleBackToChat}
              className="flex items-center gap-2 text-sm text-[#666] hover:text-[#1a1a1a] transition-colors"
            >
              <IconArrowLeft size={18} />
              New Query
            </button>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#1a1a1a]">
              <IconNews size={16} className="text-white" />
            </div>
            <span className="font-display text-lg text-[#1a1a1a]">Tomorrow&apos;s Paper</span>
          </div>
          <div className="w-32" />
        </header>

        {/* Paper Content */}
        <div className="flex-1 py-8 px-4">
          <PaperLayout
            date={generatedPaper.date}
            headline={generatedPaper.headline}
            subheadline={generatedPaper.subheadline}
            coverImageUrl={generatedPaper.cover_image_url || undefined}
            secondaryImageUrl={generatedPaper.secondary_image_url || undefined}
            tertiaryImageUrl={generatedPaper.tertiary_image_url || undefined}
            articles={generatedPaper.articles.map((a) => ({
              ...a,
              imageUrl: undefined,
            }))}
            marketSnapshot={generatedPaper.market_snapshot}
            trendingTopics={generatedPaper.trending_topics}
            newsContext={generatedPaper.news_context}
            query={generatedPaper.query}
            onTopicClick={handleTopicClick}
          />
        </div>
      </main>
    );
  }

  return (
    <main className="relative min-h-screen flex flex-col">
      {/* Thread Sidebar */}
      <ThreadSidebar 
        isOpen={sidebarOpen} 
        onClose={() => setSidebarOpen(false)}
        onSelectThread={(thread) => {
          // If thread has paper data, reconstruct and display the paper
          if (thread.paperData) {
            const paperData = thread.paperData;
            const reconstructedPaper: Paper = {
              paper_id: paperData.id,
              date: paperData.date || new Date(paperData.generatedAt).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' }),
              headline: paperData.headline,
              subheadline: paperData.subheadline || `Comprehensive coverage: ${paperData.query || 'Market analysis'}`,
              query: paperData.query || thread.messages[0]?.content || '',
              cover_image_url: paperData.coverImage || null,
              secondary_image_url: paperData.secondaryImageUrl || null,
              tertiary_image_url: paperData.tertiaryImageUrl || null,
              articles: paperData.articles.map(article => ({
                id: article.id,
                title: article.title,
                content: article.content,
                summary: article.content.slice(0, 150) + '...',
                category: article.category || 'general',
                importance: 5,
              })),
              market_snapshot: paperData.marketSnapshot || [],
              trending_topics: paperData.trendingTopics || [],
              news_context: paperData.newsContext || [],
            };
            setGeneratedPaper(reconstructedPaper);
          }
        }}
      />

      {/* Background Effects */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="radial-glow absolute -top-48 -right-48 w-[600px] h-[600px]" />
        <div className="radial-glow absolute -bottom-48 -left-48 w-[600px] h-[600px]" />
      </div>

      {/* Header */}
      <header className="relative z-10 flex items-center justify-between px-6 py-4">
        <motion.div
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="flex items-center gap-3"
        >
          <ThreadToggle onClick={() => setSidebarOpen(true)} />
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-accent to-accent-secondary shadow-accent">
            <IconNews size={20} className="text-white" />
          </div>
          <span className="font-display text-xl font-normal">Tomorrow&apos;s Paper</span>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="flex items-center gap-3"
        >
          {threads.length > 0 && (
            <span className="text-xs text-muted-foreground">
              {threads.length} thread{threads.length !== 1 ? "s" : ""}
            </span>
          )}
        </motion.div>
      </header>

      {/* Main Content */}
      <div className="relative z-10 flex flex-1 flex-col items-center justify-center px-4">
        <div className="w-full max-w-3xl flex flex-col items-center">
          {/* Greeting */}
          <AnimatePresence>
            {!isGenerating && !hasActiveAgents && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
              >
                <AnimatedGreeting />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Chat Input */}
          <div className="w-full mt-4">
            <ChatInput 
              onPaperGenerated={(paper) => setGeneratedPaper(paper as Paper)}
              onEditorReady={(editor) => {
                editorRef.current = editor;
              }}
            />
          </div>

          {/* Example Prompts */}
          <AnimatePresence>
            {!isGenerating && !hasActiveAgents && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <ExamplePrompts onSelect={handlePromptSelect} />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Agent Status */}
          <AnimatePresence>
            {(isGenerating || hasActiveAgents) && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
                className="w-full mt-8"
              >
                <AgentStatus />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Footer */}
      <footer className="relative z-10 flex items-center justify-center px-6 py-4">
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="text-xs text-muted-foreground/50"
        >
          Powered by AI agents: Yutori • Tonic Fabricate • Freepik
        </motion.p>
      </footer>
    </main>
  );
}
