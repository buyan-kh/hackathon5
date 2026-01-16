"use client";

import { useEditor, EditorContent, Editor } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import Placeholder from "@tiptap/extension-placeholder";
import { useCallback, useEffect } from "react";
import { cn } from "@/lib/utils";

interface ChatEditorProps {
    onSubmit?: (content: string) => void;
    placeholder?: string;
    disabled?: boolean;
    maxHeight?: string;
    className?: string;
    onEditorReady?: (editor: Editor) => void;
}

export function ChatEditor({
    onSubmit,
    placeholder = "What happens tomorrow?",
    disabled = false,
    maxHeight = "200px",
    className,
    onEditorReady,
}: ChatEditorProps) {
    const editor = useEditor({
        extensions: [
            StarterKit.configure({
                heading: false,
                bulletList: false,
                orderedList: false,
                blockquote: false,
                codeBlock: false,
                horizontalRule: false,
            }),
            Placeholder.configure({
                placeholder,
                emptyEditorClass: "is-editor-empty",
            }),
        ],
        editorProps: {
            attributes: {
                class: cn(
                    "prose prose-slate max-w-none",
                    "min-h-[60px] w-full outline-none focus:outline-none",
                    "text-base leading-relaxed",
                    "[&>*]:outline-none [&_p]:my-0",
                    disabled && "opacity-50 cursor-not-allowed"
                ),
            },
        },
        onUpdate: ({ editor }) => {
            // Optional: handle updates
        },
        immediatelyRender: false,
    });

    useEffect(() => {
        if (editor && onEditorReady) {
            onEditorReady(editor);
        }
    }, [editor, onEditorReady]);

    const handleKeyDown = useCallback(
        (e: React.KeyboardEvent<HTMLDivElement>) => {
            if (disabled) return;

            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                const content = editor?.getText().trim();
                if (content && onSubmit) {
                    onSubmit(content);
                    editor?.commands.clearContent();
                }
            }
        },
        [editor, onSubmit, disabled]
    );

    if (!editor) {
        return (
            <div className="flex h-[60px] w-full items-center justify-center">
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-muted-foreground border-t-transparent" />
            </div>
        );
    }

    return (
        <div
            className={cn(
                "no-scrollbar w-full cursor-text overflow-y-auto",
                className
            )}
            style={{ maxHeight }}
            onKeyDown={handleKeyDown}
        >
            <EditorContent
                editor={editor}
                disabled={disabled}
                className="w-full"
            />
            <style jsx global>{`
        .ProseMirror p.is-editor-empty:first-child::before {
          content: attr(data-placeholder);
          color: var(--muted-foreground);
          opacity: 0.5;
          float: left;
          height: 0;
          pointer-events: none;
        }
      `}</style>
        </div>
    );
}

export { type Editor };
