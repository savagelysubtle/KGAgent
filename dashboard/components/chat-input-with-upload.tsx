"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { Paperclip, Send, X, FileText, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { uploadApi } from "@/lib/api";

// Types for CopilotKit Input component props
interface InputProps {
  inProgress: boolean;
  onSend: (text: string) => void;
  isVisible?: boolean;
}

// Type for previewed file content
export interface PreviewedFile {
  filename: string;
  content: string;
  chunk_count: number;
  file_type: string;
  size_bytes: number;
  file?: File; // Original file for potential upload
}

// Context for sharing previewed files with agent actions
interface FileContextValue {
  previewedFiles: PreviewedFile[];
  setPreviewedFiles: (files: PreviewedFile[]) => void;
  clearPreviewedFiles: () => void;
  isProcessing: boolean;
}

// Create a module-level store for file context (simpler than React context for this use case)
let _previewedFiles: PreviewedFile[] = [];
let _listeners: Set<() => void> = new Set();

export const fileContextStore = {
  getFiles: () => _previewedFiles,
  setFiles: (files: PreviewedFile[]) => {
    _previewedFiles = files;
    _listeners.forEach(listener => listener());
  },
  clearFiles: () => {
    _previewedFiles = [];
    _listeners.forEach(listener => listener());
  },
  subscribe: (listener: () => void) => {
    _listeners.add(listener);
    return () => _listeners.delete(listener);
  }
};

// Hook to access file context
export function useFileContext(): FileContextValue {
  const [previewedFiles, setLocalFiles] = useState<PreviewedFile[]>(_previewedFiles);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    return fileContextStore.subscribe(() => {
      setLocalFiles(fileContextStore.getFiles());
    });
  }, []);

  return {
    previewedFiles,
    setPreviewedFiles: (files) => {
      fileContextStore.setFiles(files);
    },
    clearPreviewedFiles: () => {
      fileContextStore.clearFiles();
    },
    isProcessing
  };
}

/**
 * Custom chat input component with drag-and-drop file upload.
 *
 * When files are dropped:
 * 1. Files are parsed via the preview API (no DB storage)
 * 2. Content is made available as context for the agent
 * 3. User can chat about the files immediately
 * 4. User can explicitly say "add to database" to persist
 */
export function ChatInputWithUpload({ inProgress, onSend, isVisible = true }: InputProps) {
  const [text, setText] = useState("");
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [isProcessingFiles, setIsProcessingFiles] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Handle file drop
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    setPendingFiles(prev => [...prev, ...acceptedFiles]);
    setIsProcessingFiles(true);

    try {
      // Preview files (parse without storing)
      const response = await uploadApi.previewFiles(acceptedFiles);
      const previewed = response.data.files.map((f, idx) => ({
        ...f,
        file: acceptedFiles[idx] // Keep original file for potential upload
      }));

      // Add to context store
      fileContextStore.setFiles([...fileContextStore.getFiles(), ...previewed]);

      toast.success(`${acceptedFiles.length} file(s) ready for discussion`, {
        description: "Ask questions about the files or say 'add to knowledge base' to store them."
      });
    } catch (error) {
      console.error("Failed to preview files:", error);
      toast.error("Failed to process files");
    } finally {
      setIsProcessingFiles(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    noClick: true, // Don't open file dialog on click (we have a button for that)
    noKeyboard: true,
  });

  // Handle file button click
  const handleFileButtonClick = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.multiple = true;
    input.onchange = (e) => {
      const files = Array.from((e.target as HTMLInputElement).files || []);
      if (files.length > 0) {
        onDrop(files);
      }
    };
    input.click();
  };

  // Remove a pending file
  const removeFile = (index: number) => {
    setPendingFiles(prev => prev.filter((_, i) => i !== index));
    const currentFiles = fileContextStore.getFiles();
    fileContextStore.setFiles(currentFiles.filter((_, i) => i !== index));
  };

  // Handle send
  const handleSend = () => {
    const trimmedText = text.trim();
    if (!trimmedText && pendingFiles.length === 0) return;

    // Build message with file context if files are attached
    let messageToSend = trimmedText;

    if (pendingFiles.length > 0 && !trimmedText) {
      // If only files were dropped with no text, prompt about them
      messageToSend = `I've uploaded ${pendingFiles.length} file(s): ${pendingFiles.map(f => f.name).join(", ")}. What would you like to know about them?`;
    }

    onSend(messageToSend);
    setText("");
    // Don't clear pending files - keep them for context
  };

  // Handle key press
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [text]);

  if (!isVisible) return null;

  const currentFiles = fileContextStore.getFiles();

  return (
    <div
      {...getRootProps()}
      className={`relative border-t border-purple-500/20 bg-black/60 backdrop-blur-sm transition-all ${
        isDragActive ? "ring-2 ring-purple-500 ring-inset bg-purple-500/10" : ""
      }`}
    >
      <input {...getInputProps()} />

      {/* Drag overlay */}
      {isDragActive && (
        <div className="absolute inset-0 flex items-center justify-center bg-purple-900/50 backdrop-blur-sm z-10 pointer-events-none">
          <div className="text-center">
            <FileText className="h-12 w-12 text-purple-300 mx-auto mb-2" />
            <p className="text-purple-100 font-medium">Drop files here</p>
            <p className="text-purple-200/60 text-sm">Files will be available for discussion</p>
          </div>
        </div>
      )}

      {/* Attached files preview */}
      {currentFiles.length > 0 && (
        <div className="px-4 pt-3 pb-1">
          <div className="flex flex-wrap gap-2">
            {currentFiles.map((file, index) => (
              <div
                key={index}
                className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-purple-500/20 border border-purple-500/30 text-sm"
              >
                <FileText className="h-3.5 w-3.5 text-purple-400" />
                <span className="text-purple-100 max-w-[150px] truncate">{file.filename}</span>
                <span className="text-purple-300/50 text-xs">
                  {(file.size_bytes / 1024).toFixed(0)}KB
                </span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removeFile(index);
                  }}
                  className="p-0.5 hover:bg-purple-500/30 rounded-full transition-colors"
                >
                  <X className="h-3 w-3 text-purple-300" />
                </button>
              </div>
            ))}
          </div>
          <p className="text-xs text-purple-300/50 mt-1">
            Files loaded for discussion. Say &quot;add to knowledge base&quot; to store permanently.
          </p>
        </div>
      )}

      {/* Input area */}
      <div className="flex items-end gap-2 p-4">
        {/* File attachment button */}
        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={handleFileButtonClick}
          disabled={inProgress || isProcessingFiles}
          className="h-10 w-10 shrink-0 text-purple-300 hover:text-purple-100 hover:bg-purple-500/20"
          title="Attach files"
        >
          {isProcessingFiles ? (
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : (
            <Paperclip className="h-5 w-5" />
          )}
        </Button>

        {/* Text input */}
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            currentFiles.length > 0
              ? "Ask about the files or type a message..."
              : "Type a message or drop files here..."
          }
          disabled={inProgress}
          rows={1}
          className="flex-1 resize-none bg-purple-950/30 border border-purple-500/20 rounded-lg px-4 py-2.5 text-purple-100 placeholder:text-purple-300/40 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-transparent disabled:opacity-50"
        />

        {/* Send button */}
        <Button
          type="button"
          onClick={handleSend}
          disabled={inProgress || (!text.trim() && currentFiles.length === 0)}
          className="h-10 w-10 shrink-0 bg-purple-600 hover:bg-purple-700 text-white shadow-[0_0_15px_rgba(147,51,234,0.3)]"
        >
          {inProgress ? (
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : (
            <Send className="h-5 w-5" />
          )}
        </Button>
      </div>
    </div>
  );
}

export default ChatInputWithUpload;

