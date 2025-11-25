"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { CopilotKit } from "@copilotkit/react-core";
import { CopilotSidebar } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import { ReactNode } from "react";
import { AgentActions } from "./agent-actions";
import { ChatHistory } from "./chat-history";
import { chatApi } from "@/lib/api";
import { useQueryClient } from "@tanstack/react-query";

interface Message {
  id: string;
  conversation_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  metadata: Record<string, unknown>;
  created_at: string;
}

export function CopilotProviderWithHistory({ children }: { children: ReactNode }) {
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [historyMessages, setHistoryMessages] = useState<Message[]>([]);
  const queryClient = useQueryClient();
  const pendingUserMessageRef = useRef<string | null>(null);
  const conversationIdRef = useRef<string | null>(null);

  // Keep ref in sync
  useEffect(() => {
    conversationIdRef.current = currentConversationId;
  }, [currentConversationId]);

  // Restore conversation from sessionStorage on mount
  useEffect(() => {
    const loadedConvId = sessionStorage.getItem("loadedConversationId");
    const loadedMessagesStr = sessionStorage.getItem("loadedMessages");

    if (loadedConvId && loadedMessagesStr) {
      try {
        const loadedMessages = JSON.parse(loadedMessagesStr);
        setCurrentConversationId(loadedConvId);
        setHistoryMessages(loadedMessages);
        sessionStorage.removeItem("loadedConversationId");
        sessionStorage.removeItem("loadedMessages");
      } catch (e) {
        console.error("Failed to restore conversation:", e);
      }
    }
  }, []);

  // Save user message when submitted
  const handleSubmitMessage = useCallback(async (message: string) => {
    pendingUserMessageRef.current = message;

    try {
      let convId = conversationIdRef.current;

      // Create new conversation if needed
      if (!convId) {
        const title = message.length > 50 ? message.slice(0, 50) + "..." : message;
        const response = await chatApi.createConversation({ title });
        convId = response.data.conversation.id;
        setCurrentConversationId(convId);
        conversationIdRef.current = convId;
      }

      // Save user message
      await chatApi.addMessage(convId, {
        role: "user",
        content: message,
      });

      queryClient.invalidateQueries({ queryKey: ["conversations"] });
    } catch (error) {
      console.error("Failed to save user message:", error);
    }
  }, [queryClient]);

  // Save assistant message when response completes
  const handleInProgress = useCallback(async (inProgress: boolean) => {
    // When inProgress changes from true to false, the assistant has finished
    if (!inProgress && pendingUserMessageRef.current && conversationIdRef.current) {
      // We need to capture the assistant's response
      // Unfortunately CopilotKit doesn't expose this directly via callbacks
      // We'll need to poll or use a different approach

      // For now, let's save a placeholder and update it later
      // A better approach would be to use a custom AssistantMessage component
      pendingUserMessageRef.current = null;
    }
  }, []);

  // Handle selecting a conversation from history
  const handleSelectConversation = useCallback(
    async (convId: string | null, loadedMessages: Message[]) => {
      if (convId && loadedMessages.length > 0) {
        // Store in sessionStorage and reload to show messages in CopilotKit
        sessionStorage.setItem("loadedConversationId", convId);
        sessionStorage.setItem("loadedMessages", JSON.stringify(loadedMessages));
        window.location.reload();
      } else {
        setCurrentConversationId(convId);
        setHistoryMessages(loadedMessages);
      }
    },
    []
  );

  // Handle starting a new chat
  const handleNewChat = useCallback(() => {
    setCurrentConversationId(null);
    setHistoryMessages([]);
    conversationIdRef.current = null;
    // Reload to clear CopilotKit's internal state
    window.location.reload();
  }, []);

  // Build initial message from history if available
  const initialMessage = historyMessages.length > 0
    ? historyMessages
        .filter(m => m.role === "assistant")
        .slice(-1)[0]?.content || "Welcome back! How can I help you?"
    : `👋 Hello! I'm your Knowledge Graph Agent powered by Pydantic AI and LM Studio.

I can help you with:
• **Search** - "Search for documents about machine learning"
• **Ask questions** - "Ask the agent what we know about Python"
• **Start crawls** - "Start a crawl for https://example.com"
• **List documents** - "List all documents" or "Show web crawl documents"
• **Delete data** - "Delete document [id]" or "Delete documents from example.com"
• **Stats** - "Show document statistics" or "Check agent health"

What would you like to do?`;

  return (
    <CopilotKit runtimeUrl="/api/copilotkit">
      <AgentActions />
      <ChatHistory
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewChat={handleNewChat}
      />
      <CopilotSidebar
        defaultOpen={false}
        labels={{
          title: "KG Agent",
          initial: initialMessage,
        }}
        onSubmitMessage={handleSubmitMessage}
        onInProgress={handleInProgress}
      >
        {children}
      </CopilotSidebar>
    </CopilotKit>
  );
}
