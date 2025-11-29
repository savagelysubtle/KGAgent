'use client';

import { CopilotKit } from '@copilotkit/react-core';
import { CopilotSidebar } from '@copilotkit/react-ui';
import '@copilotkit/react-ui/styles.css';
import { ReactNode } from 'react';
import { AgentActions } from './agent-actions';
import { ChatInputWithUpload } from './chat-input-with-upload';
// NEW: Import reasoning components
import { AgentReasoningRenderer } from './agent-reasoning';
import { AgentStatusPanel } from './agent-status-panel';

export function CopilotProvider({ children }: { children: ReactNode }) {
  return (
    <CopilotKit
      runtimeUrl="/api/copilotkit"
      // Optional: Lock to specific agent (vs router mode)
      // agent="kg_multi_agent"
    >
      {/* Existing actions */}
      <AgentActions />

      {/* NEW: Register reasoning renderer (renders in chat) */}
      <AgentReasoningRenderer />

      {/* NEW: Status panel (renders fixed position outside chat) */}
      <AgentStatusPanel position="bottom-right" />

      <CopilotSidebar
        defaultOpen={false}
        labels={{
          title: 'KG Agent',
          initial: `👋 Hello! I'm your Knowledge Graph Agent powered by Pydantic AI and LM Studio.

I can help you with:
• **Search** - "Search for documents about machine learning"
• **Ask questions** - "Ask the agent what we know about Python"
• **Upload files** - Drag & drop files to discuss them immediately
• **Add to KB** - "Add these files to the knowledge base" to store permanently
• **Start crawls** - "Start a crawl for https://example.com"
• **List documents** - "List all documents" or "Show web crawl documents"
• **Delete data** - "Delete document [id]" or "Delete documents from example.com"
• **Stats** - "Show document statistics" or "Check agent health"

📎 **Tip:** Drop files into this chat to discuss them instantly!`,
        }}
        Input={ChatInputWithUpload}
      >
        {children}
      </CopilotSidebar>
    </CopilotKit>
  );
}

