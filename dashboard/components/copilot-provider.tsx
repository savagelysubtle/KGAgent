"use client";

import { CopilotKit } from "@copilotkit/react-core";
import { CopilotSidebar } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import { ReactNode } from "react";
import { AgentActions } from "./agent-actions";

export function CopilotProvider({ children }: { children: ReactNode }) {
  return (
    <CopilotKit runtimeUrl="/api/copilotkit">
      <AgentActions />
      <CopilotSidebar
        defaultOpen={false}
        labels={{
          title: "KG Agent",
          initial: `👋 Hello! I'm your Knowledge Graph Agent powered by Pydantic AI and LM Studio.

I can help you with:
• **Search** - "Search for documents about machine learning"
• **Ask questions** - "Ask the agent what we know about Python"
• **Start crawls** - "Start a crawl for https://example.com"
• **List documents** - "List all documents" or "Show web crawl documents"
• **Delete data** - "Delete document [id]" or "Delete documents from example.com"
• **Stats** - "Show document statistics" or "Check agent health"

What would you like to do?`,
        }}
      >
        {children}
      </CopilotSidebar>
    </CopilotKit>
  );
}

