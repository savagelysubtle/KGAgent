# Phase 6: Frontend Integration - Reasoning UI

> **Parent Plan:** [multi_agent.plan.md](./multi_agent.plan.md) > **Status:**
> Not Started **Estimated Effort:** 3-4 hours **Dependencies:** Phase 5 complete
> (API working) **Last Updated:** November 29, 2025

---

## 📚 Technical Research Summary

### CopilotKit React Hooks API (v1.10.x)

#### `useCoAgentStateRender` Hook

Renders agent state **inside the chat** as messages stream.

```typescript
// Render Props Interface
type CoAgentStateRenderProps<T> = {
  state: T;              // Current agent state
  nodeName: string;      // Current LangGraph node name
  status: "inProgress" | "complete";  // NOT "thinking"/"executing"!
};

// Hook Configuration
useCoAgentStateRender<AgentState>({
  name: "kg_multi_agent",      // Must match backend agent name
  nodeName?: "chat_node",      // Optional: filter to specific node
  render: ({ state, nodeName, status }) => ReactElement | null,
  handler?: (props) => void,   // Optional: side effects
});
```

#### `useCoAgent` Hook

Access agent state **anywhere in app** (outside chat).

```typescript
const {
  name,      // Agent name
  nodeName,  // Current LangGraph node
  threadId,  // Session thread ID
  running,   // boolean - is agent currently executing?
  state,     // T - reactive state object
  setState,  // (newState | (prev) => newState) => void
  start,     // () => void - start agent
  stop,      // () => void - stop agent
  run,       // (hint?) => Promise<void> - re-run agent
} = useCoAgent<AgentState>({
  name: "kg_multi_agent",
  initialState?: { ... },
});
```

### Backend State Structure (from `state.py`)

```python
class MultiAgentState(CopilotKitState):
    thinking_steps: list[ThinkingStep]  # Streamed to UI
    current_agent: str                   # 'manager', 'research', etc.
    delegation_queue: list[DelegationRequest]
    total_llm_calls: int
    execution_path: list[str]            # Node visit order
    # ... other fields
```

### Existing Dashboard Setup

- **CopilotKit version:** 1.10.6
- **Current provider:** `CopilotSidebar` with custom `ChatInputWithUpload`
- **Existing actions:** `AgentActions` component with 15+ actions
- **Missing:** `framer-motion` (needs to be added)

---

## 🎯 Objectives

1. Create `AgentReasoning` component for in-chat reasoning display
2. Create `AgentStatusPanel` for persistent status visibility
3. Integrate with CopilotKit's `useCoAgentStateRender` hook
4. Update `CopilotProvider` to register multi-agent
5. Add visual polish and animations

---

## 📋 Prerequisites

- [ ] Phase 5 complete (API endpoints working)
- [ ] CopilotKit React packages installed (`@copilotkit/react-core`,
      `@copilotkit/react-ui`) ✅ Already v1.10.6
- [ ] Backend streaming state via `copilotkit_emit_state`
- [ ] Agent registered with name `kg_multi_agent`

---

## 🎨 UI Design

### In-Chat Reasoning Display

```
┌────────────────────────────────────────────────────────────────┐
│ User: Search for Python tutorials and remember I'm learning it │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ 🧠 Agent Reasoning                                             │
├────────────────────────────────────────────────────────────────┤
│ │ 🎯 manager [thinking]                                        │
│ │   Analyzing request: 'Search for Python tutorials...'       │
│ │                                                              │
│ │ 🎯 manager [delegating]                                      │
│ │   → delegate_to_research                                     │
│ │   Delegating to research: Search for Python tutorials       │
│ │                                                              │
│ │ 🎯 manager [delegating]                                      │
│ │   → delegate_to_memory                                       │
│ │   Delegating to memory: Remember user is learning Python    │
│ │                                                              │
│ │ 🔍 research [executing]                                      │
│ │   → search_knowledge_base                                    │
│ │   Searching (hybrid): Python tutorials...                   │
│ │                                                              │
│ │ 🔍 research [complete] ✓                                     │
│ │   Research complete                                          │
│ │                                                              │
│ │ 🧠 memory [executing]                                        │
│ │   → remember_about_user                                      │
│ │   Storing: User is learning Python...                       │
│ │                                                              │
│ │ 🧠 memory [complete] ✓                                       │
│ │   Memory task complete                                       │
│ │                                                              │
│ │ 🎯 manager [complete] ✓                                      │
│ │   Response ready                                             │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ Assistant: I found several Python tutorials for you...         │
│                                                                │
│ **Research:**                                                   │
│ Found 5 Python tutorials including...                          │
│                                                                │
│ **Memory:**                                                     │
│ I've noted that you're learning Python!                        │
└────────────────────────────────────────────────────────────────┘
```

### Status Panel (Fixed Position)

```
┌─────────────────────────────────┐
│ 🤖 Agent Status                 │
├─────────────────────────────────┤
│ Current: 🔍 research            │
│ → Delegating to: memory         │
│                                 │
│ Steps: 4                        │
│ LLM Calls: 2                    │
│ Session: abc123                 │
└─────────────────────────────────┘
```

---

## 📁 Task 1: Create Agent Reasoning Component

### File: `dashboard/components/agent-reasoning.tsx`

> **Note:** This component uses `useCoAgentStateRender` which renders content
> **inside the chat stream**. The `status` prop from CopilotKit is either
> `"inProgress"` or `"complete"` - NOT the same as our backend
> `ThinkingStep.status`.

```tsx
'use client';

/**
 * AgentReasoning - Displays real-time agent thinking steps in chat.
 *
 * Uses CopilotKit's useCoAgentStateRender to render agent state
 * as a UI component within the chat stream.
 *
 * IMPORTANT: The render prop receives:
 * - state: The agent's current state (MultiAgentState from backend)
 * - nodeName: Current LangGraph node name
 * - status: "inProgress" | "complete" (CopilotKit's execution status)
 */

import { useCoAgentStateRender } from '@copilotkit/react-core';
import {
  Brain,
  Search,
  Database,
  FileText,
  Target,
  Loader2,
  CheckCircle2,
  AlertCircle,
  ChevronRight,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// NOTE: framer-motion is optional - CSS animations work fine
// import { motion, AnimatePresence } from 'framer-motion';

// === Types ===

type ThinkingStatus =
  | 'thinking'
  | 'delegating'
  | 'executing'
  | 'complete'
  | 'error';

interface ThinkingStep {
  agent: string;
  thought: string;
  action?: string;
  result?: string;
  status: ThinkingStatus;
  timestamp: string;
}

// Must match backend MultiAgentState from state.py
interface AgentState {
  thinking_steps: ThinkingStep[];
  current_agent: string;
  delegation_queue?: Array<{ target: string; task: string }>;
  total_llm_calls: number;
  execution_path: string[];
}

// === Constants ===

const AGENT_ICONS: Record<string, React.ReactNode> = {
  manager: <Target className="w-4 h-4" />,
  research: <Search className="w-4 h-4" />,
  memory: <Brain className="w-4 h-4" />,
  knowledge: <Database className="w-4 h-4" />,
  documents: <FileText className="w-4 h-4" />,
};

const AGENT_COLORS: Record<string, string> = {
  manager: 'text-amber-500 border-amber-500/30 bg-amber-500/10',
  research: 'text-blue-500 border-blue-500/30 bg-blue-500/10',
  memory: 'text-purple-500 border-purple-500/30 bg-purple-500/10',
  knowledge: 'text-green-500 border-green-500/30 bg-green-500/10',
  documents: 'text-orange-500 border-orange-500/30 bg-orange-500/10',
};

const STATUS_ICONS: Record<ThinkingStatus, React.ReactNode> = {
  thinking: <Loader2 className="w-3 h-3 animate-spin" />,
  delegating: <ChevronRight className="w-3 h-3" />,
  executing: <Loader2 className="w-3 h-3 animate-spin" />,
  complete: <CheckCircle2 className="w-3 h-3" />,
  error: <AlertCircle className="w-3 h-3" />,
};

const STATUS_COLORS: Record<ThinkingStatus, string> = {
  thinking: 'text-yellow-500',
  delegating: 'text-blue-500',
  executing: 'text-purple-500',
  complete: 'text-green-500',
  error: 'text-red-500',
};

// === Components ===

interface ThinkingStepItemProps {
  step: ThinkingStep;
  index: number;
  isLatest: boolean;
}

function ThinkingStepItem({ step, index, isLatest }: ThinkingStepItemProps) {
  const agentColor =
    AGENT_COLORS[step.agent] ||
    'text-gray-500 border-gray-500/30 bg-gray-500/10';
  const statusColor = STATUS_COLORS[step.status];

  // Using CSS animations instead of framer-motion for simplicity
  return (
    <div
      className={cn(
        'flex items-start gap-2 py-2 px-3 rounded-lg border-l-2 ml-2',
        'animate-in fade-in slide-in-from-left-2 duration-200',
        agentColor,
        isLatest && step.status !== 'complete' && 'animate-pulse',
      )}
      style={{ animationDelay: `${index * 50}ms` }}
    >
      {/* Agent Icon */}
      <div className="mt-0.5">
        {AGENT_ICONS[step.agent] || <Target className="w-4 h-4" />}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        {/* Header */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-semibold text-sm capitalize">{step.agent}</span>
          <span className={cn('flex items-center gap-1 text-xs', statusColor)}>
            {STATUS_ICONS[step.status]}[{step.status}]
          </span>
        </div>

        {/* Thought */}
        <p className="text-sm text-gray-300 mt-1 break-words">{step.thought}</p>

        {/* Action */}
        {step.action && (
          <p className="text-xs text-blue-400 mt-1 flex items-center gap-1">
            <ChevronRight className="w-3 h-3" />
            {step.action}
          </p>
        )}

        {/* Result Preview */}
        {step.result && step.status === 'complete' && (
          <p className="text-xs text-gray-400 mt-1 truncate">
            Result: {step.result}
          </p>
        )}
      </div>
    </div>
  );
}

// === Main Component ===

export function AgentReasoningRenderer() {
  useCoAgentStateRender<AgentState>({
    name: 'kg_multi_agent',
    // nodeName: 'manager', // Uncomment to filter to specific node
    render: ({ state, nodeName, status }) => {
      // status is CopilotKit's execution status: "inProgress" | "complete"
      const isAgentRunning = status === 'inProgress';
      // Don't render if no thinking steps
      if (!state.thinking_steps?.length) {
        return null;
      }

      // Only show last N steps to avoid overwhelming UI
      const MAX_VISIBLE_STEPS = 10;
      const visibleSteps = state.thinking_steps.slice(-MAX_VISIBLE_STEPS);
      const hiddenCount = state.thinking_steps.length - visibleSteps.length;

      return (
        <div className="my-4 rounded-xl bg-gray-900/50 border border-gray-800 overflow-hidden">
          {/* Header */}
          <div className="px-4 py-2 bg-gray-800/50 border-b border-gray-700 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Brain className="w-4 h-4 text-purple-500" />
              <span className="text-sm font-semibold text-gray-200">
                Agent Reasoning
              </span>
            </div>
            <div className="flex items-center gap-3 text-xs text-gray-400">
              <span>Node: {nodeName}</span>
              <span>Steps: {state.thinking_steps.length}</span>
              <span>LLM: {state.total_llm_calls || 0}</span>
              {isAgentRunning && (
                <Loader2 className="w-3 h-3 animate-spin text-blue-500" />
              )}
            </div>
          </div>

          {/* Hidden count indicator */}
          {hiddenCount > 0 && (
            <div className="px-4 py-1 text-xs text-gray-500 border-b border-gray-800">
              ... {hiddenCount} earlier step(s) hidden
            </div>
          )}

          {/* Steps - using CSS animations instead of AnimatePresence */}
          <div className="p-3 space-y-2 max-h-[400px] overflow-y-auto">
            {visibleSteps.map((step, index) => (
              <ThinkingStepItem
                key={`${step.timestamp}-${index}`}
                step={step}
                index={index}
                isLatest={index === visibleSteps.length - 1}
              />
            ))}
          </div>

          {/* Current Agent Indicator - only show when running */}
          {isAgentRunning && state.current_agent && (
            <div className="px-4 py-2 bg-gray-800/30 border-t border-gray-800 flex items-center gap-2 text-sm">
              <Loader2 className="w-3 h-3 animate-spin text-blue-500" />
              <span className="text-gray-400">
                Currently:{' '}
                <span className="text-white capitalize">
                  {state.current_agent}
                </span>
              </span>
              {state.delegation_queue && state.delegation_queue.length > 0 && (
                <span className="text-gray-500 text-xs">
                  ({state.delegation_queue.length} pending)
                </span>
              )}
            </div>
          )}
        </div>
      );
    },
  });

  // This component doesn't render anything directly
  // It registers a renderer with CopilotKit
  return null;
}

export default AgentReasoningRenderer;
```

---

## 📁 Task 2: Create Agent Status Panel

### File: `dashboard/components/agent-status-panel.tsx`

> **Note:** This component uses `useCoAgent` to access state **outside the
> chat**. The `running` boolean indicates if the agent is currently executing.

```tsx
'use client';

/**
 * AgentStatusPanel - Fixed position panel showing current agent status.
 *
 * Uses useCoAgent hook to access agent state anywhere in the app.
 * The `running` property is a boolean from CopilotKit indicating execution status.
 *
 * Displays:
 * - Current active agent
 * - Delegation queue status
 * - Step count
 * - LLM call count
 * - Execution path
 */

import { useCoAgent } from '@copilotkit/react-core';
import {
  Bot,
  Target,
  Search,
  Brain,
  Database,
  FileText,
  Activity,
  Zap,
  Clock,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// === Types (must match backend MultiAgentState) ===

interface AgentState {
  current_agent: string;
  delegation_queue?: Array<{ target: string; task: string }>;
  thinking_steps: Array<{
    agent: string;
    thought: string;
    status: string;
    timestamp: string;
  }>;
  total_llm_calls: number;
  session_id?: string;
  execution_path: string[];
}

// === Constants ===

const AGENT_ICONS: Record<string, React.ReactNode> = {
  manager: <Target className="w-5 h-5 text-amber-500" />,
  research: <Search className="w-5 h-5 text-blue-500" />,
  memory: <Brain className="w-5 h-5 text-purple-500" />,
  knowledge: <Database className="w-5 h-5 text-green-500" />,
  documents: <FileText className="w-5 h-5 text-orange-500" />,
};

// === Component ===

interface AgentStatusPanelProps {
  className?: string;
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left';
}

export function AgentStatusPanel({
  className,
  position = 'bottom-right',
}: AgentStatusPanelProps) {
  // useCoAgent returns: name, nodeName, threadId, running, state, setState, start, stop, run
  const { state, running, nodeName, threadId } = useCoAgent<AgentState>({
    name: 'kg_multi_agent',
    // initialState: { ... }, // Optional: set initial state
  });

  // Position classes
  const positionClasses = {
    'bottom-right': 'bottom-4 right-4',
    'bottom-left': 'bottom-4 left-4',
    'top-right': 'top-4 right-4',
    'top-left': 'top-4 left-4',
  };

  // Only show when there's activity
  const isActive = running || state?.thinking_steps?.length > 0;

  // Don't render if not active
  if (!isActive) return null;

  return (
    <div
      className={cn(
        'fixed z-50 w-72 animate-in fade-in slide-in-from-bottom-4 duration-300',
        positionClasses[position],
        className,
      )}
    >
      <div className="bg-gray-900 border border-gray-700 rounded-xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="px-4 py-3 bg-gradient-to-r from-gray-800 to-gray-900 border-b border-gray-700">
          <div className="flex items-center gap-2">
            <Bot className="w-5 h-5 text-blue-500" />
            <span className="font-semibold text-white">Agent Status</span>
            {running && (
              <span className="ml-auto flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-xs text-green-400">Active</span>
              </span>
            )}
          </div>
        </div>

        {/* Content */}
        <div className="p-4 space-y-3">
          {/* Current Agent */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-400 flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Current
            </span>
            <div className="flex items-center gap-2">
              {AGENT_ICONS[state?.current_agent || 'manager']}
              <span className="text-white font-medium capitalize">
                {state?.current_agent || 'idle'}
              </span>
            </div>
          </div>

          {/* Delegation Queue */}
          {state?.delegation_queue && state.delegation_queue.length > 0 && (
            <div className="flex items-center justify-between text-blue-400">
              <span className="text-sm">→ Delegating to</span>
              <span className="font-medium capitalize">
                {state.delegation_queue[0]?.target}
              </span>
            </div>
          )}

          {/* Thread ID (for debugging) */}
          {threadId && (
            <div className="text-xs text-gray-500 truncate">
              Thread: {threadId.slice(0, 8)}...
            </div>
          )}

          {/* Stats */}
          <div className="pt-2 border-t border-gray-800 grid grid-cols-2 gap-2">
            <div className="flex items-center gap-2 text-sm">
              <Clock className="w-4 h-4 text-gray-500" />
              <span className="text-gray-400">Steps:</span>
              <span className="text-white">
                {state?.thinking_steps?.length || 0}
              </span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <Zap className="w-4 h-4 text-yellow-500" />
              <span className="text-gray-400">LLM:</span>
              <span className="text-white">{state?.total_llm_calls || 0}</span>
            </div>
          </div>

          {/* Execution Path */}
          {state?.execution_path && state.execution_path.length > 0 && (
            <div className="pt-2 border-t border-gray-800">
              <span className="text-xs text-gray-500">Path:</span>
              <div className="flex flex-wrap gap-1 mt-1">
                {state.execution_path.slice(-5).map((node, i) => (
                  <span
                    key={i}
                    className="text-xs px-2 py-0.5 rounded bg-gray-800 text-gray-300 capitalize"
                  >
                    {node}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default AgentStatusPanel;
```

---

## 📁 Task 3: Update CopilotProvider

### File: `dashboard/components/copilot-provider.tsx` (update)

> **Note:** The existing provider already has `AgentActions` registered. We add
> `AgentReasoningRenderer` (for in-chat UI) and optionally `AgentStatusPanel`.

```tsx
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
          initial: `👋 Hello! I'm your Knowledge Graph Agent...`,
        }}
        Input={ChatInputWithUpload}
      >
        {children}
      </CopilotSidebar>
    </CopilotKit>
  );
}

export default CopilotProvider;
```

### Key Changes from Existing Provider

1. **Added `AgentReasoningRenderer`** - Registers with CopilotKit to render
   thinking steps in chat
2. **Added `AgentStatusPanel`** - Fixed position panel for persistent status
   visibility
3. **Optional `agent` prop** - Can lock to `kg_multi_agent` or let CopilotKit
   route

---

## 📁 Task 4: Animation Dependencies (Optional)

### Option A: Use CSS Animations (Recommended)

The components above use Tailwind CSS animations which are already available.
The `tw-animate-css` package is already installed in the dashboard.

Add these utility classes to `globals.css` if not present:

```css
/* Animate-in utilities (may already exist from tw-animate-css) */
@keyframes fade-in {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

@keyframes slide-in-from-left-2 {
  from {
    transform: translateX(-0.5rem);
  }
  to {
    transform: translateX(0);
  }
}

@keyframes slide-in-from-bottom-4 {
  from {
    transform: translateY(1rem);
  }
  to {
    transform: translateY(0);
  }
}

.animate-in {
  animation-duration: 200ms;
  animation-timing-function: ease-out;
  animation-fill-mode: both;
}

.fade-in {
  animation-name: fade-in;
}
.slide-in-from-left-2 {
  animation-name: slide-in-from-left-2;
}
.slide-in-from-bottom-4 {
  animation-name: slide-in-from-bottom-4;
}
```

### Option B: Add framer-motion (For Complex Animations)

Only install if you need `AnimatePresence` for exit animations:

```bash
cd dashboard && npm install framer-motion
```

Then update components to use `motion.div` instead of `div`.

---

## 📁 Task 5: Create Loading States Component

### File: `dashboard/components/agent-loading.tsx`

```tsx
'use client';

/**
 * AgentLoading - Loading states for agent operations.
 * Uses CSS animations - no framer-motion required.
 */

import { Brain, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface AgentLoadingProps {
  message?: string;
  agentName?: string;
  className?: string;
}

export function AgentLoading({
  message = 'Thinking...',
  agentName,
  className,
}: AgentLoadingProps) {
  return (
    <div
      className={cn(
        'flex items-center gap-3 p-4 rounded-lg bg-gray-800/50 border border-gray-700',
        'animate-in fade-in duration-200',
        className,
      )}
    >
      <div className="relative">
        <Brain className="w-8 h-8 text-purple-500" />
        <div className="absolute inset-0 flex items-center justify-center">
          <Loader2 className="w-10 h-10 text-purple-500/30 animate-spin" />
        </div>
      </div>

      <div>
        {agentName && (
          <p className="text-sm text-gray-400 capitalize">{agentName}</p>
        )}
        <p className="text-white font-medium">{message}</p>
      </div>
    </div>
  );
}

// Skeleton for thinking steps
export function ThinkingStepSkeleton() {
  return (
    <div className="space-y-2 p-3">
      {[1, 2, 3].map((i) => (
        <div
          key={i}
          className="flex items-start gap-2 p-3 rounded-lg bg-gray-800/30 animate-in fade-in"
          style={{ animationDelay: `${i * 100}ms` }}
        >
          <div className="w-4 h-4 rounded bg-gray-700 animate-pulse" />
          <div className="flex-1 space-y-2">
            <div className="h-4 w-24 rounded bg-gray-700 animate-pulse" />
            <div className="h-3 w-full rounded bg-gray-700/50 animate-pulse" />
          </div>
        </div>
      ))}
    </div>
  );
}

export default AgentLoading;
```

---

## 📁 Task 6: Update Chat Component for Multi-Agent

### File: `dashboard/components/chat-with-history.tsx` (updates)

Add to the existing chat component:

```tsx
// Import the reasoning renderer
import { AgentReasoningRenderer } from './agent-reasoning';

// Inside the component, before the message list:
<AgentReasoningRenderer />;

// Or use the CopilotKit chat with custom render:
import { CopilotChat } from '@copilotkit/react-ui';

export function ChatWithMultiAgent() {
  return (
    <div className="flex flex-col h-full">
      <CopilotChat
        labels={{
          title: 'KGAgent',
          placeholder: 'Ask me anything...',
        }}
        className="flex-1"
      />
    </div>
  );
}
```

---

## 📁 Task 7: CSS Enhancements

### File: `dashboard/app/globals.css` (additions)

```css
/* Agent Reasoning Animations */
@keyframes pulse-border {
  0%,
  100% {
    border-color: rgba(168, 85, 247, 0.3);
  }
  50% {
    border-color: rgba(168, 85, 247, 0.6);
  }
}

.agent-step-active {
  animation: pulse-border 2s ease-in-out infinite;
}

/* Status panel shadow */
.agent-status-panel {
  box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.1), 0 4px 6px -1px rgba(0, 0, 0, 0.1),
    0 2px 4px -1px rgba(0, 0, 0, 0.06), 0 0 40px rgba(59, 130, 246, 0.1);
}

/* Thinking step hover */
.thinking-step:hover {
  background-color: rgba(255, 255, 255, 0.05);
}

/* Agent icon glow */
.agent-icon-active {
  filter: drop-shadow(0 0 8px currentColor);
}
```

---

## ✅ Phase 6 Definition of Done

- [ ] Components created:

  - [ ] `agent-reasoning.tsx` - In-chat reasoning display using
        `useCoAgentStateRender`
  - [ ] `agent-status-panel.tsx` - Fixed position status using `useCoAgent`
  - [ ] `agent-loading.tsx` - Loading states (CSS animations)

- [ ] CopilotKit integration:

  - [ ] `useCoAgentStateRender` renders thinking steps in chat
  - [ ] `useCoAgent` provides state outside chat (status panel)
  - [ ] Agent name matches backend (`kg_multi_agent`)
  - [ ] Provider updated to include new components

- [ ] Visual polish:

  - [ ] CSS animations working (or framer-motion if installed)
  - [ ] Agent colors/icons consistent across components
  - [ ] Status indicators show `running` state from `useCoAgent`

- [ ] Manual verification:
  - [ ] Send message in UI
  - [ ] See reasoning steps appear in real-time via `copilotkit_emit_state`
  - [ ] See status panel update with `running` indicator
  - [ ] Final response displays correctly
  - [ ] Status panel shows execution path

---

## 🔗 Next Phase

→ [Phase 7: Testing & Polish](./phase7-testing.md) - Integration tests and final
polish

---

## 📚 References

- [CopilotKit useCoAgent Hook](https://docs.copilotkit.ai/reference/hooks/useCoAgent)
- [CopilotKit useCoAgentStateRender Hook](https://docs.copilotkit.ai/reference/hooks/useCoAgentStateRender)
- [Predictive State Updates](https://docs.copilotkit.ai/langgraph/shared-state/predictive-state-updates)
- [Agentic Generative UI](https://docs.copilotkit.ai/langgraph/generative-ui/agentic)

---

_Last Updated: November 29, 2025_
