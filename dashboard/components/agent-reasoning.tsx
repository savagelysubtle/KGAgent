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

