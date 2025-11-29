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
  const { state, running, threadId } = useCoAgent<AgentState>({
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
  const isActive = running || (state?.thinking_steps?.length ?? 0) > 0;

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
              <span className="text-white">
                {state?.total_llm_calls || 0}
              </span>
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

