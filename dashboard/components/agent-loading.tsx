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

