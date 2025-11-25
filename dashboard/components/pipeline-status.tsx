"use client";

import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { statsApi } from "@/lib/api";
import {
  Loader2,
  CheckCircle2,
  Circle,
  Clock,
  AlertCircle,
  Database,
  FileText,
  Layers,
  Network,
  Activity,
  RefreshCw
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

interface JobInfo {
  id: string;
  type: string;
  stage: string;
  timestamp: string;
}

interface ServiceStatus {
  status: string;
  chunks_stored?: number;
  nodes?: number;
  edges?: number;
  entity_types?: Record<string, number>;
}

interface StatsOverview {
  status: string;
  timestamp: string;
  metrics: {
    documents: number;
    chunks: number;
    entities: number;
    edges: number;
    active_jobs: number;
  };
  services: {
    chromadb: ServiceStatus;
    neo4j: ServiceStatus;
  };
  recent_jobs: JobInfo[];
}

const stageOrder = ["raw", "parsed", "chunked", "embedded", "graphed"];

function getStageProgress(stage: string): number {
  const index = stageOrder.indexOf(stage);
  if (index === -1) return 0;
  return ((index + 1) / stageOrder.length) * 100;
}

function getStageIcon(stage: string) {
  switch (stage) {
    case "chunked":
    case "embedded":
    case "graphed":
      return <CheckCircle2 className="h-4 w-4 text-green-500" />;
    case "parsed":
      return <Loader2 className="h-4 w-4 animate-spin text-blue-500" />;
    case "raw":
      return <Circle className="h-4 w-4 text-yellow-500" />;
    default:
      return <Circle className="h-4 w-4 text-gray-500" />;
  }
}

function formatTimestamp(timestamp: string): string {
  try {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins} min${diffMins > 1 ? 's' : ''} ago`;

    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;

    return date.toLocaleDateString();
  } catch {
    return timestamp;
  }
}

function getServiceStatusBadge(status: string) {
  if (status === "connected") {
    return <Badge variant="outline" className="bg-green-500/20 text-green-400 border-green-500/30">Connected</Badge>;
  }
  if (status === "disconnected") {
    return <Badge variant="outline" className="bg-yellow-500/20 text-yellow-400 border-yellow-500/30">Disconnected</Badge>;
  }
  if (status.startsWith("error")) {
    return <Badge variant="outline" className="bg-red-500/20 text-red-400 border-red-500/30">Error</Badge>;
  }
  return <Badge variant="outline" className="bg-gray-500/20 text-gray-400 border-gray-500/30">Unknown</Badge>;
}

export function PipelineStatus() {
  const { data, isLoading, error, refetch, isRefetching } = useQuery<StatsOverview>({
    queryKey: ['systemStats'],
    queryFn: async () => {
      const response = await statsApi.getOverview();
      return response.data;
    },
    refetchInterval: 10000, // Refresh every 10 seconds
    retry: 2,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-8 w-8 animate-spin text-purple-400" />
        <span className="ml-2 text-purple-200">Loading system stats...</span>
      </div>
    );
  }

  if (error) {
    return (
      <Card className="bg-red-950/20 border-red-500/30">
        <CardContent className="p-6">
          <div className="flex items-center gap-2 text-red-400">
            <AlertCircle className="h-5 w-5" />
            <span>Failed to load system stats. Is the backend running?</span>
          </div>
          <Button
            variant="outline"
            size="sm"
            className="mt-4 border-red-500/30 text-red-400 hover:bg-red-500/10"
            onClick={() => refetch()}
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  const metrics = data?.metrics ?? { documents: 0, chunks: 0, entities: 0, edges: 0, active_jobs: 0 };
  const services = data?.services ?? { chromadb: { status: "unknown" }, neo4j: { status: "unknown" } };
  const recentJobs = data?.recent_jobs ?? [];

  return (
    <div className="space-y-6">
      {/* Refresh Button */}
      <div className="flex justify-end">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => refetch()}
          disabled={isRefetching}
          className="text-purple-300 hover:text-purple-100 hover:bg-purple-500/10"
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${isRefetching ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm hover:border-purple-500/40 transition-colors">
          <CardContent className="p-6">
            <div className="flex items-center gap-2 mb-2">
              <FileText className="h-4 w-4 text-blue-400" />
              <p className="text-sm text-purple-200/60 font-medium">Documents</p>
            </div>
            <p className="text-2xl font-bold text-blue-400">{metrics.documents}</p>
          </CardContent>
        </Card>

        <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm hover:border-purple-500/40 transition-colors">
          <CardContent className="p-6">
            <div className="flex items-center gap-2 mb-2">
              <Layers className="h-4 w-4 text-purple-400" />
              <p className="text-sm text-purple-200/60 font-medium">Chunks</p>
            </div>
            <p className="text-2xl font-bold text-purple-400">{metrics.chunks}</p>
          </CardContent>
        </Card>

        <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm hover:border-purple-500/40 transition-colors">
          <CardContent className="p-6">
            <div className="flex items-center gap-2 mb-2">
              <Database className="h-4 w-4 text-pink-400" />
              <p className="text-sm text-purple-200/60 font-medium">Entities</p>
            </div>
            <p className="text-2xl font-bold text-pink-400">{metrics.entities}</p>
          </CardContent>
        </Card>

        <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm hover:border-purple-500/40 transition-colors">
          <CardContent className="p-6">
            <div className="flex items-center gap-2 mb-2">
              <Network className="h-4 w-4 text-amber-400" />
              <p className="text-sm text-purple-200/60 font-medium">Edges</p>
            </div>
            <p className="text-2xl font-bold text-amber-400">{metrics.edges}</p>
          </CardContent>
        </Card>
      </div>

      {/* Service Status */}
      <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-purple-100 flex items-center gap-2 text-lg">
            <Database className="h-5 w-5 text-purple-400" />
            Database Status
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-3 rounded-lg bg-purple-950/30">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-md bg-purple-500/20">
                <Layers className="h-4 w-4 text-purple-400" />
              </div>
              <div>
                <p className="text-sm font-medium text-purple-100">ChromaDB</p>
                <p className="text-xs text-purple-200/60">Vector Store</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-sm text-purple-200/80">
                {services.chromadb.chunks_stored ?? 0} vectors
              </span>
              {getServiceStatusBadge(services.chromadb.status)}
            </div>
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg bg-purple-950/30">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-md bg-blue-500/20">
                <Network className="h-4 w-4 text-blue-400" />
              </div>
              <div>
                <p className="text-sm font-medium text-purple-100">Neo4j</p>
                <p className="text-xs text-purple-200/60">Knowledge Graph</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-sm text-purple-200/80">
                {services.neo4j.nodes ?? 0} nodes, {services.neo4j.edges ?? 0} edges
              </span>
              {getServiceStatusBadge(services.neo4j.status)}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Recent Jobs / Pipeline Activity */}
      <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-purple-100 flex items-center gap-2 text-lg">
            <Activity className="h-5 w-5 text-purple-400" />
            Pipeline Activity
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {recentJobs.length === 0 ? (
            <div className="text-center py-8 text-purple-200/60">
              <Activity className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>No recent pipeline activity</p>
              <p className="text-sm mt-1">Start a crawl or upload files to begin processing</p>
            </div>
          ) : (
            recentJobs.map((job) => (
              <div key={job.id} className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2 text-purple-100">
                    {getStageIcon(job.stage)}
                    <span className="font-medium">{job.type === "crawl" ? "Crawl" : "Upload"}: </span>
                    <span className="text-purple-200/80 truncate max-w-[200px]" title={job.id}>
                      {job.id.substring(0, 30)}...
                    </span>
                  </div>
                  <span className="text-purple-200/60 text-xs flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {formatTimestamp(job.timestamp)}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <Progress
                    value={getStageProgress(job.stage)}
                    className="h-2 bg-purple-950/50 flex-1"
                  />
                  <span className="text-xs text-purple-200/60 capitalize w-16 text-right">
                    {job.stage}
                  </span>
                </div>
              </div>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}
