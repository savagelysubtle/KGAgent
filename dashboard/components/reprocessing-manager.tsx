"use client";

import { useState, useEffect, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { documentsApi, reprocessApi } from "@/lib/api";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Loader2,
  RefreshCw,
  Sparkles,
  Brain,
  Network,
  User,
  Building2,
  MapPin,
  Lightbulb,
  Calendar,
  Cpu,
  Package,
  ArrowRight,
  CheckCircle2,
  XCircle,
  Clock,
  Zap,
  Pause,
  Play,
  StopCircle,
  Trash2,
  ListTodo,
} from "lucide-react";

// Types
interface Document {
  id: string;
  title: string;
  source_type: string;
  status: string;
  chunk_count: number;
  vector_ids: string[];
  metadata?: {
    entities_count?: number;
    relationships_count?: number;
    reprocessed_at?: string;
  };
  created_at: string;
}

interface Entity {
  name: string;
  type: string;
  description: string;
  confidence: number;
}

interface Relationship {
  related_entity: string;
  type: string;
  description: string;
}

interface GraphStats {
  status: string;
  total_entities: number;
  total_relationships: number;
  documents_with_entities: number;
  entities_by_type: Record<string, number>;
  relationships_by_type: Record<string, number>;
}

interface ReprocessingResult {
  doc_id: string;
  status: string;
  entities_extracted: number;
  entities_after_dedup: number;
  relationships_extracted: number;
  relationships_after_dedup: number;
  nodes_created: number;
  edges_created: number;
  processing_time: number;
  error?: string;
}

interface ProcessingJob {
  job_id: string;
  doc_id: string;
  status: string;
  progress_percent: number;
  processed_chunks: number;
  total_chunks: number;
  entities_extracted: number;
  relationships_extracted: number;
  created_at: string;
  updated_at: string;
  started_at?: string;
  paused_at?: string;
  can_resume?: boolean;
}

// Entity type icon mapping
const entityTypeIcons: Record<string, React.ReactNode> = {
  Person: <User className="h-4 w-4" />,
  Organization: <Building2 className="h-4 w-4" />,
  Location: <MapPin className="h-4 w-4" />,
  Concept: <Lightbulb className="h-4 w-4" />,
  Event: <Calendar className="h-4 w-4" />,
  Technology: <Cpu className="h-4 w-4" />,
  Product: <Package className="h-4 w-4" />,
};

// Entity type colors
const entityTypeColors: Record<string, string> = {
  Person: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  Organization: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  Location: "bg-green-500/20 text-green-400 border-green-500/30",
  Concept: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  Event: "bg-orange-500/20 text-orange-400 border-orange-500/30",
  Technology: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
  Product: "bg-pink-500/20 text-pink-400 border-pink-500/30",
};

export function ReprocessingManager() {
  const queryClient = useQueryClient();
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [processingDocId, setProcessingDocId] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<ReprocessingResult | null>(null);
  const [selectedEntity, setSelectedEntity] = useState<Entity | null>(null);

  // Fetch documents
  const { data: documentsData, isLoading: docsLoading } = useQuery({
    queryKey: ["documents"],
    queryFn: async () => {
      const response = await documentsApi.list({ limit: 100 });
      return response.data;
    },
    refetchInterval: 5000,
  });

  // Fetch graph stats
  const { data: graphStats, isLoading: statsLoading } = useQuery({
    queryKey: ["reprocess-graph-stats"],
    queryFn: async () => {
      const response = await reprocessApi.getGraphStats();
      return response.data as GraphStats;
    },
    refetchInterval: 10000,
  });

  // Fetch all entities
  const { data: entitiesData, isLoading: entitiesLoading } = useQuery({
    queryKey: ["entities"],
    queryFn: async () => {
      const response = await reprocessApi.listEntities({ limit: 100 });
      return response.data;
    },
    refetchInterval: 10000,
  });

  // Reprocess single document mutation
  const reprocessMutation = useMutation({
    mutationFn: async (docId: string) => {
      setProcessingDocId(docId);
      const response = await reprocessApi.reprocessDocument(docId, {
        extract_entities: true,
        extract_relationships: true,
        update_existing_graph: true,
        batch_size: 3,
      });
      return response.data as ReprocessingResult;
    },
    onSuccess: (result) => {
      setLastResult(result);
      setProcessingDocId(null);
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: ["reprocess-graph-stats"] });
      queryClient.invalidateQueries({ queryKey: ["entities"] });
    },
    onError: () => {
      setProcessingDocId(null);
    },
  });

  // Batch reprocess mutation
  const batchReprocessMutation = useMutation({
    mutationFn: async (docIds: string[]) => {
      const response = await reprocessApi.reprocessBatch(docIds, {
        extract_entities: true,
        extract_relationships: true,
        update_existing_graph: true,
        batch_size: 3,
      });
      return response.data;
    },
    onSuccess: () => {
      setSelectedDocs([]);
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: ["reprocess-graph-stats"] });
      queryClient.invalidateQueries({ queryKey: ["entities"] });
    },
  });

  // Fetch entity relationships
  const { data: relationshipsData, isLoading: relationshipsLoading } = useQuery({
    queryKey: ["entity-relationships", selectedEntity?.name],
    queryFn: async () => {
      if (!selectedEntity) return null;
      const response = await reprocessApi.getEntityRelationships(selectedEntity.name);
      return response.data;
    },
    enabled: !!selectedEntity,
  });

  // Fetch processing jobs
  const { data: jobsData, isLoading: jobsLoading } = useQuery({
    queryKey: ["processing-jobs"],
    queryFn: async () => {
      const response = await reprocessApi.listJobs({});
      return response.data;
    },
    refetchInterval: 3000, // Refresh every 3 seconds
  });

  // Start job mutation
  const startJobMutation = useMutation({
    mutationFn: async (docId: string) => {
      const response = await reprocessApi.startJob(docId, { batch_size: 3 });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["processing-jobs"] });
    },
  });

  // Pause job mutation
  const pauseJobMutation = useMutation({
    mutationFn: async (jobId: string) => {
      const response = await reprocessApi.pauseJob(jobId);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["processing-jobs"] });
    },
  });

  // Resume job mutation
  const resumeJobMutation = useMutation({
    mutationFn: async (jobId: string) => {
      const response = await reprocessApi.resumeJob(jobId);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["processing-jobs"] });
    },
  });

  // Cancel job mutation
  const cancelJobMutation = useMutation({
    mutationFn: async (jobId: string) => {
      const response = await reprocessApi.cancelJob(jobId);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["processing-jobs"] });
    },
  });

  // Delete job mutation
  const deleteJobMutation = useMutation({
    mutationFn: async (jobId: string) => {
      const response = await reprocessApi.deleteJob(jobId);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["processing-jobs"] });
    },
  });

  const jobs = jobsData?.jobs || [];

  const documents = documentsData?.documents || [];
  const entities = entitiesData?.entities || [];

  // Filter documents that can be reprocessed (completed status with chunks)
  const reprocessableDocuments = documents.filter(
    (doc: Document) => doc.status === "completed" && doc.chunk_count > 0
  );

  const handleSelectAll = useCallback(() => {
    if (selectedDocs.length === reprocessableDocuments.length) {
      setSelectedDocs([]);
    } else {
      setSelectedDocs(reprocessableDocuments.map((doc: Document) => doc.id));
    }
  }, [selectedDocs, reprocessableDocuments]);

  const handleSelectDoc = useCallback((docId: string) => {
    setSelectedDocs((prev) =>
      prev.includes(docId)
        ? prev.filter((id) => id !== docId)
        : [...prev, docId]
    );
  }, []);

  const formatTime = (seconds: number) => {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs.toFixed(0)}s`;
  };

  const getJobStatusColor = (status: string) => {
    switch (status) {
      case "running":
        return "bg-blue-500/20 text-blue-400 border-blue-500/30";
      case "paused":
        return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
      case "completed":
        return "bg-green-500/20 text-green-400 border-green-500/30";
      case "failed":
        return "bg-red-500/20 text-red-400 border-red-500/30";
      case "cancelled":
        return "bg-gray-500/20 text-gray-400 border-gray-500/30";
      case "pending":
        return "bg-purple-500/20 text-purple-400 border-purple-500/30";
      default:
        return "bg-gray-500/20 text-gray-400 border-gray-500/30";
    }
  };

  const getJobStatusIcon = (status: string) => {
    switch (status) {
      case "running":
        return <Loader2 className="h-4 w-4 animate-spin" />;
      case "paused":
        return <Pause className="h-4 w-4" />;
      case "completed":
        return <CheckCircle2 className="h-4 w-4" />;
      case "failed":
        return <XCircle className="h-4 w-4" />;
      case "cancelled":
        return <StopCircle className="h-4 w-4" />;
      case "pending":
        return <Clock className="h-4 w-4" />;
      default:
        return <Clock className="h-4 w-4" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-400">
              Total Entities
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-purple-400" />
              <span className="text-2xl font-bold text-purple-400">
                {statsLoading ? "..." : graphStats?.total_entities || 0}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-400">
              Relationships
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Network className="h-5 w-5 text-cyan-400" />
              <span className="text-2xl font-bold text-cyan-400">
                {statsLoading ? "..." : graphStats?.total_relationships || 0}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-400">
              Documents Ready
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-green-400" />
              <span className="text-2xl font-bold text-green-400">
                {reprocessableDocuments.length}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-400">
              Entity Types
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-yellow-400" />
              <span className="text-2xl font-bold text-yellow-400">
                {statsLoading
                  ? "..."
                  : Object.keys(graphStats?.entities_by_type || {}).length}
              </span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Entity Type Distribution */}
      {graphStats && Object.keys(graphStats.entities_by_type).length > 0 && (
        <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-lg">Entity Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-3">
              {Object.entries(graphStats.entities_by_type).map(
                ([type, count]) => (
                  <Badge
                    key={type}
                    variant="outline"
                    className={`${entityTypeColors[type] || "bg-gray-500/20 text-gray-400"} px-3 py-1.5`}
                  >
                    <span className="mr-2">
                      {entityTypeIcons[type] || <Lightbulb className="h-4 w-4" />}
                    </span>
                    {type}: {count}
                  </Badge>
                )
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Last Reprocessing Result */}
      {lastResult && (
        <Card className="bg-black/40 border-green-500/30 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-green-400" />
                Last Reprocessing Result
              </CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setLastResult(null)}
              >
                Dismiss
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-gray-400">Entities Extracted:</span>
                <span className="ml-2 text-white font-medium">
                  {lastResult.entities_extracted} → {lastResult.entities_after_dedup}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Relationships:</span>
                <span className="ml-2 text-white font-medium">
                  {lastResult.relationships_extracted} → {lastResult.relationships_after_dedup}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Nodes Created:</span>
                <span className="ml-2 text-green-400 font-medium">
                  {lastResult.nodes_created}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Processing Time:</span>
                <span className="ml-2 text-cyan-400 font-medium">
                  {formatTime(lastResult.processing_time)}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Main Tabs */}
      <Tabs defaultValue="jobs" className="space-y-4">
        <TabsList className="bg-black/40 border border-purple-500/20">
          <TabsTrigger value="jobs" className="flex items-center gap-2">
            <ListTodo className="h-4 w-4" />
            Jobs
            {jobs.filter((j: ProcessingJob) => j.status === "running" || j.status === "paused").length > 0 && (
              <Badge variant="outline" className="ml-1 bg-purple-500/20 text-purple-400 border-purple-500/30">
                {jobs.filter((j: ProcessingJob) => j.status === "running" || j.status === "paused").length}
              </Badge>
            )}
          </TabsTrigger>
          <TabsTrigger value="documents">Documents</TabsTrigger>
          <TabsTrigger value="entities">Entities</TabsTrigger>
          <TabsTrigger value="relationships">Relationships</TabsTrigger>
        </TabsList>

        {/* Jobs Tab */}
        <TabsContent value="jobs" className="space-y-4">
          <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <ListTodo className="h-5 w-5 text-purple-400" />
                    Processing Jobs
                  </CardTitle>
                  <CardDescription>
                    Monitor and control entity extraction jobs. Jobs can be paused and resumed.
                  </CardDescription>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    queryClient.invalidateQueries({ queryKey: ["processing-jobs"] });
                    queryClient.refetchQueries({ queryKey: ["processing-jobs"] });
                  }}
                  className="border-purple-500/30"
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Refresh
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {jobsLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-8 w-8 animate-spin text-purple-400" />
                </div>
              ) : jobs.length === 0 ? (
                <div className="text-center py-8 text-gray-400">
                  <ListTodo className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  No processing jobs found.
                  <br />
                  Start a resumable job from the Documents tab using the "Start Resumable" button.
                </div>
              ) : (
                <ScrollArea className="h-[500px]">
                  <div className="space-y-4">
                    {jobs.map((job: ProcessingJob) => (
                      <Card
                        key={job.job_id}
                        className={`bg-black/20 border ${
                          job.status === "running"
                            ? "border-blue-500/40"
                            : job.status === "paused"
                            ? "border-yellow-500/40"
                            : "border-purple-500/20"
                        }`}
                      >
                        <CardContent className="p-4">
                          <div className="flex items-start justify-between mb-4">
                            <div className="flex items-center gap-3">
                              <Badge
                                variant="outline"
                                className={`${getJobStatusColor(job.status)} flex items-center gap-1`}
                              >
                                {getJobStatusIcon(job.status)}
                                {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                              </Badge>
                              <span className="text-sm text-gray-400">
                                Job: {job.job_id.slice(0, 8)}...
                              </span>
                            </div>
                            <div className="flex items-center gap-2">
                              {/* Pause button - only for running jobs */}
                              {job.status === "running" && (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => pauseJobMutation.mutate(job.job_id)}
                                  disabled={pauseJobMutation.isPending}
                                  className="border-yellow-500/30 text-yellow-400 hover:bg-yellow-500/10"
                                >
                                  {pauseJobMutation.isPending ? (
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                  ) : (
                                    <>
                                      <Pause className="h-4 w-4 mr-1" />
                                      Pause
                                    </>
                                  )}
                                </Button>
                              )}

                              {/* Resume button - for paused or pending jobs */}
                              {(job.status === "paused" || job.status === "pending") && (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => resumeJobMutation.mutate(job.job_id)}
                                  disabled={resumeJobMutation.isPending}
                                  className="border-green-500/30 text-green-400 hover:bg-green-500/10"
                                >
                                  {resumeJobMutation.isPending ? (
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                  ) : (
                                    <>
                                      <Play className="h-4 w-4 mr-1" />
                                      Resume
                                    </>
                                  )}
                                </Button>
                              )}

                              {/* Cancel button - for running, paused, or pending jobs */}
                              {["running", "paused", "pending"].includes(job.status) && (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => cancelJobMutation.mutate(job.job_id)}
                                  disabled={cancelJobMutation.isPending}
                                  className="border-red-500/30 text-red-400 hover:bg-red-500/10"
                                >
                                  {cancelJobMutation.isPending ? (
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                  ) : (
                                    <>
                                      <StopCircle className="h-4 w-4 mr-1" />
                                      Cancel
                                    </>
                                  )}
                                </Button>
                              )}

                              {/* Delete button - for completed, failed, or cancelled jobs */}
                              {["completed", "failed", "cancelled"].includes(job.status) && (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => deleteJobMutation.mutate(job.job_id)}
                                  disabled={deleteJobMutation.isPending}
                                  className="border-gray-500/30 text-gray-400 hover:bg-gray-500/10"
                                >
                                  {deleteJobMutation.isPending ? (
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                  ) : (
                                    <>
                                      <Trash2 className="h-4 w-4 mr-1" />
                                      Delete
                                    </>
                                  )}
                                </Button>
                              )}
                            </div>
                          </div>

                          {/* Progress bar */}
                          <div className="mb-3">
                            <div className="flex justify-between text-sm mb-1">
                              <span className="text-gray-400">Progress</span>
                              <span className="text-white font-medium">
                                {job.processed_chunks} / {job.total_chunks} chunks ({job.progress_percent.toFixed(1)}%)
                              </span>
                            </div>
                            <Progress
                              value={job.progress_percent}
                              className="h-2 bg-black/40"
                            />
                          </div>

                          {/* Stats */}
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                            <div>
                              <span className="text-gray-400">Document:</span>
                              <span className="ml-2 text-white font-medium truncate">
                                {job.doc_id.slice(0, 8)}...
                              </span>
                            </div>
                            <div>
                              <span className="text-gray-400">Entities:</span>
                              <span className="ml-2 text-purple-400 font-medium">
                                {job.entities_extracted}
                              </span>
                            </div>
                            <div>
                              <span className="text-gray-400">Relationships:</span>
                              <span className="ml-2 text-cyan-400 font-medium">
                                {job.relationships_extracted}
                              </span>
                            </div>
                            <div>
                              <span className="text-gray-400">Updated:</span>
                              <span className="ml-2 text-gray-300 text-xs">
                                {new Date(job.updated_at).toLocaleTimeString()}
                              </span>
                            </div>
                          </div>

                          {/* Paused message */}
                          {job.status === "paused" && (
                            <div className="mt-3 p-2 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                              <div className="flex items-center gap-2 text-yellow-400 text-sm">
                                <Pause className="h-4 w-4" />
                                <span>
                                  Job paused at chunk {job.processed_chunks}. Click Resume to continue.
                                </span>
                              </div>
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </ScrollArea>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Documents Tab */}
        <TabsContent value="documents" className="space-y-4">
          <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Document Reprocessing</CardTitle>
                  <CardDescription>
                    Select documents to extract entities and build knowledge graph
                  </CardDescription>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleSelectAll}
                    className="border-purple-500/30"
                  >
                    {selectedDocs.length === reprocessableDocuments.length
                      ? "Deselect All"
                      : "Select All"}
                  </Button>
                  <Button
                    size="sm"
                    disabled={
                      selectedDocs.length === 0 ||
                      batchReprocessMutation.isPending
                    }
                    onClick={() => batchReprocessMutation.mutate(selectedDocs)}
                    className="bg-purple-600 hover:bg-purple-700"
                  >
                    {batchReprocessMutation.isPending ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-4 w-4 mr-2" />
                        Reprocess Selected ({selectedDocs.length})
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {docsLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-8 w-8 animate-spin text-purple-400" />
                </div>
              ) : reprocessableDocuments.length === 0 ? (
                <div className="text-center py-8 text-gray-400">
                  No documents available for reprocessing.
                  <br />
                  Upload and process documents first.
                </div>
              ) : (
                <ScrollArea className="h-[400px]">
                  <Table>
                    <TableHeader>
                      <TableRow className="border-purple-500/20">
                        <TableHead className="w-12"></TableHead>
                        <TableHead>Document</TableHead>
                        <TableHead>Chunks</TableHead>
                        <TableHead>Entities</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead className="text-right">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {reprocessableDocuments.map((doc: Document) => (
                        <TableRow
                          key={doc.id}
                          className="border-purple-500/10"
                        >
                          <TableCell>
                            <Checkbox
                              checked={selectedDocs.includes(doc.id)}
                              onCheckedChange={() => handleSelectDoc(doc.id)}
                            />
                          </TableCell>
                          <TableCell>
                            <div>
                              <div className="font-medium text-white">
                                {doc.title}
                              </div>
                              <div className="text-xs text-gray-500">
                                {doc.source_type}
                              </div>
                            </div>
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline" className="border-cyan-500/30 text-cyan-400">
                              {doc.chunk_count} chunks
                            </Badge>
                          </TableCell>
                          <TableCell>
                            {doc.metadata?.entities_count !== undefined ? (
                              <Badge variant="outline" className="border-green-500/30 text-green-400">
                                {doc.metadata.entities_count} entities
                              </Badge>
                            ) : (
                              <Badge variant="outline" className="border-gray-500/30 text-gray-400">
                                Not processed
                              </Badge>
                            )}
                          </TableCell>
                          <TableCell>
                            {doc.metadata?.reprocessed_at ? (
                              <div className="flex items-center gap-1 text-green-400 text-sm">
                                <CheckCircle2 className="h-4 w-4" />
                                <span>Processed</span>
                              </div>
                            ) : (
                              <div className="flex items-center gap-1 text-gray-400 text-sm">
                                <Clock className="h-4 w-4" />
                                <span>Pending</span>
                              </div>
                            )}
                          </TableCell>
                          <TableCell className="text-right">
                            <div className="flex items-center justify-end gap-2">
                              <Button
                                size="sm"
                                variant="ghost"
                                disabled={
                                  processingDocId === doc.id ||
                                  reprocessMutation.isPending
                                }
                                onClick={() => reprocessMutation.mutate(doc.id)}
                                className="text-purple-400 hover:text-purple-300"
                                title="Quick reprocess (non-resumable)"
                              >
                                {processingDocId === doc.id ? (
                                  <>
                                    <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                                    Processing...
                                  </>
                                ) : (
                                  <>
                                    <RefreshCw className="h-4 w-4 mr-1" />
                                    Quick
                                  </>
                                )}
                              </Button>
                              <Button
                                size="sm"
                                variant="outline"
                                disabled={startJobMutation.isPending}
                                onClick={() => startJobMutation.mutate(doc.id)}
                                className="border-green-500/30 text-green-400 hover:bg-green-500/10"
                                title="Start resumable job (can pause/resume)"
                              >
                                {startJobMutation.isPending ? (
                                  <Loader2 className="h-4 w-4 animate-spin" />
                                ) : (
                                  <>
                                    <Play className="h-4 w-4 mr-1" />
                                    Resumable
                                  </>
                                )}
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </ScrollArea>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Entities Tab */}
        <TabsContent value="entities" className="space-y-4">
          <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
            <CardHeader>
              <CardTitle>Extracted Entities</CardTitle>
              <CardDescription>
                All entities extracted from your documents
              </CardDescription>
            </CardHeader>
            <CardContent>
              {entitiesLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-8 w-8 animate-spin text-purple-400" />
                </div>
              ) : entities.length === 0 ? (
                <div className="text-center py-8 text-gray-400">
                  No entities extracted yet.
                  <br />
                  Reprocess documents to extract entities.
                </div>
              ) : (
                <ScrollArea className="h-[500px]">
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {entities.map((entity: Entity, index: number) => (
                      <Card
                        key={`${entity.name}-${index}`}
                        className={`bg-black/20 border cursor-pointer transition-all hover:scale-[1.02] ${
                          entityTypeColors[entity.type]?.replace("bg-", "border-").replace("/20", "/40") ||
                          "border-gray-500/40"
                        }`}
                        onClick={() => setSelectedEntity(entity)}
                      >
                        <CardContent className="p-4">
                          <div className="flex items-start gap-3">
                            <div
                              className={`p-2 rounded-lg ${
                                entityTypeColors[entity.type] ||
                                "bg-gray-500/20"
                              }`}
                            >
                              {entityTypeIcons[entity.type] || (
                                <Lightbulb className="h-4 w-4" />
                              )}
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="font-medium text-white truncate">
                                {entity.name}
                              </div>
                              <Badge
                                variant="outline"
                                className={`mt-1 text-xs ${
                                  entityTypeColors[entity.type] ||
                                  "bg-gray-500/20 text-gray-400"
                                }`}
                              >
                                {entity.type}
                              </Badge>
                              {entity.description && (
                                <p className="mt-2 text-xs text-gray-400 line-clamp-2">
                                  {entity.description}
                                </p>
                              )}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </ScrollArea>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Relationships Tab */}
        <TabsContent value="relationships" className="space-y-4">
          <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
            <CardHeader>
              <CardTitle>Entity Relationships</CardTitle>
              <CardDescription>
                {selectedEntity
                  ? `Relationships for: ${selectedEntity.name}`
                  : "Select an entity from the Entities tab to view its relationships"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!selectedEntity ? (
                <div className="text-center py-8 text-gray-400">
                  <Network className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  Click on an entity in the Entities tab to view its relationships
                </div>
              ) : relationshipsLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-8 w-8 animate-spin text-purple-400" />
                </div>
              ) : !relationshipsData?.relationships?.length ? (
                <div className="text-center py-8 text-gray-400">
                  No relationships found for this entity
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center gap-3 p-4 bg-black/20 rounded-lg border border-purple-500/20">
                    <div
                      className={`p-3 rounded-lg ${
                        entityTypeColors[selectedEntity.type] || "bg-gray-500/20"
                      }`}
                    >
                      {entityTypeIcons[selectedEntity.type] || (
                        <Lightbulb className="h-5 w-5" />
                      )}
                    </div>
                    <div>
                      <div className="font-medium text-white text-lg">
                        {selectedEntity.name}
                      </div>
                      <div className="text-sm text-gray-400">
                        {selectedEntity.description}
                      </div>
                    </div>
                  </div>

                  <ScrollArea className="h-[400px]">
                    <div className="space-y-2">
                      {relationshipsData.relationships.map(
                        (rel: Relationship, index: number) => (
                          <div
                            key={index}
                            className="flex items-center gap-3 p-3 bg-black/20 rounded-lg border border-purple-500/10"
                          >
                            <div className="flex-1">
                              <div className="flex items-center gap-2">
                                <span className="text-white font-medium">
                                  {selectedEntity.name}
                                </span>
                                <ArrowRight className="h-4 w-4 text-purple-400" />
                                <Badge
                                  variant="outline"
                                  className="border-purple-500/30 text-purple-400"
                                >
                                  {rel.type}
                                </Badge>
                                <ArrowRight className="h-4 w-4 text-purple-400" />
                                <span className="text-cyan-400 font-medium">
                                  {rel.related_entity}
                                </span>
                              </div>
                              {rel.description && (
                                <p className="mt-1 text-xs text-gray-400">
                                  {rel.description}
                                </p>
                              )}
                            </div>
                          </div>
                        )
                      )}
                    </div>
                  </ScrollArea>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

