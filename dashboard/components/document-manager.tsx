"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { documentsApi } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { toast } from "sonner";
import {
  Trash2,
  RefreshCw,
  Search,
  FileText,
  Globe,
  Upload,
  Database,
  AlertTriangle,
  CheckCircle,
  Clock,
  XCircle,
  Loader2,
} from "lucide-react";

interface Document {
  id: string;
  title: string;
  source_url: string | null;
  source_type: string;
  status: string;
  chunk_count: number;
  vector_ids: string[];
  graph_node_ids: string[];
  created_at: string;
  updated_at: string;
}

interface DocumentStats {
  total_documents: number;
  by_status: Record<string, number>;
  by_source_type: Record<string, number>;
  total_vectors: number;
  total_graph_nodes: number;
  total_graph_edges?: number;
  total_episodes?: number;
  chromadb_connected?: boolean;
  falkordb_connected?: boolean;
}

const statusIcons: Record<string, React.ReactNode> = {
  completed: <CheckCircle className="h-4 w-4 text-green-500" />,
  pending: <Clock className="h-4 w-4 text-yellow-500" />,
  processing: <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />,
  failed: <XCircle className="h-4 w-4 text-red-500" />,
  deleted: <Trash2 className="h-4 w-4 text-gray-500" />,
};

const sourceIcons: Record<string, React.ReactNode> = {
  web_crawl: <Globe className="h-4 w-4" />,
  file_upload: <Upload className="h-4 w-4" />,
  api: <Database className="h-4 w-4" />,
};

export function DocumentManager() {
  const queryClient = useQueryClient();
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [sourceFilter, setSourceFilter] = useState<string>("all");
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [deleteSourcePattern, setDeleteSourcePattern] = useState("");

  // Fetch documents
  const { data: documentsData, isLoading: isLoadingDocs, refetch: refetchDocs } = useQuery({
    queryKey: ["documents", search, statusFilter, sourceFilter],
    queryFn: async () => {
      const params: Record<string, string | number> = { limit: 100 };
      if (search) params.search = search;
      if (statusFilter !== "all") params.status = statusFilter;
      if (sourceFilter !== "all") params.source_type = sourceFilter;
      const response = await documentsApi.list(params);
      return response.data;
    },
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  // Fetch stats
  const { data: statsData, isLoading: isLoadingStats } = useQuery({
    queryKey: ["document-stats"],
    queryFn: async () => {
      const response = await documentsApi.getStats();
      return response.data as DocumentStats;
    },
    refetchInterval: 10000,
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: async (docId: string) => {
      return documentsApi.delete(docId, {
        delete_vectors: true,
        delete_graph_nodes: true,
      });
    },
    onSuccess: () => {
      toast.success("Document deleted successfully");
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: ["document-stats"] });
    },
    onError: (error) => {
      toast.error(`Failed to delete document: ${error}`);
    },
  });

  // Bulk delete mutation
  const bulkDeleteMutation = useMutation({
    mutationFn: async (docIds: string[]) => {
      return documentsApi.bulkDelete(docIds, {
        delete_vectors: true,
        delete_graph_nodes: true,
      });
    },
    onSuccess: (response) => {
      const result = response.data;
      toast.success(
        `Deleted ${result.documents_deleted} documents, ${result.vectors_deleted} vectors, ${result.graph_nodes_deleted} graph nodes`
      );
      setSelectedDocs([]);
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: ["document-stats"] });
    },
    onError: (error) => {
      toast.error(`Failed to delete documents: ${error}`);
    },
  });

  // Delete by source mutation
  const deleteBySourceMutation = useMutation({
    mutationFn: async (sourcePattern: string) => {
      return documentsApi.deleteBySource(sourcePattern, {
        delete_vectors: true,
        delete_graph_nodes: true,
      });
    },
    onSuccess: (response) => {
      const result = response.data;
      toast.success(
        `Deleted ${result.documents_deleted} documents from source`
      );
      setDeleteSourcePattern("");
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: ["document-stats"] });
    },
    onError: (error) => {
      toast.error(`Failed to delete by source: ${error}`);
    },
  });

  // Clear all mutation
  const clearAllMutation = useMutation({
    mutationFn: async () => {
      return documentsApi.clearAll(true);
    },
    onSuccess: (response) => {
      const result = response.data;
      const graphCleared = result.graph_cleared || {};
      const nodesDeleted = graphCleared.nodes_deleted || 0;
      const edgesDeleted = graphCleared.edges_deleted || 0;

      toast.success(
        `Cleared all data: ${result.documents_cleared} documents, ${result.vectors_cleared} vectors, ${nodesDeleted} graph nodes, ${edgesDeleted} edges`
      );

      // Show any errors that occurred
      if (result.errors && result.errors.length > 0) {
        toast.warning(`Some errors occurred: ${result.errors.join(', ')}`);
      }

      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: ["document-stats"] });
    },
    onError: (error) => {
      toast.error(`Failed to clear all data: ${error}`);
    },
  });

  // Sync existing data mutation
  const syncMutation = useMutation({
    mutationFn: async () => {
      return documentsApi.sync();
    },
    onSuccess: (response) => {
      const result = response.data;
      toast.success(
        `Synced: ${result.documents_created} documents, ${result.vectors_linked} vectors, ${result.graph_nodes_linked} graph nodes`
      );
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: ["document-stats"] });
    },
    onError: (error) => {
      toast.error(`Failed to sync data: ${error}`);
    },
  });

  // Fix titles mutation
  const fixTitlesMutation = useMutation({
    mutationFn: async () => {
      return documentsApi.fixTitles();
    },
    onSuccess: (response) => {
      const result = response.data;
      if (result.fixed > 0) {
        toast.success(`Fixed ${result.fixed} document titles`);
      } else {
        toast.info(`Checked ${result.checked} documents, no titles needed fixing`);
      }
      queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
    onError: (error) => {
      toast.error(`Failed to fix titles: ${error}`);
    },
  });

  // Delete stale mutation
  const deleteStaleMutation = useMutation({
    mutationFn: async () => {
      return documentsApi.deleteStale();
    },
    onSuccess: (response) => {
      const result = response.data;
      if (result.deleted > 0) {
        toast.success(`Deleted ${result.deleted} stale documents`);
      } else {
        toast.info(`Checked ${result.checked} documents, no stale documents found`);
      }
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: ["document-stats"] });
    },
    onError: (error) => {
      toast.error(`Failed to delete stale documents: ${error}`);
    },
  });

  const documents: Document[] = documentsData?.documents || [];
  const stats = statsData;

  const toggleDocSelection = (docId: string) => {
    setSelectedDocs((prev) =>
      prev.includes(docId)
        ? prev.filter((id) => id !== docId)
        : [...prev, docId]
    );
  };

  const toggleAllSelection = () => {
    if (selectedDocs.length === documents.length) {
      setSelectedDocs([]);
    } else {
      setSelectedDocs(documents.map((d) => d.id));
    }
  };

  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-400">
              Total Documents
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-purple-400">
              {isLoadingStats ? "..." : stats?.total_documents || 0}
            </p>
          </CardContent>
        </Card>

        <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-400 flex items-center gap-2">
              ChromaDB Vectors
              {stats?.chromadb_connected !== undefined && (
                <span className={`w-2 h-2 rounded-full ${stats.chromadb_connected ? 'bg-green-500' : 'bg-red-500'}`} />
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-blue-400">
              {isLoadingStats ? "..." : stats?.total_vectors?.toLocaleString() || 0}
            </p>
          </CardContent>
        </Card>

        <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-400 flex items-center gap-2">
              FalkorDB Graph
              {stats?.falkordb_connected !== undefined && (
                <span className={`w-2 h-2 rounded-full ${stats.falkordb_connected ? 'bg-green-500' : 'bg-red-500'}`} />
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-400">
              {isLoadingStats ? "..." : (
                <>
                  {stats?.total_graph_nodes?.toLocaleString() || 0}
                  <span className="text-sm text-gray-400 ml-1">nodes</span>
                  {stats?.total_graph_edges !== undefined && (
                    <span className="text-sm text-purple-400 ml-2">
                      {stats.total_graph_edges.toLocaleString()} edges
                    </span>
                  )}
                </>
              )}
            </p>
          </CardContent>
        </Card>

        <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-400">
              By Status
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            {stats?.by_status &&
              Object.entries(stats.by_status).map(([status, count]) => (
                <Badge
                  key={status}
                  variant="outline"
                  className="border-purple-500/30"
                >
                  {status}: {count}
                </Badge>
              ))}
          </CardContent>
        </Card>
      </div>

      {/* Main Document Management Card */}
      <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-xl text-purple-300">
                <FileText className="inline-block mr-2 h-5 w-5" />
                Document Manager
              </CardTitle>
              <CardDescription className="text-gray-400">
                Manage ingested documents and their data across all databases
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => syncMutation.mutate()}
                disabled={syncMutation.isPending}
                className="border-green-500/30 text-green-400 hover:bg-green-500/20"
                title="Import data from ChromaDB/FalkorDB into tracker"
              >
                {syncMutation.isPending ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Database className="h-4 w-4 mr-2" />
                )}
                Sync Data
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => fixTitlesMutation.mutate()}
                disabled={fixTitlesMutation.isPending}
                className="border-yellow-500/30 text-yellow-400 hover:bg-yellow-500/20"
                title="Fix hash-based titles using original filenames from metadata"
              >
                {fixTitlesMutation.isPending ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <FileText className="h-4 w-4 mr-2" />
                )}
                Fix Titles
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => deleteStaleMutation.mutate()}
                disabled={deleteStaleMutation.isPending}
                className="border-orange-500/30 text-orange-400 hover:bg-orange-500/20"
                title="Delete stuck/failed documents with no chunks"
              >
                {deleteStaleMutation.isPending ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Trash2 className="h-4 w-4 mr-2" />
                )}
                Clean Stale
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => refetchDocs()}
                className="border-purple-500/30 hover:bg-purple-500/20"
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {/* Filters */}
          <div className="flex flex-wrap gap-4">
            <div className="flex-1 min-w-[200px]">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <Input
                  placeholder="Search documents..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="pl-10 bg-black/20 border-purple-500/30"
                />
              </div>
            </div>

            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-[150px] bg-black/20 border-purple-500/30">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="completed">Completed</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
                <SelectItem value="processing">Processing</SelectItem>
                <SelectItem value="failed">Failed</SelectItem>
                <SelectItem value="deleted">Deleted</SelectItem>
              </SelectContent>
            </Select>

            <Select value={sourceFilter} onValueChange={setSourceFilter}>
              <SelectTrigger className="w-[150px] bg-black/20 border-purple-500/30">
                <SelectValue placeholder="Source" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Sources</SelectItem>
                <SelectItem value="web_crawl">Web Crawl</SelectItem>
                <SelectItem value="file_upload">File Upload</SelectItem>
                <SelectItem value="api">API</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Bulk Actions */}
          {selectedDocs.length > 0 && (
            <div className="flex items-center gap-4 p-3 bg-purple-500/10 rounded-lg border border-purple-500/30">
              <span className="text-sm text-purple-300">
                {selectedDocs.length} document(s) selected
              </span>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button
                    variant="destructive"
                    size="sm"
                    disabled={bulkDeleteMutation.isPending}
                  >
                    {bulkDeleteMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Trash2 className="h-4 w-4 mr-2" />
                    )}
                    Delete Selected
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent className="bg-slate-900 border-purple-500/30">
                  <AlertDialogHeader>
                    <AlertDialogTitle>Delete Selected Documents?</AlertDialogTitle>
                    <AlertDialogDescription>
                      This will delete {selectedDocs.length} document(s) and their
                      associated vectors and graph nodes. This action cannot be
                      undone.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction
                      onClick={() => bulkDeleteMutation.mutate(selectedDocs)}
                      className="bg-red-600 hover:bg-red-700"
                    >
                      Delete
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSelectedDocs([])}
              >
                Clear Selection
              </Button>
            </div>
          )}

          {/* Delete by Source */}
          <div className="flex items-center gap-4 p-3 bg-black/20 rounded-lg border border-purple-500/20">
            <Input
              placeholder="Enter source pattern (e.g., example.com)"
              value={deleteSourcePattern}
              onChange={(e) => setDeleteSourcePattern(e.target.value)}
              className="flex-1 bg-black/20 border-purple-500/30"
            />
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={!deleteSourcePattern || deleteBySourceMutation.isPending}
                  className="border-orange-500/30 text-orange-400 hover:bg-orange-500/20"
                >
                  {deleteBySourceMutation.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4 mr-2" />
                  )}
                  Delete by Source
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent className="bg-slate-900 border-purple-500/30">
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete Documents by Source?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This will delete all documents matching &quot;{deleteSourcePattern}&quot;
                    and their associated data. This action cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={() => deleteBySourceMutation.mutate(deleteSourcePattern)}
                    className="bg-orange-600 hover:bg-orange-700"
                  >
                    Delete
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>

          {/* Documents Table */}
          <div className="rounded-lg border border-purple-500/20 overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow className="border-purple-500/20 hover:bg-purple-500/5">
                  <TableHead className="w-[50px]">
                    <input
                      type="checkbox"
                      checked={selectedDocs.length === documents.length && documents.length > 0}
                      onChange={toggleAllSelection}
                      className="rounded border-purple-500/30"
                    />
                  </TableHead>
                  <TableHead>Title</TableHead>
                  <TableHead>Source</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Chunks</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoadingDocs ? (
                  <TableRow>
                    <TableCell colSpan={8} className="text-center py-8">
                      <Loader2 className="h-6 w-6 animate-spin mx-auto text-purple-400" />
                    </TableCell>
                  </TableRow>
                ) : documents.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={8} className="text-center py-8 text-gray-400">
                      No documents found
                    </TableCell>
                  </TableRow>
                ) : (
                  documents.map((doc) => (
                    <TableRow
                      key={doc.id}
                      className="border-purple-500/20 hover:bg-purple-500/5"
                    >
                      <TableCell>
                        <input
                          type="checkbox"
                          checked={selectedDocs.includes(doc.id)}
                          onChange={() => toggleDocSelection(doc.id)}
                          className="rounded border-purple-500/30"
                        />
                      </TableCell>
                      <TableCell className="font-medium max-w-[200px] truncate">
                        {doc.title}
                      </TableCell>
                      <TableCell className="max-w-[200px] truncate text-gray-400">
                        {doc.source_url || "-"}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          {sourceIcons[doc.source_type]}
                          <span className="text-sm">{doc.source_type}</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          {statusIcons[doc.status]}
                          <span className="text-sm">{doc.status}</span>
                        </div>
                      </TableCell>
                      <TableCell>{doc.chunk_count}</TableCell>
                      <TableCell className="text-gray-400 text-sm">
                        {new Date(doc.created_at).toLocaleDateString()}
                      </TableCell>
                      <TableCell className="text-right">
                        <AlertDialog>
                          <AlertDialogTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="text-red-400 hover:text-red-300 hover:bg-red-500/20"
                              disabled={deleteMutation.isPending}
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </AlertDialogTrigger>
                          <AlertDialogContent className="bg-slate-900 border-purple-500/30">
                            <AlertDialogHeader>
                              <AlertDialogTitle>Delete Document?</AlertDialogTitle>
                              <AlertDialogDescription>
                                This will delete &quot;{doc.title}&quot; and its associated
                                vectors ({doc.vector_ids.length}) and graph nodes (
                                {doc.graph_node_ids.length}). This action cannot be
                                undone.
                              </AlertDialogDescription>
                            </AlertDialogHeader>
                            <AlertDialogFooter>
                              <AlertDialogCancel>Cancel</AlertDialogCancel>
                              <AlertDialogAction
                                onClick={() => deleteMutation.mutate(doc.id)}
                                className="bg-red-600 hover:bg-red-700"
                              >
                                Delete
                              </AlertDialogAction>
                            </AlertDialogFooter>
                          </AlertDialogContent>
                        </AlertDialog>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>

          {/* Clear All Data */}
          <div className="flex justify-end pt-4 border-t border-purple-500/20">
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  className="border-red-500/30 text-red-400 hover:bg-red-500/20"
                  disabled={clearAllMutation.isPending}
                >
                  {clearAllMutation.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <AlertTriangle className="h-4 w-4 mr-2" />
                  )}
                  Clear All Data
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent className="bg-slate-900 border-red-500/30">
                <AlertDialogHeader>
                  <AlertDialogTitle className="text-red-400">
                    ⚠️ Clear All Data?
                  </AlertDialogTitle>
                  <AlertDialogDescription>
                    This will permanently delete ALL documents, vectors, and graph
                    nodes from all databases. This action cannot be undone!
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={() => clearAllMutation.mutate()}
                    className="bg-red-600 hover:bg-red-700"
                  >
                    Yes, Clear Everything
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

