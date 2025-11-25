"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";
import { graphApi } from "@/lib/api";
import { Loader2, Search, Database, Network } from "lucide-react";

interface GraphStats {
  total_nodes: number;
  total_edges: number;
  entity_types: Record<string, number>;
}

interface SearchResult {
  query: string;
  results: any[];
  count: number;
}

export function GraphExplorer() {
  const [stats, setStats] = useState<GraphStats | null>(null);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult | null>(null);
  const [loadingStats, setLoadingStats] = useState(true);
  const [searching, setSearching] = useState(false);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const response = await graphApi.getStats();
      setStats(response.data);
    } catch (error) {
      console.error("Failed to load graph stats:", error);
      toast.error("Failed to load graph statistics");
    } finally {
      setLoadingStats(false);
    }
  };

  const handleSearch = async () => {
    if (!query.trim()) return;

    setSearching(true);
    try {
      const response = await graphApi.query(query);
      setResults(response.data);
    } catch (error) {
      console.error("Search failed:", error);
      toast.error("Failed to search knowledge graph");
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Stats Overview */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-purple-100">
              Total Nodes
            </CardTitle>
            <Database className="h-4 w-4 text-purple-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-purple-100">
              {loadingStats ? <Loader2 className="h-4 w-4 animate-spin" /> : stats?.total_nodes || 0}
            </div>
          </CardContent>
        </Card>
        <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-purple-100">
              Total Edges
            </CardTitle>
            <Network className="h-4 w-4 text-purple-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-purple-100">
              {loadingStats ? <Loader2 className="h-4 w-4 animate-spin" /> : stats?.total_edges || 0}
            </div>
          </CardContent>
        </Card>
        <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-purple-100">
              Entity Types
            </CardTitle>
            <ActivityIcon />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-purple-100">
              {loadingStats ? <Loader2 className="h-4 w-4 animate-spin" /> : Object.keys(stats?.entity_types || {}).length}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Search Interface */}
      <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-purple-100">Graph Search</CardTitle>
          <CardDescription className="text-purple-200/60">
            Query the knowledge graph using natural language.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Input
              placeholder="Search for concepts (e.g., 'What are the key features of Crawl4AI?')"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              className="bg-black/20 border-purple-500/20 text-purple-100"
            />
            <Button
              onClick={handleSearch}
              disabled={searching}
              className="bg-purple-600 hover:bg-purple-700"
            >
              {searching ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
            </Button>
          </div>

          {results && (
            <div className="space-y-4 mt-4">
              <h3 className="text-sm font-medium text-purple-100">
                Results ({results.count})
              </h3>
              <div className="grid gap-3">
                {results.results.length === 0 ? (
                  <p className="text-sm text-purple-200/60">No results found.</p>
                ) : (
                  results.results.map((result, i) => (
                    <Card key={i} className="bg-purple-500/5 border-purple-500/10">
                      <CardContent className="p-4">
                        <pre className="text-xs text-purple-100 whitespace-pre-wrap overflow-auto">
                          {JSON.stringify(result, null, 2)}
                        </pre>
                      </CardContent>
                    </Card>
                  ))
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function ActivityIcon() {
  return (
    <svg
      className="h-4 w-4 text-purple-400"
      fill="none"
      height="24"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      viewBox="0 0 24 24"
      width="24"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
    </svg>
  )
}

