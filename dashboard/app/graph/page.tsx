import { GraphExplorer } from "@/components/graph-explorer";

export default function GraphPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-purple-100 mb-2">Knowledge Graph</h1>
        <p className="text-purple-200/60">
          Explore and query the constructed Knowledge Graph.
        </p>
      </div>

      <GraphExplorer />
    </div>
  );
}

