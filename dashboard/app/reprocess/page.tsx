"use client";

import { ReprocessingManager } from "@/components/reprocessing-manager";

export default function ReprocessPage() {
  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
          Entity Extraction & Reprocessing
        </h1>
        <p className="text-gray-400 mt-2">
          Use AI to extract entities and relationships from your documents to build a rich knowledge graph
        </p>
      </div>

      <ReprocessingManager />
    </div>
  );
}

