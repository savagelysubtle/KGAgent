"use client";

import { DocumentManager } from "@/components/document-manager";

export default function DocumentsPage() {
  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
          Document Management
        </h1>
        <p className="text-gray-400 mt-2">
          View, manage, and delete ingested documents and their data across ChromaDB and Neo4j
        </p>
      </div>
      <DocumentManager />
    </div>
  );
}

