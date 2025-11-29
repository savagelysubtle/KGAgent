"use client";

import { useCopilotAction, useCopilotReadable } from "@copilotkit/react-core";
import { crawlApi, graphApi, agentApi, documentsApi, uploadApi } from "@/lib/api";
import { toast } from "sonner";
import { useEffect, useState } from "react";
import { fileContextStore, type PreviewedFile } from "./chat-input-with-upload";

export function AgentActions() {
  const [dbStats, setDbStats] = useState<{
    vectorStore?: { total_chunks: number };
    graphDatabase?: { total_nodes: number; total_edges: number; connected: boolean };
  }>({});

  // Track previewed files for agent context
  const [previewedFiles, setPreviewedFiles] = useState<PreviewedFile[]>([]);

  // Subscribe to file context changes
  useEffect(() => {
    // Initial load
    setPreviewedFiles(fileContextStore.getFiles());

    // Subscribe to changes
    return fileContextStore.subscribe(() => {
      setPreviewedFiles(fileContextStore.getFiles());
    });
  }, []);

  // Fetch database stats periodically to provide context
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await agentApi.getStats();
        setDbStats({
          vectorStore: response.data.vector_store,
          graphDatabase: response.data.graph_database,
        });
      } catch (error) {
        console.error("Failed to fetch agent stats:", error);
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  // Provide database context to the Copilot
  useCopilotReadable({
    description: "Current knowledge base statistics",
    value: dbStats,
  });

  // Provide uploaded/previewed file content as context to the Copilot
  // This allows the agent to discuss files immediately without storing them
  useCopilotReadable({
    description: "Files uploaded by the user for discussion (not yet stored in knowledge base). The agent can answer questions about these files immediately.",
    value: previewedFiles.length > 0 ? {
      file_count: previewedFiles.length,
      files: previewedFiles.map(f => ({
        filename: f.filename,
        file_type: f.file_type,
        size_bytes: f.size_bytes,
        content: f.content,
        chunk_count: f.chunk_count,
      }))
    } : null,
  });

  // Action to start a crawl
  useCopilotAction({
    name: "startCrawl",
    description: "Start a web crawl for a given URL or list of URLs to add content to the knowledge base",
    parameters: [
      {
        name: "urls",
        type: "string[]",
        description: "The list of URLs to crawl",
        required: true,
      },
      {
        name: "depth",
        type: "number",
        description: "The depth of the crawl (default: 1)",
        required: false,
      },
    ],
    handler: async ({ urls, depth }) => {
      try {
        toast.info(`Starting crawl for ${urls.length} URLs...`);
        await crawlApi.startCrawl(urls, depth || 1);
        toast.success("Crawl job started!");
        return "Crawl job started successfully. The content will be processed and added to the knowledge base.";
      } catch (error) {
        console.error(error);
        toast.error("Failed to start crawl.");
        return "Failed to start crawl job.";
      }
    },
  });

  // Action to search the knowledge base using the Pydantic AI agent
  useCopilotAction({
    name: "searchKnowledgeBase",
    description: "Search the knowledge base for information using semantic (vector) search, structured (graph) search, or both (hybrid)",
    parameters: [
      {
        name: "query",
        type: "string",
        description: "The search query or topic to look for",
        required: true,
      },
      {
        name: "searchType",
        type: "string",
        description: "Type of search: 'vector' for semantic, 'graph' for structured, 'hybrid' for both (default: hybrid)",
        required: false,
      },
    ],
    handler: async ({ query, searchType }) => {
      try {
        const type = (searchType as 'vector' | 'graph' | 'hybrid') || 'hybrid';
        const result = await agentApi.search(query, type);
        return JSON.stringify(result.data, null, 2);
      } catch (error) {
        console.error(error);
        return "Failed to search the knowledge base.";
      }
    },
  });

  // Action to chat with the Pydantic AI agent
  useCopilotAction({
    name: "askAgent",
    description: "Ask the Knowledge Graph AI agent a question. The agent can search the knowledge base and provide intelligent answers.",
    parameters: [
      {
        name: "question",
        type: "string",
        description: "The question to ask the AI agent",
        required: true,
      },
    ],
    handler: async ({ question }) => {
      try {
        toast.info("Asking the AI agent...");
        const result = await agentApi.chat(question);
        return result.data.response;
      } catch (error) {
        console.error(error);
        toast.error("Failed to get response from agent.");
        return "Failed to get a response from the AI agent. Please ensure LM Studio is running with a model loaded.";
      }
    },
  });

  // Action to get database statistics
  useCopilotAction({
    name: "getDatabaseStats",
    description: "Get current statistics about the knowledge base including vector store and graph database metrics",
    parameters: [],
    handler: async () => {
      try {
        const result = await agentApi.getStats();
        const stats = result.data;

        return `📊 Knowledge Base Statistics:

Vector Store (ChromaDB):
- Collection: ${stats.vector_store.collection_name}
- Total Chunks: ${stats.vector_store.total_chunks}

Knowledge Graph (FalkorDB/Graphiti):
- Connected: ${stats.graph_database.connected ? '✅ Yes' : '❌ No'}
- Total Nodes: ${stats.graph_database.total_nodes}
- Total Edges: ${stats.graph_database.total_edges}
- Entity Types: ${JSON.stringify(stats.graph_database.entity_types)}
- Relationship Types: ${JSON.stringify(stats.graph_database.relationship_types)}`;
      } catch (error) {
        console.error(error);
        return "Failed to get database statistics.";
      }
    },
  });

  // Action to check agent health
  useCopilotAction({
    name: "checkAgentHealth",
    description: "Check the health status of the AI agent and its connections to LM Studio, ChromaDB, and FalkorDB",
    parameters: [],
    handler: async () => {
      try {
        const result = await agentApi.health();
        const health = result.data;

        return `🏥 Agent Health Status: ${health.status.toUpperCase()}

Components:
- LM Studio: ${health.components.llm_studio.status} (${health.components.llm_studio.base_url})
- Vector Store: ${health.components.vector_store.status}
- Graph Database: ${health.components.graph_database.status}`;
      } catch (error) {
        console.error(error);
        return "Failed to check agent health. The backend API might be unavailable.";
      }
    },
  });

  // ==================== Document Management Actions ====================

  // Action to list documents
  useCopilotAction({
    name: "listDocuments",
    description: "List ingested documents in the knowledge base with optional filtering by status or source type",
    parameters: [
      {
        name: "status",
        type: "string",
        description: "Filter by status: 'completed', 'pending', 'processing', 'failed', 'deleted'",
        required: false,
      },
      {
        name: "sourceType",
        type: "string",
        description: "Filter by source type: 'web_crawl', 'file_upload', 'api'",
        required: false,
      },
      {
        name: "search",
        type: "string",
        description: "Search in document titles and source URLs",
        required: false,
      },
    ],
    handler: async ({ status, sourceType, search }) => {
      try {
        const params: Record<string, string | number> = { limit: 20 };
        if (status) params.status = status;
        if (sourceType) params.source_type = sourceType;
        if (search) params.search = search;

        const result = await documentsApi.list(params);
        const docs = result.data.documents;

        if (!docs || docs.length === 0) {
          return "No documents found matching the criteria.";
        }

        let output = `📄 **Documents** (${docs.length} found)\n\n`;
        for (const doc of docs) {
          output += `**${doc.title}**\n`;
          output += `- ID: \`${doc.id}\`\n`;
          output += `- Source: ${doc.source_url || 'N/A'}\n`;
          output += `- Type: ${doc.source_type}\n`;
          output += `- Status: ${doc.status}\n`;
          output += `- Chunks: ${doc.chunk_count}\n\n`;
        }
        return output;
      } catch (error) {
        console.error(error);
        return "Failed to list documents.";
      }
    },
  });

  // Action to get document statistics
  useCopilotAction({
    name: "getDocumentStats",
    description: "Get statistics about tracked documents including counts by status and source type",
    parameters: [],
    handler: async () => {
      try {
        const result = await documentsApi.getStats();
        const stats = result.data;

        let output = `📊 **Document Statistics**\n\n`;
        output += `**Total Documents:** ${stats.total_documents}\n\n`;

        if (stats.by_status) {
          output += `**By Status:**\n`;
          for (const [status, count] of Object.entries(stats.by_status)) {
            output += `- ${status}: ${count}\n`;
          }
          output += `\n`;
        }

        if (stats.by_source_type) {
          output += `**By Source Type:**\n`;
          for (const [type, count] of Object.entries(stats.by_source_type)) {
            output += `- ${type}: ${count}\n`;
          }
          output += `\n`;
        }

        output += `**Total Vectors:** ${stats.total_vectors}\n`;
        output += `**Total Graph Nodes:** ${stats.total_graph_nodes}\n`;

        return output;
      } catch (error) {
        console.error(error);
        return "Failed to get document statistics.";
      }
    },
  });

  // Action to delete a document
  useCopilotAction({
    name: "deleteDocument",
    description: "Delete a specific document and its associated data from ChromaDB and FalkorDB",
    parameters: [
      {
        name: "documentId",
        type: "string",
        description: "The ID of the document to delete",
        required: true,
      },
    ],
    handler: async ({ documentId }) => {
      try {
        toast.info("Deleting document...");
        const result = await documentsApi.delete(documentId, {
          delete_vectors: true,
          delete_graph_nodes: true,
        });

        const data = result.data;
        toast.success("Document deleted!");

        return `✅ **Document Deleted**
- Vectors removed: ${data.vectors_deleted}
- Graph nodes removed: ${data.graph_nodes_deleted}`;
      } catch (error) {
        console.error(error);
        toast.error("Failed to delete document.");
        return "Failed to delete the document. It may not exist or there was an error.";
      }
    },
  });

  // Action to delete documents by source
  useCopilotAction({
    name: "deleteBySource",
    description: "Delete all documents from a specific source (e.g., 'example.com')",
    parameters: [
      {
        name: "sourcePattern",
        type: "string",
        description: "Pattern to match source URLs (e.g., 'example.com')",
        required: true,
      },
    ],
    handler: async ({ sourcePattern }) => {
      try {
        toast.info(`Deleting documents from ${sourcePattern}...`);
        const result = await documentsApi.deleteBySource(sourcePattern, {
          delete_vectors: true,
          delete_graph_nodes: true,
        });

        const data = result.data;
        toast.success(`Deleted ${data.documents_deleted} documents!`);

        return `✅ **Documents Deleted by Source**
- Source pattern: '${sourcePattern}'
- Documents removed: ${data.documents_deleted}
- Vectors removed: ${data.vectors_deleted}
- Graph nodes removed: ${data.graph_nodes_deleted}`;
      } catch (error) {
        console.error(error);
        toast.error("Failed to delete documents by source.");
        return "Failed to delete documents by source.";
      }
    },
  });

  // Action to clear all data
  useCopilotAction({
    name: "clearAllData",
    description: "⚠️ DANGER: Clear ALL data from all databases. This action cannot be undone!",
    parameters: [
      {
        name: "confirm",
        type: "boolean",
        description: "Must be true to confirm deletion of all data",
        required: true,
      },
    ],
    handler: async ({ confirm }) => {
      if (!confirm) {
        return "⚠️ Clear all data was NOT confirmed. Set confirm to true to delete all data.";
      }

      try {
        toast.warning("Clearing all data...");
        const result = await documentsApi.clearAll(true);

        const data = result.data;
        toast.success("All data cleared!");

        return `🗑️ **All Data Cleared**
- Documents cleared: ${data.documents_cleared}
- Vectors cleared: ${data.vectors_cleared}
- Graph nodes cleared: ${data.graph_cleared?.nodes_deleted || 0}
- Graph edges cleared: ${data.graph_cleared?.edges_deleted || 0}`;
      } catch (error) {
        console.error(error);
        toast.error("Failed to clear all data.");
        return "Failed to clear all data.";
      }
    },
  });

  // ==================== Document Upload Actions ====================

  // Action to add uploaded/previewed files to the knowledge base
  useCopilotAction({
    name: "addDocumentsToKnowledgeBase",
    description: "Add the currently uploaded/previewed files to the knowledge base permanently. Use this when the user says 'add to database', 'save to knowledge base', 'store these files', etc.",
    parameters: [],
    handler: async () => {
      const files = fileContextStore.getFiles();

      if (files.length === 0) {
        return "No files have been uploaded yet. Please drag and drop files into the chat first, or use the attachment button.";
      }

      // Get the original File objects
      const originalFiles = files
        .filter(f => f.file)
        .map(f => f.file as File);

      if (originalFiles.length === 0) {
        return "The uploaded files are no longer available for storage. Please upload them again.";
      }

      try {
        toast.info(`Adding ${originalFiles.length} file(s) to knowledge base...`);

        const result = await uploadApi.uploadFiles(originalFiles);
        const data = result.data;

        // Clear the previewed files after successful upload
        fileContextStore.clearFiles();

        toast.success(`${originalFiles.length} file(s) added to knowledge base!`);

        return `✅ **Files Added to Knowledge Base**
- Job ID: \`${data.job_id}\`
- Files received: ${data.files_received}
- Status: ${data.status}
- Message: ${data.message}

The files are now being processed (parsed, chunked, and indexed). You can check the pipeline status or search for this content once processing completes.`;
      } catch (error) {
        console.error(error);
        toast.error("Failed to add files to knowledge base.");
        return "Failed to add files to the knowledge base. Please try again.";
      }
    },
  });

  // Action to clear uploaded files without storing
  useCopilotAction({
    name: "clearUploadedFiles",
    description: "Clear/remove the currently uploaded files from the chat context without adding them to the knowledge base",
    parameters: [],
    handler: async () => {
      const files = fileContextStore.getFiles();

      if (files.length === 0) {
        return "No files are currently uploaded.";
      }

      const count = files.length;
      fileContextStore.clearFiles();
      toast.info(`Cleared ${count} file(s) from chat context`);

      return `✅ Cleared ${count} file(s) from the chat context. They were NOT added to the knowledge base.`;
    },
  });

  // Action to get info about currently uploaded files
  useCopilotAction({
    name: "getUploadedFilesInfo",
    description: "Get information about files currently uploaded in the chat (but not yet stored in knowledge base)",
    parameters: [],
    handler: async () => {
      const files = fileContextStore.getFiles();

      if (files.length === 0) {
        return "No files are currently uploaded. You can drag and drop files into the chat or use the attachment button.";
      }

      let output = `📎 **Currently Uploaded Files** (${files.length} file(s))\n\n`;

      for (const file of files) {
        output += `**${file.filename}**\n`;
        output += `- Type: ${file.file_type}\n`;
        output += `- Size: ${(file.size_bytes / 1024).toFixed(1)} KB\n`;
        output += `- Content length: ${file.content.length.toLocaleString()} chars\n`;
        output += `- Estimated chunks: ${file.chunk_count}\n\n`;
      }

      output += `\n💡 **Tip:** Say "add to knowledge base" to store these files permanently, or ask questions about them directly.`;

      return output;
    },
  });

  return null; // Headless component
}

