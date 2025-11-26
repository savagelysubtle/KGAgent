import axios from 'axios';

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

export const crawlApi = {
  startCrawl: (urls: string[], depth: number = 1) =>
    api.post('/api/v1/crawl', { urls, depth }),
};

export const uploadApi = {
  uploadFiles: (files: File[]) => {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    return api.post('/api/v1/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
};

export const pipelineApi = {
    getStatus: () => api.get('/api/v1/pipeline/status'),
    runPipeline: (urls: string[]) => api.post('/api/v1/pipeline/run', { urls }),
};

export const statsApi = {
    getOverview: () => api.get('/api/v1/stats/overview'),
    getChromaDBStats: () => api.get('/api/v1/stats/chromadb'),
    getFalkorDBStats: () => api.get('/api/v1/stats/falkordb'),
    getNeo4jStats: () => api.get('/api/v1/stats/neo4j'), // Legacy, redirects to FalkorDB
    getJobs: () => api.get('/api/v1/stats/jobs'),
};

export const graphApi = {
    query: (query: string) => api.post('/api/v1/graph/search', { query, limit: 10 }),
    getStats: () => api.get('/api/v1/graph/stats'),
    build: (jobId?: string, sourceUrls?: string[]) => api.post('/api/v1/graph/build', { job_id: jobId, source_urls: sourceUrls }),
};

// Pydantic AI Agent API
export const agentApi = {
    // Chat with the agent
    chat: (message: string, userContext?: Record<string, unknown>) =>
        api.post('/api/v1/agent/chat', { message, user_context: userContext }),

    // Stream chat response (returns SSE stream)
    chatStream: (message: string, userContext?: Record<string, unknown>) =>
        api.post('/api/v1/agent/chat/stream', { message, user_context: userContext }, {
            responseType: 'stream',
        }),

    // Direct tool calls
    search: (query: string, searchType: 'vector' | 'graph' | 'hybrid' = 'hybrid', limit: number = 5) =>
        api.post('/api/v1/agent/tools/search', null, {
            params: { query, search_type: searchType, limit }
        }),

    // Get knowledge base stats
    getStats: () => api.get('/api/v1/agent/tools/stats'),

    // Health check
    health: () => api.get('/api/v1/agent/health'),
};

// Document Management API
export const documentsApi = {
    // List documents with optional filtering
    list: (params?: {
        status?: string;
        source_type?: string;
        search?: string;
        limit?: number;
        offset?: number;
    }) => api.get('/api/v1/documents/', { params }),

    // Get document statistics
    getStats: () => api.get('/api/v1/documents/stats'),

    // Get a specific document
    get: (docId: string) => api.get(`/api/v1/documents/${docId}`),

    // Create a new document record
    create: (data: {
        title: string;
        source_type: string;
        source_url?: string;
        file_path?: string;
        content_hash?: string;
        metadata?: Record<string, unknown>;
    }) => api.post('/api/v1/documents/', data),

    // Update a document
    update: (docId: string, data: {
        status?: string;
        chunk_count?: number;
        error_message?: string;
        metadata?: Record<string, unknown>;
    }) => api.patch(`/api/v1/documents/${docId}`, data),

    // Delete a document
    delete: (docId: string, options?: {
        delete_vectors?: boolean;
        delete_graph_nodes?: boolean;
        soft_delete?: boolean;
    }) => api.delete(`/api/v1/documents/${docId}`, {
        params: {
            delete_vectors: options?.delete_vectors ?? true,
            delete_graph_nodes: options?.delete_graph_nodes ?? true,
            soft_delete: options?.soft_delete ?? false,
        }
    }),

    // Bulk delete documents
    bulkDelete: (documentIds: string[], options?: {
        delete_vectors?: boolean;
        delete_graph_nodes?: boolean;
    }) => api.post('/api/v1/documents/bulk-delete', {
        document_ids: documentIds,
        ...options
    }),

    // Delete documents by source pattern
    deleteBySource: (sourcePattern: string, options?: {
        delete_vectors?: boolean;
        delete_graph_nodes?: boolean;
    }) => api.post('/api/v1/documents/delete-by-source', {
        source_pattern: sourcePattern,
        ...options
    }),

    // Clear all data (requires confirmation)
    clearAll: (confirm: boolean = false) =>
        api.post('/api/v1/documents/clear-all', null, { params: { confirm } }),

    // Get document processing history
    getHistory: (docId: string) => api.get(`/api/v1/documents/${docId}/history`),

    // Sync existing data from ChromaDB and FalkorDB into document tracker
    sync: () => api.post('/api/v1/documents/sync'),
};

// Reprocessing API - Entity Extraction and Knowledge Graph Enhancement
export const reprocessApi = {
    // Reprocess a single document (synchronous - not resumable)
    reprocessDocument: (docId: string, options?: {
        extract_entities?: boolean;
        extract_relationships?: boolean;
        update_existing_graph?: boolean;
        batch_size?: number;
        include_chunk_nodes?: boolean;
    }) => api.post(`/api/v1/reprocess/${docId}`, options || {}),

    // Batch reprocess multiple documents
    reprocessBatch: (documentIds: string[], options?: {
        extract_entities?: boolean;
        extract_relationships?: boolean;
        update_existing_graph?: boolean;
        batch_size?: number;
        include_chunk_nodes?: boolean;
    }) => api.post('/api/v1/reprocess/batch', {
        document_ids: documentIds,
        options: options || {}
    }),

    // Get reprocessing status for a document
    getStatus: (docId: string) => api.get(`/api/v1/reprocess/${docId}/status`),

    // Get entities for a specific document
    getDocumentEntities: (docId: string) => api.get(`/api/v1/reprocess/${docId}/entities`),

    // Get all entities in the knowledge graph
    listEntities: (params?: {
        limit?: number;
        entity_type?: string;
    }) => api.get('/api/v1/reprocess/entities/all', { params }),

    // Get relationships for an entity
    getEntityRelationships: (entityName: string) =>
        api.get(`/api/v1/reprocess/entities/${encodeURIComponent(entityName)}/relationships`),

    // Get entity graph statistics
    getGraphStats: () => api.get('/api/v1/reprocess/graph/stats'),

    // ==================== Resumable Job APIs ====================

    // Start a new extraction job (resumable)
    startJob: (docId: string, options?: {
        batch_size?: number;
    }) => api.post(`/api/v1/reprocess/jobs/${docId}/start`, { options }),

    // Pause a running job
    pauseJob: (jobId: string) => api.post(`/api/v1/reprocess/jobs/${jobId}/pause`),

    // Resume a paused job
    resumeJob: (jobId: string) => api.post(`/api/v1/reprocess/jobs/${jobId}/resume`),

    // Cancel a job
    cancelJob: (jobId: string) => api.post(`/api/v1/reprocess/jobs/${jobId}/cancel`),

    // Get job status
    getJobStatus: (jobId: string) => api.get(`/api/v1/reprocess/jobs/${jobId}`),

    // List all jobs
    listJobs: (params?: {
        doc_id?: string;
        status?: string;
    }) => api.get('/api/v1/reprocess/jobs', { params }),

    // List resumable jobs (paused or pending)
    listResumableJobs: () => api.get('/api/v1/reprocess/jobs/resumable'),

    // Delete a completed/failed/cancelled job
    deleteJob: (jobId: string) => api.delete(`/api/v1/reprocess/jobs/${jobId}`),
};

// Chat History API
export const chatApi = {
    // List conversations
    listConversations: (params?: {
        limit?: number;
        offset?: number;
        search?: string;
    }) => api.get('/api/v1/chat/conversations', { params }),

    // Create a new conversation
    createConversation: (data?: {
        title?: string;
        summary?: string;
        metadata?: Record<string, unknown>;
    }) => api.post('/api/v1/chat/conversations', data || {}),

    // Get a specific conversation
    getConversation: (convId: string) =>
        api.get(`/api/v1/chat/conversations/${convId}`),

    // Update a conversation
    updateConversation: (convId: string, data: {
        title?: string;
        summary?: string;
        metadata?: Record<string, unknown>;
    }) => api.patch(`/api/v1/chat/conversations/${convId}`, data),

    // Delete a conversation
    deleteConversation: (convId: string) =>
        api.delete(`/api/v1/chat/conversations/${convId}`),

    // Get messages for a conversation
    getMessages: (convId: string, params?: {
        limit?: number;
        offset?: number;
        order?: 'asc' | 'desc';
    }) => api.get(`/api/v1/chat/conversations/${convId}/messages`, { params }),

    // Add a message to a conversation
    addMessage: (convId: string, data: {
        role: 'user' | 'assistant' | 'system';
        content: string;
        metadata?: Record<string, unknown>;
    }) => api.post(`/api/v1/chat/conversations/${convId}/messages`, data),

    // Delete a message
    deleteMessage: (msgId: string) =>
        api.delete(`/api/v1/chat/messages/${msgId}`),

    // Get chat statistics
    getStats: () => api.get('/api/v1/chat/stats'),

    // Clear all chat history
    clearAll: (confirm: boolean = false) =>
        api.delete('/api/v1/chat/clear', { params: { confirm } }),
};

export default api;
