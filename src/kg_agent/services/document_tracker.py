"""SQLite-based document tracking service for managing ingested documents."""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from ..core.config import settings
from ..core.logging import logger


class DocumentStatus(str, Enum):
    """Status of a document in the system."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class DocumentSource(str, Enum):
    """Source type of a document."""
    WEB_CRAWL = "web_crawl"
    FILE_UPLOAD = "file_upload"
    API = "api"


@dataclass
class TrackedDocument:
    """Represents a tracked document in the system."""
    id: str
    title: str
    source_url: Optional[str]
    source_type: str
    file_path: Optional[str]
    content_hash: Optional[str]
    chunk_count: int
    vector_ids: List[str]
    graph_node_ids: List[str]
    status: str
    error_message: Optional[str]
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str
    processed_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DocumentTrackerService:
    """
    SQLite-based service for tracking ingested documents.
    Provides CRUD operations and links to ChromaDB and FalkorDB data.
    """

    def __init__(self, db_path: str = None):
        """Initialize the document tracker service."""
        self.db_path = db_path or str(Path(settings.STORAGE_DIR) / "documents.db")
        self._ensure_db_directory()
        self._init_database()
        logger.info(f"DocumentTrackerService initialized with database at {self.db_path}")

    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Main documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_url TEXT,
                    source_type TEXT NOT NULL,
                    file_path TEXT,
                    content_hash TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    processed_at TEXT
                )
            """)

            # Document-Vector mapping table (ChromaDB IDs)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    vector_id TEXT NOT NULL,
                    chunk_index INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                    UNIQUE(document_id, vector_id)
                )
            """)

            # Document-Graph mapping table (FalkorDB node IDs)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_graph_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    node_type TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                    UNIQUE(document_id, node_id)
                )
            """)

            # Processing history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)

            # Create indexes for faster queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_source_type ON documents(source_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_vectors_document_id ON document_vectors(document_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_graph_nodes_document_id ON document_graph_nodes(document_id)")

            conn.commit()
            logger.info("Database schema initialized")

    def create_document(
        self,
        title: str,
        source_type: str,
        source_url: Optional[str] = None,
        file_path: Optional[str] = None,
        content_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TrackedDocument:
        """
        Create a new document record.

        Args:
            title: Document title
            source_type: Source type (web_crawl, file_upload, api)
            source_url: URL if web-sourced
            file_path: Local file path if file-uploaded
            content_hash: Hash of content for deduplication
            metadata: Additional metadata

        Returns:
            TrackedDocument: The created document
        """
        import json

        doc_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents (id, title, source_url, source_type, file_path,
                                       content_hash, status, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id, title, source_url, source_type, file_path,
                content_hash, DocumentStatus.PENDING.value,
                json.dumps(metadata or {}), now, now
            ))
            conn.commit()

        logger.info(f"Created document: {doc_id} - {title}")
        return self.get_document(doc_id)

    def get_document(self, doc_id: str) -> Optional[TrackedDocument]:
        """Get a document by ID."""
        import json

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()

            if not row:
                return None

            # Get associated vector IDs
            cursor.execute("SELECT vector_id FROM document_vectors WHERE document_id = ?", (doc_id,))
            vector_ids = [r["vector_id"] for r in cursor.fetchall()]

            # Get associated graph node IDs
            cursor.execute("SELECT node_id FROM document_graph_nodes WHERE document_id = ?", (doc_id,))
            graph_node_ids = [r["node_id"] for r in cursor.fetchall()]

            return TrackedDocument(
                id=row["id"],
                title=row["title"],
                source_url=row["source_url"],
                source_type=row["source_type"],
                file_path=row["file_path"],
                content_hash=row["content_hash"],
                chunk_count=row["chunk_count"],
                vector_ids=vector_ids,
                graph_node_ids=graph_node_ids,
                status=row["status"],
                error_message=row["error_message"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                processed_at=row["processed_at"]
            )

    def list_documents(
        self,
        status: Optional[str] = None,
        source_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        search: Optional[str] = None
    ) -> List[TrackedDocument]:
        """
        List documents with optional filtering.

        Args:
            status: Filter by status
            source_type: Filter by source type
            limit: Maximum number of results
            offset: Pagination offset
            search: Search in title and source_url

        Returns:
            List of TrackedDocument objects
        """
        query = "SELECT id FROM documents WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)

        if search:
            query += " AND (title LIKE ? OR source_url LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            doc_ids = [row["id"] for row in cursor.fetchall()]

        return [self.get_document(doc_id) for doc_id in doc_ids if doc_id]

    def update_document(
        self,
        doc_id: str,
        status: Optional[str] = None,
        chunk_count: Optional[int] = None,
        error_message: Optional[str] = None,
        processed_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[TrackedDocument]:
        """Update a document record."""
        import json

        updates = []
        params = []

        if status:
            updates.append("status = ?")
            params.append(status)

        if chunk_count is not None:
            updates.append("chunk_count = ?")
            params.append(chunk_count)

        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)

        if processed_at:
            updates.append("processed_at = ?")
            params.append(processed_at)

        if metadata:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))

        if not updates:
            return self.get_document(doc_id)

        updates.append("updated_at = ?")
        params.append(datetime.utcnow().isoformat())
        params.append(doc_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE documents SET {', '.join(updates)} WHERE id = ?
            """, params)
            conn.commit()

        return self.get_document(doc_id)

    def add_vector_ids(self, doc_id: str, vector_ids: List[str]):
        """Add vector IDs (ChromaDB) associated with a document."""
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            for idx, vector_id in enumerate(vector_ids):
                cursor.execute("""
                    INSERT OR IGNORE INTO document_vectors (document_id, vector_id, chunk_index, created_at)
                    VALUES (?, ?, ?, ?)
                """, (doc_id, vector_id, idx, now))
            conn.commit()

        logger.info(f"Added {len(vector_ids)} vector IDs to document {doc_id}")

    def add_graph_node_ids(self, doc_id: str, node_ids: List[str], node_type: str = "Document"):
        """Add graph node IDs (FalkorDB) associated with a document."""
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            for node_id in node_ids:
                cursor.execute("""
                    INSERT OR IGNORE INTO document_graph_nodes (document_id, node_id, node_type, created_at)
                    VALUES (?, ?, ?, ?)
                """, (doc_id, node_id, node_type, now))
            conn.commit()

        logger.info(f"Added {len(node_ids)} graph node IDs to document {doc_id}")

    def get_vector_ids_for_document(self, doc_id: str) -> List[str]:
        """Get all vector IDs associated with a document."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT vector_id FROM document_vectors WHERE document_id = ? ORDER BY chunk_index",
                (doc_id,)
            )
            return [row["vector_id"] for row in cursor.fetchall()]

    def get_graph_node_ids_for_document(self, doc_id: str) -> List[str]:
        """Get all graph node IDs associated with a document."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT node_id FROM document_graph_nodes WHERE document_id = ?",
                (doc_id,)
            )
            return [row["node_id"] for row in cursor.fetchall()]

    def delete_document(self, doc_id: str, soft_delete: bool = False) -> bool:
        """
        Delete a document and its associated data.

        Args:
            doc_id: Document ID to delete
            soft_delete: If True, mark as deleted instead of removing

        Returns:
            bool: True if successful
        """
        if soft_delete:
            self.update_document(doc_id, status=DocumentStatus.DELETED.value)
            self._add_history(doc_id, "soft_delete", "completed", "Document marked as deleted")
            return True

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Due to CASCADE, related records will be deleted automatically
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            deleted = cursor.rowcount > 0
            conn.commit()

        if deleted:
            logger.info(f"Deleted document: {doc_id}")

        return deleted

    def _add_history(self, doc_id: str, action: str, status: str, details: Optional[str] = None):
        """Add a processing history entry."""
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO processing_history (document_id, action, status, details, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (doc_id, action, status, details, now))
            conn.commit()

    def get_document_history(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get processing history for a document."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT action, status, details, created_at
                FROM processing_history
                WHERE document_id = ?
                ORDER BY created_at DESC
            """, (doc_id,))

            return [dict(row) for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total documents
            cursor.execute("SELECT COUNT(*) as count FROM documents")
            total = cursor.fetchone()["count"]

            # By status
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM documents
                GROUP BY status
            """)
            by_status = {row["status"]: row["count"] for row in cursor.fetchall()}

            # By source type
            cursor.execute("""
                SELECT source_type, COUNT(*) as count
                FROM documents
                GROUP BY source_type
            """)
            by_source = {row["source_type"]: row["count"] for row in cursor.fetchall()}

            # Total vectors and nodes
            cursor.execute("SELECT COUNT(*) as count FROM document_vectors")
            total_vectors = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM document_graph_nodes")
            total_nodes = cursor.fetchone()["count"]

            # Total chunks
            cursor.execute("SELECT SUM(chunk_count) as total FROM documents")
            total_chunks = cursor.fetchone()["total"] or 0

            return {
                "total_documents": total,
                "by_status": by_status,
                "by_source_type": by_source,
                "total_vectors": total_vectors,
                "total_graph_nodes": total_nodes,
                "total_chunks": total_chunks
            }

    def find_by_content_hash(self, content_hash: str) -> Optional[TrackedDocument]:
        """Find a document by its content hash (for deduplication)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM documents WHERE content_hash = ?", (content_hash,))
            row = cursor.fetchone()

            if row:
                return self.get_document(row["id"])
            return None


# Singleton instance
_document_tracker_instance: Optional[DocumentTrackerService] = None


def get_document_tracker() -> DocumentTrackerService:
    """Get or create the singleton DocumentTrackerService instance."""
    global _document_tracker_instance
    if _document_tracker_instance is None:
        _document_tracker_instance = DocumentTrackerService()
    return _document_tracker_instance

