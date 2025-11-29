"""
SQLite-based processing job tracker for resumable entity extraction.

Tracks the progress of long-running jobs so they can be paused and resumed,
even after application shutdown.
"""

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..core.config import settings
from ..core.logging import logger


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            # Add Z suffix to indicate UTC timezone
            return obj.isoformat() + "Z"
        return super().default(obj)


def utc_now_iso() -> str:
    """Return current UTC time as ISO string with Z suffix."""
    return datetime.utcnow().isoformat() + "Z"


class JobStatus(str, Enum):
    """Status of a processing job."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Type of processing job."""

    ENTITY_EXTRACTION = "entity_extraction"
    GRAPH_BUILDING = "graph_building"
    EMBEDDING = "embedding"


@dataclass
class ProcessingJob:
    """Represents a processing job that can be paused and resumed."""

    id: str
    doc_id: str
    job_type: str
    status: str
    total_chunks: int
    processed_chunks: int
    entities_extracted: int
    relationships_extracted: int
    current_chunk_index: int  # Where to resume from
    options: Dict[str, Any]
    error_message: Optional[str]
    started_at: Optional[str]
    paused_at: Optional[str]
    completed_at: Optional[str]
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def progress_percent(self) -> float:
        """Get progress as percentage."""
        if self.total_chunks == 0:
            return 0.0
        return (self.processed_chunks / self.total_chunks) * 100


@dataclass
class ProcessedChunk:
    """Tracks which chunks have been processed for a job."""

    job_id: str
    chunk_id: str
    chunk_index: int
    entities_found: int
    relationships_found: int
    processed_at: str


class ProcessingJobTracker:
    """
    SQLite-based service for tracking processing jobs.
    Enables pause/resume functionality for long-running entity extraction.
    """

    def __init__(self, db_path: str = None):
        """Initialize the job tracker service."""
        self.db_path = db_path or str(Path(settings.STORAGE_DIR) / "processing_jobs.db")
        self._ensure_db_directory()
        self._init_database()
        logger.info(f"ProcessingJobTracker initialized with database at {self.db_path}")

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

            # Processing jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    total_chunks INTEGER DEFAULT 0,
                    processed_chunks INTEGER DEFAULT 0,
                    entities_extracted INTEGER DEFAULT 0,
                    relationships_extracted INTEGER DEFAULT 0,
                    current_chunk_index INTEGER DEFAULT 0,
                    options TEXT DEFAULT '{}',
                    error_message TEXT,
                    started_at TEXT,
                    paused_at TEXT,
                    completed_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    pause_requested INTEGER DEFAULT 0
                )
            """)

            # Add pause_requested column if it doesn't exist (migration)
            try:
                cursor.execute(
                    "ALTER TABLE processing_jobs ADD COLUMN pause_requested INTEGER DEFAULT 0"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Processed chunks table - tracks which chunks have been done
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processed_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    entities_found INTEGER DEFAULT 0,
                    relationships_found INTEGER DEFAULT 0,
                    processed_at TEXT NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES processing_jobs(id),
                    UNIQUE(job_id, chunk_id)
                )
            """)

            # Extracted entities table - persists entities before graph insertion
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS extracted_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    entity_data TEXT NOT NULL,
                    inserted_to_graph INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES processing_jobs(id)
                )
            """)

            # Extracted relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS extracted_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    relationship_data TEXT NOT NULL,
                    inserted_to_graph INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES processing_jobs(id)
                )
            """)

            # Indexes for performance
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_doc_id ON processing_jobs(doc_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status ON processing_jobs(status)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_job_id ON processed_chunks(job_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_job_id ON extracted_entities(job_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_relationships_job_id ON extracted_relationships(job_id)"
            )

            conn.commit()
            logger.info("Processing job database schema initialized")

        # Reset any "zombie" running jobs from previous server instances
        self._reset_zombie_jobs()

    def _reset_zombie_jobs(self):
        """Reset jobs that were running when the server died.

        On startup, any job marked as "running" is a zombie (the worker that was
        processing it is gone). Reset these to "paused" so they can be resumed.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE processing_jobs
                SET status = ?, paused_at = ?, pause_requested = 0,
                    error_message = 'Server restarted - job paused automatically'
                WHERE status = ?
            """,
                (JobStatus.PAUSED.value, utc_now_iso(), JobStatus.RUNNING.value),
            )

            if cursor.rowcount > 0:
                logger.warning(
                    f"Reset {cursor.rowcount} zombie job(s) from 'running' to 'paused' "
                    "(server was restarted while they were processing)"
                )
            conn.commit()

    # ==================== Job Management ====================

    def create_job(
        self,
        doc_id: str,
        job_type: JobType,
        total_chunks: int,
        options: Dict[str, Any] = None,
    ) -> ProcessingJob:
        """Create a new processing job."""
        now = utc_now_iso()
        job_id = str(uuid.uuid4())

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO processing_jobs
                (id, doc_id, job_type, status, total_chunks, options, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    job_id,
                    doc_id,
                    job_type.value if isinstance(job_type, JobType) else job_type,
                    JobStatus.PENDING.value,
                    total_chunks,
                    json.dumps(options or {}),
                    now,
                    now,
                ),
            )
            conn.commit()

        logger.info(f"Created processing job {job_id} for document {doc_id}")
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get a job by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM processing_jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return ProcessingJob(
                id=row["id"],
                doc_id=row["doc_id"],
                job_type=row["job_type"],
                status=row["status"],
                total_chunks=row["total_chunks"],
                processed_chunks=row["processed_chunks"],
                entities_extracted=row["entities_extracted"],
                relationships_extracted=row["relationships_extracted"],
                current_chunk_index=row["current_chunk_index"],
                options=json.loads(row["options"]),
                error_message=row["error_message"],
                started_at=row["started_at"],
                paused_at=row["paused_at"],
                completed_at=row["completed_at"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def update_job_options(self, job_id: str, options: Dict[str, Any]) -> bool:
        """Update the options for a job (e.g., to change batch_size).

        Only works for pending or paused jobs.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE processing_jobs
                SET options = ?, updated_at = ?
                WHERE id = ? AND status IN (?, ?)
                """,
                (
                    json.dumps(options),
                    utc_now_iso(),
                    job_id,
                    JobStatus.PENDING.value,
                    JobStatus.PAUSED.value,
                ),
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Updated options for job {job_id}: {options}")
                return True
            return False

    def get_jobs_for_document(self, doc_id: str) -> List[ProcessingJob]:
        """Get all jobs for a document."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM processing_jobs WHERE doc_id = ? ORDER BY created_at DESC",
                (doc_id,),
            )
            rows = cursor.fetchall()
            return [self._row_to_job(row) for row in rows]

    def get_active_job_for_document(self, doc_id: str) -> Optional[ProcessingJob]:
        """Get the active (running or paused) job for a document."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM processing_jobs
                WHERE doc_id = ? AND status IN (?, ?, ?)
                ORDER BY created_at DESC LIMIT 1
            """,
                (
                    doc_id,
                    JobStatus.PENDING.value,
                    JobStatus.RUNNING.value,
                    JobStatus.PAUSED.value,
                ),
            )
            row = cursor.fetchone()
            return self._row_to_job(row) if row else None

    def get_resumable_jobs(self) -> List[ProcessingJob]:
        """Get all jobs that can be resumed (paused or pending)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM processing_jobs
                WHERE status IN (?, ?)
                ORDER BY updated_at DESC
            """,
                (JobStatus.PAUSED.value, JobStatus.PENDING.value),
            )
            rows = cursor.fetchall()
            return [self._row_to_job(row) for row in rows]

    def get_all_active_jobs(self) -> List[ProcessingJob]:
        """Get all active jobs (pending, running, or paused)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM processing_jobs
                WHERE status IN (?, ?, ?)
                ORDER BY updated_at DESC
            """,
                (
                    JobStatus.PENDING.value,
                    JobStatus.RUNNING.value,
                    JobStatus.PAUSED.value,
                ),
            )
            rows = cursor.fetchall()
            return [self._row_to_job(row) for row in rows]

    def get_all_jobs(self, limit: int = 50) -> List[ProcessingJob]:
        """Get all jobs, ordered by most recent first."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM processing_jobs
                ORDER BY updated_at DESC
                LIMIT ?
            """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [self._row_to_job(row) for row in rows]

    def _row_to_job(self, row) -> ProcessingJob:
        """Convert a database row to a ProcessingJob."""
        return ProcessingJob(
            id=row["id"],
            doc_id=row["doc_id"],
            job_type=row["job_type"],
            status=row["status"],
            total_chunks=row["total_chunks"],
            processed_chunks=row["processed_chunks"],
            entities_extracted=row["entities_extracted"],
            relationships_extracted=row["relationships_extracted"],
            current_chunk_index=row["current_chunk_index"],
            options=json.loads(row["options"]),
            error_message=row["error_message"],
            started_at=row["started_at"],
            paused_at=row["paused_at"],
            completed_at=row["completed_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ==================== Job State Transitions ====================

    def start_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Start or resume a job."""
        now = utc_now_iso()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE processing_jobs
                SET status = ?, started_at = COALESCE(started_at, ?), paused_at = NULL,
                    updated_at = ?, pause_requested = 0
                WHERE id = ? AND status IN (?, ?)
            """,
                (
                    JobStatus.RUNNING.value,
                    now,
                    now,
                    job_id,
                    JobStatus.PENDING.value,
                    JobStatus.PAUSED.value,
                ),
            )
            conn.commit()

        logger.info(f"Started job {job_id}")
        return self.get_job(job_id)

    def pause_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Pause a running job."""
        now = utc_now_iso()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE processing_jobs
                SET status = ?, paused_at = ?, updated_at = ?, pause_requested = 0
                WHERE id = ? AND status = ?
            """,
                (JobStatus.PAUSED.value, now, now, job_id, JobStatus.RUNNING.value),
            )
            conn.commit()

        logger.info(f"Paused job {job_id}")
        return self.get_job(job_id)

    def request_pause(self, job_id: str) -> bool:
        """Request a job to pause. Sets a flag that the worker will check.

        This is used for cross-worker communication when running with multiple workers.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE processing_jobs
                SET pause_requested = 1, updated_at = ?
                WHERE id = ? AND status = ?
            """,
                (utc_now_iso(), job_id, JobStatus.RUNNING.value),
            )
            conn.commit()
            updated = cursor.rowcount > 0

        if updated:
            logger.info(f"Pause requested for job {job_id}")
        return updated

    def is_pause_requested(self, job_id: str) -> bool:
        """Check if a pause has been requested for a job.

        Used by workers to check if they should pause.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT pause_requested FROM processing_jobs WHERE id = ?", (job_id,)
            )
            row = cursor.fetchone()
            return bool(row and row[0])

    def clear_pause_request(self, job_id: str):
        """Clear the pause request flag after pausing or resuming."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE processing_jobs
                SET pause_requested = 0
                WHERE id = ?
            """,
                (job_id,),
            )
            conn.commit()

    def complete_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Mark a job as completed."""
        now = utc_now_iso()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE processing_jobs
                SET status = ?, completed_at = ?, updated_at = ?
                WHERE id = ?
            """,
                (JobStatus.COMPLETED.value, now, now, job_id),
            )
            conn.commit()

        logger.info(f"Completed job {job_id}")
        return self.get_job(job_id)

    def fail_job(self, job_id: str, error_message: str) -> Optional[ProcessingJob]:
        """Mark a job as failed."""
        now = utc_now_iso()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE processing_jobs
                SET status = ?, error_message = ?, updated_at = ?
                WHERE id = ?
            """,
                (JobStatus.FAILED.value, error_message, now, job_id),
            )
            conn.commit()

        logger.error(f"Job {job_id} failed: {error_message}")
        return self.get_job(job_id)

    def cancel_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Cancel a job."""
        now = utc_now_iso()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE processing_jobs
                SET status = ?, updated_at = ?
                WHERE id = ? AND status IN (?, ?, ?)
            """,
                (
                    JobStatus.CANCELLED.value,
                    now,
                    job_id,
                    JobStatus.PENDING.value,
                    JobStatus.RUNNING.value,
                    JobStatus.PAUSED.value,
                ),
            )
            conn.commit()

        logger.info(f"Cancelled job {job_id}")
        return self.get_job(job_id)

    # ==================== Progress Tracking ====================

    def update_progress(
        self,
        job_id: str,
        processed_chunks: int = None,
        current_chunk_index: int = None,
        entities_extracted: int = None,
        relationships_extracted: int = None,
        error_message: str = None,
    ):
        """Update job progress and optionally set error message."""
        now = utc_now_iso()

        updates = ["updated_at = ?"]
        params = [now]

        if processed_chunks is not None:
            updates.append("processed_chunks = ?")
            params.append(processed_chunks)
        if current_chunk_index is not None:
            updates.append("current_chunk_index = ?")
            params.append(current_chunk_index)
        if entities_extracted is not None:
            updates.append("entities_extracted = ?")
            params.append(entities_extracted)
        if relationships_extracted is not None:
            updates.append("relationships_extracted = ?")
            params.append(relationships_extracted)
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)

        params.append(job_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                UPDATE processing_jobs
                SET {", ".join(updates)}
                WHERE id = ?
            """,
                params,
            )
            conn.commit()

    def record_processed_chunk(
        self,
        job_id: str,
        chunk_id: str,
        chunk_index: int,
        entities_found: int = 0,
        relationships_found: int = 0,
    ):
        """Record that a chunk has been processed."""
        now = utc_now_iso()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO processed_chunks
                (job_id, chunk_id, chunk_index, entities_found, relationships_found, processed_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    job_id,
                    chunk_id,
                    chunk_index,
                    entities_found,
                    relationships_found,
                    now,
                ),
            )
            conn.commit()

    def get_processed_chunk_ids(self, job_id: str) -> Set[str]:
        """Get IDs of chunks that have already been processed for a job."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT chunk_id FROM processed_chunks WHERE job_id = ?", (job_id,)
            )
            return {row["chunk_id"] for row in cursor.fetchall()}

    def get_next_chunk_index(self, job_id: str) -> int:
        """Get the next chunk index to process."""
        job = self.get_job(job_id)
        return job.current_chunk_index if job else 0

    # ==================== Entity/Relationship Storage ====================

    def store_extracted_entity(
        self, job_id: str, chunk_id: str, entity_data: Dict[str, Any]
    ):
        """Store an extracted entity for later graph insertion."""
        now = utc_now_iso()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO extracted_entities (job_id, chunk_id, entity_data, created_at)
                VALUES (?, ?, ?, ?)
            """,
                (job_id, chunk_id, json.dumps(entity_data, cls=DateTimeEncoder), now),
            )
            conn.commit()

    def store_extracted_relationship(
        self, job_id: str, chunk_id: str, relationship_data: Dict[str, Any]
    ):
        """Store an extracted relationship for later graph insertion."""
        now = utc_now_iso()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO extracted_relationships (job_id, chunk_id, relationship_data, created_at)
                VALUES (?, ?, ?, ?)
            """,
                (
                    job_id,
                    chunk_id,
                    json.dumps(relationship_data, cls=DateTimeEncoder),
                    now,
                ),
            )
            conn.commit()

    def get_extracted_entities(
        self, job_id: str, not_inserted_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get extracted entities for a job."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if not_inserted_only:
                cursor.execute(
                    "SELECT entity_data FROM extracted_entities WHERE job_id = ? AND inserted_to_graph = 0",
                    (job_id,),
                )
            else:
                cursor.execute(
                    "SELECT entity_data FROM extracted_entities WHERE job_id = ?",
                    (job_id,),
                )
            return [json.loads(row["entity_data"]) for row in cursor.fetchall()]

    def get_extracted_relationships(
        self, job_id: str, not_inserted_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get extracted relationships for a job."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if not_inserted_only:
                cursor.execute(
                    "SELECT relationship_data FROM extracted_relationships WHERE job_id = ? AND inserted_to_graph = 0",
                    (job_id,),
                )
            else:
                cursor.execute(
                    "SELECT relationship_data FROM extracted_relationships WHERE job_id = ?",
                    (job_id,),
                )
            return [json.loads(row["relationship_data"]) for row in cursor.fetchall()]

    def mark_entities_inserted(self, job_id: str):
        """Mark all entities for a job as inserted to graph."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE extracted_entities SET inserted_to_graph = 1 WHERE job_id = ?",
                (job_id,),
            )
            conn.commit()

    def mark_relationships_inserted(self, job_id: str):
        """Mark all relationships for a job as inserted to graph."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE extracted_relationships SET inserted_to_graph = 1 WHERE job_id = ?",
                (job_id,),
            )
            conn.commit()

    # ==================== Cleanup ====================

    def delete_job(self, job_id: str):
        """Delete a job and all its associated data."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM processed_chunks WHERE job_id = ?", (job_id,))
            cursor.execute("DELETE FROM extracted_entities WHERE job_id = ?", (job_id,))
            cursor.execute(
                "DELETE FROM extracted_relationships WHERE job_id = ?", (job_id,)
            )
            cursor.execute("DELETE FROM processing_jobs WHERE id = ?", (job_id,))
            conn.commit()

        logger.info(f"Deleted job {job_id} and all associated data")

    def cleanup_old_completed_jobs(self, days_old: int = 30):
        """Clean up completed jobs older than specified days."""
        from datetime import timedelta

        cutoff = (datetime.utcnow() - timedelta(days=days_old)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get old completed job IDs
            cursor.execute(
                """
                SELECT id FROM processing_jobs
                WHERE status = ? AND completed_at < ?
            """,
                (JobStatus.COMPLETED.value, cutoff),
            )
            job_ids = [row["id"] for row in cursor.fetchall()]

            for job_id in job_ids:
                self.delete_job(job_id)

            logger.info(f"Cleaned up {len(job_ids)} old completed jobs")


# Singleton instance
_job_tracker_instance: Optional[ProcessingJobTracker] = None


def get_job_tracker() -> ProcessingJobTracker:
    """Get or create the singleton ProcessingJobTracker instance."""
    global _job_tracker_instance
    if _job_tracker_instance is None:
        _job_tracker_instance = ProcessingJobTracker()
    return _job_tracker_instance
