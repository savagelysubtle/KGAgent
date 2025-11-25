"""SQLite-based chat history service for storing conversation history."""

import sqlite3
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from ..core.config import settings
from ..core.logging import logger


class MessageRole(str, Enum):
    """Role of a message in a conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """Represents a single message in a conversation."""
    id: str
    conversation_id: str
    role: str
    content: str
    metadata: Dict[str, Any]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Conversation:
    """Represents a conversation/chat session."""
    id: str
    title: str
    summary: Optional[str]
    message_count: int
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str
    last_message_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ChatHistoryService:
    """
    SQLite-based service for storing and retrieving chat history.
    Provides CRUD operations for conversations and messages.
    """

    def __init__(self, db_path: str = None):
        """Initialize the chat history service."""
        self.db_path = db_path or str(Path(settings.STORAGE_DIR) / "chat_history.db")
        self._ensure_db_directory()
        self._init_database()
        logger.info(f"ChatHistoryService initialized with database at {self.db_path}")

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

            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    summary TEXT,
                    message_count INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_message_at TEXT
                )
            """)

            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)")

            conn.commit()
            logger.info("Chat history database schema initialized")

    # ==================== Conversation Methods ====================

    def create_conversation(
        self,
        title: str = "New Chat",
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """
        Create a new conversation.

        Args:
            title: Conversation title
            summary: Optional summary
            metadata: Additional metadata

        Returns:
            Conversation object
        """
        conv_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations (id, title, summary, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                conv_id, title, summary,
                json.dumps(metadata or {}), now, now
            ))
            conn.commit()

        logger.info(f"Created conversation: {conv_id} - {title}")
        return self.get_conversation(conv_id)

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return Conversation(
                id=row["id"],
                title=row["title"],
                summary=row["summary"],
                message_count=row["message_count"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                last_message_at=row["last_message_at"]
            )

    def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
        search: Optional[str] = None
    ) -> List[Conversation]:
        """
        List conversations with optional filtering.

        Args:
            limit: Maximum number of results
            offset: Pagination offset
            search: Search in title and summary

        Returns:
            List of Conversation objects
        """
        query = "SELECT * FROM conversations WHERE 1=1"
        params = []

        if search:
            query += " AND (title LIKE ? OR summary LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])

        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [
            Conversation(
                id=row["id"],
                title=row["title"],
                summary=row["summary"],
                message_count=row["message_count"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                last_message_at=row["last_message_at"]
            )
            for row in rows
        ]

    def update_conversation(
        self,
        conv_id: str,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Conversation]:
        """Update a conversation."""
        updates = []
        params = []

        if title is not None:
            updates.append("title = ?")
            params.append(title)

        if summary is not None:
            updates.append("summary = ?")
            params.append(summary)

        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))

        if not updates:
            return self.get_conversation(conv_id)

        updates.append("updated_at = ?")
        params.append(datetime.utcnow().isoformat())
        params.append(conv_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE conversations SET {', '.join(updates)} WHERE id = ?
            """, params)
            conn.commit()

        return self.get_conversation(conv_id)

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation and all its messages."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Messages will be deleted via CASCADE
            cursor.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
            deleted = cursor.rowcount > 0
            conn.commit()

        if deleted:
            logger.info(f"Deleted conversation: {conv_id}")

        return deleted

    # ==================== Message Methods ====================

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatMessage:
        """
        Add a message to a conversation.

        Args:
            conversation_id: ID of the conversation
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Additional metadata

        Returns:
            ChatMessage object
        """
        msg_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Insert message
            cursor.execute("""
                INSERT INTO messages (id, conversation_id, role, content, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                msg_id, conversation_id, role, content,
                json.dumps(metadata or {}), now
            ))

            # Update conversation
            cursor.execute("""
                UPDATE conversations
                SET message_count = message_count + 1,
                    updated_at = ?,
                    last_message_at = ?
                WHERE id = ?
            """, (now, now, conversation_id))

            conn.commit()

        logger.debug(f"Added message to conversation {conversation_id}: {role}")
        return self.get_message(msg_id)

    def get_message(self, msg_id: str) -> Optional[ChatMessage]:
        """Get a message by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM messages WHERE id = ?", (msg_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return ChatMessage(
                id=row["id"],
                conversation_id=row["conversation_id"],
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"]
            )

    def get_messages(
        self,
        conversation_id: str,
        limit: int = 100,
        offset: int = 0,
        order: str = "asc"
    ) -> List[ChatMessage]:
        """
        Get messages for a conversation.

        Args:
            conversation_id: ID of the conversation
            limit: Maximum number of messages
            offset: Pagination offset
            order: Sort order ('asc' or 'desc')

        Returns:
            List of ChatMessage objects
        """
        order_dir = "ASC" if order.lower() == "asc" else "DESC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at {order_dir}
                LIMIT ? OFFSET ?
            """, (conversation_id, limit, offset))
            rows = cursor.fetchall()

        return [
            ChatMessage(
                id=row["id"],
                conversation_id=row["conversation_id"],
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"]
            )
            for row in rows
        ]

    def delete_message(self, msg_id: str) -> bool:
        """Delete a specific message."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get conversation ID first
            cursor.execute("SELECT conversation_id FROM messages WHERE id = ?", (msg_id,))
            row = cursor.fetchone()

            if not row:
                return False

            conv_id = row["conversation_id"]

            # Delete message
            cursor.execute("DELETE FROM messages WHERE id = ?", (msg_id,))

            # Update conversation message count
            cursor.execute("""
                UPDATE conversations
                SET message_count = message_count - 1,
                    updated_at = ?
                WHERE id = ?
            """, (datetime.utcnow().isoformat(), conv_id))

            conn.commit()

        return True

    # ==================== Utility Methods ====================

    def generate_title_from_message(self, content: str, max_length: int = 50) -> str:
        """Generate a conversation title from the first message."""
        # Take first line or first N characters
        title = content.split('\n')[0].strip()
        if len(title) > max_length:
            title = title[:max_length - 3] + "..."
        return title or "New Chat"

    def get_stats(self) -> Dict[str, Any]:
        """Get chat history statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total conversations
            cursor.execute("SELECT COUNT(*) as count FROM conversations")
            total_conversations = cursor.fetchone()["count"]

            # Total messages
            cursor.execute("SELECT COUNT(*) as count FROM messages")
            total_messages = cursor.fetchone()["count"]

            # Messages by role
            cursor.execute("""
                SELECT role, COUNT(*) as count
                FROM messages
                GROUP BY role
            """)
            messages_by_role = {row["role"]: row["count"] for row in cursor.fetchall()}

            # Recent activity (last 7 days)
            cursor.execute("""
                SELECT COUNT(*) as count FROM conversations
                WHERE created_at > datetime('now', '-7 days')
            """)
            recent_conversations = cursor.fetchone()["count"]

            return {
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "messages_by_role": messages_by_role,
                "recent_conversations": recent_conversations
            }

    def clear_all(self) -> Dict[str, int]:
        """Clear all chat history."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get counts
            cursor.execute("SELECT COUNT(*) as count FROM conversations")
            conv_count = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM messages")
            msg_count = cursor.fetchone()["count"]

            # Delete all
            cursor.execute("DELETE FROM messages")
            cursor.execute("DELETE FROM conversations")
            conn.commit()

        logger.info(f"Cleared chat history: {conv_count} conversations, {msg_count} messages")
        return {"conversations_deleted": conv_count, "messages_deleted": msg_count}


# Singleton instance
_chat_history_instance: Optional[ChatHistoryService] = None


def get_chat_history() -> ChatHistoryService:
    """Get or create the singleton ChatHistoryService instance."""
    global _chat_history_instance
    if _chat_history_instance is None:
        _chat_history_instance = ChatHistoryService()
    return _chat_history_instance

