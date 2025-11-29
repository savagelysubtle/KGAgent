"""Session management for multi-agent conversations."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...core.logging import logger


class SessionManager:
    """
    Manages conversation sessions for the multi-agent system.

    Provides:
    - Session ID generation
    - Session metadata storage
    - History retrieval
    """

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new session.

        Args:
            user_id: Optional user identifier
            metadata: Optional session metadata

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())

        self._sessions[session_id] = {
            "id": session_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "message_count": 0,
            "metadata": metadata or {},
        }

        logger.info(f"Created session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session info."""
        return self._sessions.get(session_id)

    def update_session(self, session_id: str, **updates) -> bool:
        """
        Update session metadata.

        Args:
            session_id: Session to update
            **updates: Key-value pairs to update

        Returns:
            True if session was updated, False if not found
        """
        if session_id in self._sessions:
            self._sessions[session_id]["last_active"] = datetime.now().isoformat()
            self._sessions[session_id].update(updates)
            return True
        return False

    def increment_message_count(self, session_id: str) -> bool:
        """
        Increment the message count for a session.

        Args:
            session_id: Session to update

        Returns:
            True if session was updated, False if not found
        """
        if session_id in self._sessions:
            self._sessions[session_id]["message_count"] += 1
            self._sessions[session_id]["last_active"] = datetime.now().isoformat()
            return True
        return False

    def list_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List sessions, optionally filtered by user.

        Args:
            user_id: Filter by user ID (optional)

        Returns:
            List of session dicts, sorted by last_active (newest first)
        """
        sessions = list(self._sessions.values())
        if user_id:
            sessions = [s for s in sessions if s.get("user_id") == user_id]
        return sorted(sessions, key=lambda s: s["last_active"], reverse=True)

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session to delete

        Returns:
            True if session was deleted, False if not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False

    def get_session_count(self, user_id: Optional[str] = None) -> int:
        """
        Get the number of sessions.

        Args:
            user_id: Filter by user ID (optional)

        Returns:
            Number of sessions
        """
        if user_id:
            return len(
                [s for s in self._sessions.values() if s.get("user_id") == user_id]
            )
        return len(self._sessions)

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Remove sessions older than max_age_hours.

        Args:
            max_age_hours: Maximum session age in hours

        Returns:
            Number of sessions deleted
        """
        now = datetime.now()
        to_delete = []

        for session_id, session in self._sessions.items():
            last_active = datetime.fromisoformat(session["last_active"])
            age_hours = (now - last_active).total_seconds() / 3600
            if age_hours > max_age_hours:
                to_delete.append(session_id)

        for session_id in to_delete:
            del self._sessions[session_id]

        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} old sessions")

        return len(to_delete)


# Singleton instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the singleton session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
