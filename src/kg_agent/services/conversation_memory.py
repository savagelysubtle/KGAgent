"""
Conversation Memory Service - Continuous learning from chat interactions.

This service automatically persists conversation data to the knowledge graph,
enabling the agent to:
- Remember past conversations and their context
- Learn user preferences over time
- Build a knowledge base about the user
- Recall relevant past discussions when answering new questions
"""

import asyncio
import json
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

from ..core.config import settings
from ..core.logging import logger
from .chat_history import get_chat_history, ChatHistoryService, ChatMessage, Conversation
from .graphiti_service import get_graphiti_service, GraphitiService


@dataclass
class UserProfile:
    """Represents learned information about the user."""
    name: Optional[str] = None
    preferences: Dict[str, Any] = None
    topics_of_interest: List[str] = None
    interaction_count: int = 0
    first_interaction: Optional[str] = None
    last_interaction: Optional[str] = None

    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}
        if self.topics_of_interest is None:
            self.topics_of_interest = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationSummary:
    """Summary of a conversation for knowledge graph storage."""
    conversation_id: str
    title: str
    summary: str
    topics: List[str]
    user_questions: List[str]
    key_insights: List[str]
    message_count: int
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConversationMemoryService:
    """
    Service for persisting conversation memory to the knowledge graph.

    Automatically:
    - Saves conversations as episodes to FalkorDB via Graphiti
    - Extracts and stores user information
    - Tracks topics discussed
    - Enables recall of past conversations
    """

    # How many messages to accumulate before syncing to graph
    SYNC_THRESHOLD = 5

    # Group ID for conversation memory in Graphiti
    MEMORY_GROUP_ID = "conversation_memory"

    def __init__(self):
        self._chat_history: Optional[ChatHistoryService] = None
        self._graphiti: Optional[GraphitiService] = None
        self._initialized = False
        self._pending_messages: Dict[str, List[ChatMessage]] = {}  # conv_id -> messages
        self._user_profile = UserProfile()

    async def initialize(self) -> bool:
        """Initialize the conversation memory service."""
        if self._initialized:
            return True

        try:
            self._chat_history = get_chat_history()
            self._graphiti = get_graphiti_service()

            if not self._graphiti._initialized:
                await self._graphiti.initialize()

            # Load existing user profile from graph if available
            await self._load_user_profile()

            self._initialized = True
            logger.info("ConversationMemoryService initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ConversationMemoryService: {e}")
            return False

    async def _load_user_profile(self):
        """Load user profile from the knowledge graph."""
        try:
            if not self._graphiti or not self._graphiti._graphiti:
                return

            # Search for user profile in graph
            results = await self._graphiti.search(
                query="user profile preferences information",
                limit=5
            )

            # Extract user info from results if found
            if results.get("nodes"):
                for node in results["nodes"]:
                    if "User" in str(node.get("labels", [])):
                        # Found user node, extract properties
                        props = node.get("properties", {})
                        if props.get("name"):
                            self._user_profile.name = props["name"]
                        if props.get("preferences"):
                            self._user_profile.preferences = props["preferences"]

            logger.info(f"Loaded user profile: {self._user_profile.name or 'Anonymous'}")

        except Exception as e:
            logger.debug(f"Could not load user profile: {e}")

    async def record_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record a message and potentially sync to knowledge graph.

        Args:
            conversation_id: The conversation this message belongs to
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional metadata

        Returns:
            True if message was recorded successfully
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Store in SQLite chat history
            message = self._chat_history.add_message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                metadata=metadata
            )

            # Track pending messages for batch sync
            if conversation_id not in self._pending_messages:
                self._pending_messages[conversation_id] = []
            self._pending_messages[conversation_id].append(message)

            # Update user profile interaction count
            self._user_profile.interaction_count += 1
            now = datetime.now(timezone.utc).isoformat()
            if not self._user_profile.first_interaction:
                self._user_profile.first_interaction = now
            self._user_profile.last_interaction = now

            # Check if we should sync to graph
            if len(self._pending_messages[conversation_id]) >= self.SYNC_THRESHOLD:
                await self._sync_conversation_to_graph(conversation_id)

            return True

        except Exception as e:
            logger.error(f"Failed to record message: {e}")
            return False

    async def _sync_conversation_to_graph(self, conversation_id: str) -> bool:
        """
        Sync pending messages to the knowledge graph.

        This creates an episode in Graphiti containing the conversation
        content, allowing it to be searched and recalled later.
        """
        if conversation_id not in self._pending_messages:
            return True

        messages = self._pending_messages[conversation_id]
        if not messages:
            return True

        try:
            # Get conversation info
            conversation = self._chat_history.get_conversation(conversation_id)
            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found")
                return False

            # Format messages as conversation text
            conversation_text = self._format_conversation_for_graph(
                conversation, messages
            )

            # Create episode in Graphiti
            result = await self._graphiti.add_episode(
                content=conversation_text,
                name=f"conversation_{conversation_id[:8]}",
                source_description=f"Chat conversation: {conversation.title}",
                source_type="message",
                reference_time=datetime.now(timezone.utc),
                group_id=self.MEMORY_GROUP_ID
            )

            # Clear pending messages
            self._pending_messages[conversation_id] = []

            nodes_created = result.get("nodes_created", 0) if result else 0
            edges_created = result.get("edges_created", 0) if result else 0

            logger.info(
                f"Synced conversation {conversation_id[:8]} to graph: "
                f"{nodes_created} nodes, {edges_created} edges"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to sync conversation to graph: {e}")
            return False

    def _format_conversation_for_graph(
        self,
        conversation: Conversation,
        messages: List[ChatMessage]
    ) -> str:
        """Format conversation messages for storage in knowledge graph."""
        parts = [
            f"Conversation: {conversation.title}",
            f"Date: {conversation.created_at}",
            f"ID: {conversation.id}",
            "",
            "Messages:"
        ]

        for msg in messages:
            role_label = {
                "user": "User",
                "assistant": "Assistant",
                "system": "System"
            }.get(msg.role, msg.role.title())

            parts.append(f"\n{role_label}: {msg.content}")

        return "\n".join(parts)

    async def sync_all_pending(self) -> Dict[str, int]:
        """Sync all pending conversations to the knowledge graph."""
        results = {"synced": 0, "failed": 0}

        for conv_id in list(self._pending_messages.keys()):
            if await self._sync_conversation_to_graph(conv_id):
                results["synced"] += 1
            else:
                results["failed"] += 1

        return results

    async def learn_user_preference(
        self,
        preference_key: str,
        preference_value: Any,
        source: str = "conversation"
    ) -> bool:
        """
        Learn and store a user preference.

        Args:
            preference_key: The preference name (e.g., "favorite_language")
            preference_value: The preference value
            source: Where this preference was learned from

        Returns:
            True if preference was stored successfully
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Update local profile
            self._user_profile.preferences[preference_key] = {
                "value": preference_value,
                "source": source,
                "learned_at": datetime.now(timezone.utc).isoformat()
            }

            # Store in knowledge graph
            preference_text = (
                f"User preference learned: {preference_key} = {preference_value}. "
                f"Source: {source}. "
                f"The user prefers or uses {preference_value} for {preference_key}."
            )

            await self._graphiti.add_episode(
                content=preference_text,
                name=f"user_preference_{preference_key}",
                source_description=f"User preference: {preference_key}",
                source_type="text",
                reference_time=datetime.now(timezone.utc),
                group_id=self.MEMORY_GROUP_ID
            )

            logger.info(f"Learned user preference: {preference_key} = {preference_value}")
            return True

        except Exception as e:
            logger.error(f"Failed to learn user preference: {e}")
            return False

    async def learn_about_user(
        self,
        fact: str,
        category: str = "general"
    ) -> bool:
        """
        Store a fact about the user in the knowledge graph.

        Args:
            fact: The fact to store (e.g., "User is a Python developer")
            category: Category of the fact (general, work, interests, etc.)

        Returns:
            True if fact was stored successfully
        """
        if not self._initialized:
            await self.initialize()

        try:
            fact_text = (
                f"Fact about user ({category}): {fact}. "
                f"This information was shared by the user during conversation."
            )

            await self._graphiti.add_episode(
                content=fact_text,
                name=f"user_fact_{hashlib.md5(fact.encode()).hexdigest()[:8]}",
                source_description=f"User fact: {category}",
                source_type="text",
                reference_time=datetime.now(timezone.utc),
                group_id=self.MEMORY_GROUP_ID
            )

            logger.info(f"Learned about user: {fact[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to learn about user: {e}")
            return False

    async def recall_relevant_context(
        self,
        query: str,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Recall relevant past conversations and user information.

        Args:
            query: The query to search for relevant context
            limit: Maximum number of results

        Returns:
            Dict with relevant conversations and user info
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Search knowledge graph for relevant content
            results = await self._graphiti.search(
                query=query,
                limit=limit
            )

            # Also search chat history
            conversations = self._chat_history.list_conversations(
                search=query,
                limit=limit
            )

            return {
                "graph_results": results,
                "related_conversations": [c.to_dict() for c in conversations],
                "user_profile": self._user_profile.to_dict()
            }

        except Exception as e:
            logger.error(f"Failed to recall context: {e}")
            return {
                "graph_results": {},
                "related_conversations": [],
                "user_profile": self._user_profile.to_dict()
            }

    async def get_conversation_context(
        self,
        conversation_id: str,
        include_related: bool = True
    ) -> Dict[str, Any]:
        """
        Get full context for a conversation including related past discussions.

        Args:
            conversation_id: The conversation to get context for
            include_related: Whether to include related past conversations

        Returns:
            Dict with conversation context and related information
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Get current conversation
            conversation = self._chat_history.get_conversation(conversation_id)
            if not conversation:
                return {"error": "Conversation not found"}

            messages = self._chat_history.get_messages(conversation_id)

            context = {
                "conversation": conversation.to_dict(),
                "messages": [m.to_dict() for m in messages],
                "user_profile": self._user_profile.to_dict()
            }

            # Find related past conversations
            if include_related and messages:
                # Use first user message as query
                user_messages = [m for m in messages if m.role == "user"]
                if user_messages:
                    related = await self.recall_relevant_context(
                        user_messages[0].content,
                        limit=3
                    )
                    context["related"] = related

            return context

        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return {"error": str(e)}

    def get_user_profile(self) -> UserProfile:
        """Get the current user profile."""
        return self._user_profile

    async def export_memory(self) -> Dict[str, Any]:
        """Export all conversation memory data."""
        if not self._initialized:
            await self.initialize()

        try:
            # Get all conversations
            conversations = self._chat_history.list_conversations(limit=1000)

            # Get stats
            stats = self._chat_history.get_stats()

            return {
                "user_profile": self._user_profile.to_dict(),
                "conversations_count": len(conversations),
                "stats": stats,
                "pending_sync": {
                    conv_id: len(msgs)
                    for conv_id, msgs in self._pending_messages.items()
                }
            }

        except Exception as e:
            logger.error(f"Failed to export memory: {e}")
            return {"error": str(e)}


# Singleton instance
_conversation_memory_instance: Optional[ConversationMemoryService] = None


def get_conversation_memory() -> ConversationMemoryService:
    """Get or create the singleton ConversationMemoryService instance."""
    global _conversation_memory_instance
    if _conversation_memory_instance is None:
        _conversation_memory_instance = ConversationMemoryService()
    return _conversation_memory_instance

