"""End-to-end tests for the multi-agent system.

Tests complete workflows from user input to final response.
"""

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from kg_agent.agent.multi import (
    invoke_multi_agent,
    stream_multi_agent,
    create_initial_state,
)
from kg_agent.agent.multi.state import ThinkingStep


# === Fixtures ===


@pytest.fixture
def mock_rag_tools():
    """Create mock RAG tools that return predictable results."""
    mock = MagicMock()

    # Vector search mock
    mock.search_vectors = AsyncMock(
        return_value=[
            MagicMock(
                text="Python is a programming language", source="docs", score=0.95
            ),
            MagicMock(text="Python was created by Guido", source="wiki", score=0.85),
        ]
    )

    # Graph search mock
    mock.search_graph = AsyncMock(
        return_value=[
            MagicMock(text="Python: A versatile language", source="entity:Technology"),
        ]
    )

    # Stats mocks
    mock.get_graph_stats = AsyncMock(
        return_value=MagicMock(
            connected=True, total_nodes=100, total_edges=50, entity_types={}
        )
    )
    mock.get_vector_stats = AsyncMock(
        return_value=MagicMock(total_chunks=500, collection_name="kg_chunks")
    )

    # Entity creation mock
    mock.create_entity = AsyncMock(
        return_value=MagicMock(success=True, entity_id="ent_123", message="Created")
    )

    # Document mocks
    mock.list_documents = AsyncMock(return_value=[])
    mock.get_document_stats = AsyncMock(
        return_value=MagicMock(
            total_documents=10, by_status={"completed": 10}, by_source_type={}
        )
    )

    mock.initialize = AsyncMock(return_value=True)

    return mock


@pytest.fixture
def mock_memory():
    """Create mock conversation memory."""
    mock = MagicMock()
    mock.initialize = AsyncMock()
    mock.recall_relevant_context = AsyncMock(
        return_value={
            "related_conversations": [],
            "user_profile": {"preferences": {}},
        }
    )
    mock.learn_about_user = AsyncMock(return_value=True)
    mock.get_user_profile = MagicMock(
        return_value=MagicMock(
            name=None, interaction_count=0, preferences={}, topics_of_interest=[]
        )
    )
    return mock


@pytest.fixture
def mock_delegation_output():
    """Factory for creating mock delegation outputs."""

    def _create(
        reasoning: str,
        delegations: list[dict[str, str]],
        needs_clarification: bool = False,
        clarification_question: str | None = None,
    ):
        return MagicMock(
            output=MagicMock(
                reasoning=reasoning,
                delegations=delegations,
                needs_clarification=needs_clarification,
                clarification_question=clarification_question,
            )
        )

    return _create


# === Test Classes ===


class TestResearchWorkflow:
    """Test research-focused workflows."""

    @pytest.mark.asyncio
    async def test_simple_search_query(self, mock_rag_tools, mock_delegation_output):
        """Test a simple search query routes to research and returns results."""
        with patch(
            "kg_agent.agent.multi.research_lead.get_rag_tools",
            return_value=mock_rag_tools,
        ):
            with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
                # Manager decides to delegate to research
                mock_manager.run = AsyncMock(
                    return_value=mock_delegation_output(
                        reasoning="User wants to search",
                        delegations=[{"target": "research", "task": "Search for Python"}],
                    )
                )

                with patch(
                    "kg_agent.agent.multi.research_lead.research_agent"
                ) as mock_research:
                    mock_research.run = AsyncMock(
                        return_value=MagicMock(
                            output="Found: Python is a programming language"
                        )
                    )

                    result = await invoke_multi_agent(
                        "What is Python?",
                        session_id="test-research-workflow",
                    )

                    # Verify result structure
                    assert result["should_end"] is True
                    assert result["final_response"] is not None
                    assert "research" in result["execution_path"]
                    assert len(result["thinking_steps"]) > 0

    @pytest.mark.asyncio
    async def test_database_stats_query(self, mock_rag_tools, mock_delegation_output):
        """Test stats query routes correctly."""
        with patch(
            "kg_agent.agent.multi.research_lead.get_rag_tools",
            return_value=mock_rag_tools,
        ):
            with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
                mock_manager.run = AsyncMock(
                    return_value=mock_delegation_output(
                        reasoning="User wants stats",
                        delegations=[
                            {"target": "research", "task": "Get database statistics"}
                        ],
                    )
                )

                with patch(
                    "kg_agent.agent.multi.research_lead.research_agent"
                ) as mock_research:
                    mock_research.run = AsyncMock(
                        return_value=MagicMock(
                            output="Database has 100 nodes and 50 edges"
                        )
                    )

                    result = await invoke_multi_agent(
                        "Show me database stats",
                        session_id="test-stats-workflow",
                    )

                    assert result["should_end"] is True
                    assert "research" in result["execution_path"]


class TestMemoryWorkflow:
    """Test memory-focused workflows."""

    @pytest.mark.asyncio
    async def test_remember_user_fact(self, mock_memory, mock_delegation_output):
        """Test storing a fact about the user."""
        with patch(
            "kg_agent.agent.multi.memory_lead.get_conversation_memory",
            return_value=mock_memory,
        ):
            with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
                mock_manager.run = AsyncMock(
                    return_value=mock_delegation_output(
                        reasoning="User wants to store info",
                        delegations=[
                            {"target": "memory", "task": "Remember user's name is Steve"}
                        ],
                    )
                )

                with patch(
                    "kg_agent.agent.multi.memory_lead.memory_agent"
                ) as mock_mem_agent:
                    mock_mem_agent.run = AsyncMock(
                        return_value=MagicMock(output="Remembered: Your name is Steve")
                    )

                    result = await invoke_multi_agent(
                        "My name is Steve",
                        session_id="test-memory-workflow",
                    )

                    assert "memory" in result["execution_path"]

    @pytest.mark.asyncio
    async def test_recall_past_conversation(self, mock_memory, mock_delegation_output):
        """Test recalling past conversations."""
        with patch(
            "kg_agent.agent.multi.memory_lead.get_conversation_memory",
            return_value=mock_memory,
        ):
            with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
                mock_manager.run = AsyncMock(
                    return_value=mock_delegation_output(
                        reasoning="User wants context",
                        delegations=[
                            {"target": "memory", "task": "Recall conversation context"}
                        ],
                    )
                )

                with patch(
                    "kg_agent.agent.multi.memory_lead.memory_agent"
                ) as mock_mem_agent:
                    mock_mem_agent.run = AsyncMock(
                        return_value=MagicMock(output="Previous context recalled")
                    )

                    result = await invoke_multi_agent(
                        "What were we talking about?",
                        session_id="test-recall-workflow",
                    )

                    assert result["should_end"] is True


class TestKnowledgeWorkflow:
    """Test knowledge graph workflows."""

    @pytest.mark.asyncio
    async def test_create_entity(self, mock_rag_tools, mock_delegation_output):
        """Test creating an entity in the knowledge graph."""
        with patch(
            "kg_agent.agent.multi.knowledge_lead.get_rag_tools",
            return_value=mock_rag_tools,
        ):
            with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
                mock_manager.run = AsyncMock(
                    return_value=mock_delegation_output(
                        reasoning="User wants to create entity",
                        delegations=[
                            {"target": "knowledge", "task": "Create Python entity"}
                        ],
                    )
                )

                with patch(
                    "kg_agent.agent.multi.knowledge_lead.knowledge_agent"
                ) as mock_kg:
                    mock_kg.run = AsyncMock(
                        return_value=MagicMock(output="Created: Python (Technology)")
                    )

                    result = await invoke_multi_agent(
                        "Create an entity for Python as a Technology",
                        session_id="test-knowledge-workflow",
                    )

                    assert "knowledge" in result["execution_path"]


class TestDocumentWorkflow:
    """Test document management workflows."""

    @pytest.mark.asyncio
    async def test_list_documents(self, mock_rag_tools, mock_delegation_output):
        """Test listing documents."""
        with patch(
            "kg_agent.agent.multi.document_lead.get_rag_tools",
            return_value=mock_rag_tools,
        ):
            with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
                mock_manager.run = AsyncMock(
                    return_value=mock_delegation_output(
                        reasoning="User wants document list",
                        delegations=[{"target": "documents", "task": "List documents"}],
                    )
                )

                with patch(
                    "kg_agent.agent.multi.document_lead.document_agent"
                ) as mock_doc:
                    mock_doc.run = AsyncMock(
                        return_value=MagicMock(output="Found 10 documents")
                    )

                    result = await invoke_multi_agent(
                        "List all documents",
                        session_id="test-documents-workflow",
                    )

                    assert "documents" in result["execution_path"]

    @pytest.mark.asyncio
    async def test_delete_document_confirmation(
        self, mock_rag_tools, mock_delegation_output
    ):
        """Test that delete operations route to documents specialist."""
        with patch(
            "kg_agent.agent.multi.document_lead.get_rag_tools",
            return_value=mock_rag_tools,
        ):
            with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
                mock_manager.run = AsyncMock(
                    return_value=mock_delegation_output(
                        reasoning="User wants to delete",
                        delegations=[
                            {"target": "documents", "task": "Delete document doc_123"}
                        ],
                    )
                )

                with patch(
                    "kg_agent.agent.multi.document_lead.document_agent"
                ) as mock_doc:
                    mock_doc.run = AsyncMock(
                        return_value=MagicMock(output="Deleted document doc_123")
                    )

                    result = await invoke_multi_agent(
                        "Delete document doc_123",
                        session_id="test-delete-workflow",
                    )

                    assert result["should_end"] is True


class TestMultiDelegationWorkflow:
    """Test workflows that involve multiple specialists."""

    @pytest.mark.asyncio
    async def test_search_and_remember(
        self, mock_rag_tools, mock_memory, mock_delegation_output
    ):
        """Test query that needs both research and memory."""
        with patch(
            "kg_agent.agent.multi.research_lead.get_rag_tools",
            return_value=mock_rag_tools,
        ):
            with patch(
                "kg_agent.agent.multi.memory_lead.get_conversation_memory",
                return_value=mock_memory,
            ):
                with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
                    mock_manager.run = AsyncMock(
                        return_value=mock_delegation_output(
                            reasoning="User wants to search and remember",
                            delegations=[
                                {"target": "research", "task": "Search Python"},
                                {"target": "memory", "task": "Remember user interest"},
                            ],
                        )
                    )

                    with patch(
                        "kg_agent.agent.multi.research_lead.research_agent"
                    ) as mock_res:
                        mock_res.run = AsyncMock(
                            return_value=MagicMock(output="Found Python info")
                        )

                        with patch(
                            "kg_agent.agent.multi.memory_lead.memory_agent"
                        ) as mock_mem:
                            mock_mem.run = AsyncMock(
                                return_value=MagicMock(output="Remembered interest")
                            )

                            result = await invoke_multi_agent(
                                "Search for Python and remember I'm interested in it",
                                session_id="test-multi-delegation",
                            )

                            # Both specialists should have been called
                            assert "research" in result["execution_path"]
                            assert "memory" in result["execution_path"]


class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_specialist_failure_recovery(self, mock_delegation_output):
        """Test that system recovers gracefully from specialist errors."""
        with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
            mock_manager.run = AsyncMock(
                return_value=mock_delegation_output(
                    reasoning="Delegating",
                    delegations=[{"target": "research", "task": "Search"}],
                )
            )

            with patch(
                "kg_agent.agent.multi.research_lead.research_agent"
            ) as mock_res:
                mock_res.run = AsyncMock(side_effect=Exception("Network error"))

                result = await invoke_multi_agent(
                    "Search for something",
                    session_id="test-error-recovery",
                )

                # Should complete despite error
                assert result["should_end"] is True

                # Error should be captured gracefully - check for error in:
                # 1. last_error field, OR
                # 2. Error message in research_result, OR
                # 3. Error mentioned in final_response
                has_error = (
                    result.get("last_error") is not None
                    or "error" in str(result.get("research_result", "")).lower()
                    or "failed" in str(result.get("research_result", "")).lower()
                    or "error" in str(result.get("final_response", "")).lower()
                )
                assert has_error, f"Expected error to be captured. Result: {result}"

    @pytest.mark.asyncio
    async def test_empty_delegation_handling(self, mock_delegation_output):
        """Test handling when manager doesn't delegate to anyone."""
        with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
            mock_manager.run = AsyncMock(
                return_value=mock_delegation_output(
                    reasoning="Can't determine",
                    delegations=[],
                    needs_clarification=True,
                    clarification_question="Could you be more specific?",
                )
            )

            result = await invoke_multi_agent(
                "Do something",
                session_id="test-empty-delegation",
            )

            assert result["should_end"] is True
            # Should have a response asking for clarification
            assert result.get("final_response") is not None


class TestStreaming:
    """Test streaming functionality."""

    @pytest.mark.asyncio
    async def test_stream_yields_events(self, mock_delegation_output):
        """Test that streaming yields state updates."""
        events = []

        with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
            mock_manager.run = AsyncMock(
                return_value=mock_delegation_output(
                    reasoning="Test",
                    delegations=[{"target": "research", "task": "Test"}],
                )
            )

            with patch(
                "kg_agent.agent.multi.research_lead.research_agent"
            ) as mock_res:
                mock_res.run = AsyncMock(return_value=MagicMock(output="Done"))

                async for node, state in stream_multi_agent(
                    "Test query",
                    session_id="test-streaming",
                ):
                    events.append((node, state))

        # Should have multiple events
        assert len(events) > 0
        # Should include manager
        nodes = [e[0] for e in events]
        assert "manager" in nodes


class TestSessionPersistence:
    """Test session and checkpoint functionality."""

    @pytest.mark.asyncio
    async def test_session_continuity(self):
        """Test that sessions maintain state."""
        from kg_agent.agent.multi.session import get_session_manager

        sm = get_session_manager()
        session_id = sm.create_session(user_id="test")

        # First message
        sm.increment_message_count(session_id)

        # Second message
        sm.increment_message_count(session_id)

        session = sm.get_session(session_id)
        assert session is not None
        assert session["message_count"] == 2

        # Cleanup
        sm.delete_session(session_id)

    @pytest.mark.asyncio
    async def test_session_metadata(self):
        """Test session metadata storage."""
        from kg_agent.agent.multi.session import get_session_manager

        sm = get_session_manager()
        session_id = sm.create_session(user_id="test", metadata={"key": "value"})

        session = sm.get_session(session_id)
        assert session is not None
        assert session["metadata"]["key"] == "value"

        # Update metadata
        sm.update_session(session_id, metadata={"key": "updated"})
        session = sm.get_session(session_id)
        assert session["metadata"]["key"] == "updated"

        # Cleanup
        sm.delete_session(session_id)


class TestInitialState:
    """Test initial state creation."""

    def test_create_initial_state_minimal(self):
        """Test creating initial state with minimal args."""
        state = create_initial_state("Hello")

        assert state["messages"] is not None
        assert state["current_agent"] == "manager"
        assert state["thinking_steps"] == []
        assert state["should_end"] is False

    def test_create_initial_state_full(self):
        """Test creating initial state with all args."""
        state = create_initial_state(
            user_message="Hello",
            user_id="user-123",
            session_id="session-456",
        )

        assert state["user_id"] == "user-123"
        assert state["session_id"] == "session-456"

