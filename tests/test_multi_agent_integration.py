"""Integration tests for the multi-agent system."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.messages import HumanMessage

from kg_agent.agent.multi import (
    get_multi_agent,
    invoke_multi_agent,
    create_initial_state,
    MultiAgentState,
)
from kg_agent.agent.multi.manager import DelegationDecision


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response factory."""

    def create_response(delegations):
        return type(
            "Result",
            (),
            {
                "output": DelegationDecision(
                    reasoning="Test delegation",
                    delegations=delegations,
                )
            },
        )()

    return create_response


class TestMultiAgentFlow:
    """Test complete multi-agent flows."""

    @pytest.mark.asyncio
    async def test_single_delegation_flow(self, mock_llm_response):
        """Test flow with single specialist delegation."""
        # Force recreate graph without checkpointer for testing
        from kg_agent.agent.multi.graph import get_multi_agent
        get_multi_agent(use_checkpointer=False, force_recreate=True)

        with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
            mock_manager.run = AsyncMock(
                return_value=mock_llm_response(
                    [{"target": "research", "task": "Search for Python"}]
                )
            )

            # Also mock the research agent
            with patch(
                "kg_agent.agent.multi.research_lead.research_agent"
            ) as mock_research:
                mock_research.run = AsyncMock(
                    return_value=type("Result", (), {"output": "Found Python tutorials"})()
                )

                result = await invoke_multi_agent(
                    "Search for Python tutorials",
                    session_id="test-session-1"
                )

                assert result.get("should_end") is True
                assert result.get("final_response") is not None

    @pytest.mark.asyncio
    async def test_multi_delegation_flow(self, mock_llm_response):
        """Test flow with multiple specialist delegations."""
        # Force recreate graph without checkpointer for testing
        from kg_agent.agent.multi.graph import get_multi_agent
        get_multi_agent(use_checkpointer=False, force_recreate=True)

        with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
            mock_manager.run = AsyncMock(
                return_value=mock_llm_response(
                    [
                        {"target": "memory", "task": "Remember name is Steve"},
                        {"target": "research", "task": "Search for Steve's projects"},
                    ]
                )
            )

            with patch(
                "kg_agent.agent.multi.memory_lead.memory_agent"
            ) as mock_memory:
                mock_memory.run = AsyncMock(
                    return_value=type(
                        "Result", (), {"output": "Remembered: name is Steve"}
                    )()
                )

                with patch(
                    "kg_agent.agent.multi.research_lead.research_agent"
                ) as mock_research:
                    mock_research.run = AsyncMock(
                        return_value=type(
                            "Result", (), {"output": "Found Steve's projects"}
                        )()
                    )

                    result = await invoke_multi_agent(
                        "Remember my name is Steve and find my projects",
                        session_id="test-session-2"
                    )

                    assert result.get("should_end") is True

    @pytest.mark.asyncio
    async def test_session_persistence(self):
        """Test that sessions are persisted correctly."""
        from kg_agent.agent.multi.session import get_session_manager

        sm = get_session_manager()
        session_id = sm.create_session(user_id="test-user")

        assert sm.get_session(session_id) is not None
        assert sm.get_session(session_id)["user_id"] == "test-user"

        sm.increment_message_count(session_id)
        assert sm.get_session(session_id)["message_count"] == 1


class TestRoutingLogic:
    """Test routing decision logic."""

    def test_route_to_specialist(self):
        """Test routing to specialist when delegations exist."""
        from kg_agent.agent.multi.graph import route_from_manager
        from kg_agent.agent.multi.state import create_delegation_request

        state: MultiAgentState = {
            "messages": [],
            "delegation_queue": [
                create_delegation_request(target="research", task="test")
            ],
            "should_end": False,
        }

        result = route_from_manager(state)
        assert result == "research"

    def test_route_to_synthesize(self):
        """Test routing to synthesize when queue empty."""
        from kg_agent.agent.multi.graph import route_from_manager

        state: MultiAgentState = {
            "messages": [],
            "delegation_queue": [],
            "should_end": False,
        }

        result = route_from_manager(state)
        assert result == "synthesize"

    def test_route_to_end(self):
        """Test routing to END when should_end is True."""
        from kg_agent.agent.multi.graph import route_from_manager
        from langgraph.graph import END

        state: MultiAgentState = {
            "messages": [],
            "should_end": True,
        }

        result = route_from_manager(state)
        assert result == END

    def test_route_after_specialist_more_delegations(self):
        """Test routing after specialist with remaining delegations."""
        from kg_agent.agent.multi.graph import route_after_specialist
        from kg_agent.agent.multi.state import create_delegation_request

        state: MultiAgentState = {
            "messages": [],
            "delegation_queue": [
                create_delegation_request(target="memory", task="test")
            ],
        }

        result = route_after_specialist(state)
        assert result == "memory"

    def test_route_after_specialist_empty_queue(self):
        """Test routing after specialist with empty queue."""
        from kg_agent.agent.multi.graph import route_after_specialist

        state: MultiAgentState = {
            "messages": [],
            "delegation_queue": [],
        }

        result = route_after_specialist(state)
        assert result == "synthesize"


class TestSessionManager:
    """Test session manager functionality."""

    def test_create_and_get_session(self):
        """Test session creation and retrieval."""
        from kg_agent.agent.multi.session import SessionManager

        sm = SessionManager()
        session_id = sm.create_session(user_id="test-user", metadata={"key": "value"})

        session = sm.get_session(session_id)
        assert session is not None
        assert session["user_id"] == "test-user"
        assert session["metadata"]["key"] == "value"
        assert session["message_count"] == 0

    def test_update_session(self):
        """Test session updates."""
        from kg_agent.agent.multi.session import SessionManager

        sm = SessionManager()
        session_id = sm.create_session()

        sm.update_session(session_id, custom_field="test")
        session = sm.get_session(session_id)
        assert session["custom_field"] == "test"

    def test_list_sessions(self):
        """Test listing sessions."""
        from kg_agent.agent.multi.session import SessionManager

        sm = SessionManager()
        sm.create_session(user_id="user1")
        sm.create_session(user_id="user1")
        sm.create_session(user_id="user2")

        all_sessions = sm.list_sessions()
        assert len(all_sessions) == 3

        user1_sessions = sm.list_sessions(user_id="user1")
        assert len(user1_sessions) == 2

    def test_delete_session(self):
        """Test session deletion."""
        from kg_agent.agent.multi.session import SessionManager

        sm = SessionManager()
        session_id = sm.create_session()

        assert sm.get_session(session_id) is not None
        assert sm.delete_session(session_id) is True
        assert sm.get_session(session_id) is None


class TestErrorHandling:
    """Test error handling in the multi-agent system."""

    def test_error_handling_decorator(self):
        """Test the error handling decorator creates valid wrapper."""
        from kg_agent.agent.multi.error_handling import with_error_handling

        @with_error_handling("test_agent")
        async def failing_node(state, config):
            raise ValueError("Test error")

        import asyncio
        import inspect

        # Verify it's still async
        assert inspect.iscoroutinefunction(failing_node)

    def test_exception_classes(self):
        """Test custom exception classes."""
        from kg_agent.agent.multi.error_handling import (
            MultiAgentError,
            DelegationError,
            SpecialistError,
            ConfigurationError,
            StateValidationError,
        )

        # Test inheritance
        assert issubclass(DelegationError, MultiAgentError)
        assert issubclass(SpecialistError, MultiAgentError)
        assert issubclass(ConfigurationError, MultiAgentError)
        assert issubclass(StateValidationError, MultiAgentError)

        # Test SpecialistError message
        err = SpecialistError("research", "test message")
        assert "research" in str(err)
        assert "test message" in str(err)

