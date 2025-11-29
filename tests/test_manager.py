"""Unit tests for the Manager agent."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.messages import HumanMessage

from kg_agent.agent.multi.state import create_initial_state
from kg_agent.agent.multi.manager import (
    manager_node,
    synthesize_node,
    DelegationDecision,
    get_conversation_context,
    state_to_emittable_dict,
)


@pytest.fixture
def mock_config():
    """Create a mock RunnableConfig."""
    return {}


@pytest.fixture
def sample_state():
    """Create a sample initial state."""
    return create_initial_state("Search for Python tutorials")


@pytest.fixture
def mock_emit_state():
    """Mock copilotkit_emit_state for testing."""
    with patch("kg_agent.agent.multi.manager.copilotkit_emit_state", new_callable=AsyncMock) as mock:
        yield mock


class TestHelperFunctions:
    """Test helper functions."""

    def test_state_to_emittable_dict(self, sample_state):
        """Test state conversion for CopilotKit emission."""
        result = state_to_emittable_dict(sample_state)

        assert "current_agent" in result
        assert "thinking_steps" in result
        assert "delegation_queue" in result
        assert "should_end" in result

    def test_state_to_emittable_dict_with_overrides(self, sample_state):
        """Test state conversion with overrides."""
        result = state_to_emittable_dict(
            sample_state,
            current_agent="research",
            should_end=True,
        )

        assert result["current_agent"] == "research"
        assert result["should_end"] is True

    def test_get_conversation_context_empty(self, sample_state):
        """Test context extraction with single message."""
        context = get_conversation_context(sample_state)
        assert "User:" in context
        assert "Python tutorials" in context

    def test_get_conversation_context_no_messages(self):
        """Test context extraction with no messages."""
        state = {"messages": []}
        context = get_conversation_context(state)
        assert context == "No previous context"


class TestManagerIntentClassification:
    """Test Manager's intent classification."""

    @pytest.mark.asyncio
    async def test_research_intent(self, sample_state, mock_config, mock_emit_state):
        """Manager should delegate search queries to Research."""
        # Mock the LLM response
        mock_result = MagicMock()
        mock_result.output = DelegationDecision(
            reasoning="User wants to search for information",
            delegations=[{"target": "research", "task": "Search for Python tutorials"}],
        )

        with patch("kg_agent.agent.multi.manager.manager_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            result = await manager_node(sample_state, mock_config)

            assert len(result["delegation_queue"]) == 1
            assert result["delegation_queue"][0]["target"] == "research"
            assert result["current_agent"] == "research"

    @pytest.mark.asyncio
    async def test_multiple_delegations(self, mock_config, mock_emit_state):
        """Manager should queue multiple specialists when needed."""
        state = create_initial_state("Remember my name is Steve and search for my projects")

        mock_result = MagicMock()
        mock_result.output = DelegationDecision(
            reasoning="User wants to store info and search",
            delegations=[
                {"target": "memory", "task": "Remember user's name is Steve"},
                {"target": "research", "task": "Search for Steve's projects"},
            ],
        )

        with patch("kg_agent.agent.multi.manager.manager_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            result = await manager_node(state, mock_config)

            assert len(result["delegation_queue"]) == 2
            assert result["delegation_queue"][0]["target"] == "memory"
            assert result["delegation_queue"][1]["target"] == "research"
            # First delegation should set current_agent
            assert result["current_agent"] == "memory"

    @pytest.mark.asyncio
    async def test_clarification_needed(self, mock_config, mock_emit_state):
        """Manager should ask for clarification when needed."""
        state = create_initial_state("Do the thing")

        mock_result = MagicMock()
        mock_result.output = DelegationDecision(
            reasoning="Request is too vague",
            delegations=[],
            needs_clarification=True,
            clarification_question="What would you like me to do?",
        )

        with patch("kg_agent.agent.multi.manager.manager_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            result = await manager_node(state, mock_config)

            assert result["should_end"] is True
            assert "clarify" in result["final_response"].lower() or "do" in result["final_response"].lower()

    @pytest.mark.asyncio
    async def test_no_user_message(self, mock_config, mock_emit_state):
        """Manager should handle missing user message gracefully."""
        state = {"messages": [], "thinking_steps": [], "execution_path": []}

        result = await manager_node(state, mock_config)

        assert result["should_end"] is True
        assert "didn't receive" in result["final_response"].lower()

    @pytest.mark.asyncio
    async def test_llm_error_fallback(self, sample_state, mock_config, mock_emit_state):
        """Manager should fallback to research on LLM error."""
        with patch("kg_agent.agent.multi.manager.manager_agent") as mock_agent:
            mock_agent.run = AsyncMock(side_effect=Exception("LLM unavailable"))

            result = await manager_node(sample_state, mock_config)

            # Should fallback to research
            assert len(result["delegation_queue"]) == 1
            assert result["delegation_queue"][0]["target"] == "research"


class TestSynthesizeNode:
    """Test the synthesis node."""

    @pytest.mark.asyncio
    async def test_single_result_synthesis(self, mock_config, mock_emit_state):
        """Single specialist result should be returned directly."""
        state = {
            "messages": [],
            "thinking_steps": [],
            "research_result": "Found 5 Python tutorials",
            "execution_path": ["manager", "research"],
        }

        result = await synthesize_node(state, mock_config)

        assert result["final_response"] == "Found 5 Python tutorials"
        assert result["should_end"] is True
        assert "synthesize" in result["execution_path"]

    @pytest.mark.asyncio
    async def test_multiple_results_synthesis(self, mock_config, mock_emit_state):
        """Multiple results should be combined with headers."""
        state = {
            "messages": [],
            "thinking_steps": [],
            "research_result": "Found tutorials",
            "memory_result": "User prefers video content",
            "execution_path": ["manager", "research", "memory"],
        }

        result = await synthesize_node(state, mock_config)

        assert "**Research:**" in result["final_response"]
        assert "**Memory:**" in result["final_response"]
        assert "Found tutorials" in result["final_response"]
        assert "User prefers video content" in result["final_response"]

    @pytest.mark.asyncio
    async def test_no_results_synthesis(self, mock_config, mock_emit_state):
        """Should provide helpful message when no results."""
        state = {
            "messages": [],
            "thinking_steps": [],
            "execution_path": ["manager"],
        }

        result = await synthesize_node(state, mock_config)

        assert "rephras" in result["final_response"].lower()
        assert result["should_end"] is True

    @pytest.mark.asyncio
    async def test_all_specialist_results(self, mock_config, mock_emit_state):
        """Should combine all four specialist results."""
        state = {
            "messages": [],
            "thinking_steps": [],
            "research_result": "Research data",
            "memory_result": "Memory data",
            "knowledge_result": "Knowledge data",
            "document_result": "Document data",
            "execution_path": [],
        }

        result = await synthesize_node(state, mock_config)

        assert "**Research:**" in result["final_response"]
        assert "**Memory:**" in result["final_response"]
        assert "**Knowledge:**" in result["final_response"]
        assert "**Documents:**" in result["final_response"]


class TestThinkingSteps:
    """Test thinking step emission."""

    @pytest.mark.asyncio
    async def test_manager_emits_thinking_steps(self, sample_state, mock_config, mock_emit_state):
        """Manager should emit thinking steps during execution."""
        mock_result = MagicMock()
        mock_result.output = DelegationDecision(
            reasoning="User wants to search",
            delegations=[{"target": "research", "task": "Search for Python tutorials"}],
        )

        with patch("kg_agent.agent.multi.manager.manager_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            result = await manager_node(sample_state, mock_config)

            # Should have thinking steps
            assert len(result["thinking_steps"]) >= 2

            # Check step structure
            first_step = result["thinking_steps"][0]
            assert first_step["agent"] == "manager"
            assert first_step["status"] == "thinking"
            assert "timestamp" in first_step

            # Should have emitted state at least twice (analyzing + delegating)
            assert mock_emit_state.call_count >= 2

    @pytest.mark.asyncio
    async def test_synthesize_emits_thinking_steps(self, mock_config, mock_emit_state):
        """Synthesize should emit thinking steps."""
        state = {
            "messages": [],
            "thinking_steps": [],
            "research_result": "Found results",
            "execution_path": [],
        }

        result = await synthesize_node(state, mock_config)

        # Should have added synthesizing steps
        assert len(result["thinking_steps"]) >= 2

        # Check for "complete" status
        final_step = result["thinking_steps"][-1]
        assert final_step["status"] == "complete"

