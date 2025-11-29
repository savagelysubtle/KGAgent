# Phase 7: Testing & Polish - Integration Tests & Documentation

> **Parent Plan:** [multi_agent.plan.md](./multi_agent.plan.md)
> **Status:** Not Started
> **Estimated Effort:** 3-4 hours
> **Dependencies:** Phases 1-6 complete

---

## 🎯 Objectives

1. Write comprehensive integration tests
2. Add end-to-end tests for common workflows
3. Performance optimization
4. Error handling improvements
5. Documentation updates
6. Cleanup and refactoring

---

## 📋 Prerequisites

- [ ] All phases 1-6 complete
- [ ] System can run end-to-end
- [ ] Manual testing shows basic functionality works

---

## 🧪 Task 1: Integration Test Suite

### File: `tests/test_multi_agent_e2e.py`

```python
"""End-to-end tests for the multi-agent system.

Tests complete workflows from user input to final response.
"""

import pytest
import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from kg_agent.agent.multi import (
    invoke_multi_agent,
    stream_multi_agent,
    get_multi_agent,
    create_initial_state,
    MultiAgentState,
    ThinkingStep,
)


# === Fixtures ===

@pytest.fixture
def mock_rag_tools():
    """Create mock RAG tools that return predictable results."""
    mock = MagicMock()

    # Vector search mock
    mock.search_vectors = AsyncMock(return_value=[
        MagicMock(text="Python is a programming language", source="docs", score=0.95),
        MagicMock(text="Python was created by Guido", source="wiki", score=0.85),
    ])

    # Graph search mock
    mock.search_graph = AsyncMock(return_value=[
        MagicMock(text="Python: A versatile language", source="entity:Technology"),
    ])

    # Stats mocks
    mock.get_graph_stats = AsyncMock(return_value=MagicMock(
        connected=True, total_nodes=100, total_edges=50, entity_types={}
    ))
    mock.get_vector_stats = AsyncMock(return_value=MagicMock(
        total_chunks=500, collection_name="kg_chunks"
    ))

    # Entity creation mock
    mock.create_entity = AsyncMock(return_value=MagicMock(
        success=True, entity_id="ent_123", message="Created"
    ))

    # Document mocks
    mock.list_documents = AsyncMock(return_value=[])
    mock.get_document_stats = AsyncMock(return_value=MagicMock(
        total_documents=10, by_status={"completed": 10}, by_source_type={}
    ))

    mock.initialize = AsyncMock(return_value=True)

    return mock


@pytest.fixture
def mock_memory():
    """Create mock conversation memory."""
    mock = MagicMock()
    mock.initialize = AsyncMock()
    mock.recall_relevant_context = AsyncMock(return_value={
        "related_conversations": [],
        "user_profile": {"preferences": {}},
    })
    mock.learn_about_user = AsyncMock(return_value=True)
    mock.get_user_profile = MagicMock(return_value=MagicMock(
        name=None, interaction_count=0, preferences={}, topics_of_interest=[]
    ))
    return mock


# === Test Classes ===

class TestResearchWorkflow:
    """Test research-focused workflows."""

    @pytest.mark.asyncio
    async def test_simple_search_query(self, mock_rag_tools):
        """Test a simple search query routes to research and returns results."""
        with patch("kg_agent.agent.multi.research_lead.get_rag_tools", return_value=mock_rag_tools):
            with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
                # Manager decides to delegate to research
                mock_manager.run = AsyncMock(return_value=MagicMock(
                    output=MagicMock(
                        reasoning="User wants to search",
                        delegations=[{"target": "research", "task": "Search for Python"}],
                        needs_clarification=False,
                    )
                ))

                with patch("kg_agent.agent.multi.research_lead.research_agent") as mock_research:
                    mock_research.run = AsyncMock(return_value=MagicMock(
                        output="Found: Python is a programming language"
                    ))

                    result = await invoke_multi_agent("What is Python?")

                    # Verify result structure
                    assert result["should_end"] == True
                    assert result["final_response"] is not None
                    assert "research" in result["execution_path"]
                    assert len(result["thinking_steps"]) > 0

    @pytest.mark.asyncio
    async def test_database_stats_query(self, mock_rag_tools):
        """Test stats query routes correctly."""
        # Similar pattern - test that "show me database stats"
        # triggers research with get_database_statistics
        pass


class TestMemoryWorkflow:
    """Test memory-focused workflows."""

    @pytest.mark.asyncio
    async def test_remember_user_fact(self, mock_memory):
        """Test storing a fact about the user."""
        with patch("kg_agent.agent.multi.memory_lead.get_conversation_memory", return_value=mock_memory):
            with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
                mock_manager.run = AsyncMock(return_value=MagicMock(
                    output=MagicMock(
                        reasoning="User wants to store info",
                        delegations=[{"target": "memory", "task": "Remember user's name is Steve"}],
                        needs_clarification=False,
                    )
                ))

                with patch("kg_agent.agent.multi.memory_lead.memory_agent") as mock_mem_agent:
                    mock_mem_agent.run = AsyncMock(return_value=MagicMock(
                        output="Remembered: Your name is Steve"
                    ))

                    result = await invoke_multi_agent("My name is Steve")

                    assert "memory" in result["execution_path"]
                    assert "Steve" in result.get("final_response", "")

    @pytest.mark.asyncio
    async def test_recall_past_conversation(self, mock_memory):
        """Test recalling past conversations."""
        pass


class TestKnowledgeWorkflow:
    """Test knowledge graph workflows."""

    @pytest.mark.asyncio
    async def test_create_entity(self, mock_rag_tools):
        """Test creating an entity in the knowledge graph."""
        with patch("kg_agent.agent.multi.knowledge_lead.get_rag_tools", return_value=mock_rag_tools):
            with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
                mock_manager.run = AsyncMock(return_value=MagicMock(
                    output=MagicMock(
                        reasoning="User wants to create entity",
                        delegations=[{"target": "knowledge", "task": "Create Python entity"}],
                        needs_clarification=False,
                    )
                ))

                with patch("kg_agent.agent.multi.knowledge_lead.knowledge_agent") as mock_kg:
                    mock_kg.run = AsyncMock(return_value=MagicMock(
                        output="Created: Python (Technology)"
                    ))

                    result = await invoke_multi_agent(
                        "Create an entity for Python as a Technology"
                    )

                    assert "knowledge" in result["execution_path"]


class TestDocumentWorkflow:
    """Test document management workflows."""

    @pytest.mark.asyncio
    async def test_list_documents(self, mock_rag_tools):
        """Test listing documents."""
        pass

    @pytest.mark.asyncio
    async def test_delete_document_confirmation(self, mock_rag_tools):
        """Test that delete operations require confirmation."""
        pass


class TestMultiDelegationWorkflow:
    """Test workflows that involve multiple specialists."""

    @pytest.mark.asyncio
    async def test_search_and_remember(self, mock_rag_tools, mock_memory):
        """Test query that needs both research and memory."""
        with patch("kg_agent.agent.multi.research_lead.get_rag_tools", return_value=mock_rag_tools):
            with patch("kg_agent.agent.multi.memory_lead.get_conversation_memory", return_value=mock_memory):
                with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
                    mock_manager.run = AsyncMock(return_value=MagicMock(
                        output=MagicMock(
                            reasoning="User wants to search and remember",
                            delegations=[
                                {"target": "research", "task": "Search Python"},
                                {"target": "memory", "task": "Remember user interest"},
                            ],
                            needs_clarification=False,
                        )
                    ))

                    with patch("kg_agent.agent.multi.research_lead.research_agent") as mock_res:
                        mock_res.run = AsyncMock(return_value=MagicMock(output="Found Python info"))

                        with patch("kg_agent.agent.multi.memory_lead.memory_agent") as mock_mem:
                            mock_mem.run = AsyncMock(return_value=MagicMock(output="Remembered interest"))

                            result = await invoke_multi_agent(
                                "Search for Python and remember I'm interested in it"
                            )

                            # Both specialists should have been called
                            assert "research" in result["execution_path"]
                            assert "memory" in result["execution_path"]


class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_specialist_failure_recovery(self):
        """Test that system recovers gracefully from specialist errors."""
        with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
            mock_manager.run = AsyncMock(return_value=MagicMock(
                output=MagicMock(
                    reasoning="Delegating",
                    delegations=[{"target": "research", "task": "Search"}],
                    needs_clarification=False,
                )
            ))

            with patch("kg_agent.agent.multi.research_lead.research_agent") as mock_res:
                mock_res.run = AsyncMock(side_effect=Exception("Network error"))

                result = await invoke_multi_agent("Search for something")

                # Should complete despite error
                assert result["should_end"] == True
                # Error should be captured
                assert result.get("last_error") or "error" in result.get("research_result", "").lower()

    @pytest.mark.asyncio
    async def test_empty_delegation_handling(self):
        """Test handling when manager doesn't delegate to anyone."""
        with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
            mock_manager.run = AsyncMock(return_value=MagicMock(
                output=MagicMock(
                    reasoning="Can't determine",
                    delegations=[],
                    needs_clarification=True,
                    clarification_question="Could you be more specific?",
                )
            ))

            result = await invoke_multi_agent("Do something")

            assert result["should_end"] == True
            assert "specific" in result.get("final_response", "").lower()


class TestStreaming:
    """Test streaming functionality."""

    @pytest.mark.asyncio
    async def test_stream_yields_events(self):
        """Test that streaming yields state updates."""
        events = []

        with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
            mock_manager.run = AsyncMock(return_value=MagicMock(
                output=MagicMock(
                    reasoning="Test",
                    delegations=[{"target": "research", "task": "Test"}],
                    needs_clarification=False,
                )
            ))

            with patch("kg_agent.agent.multi.research_lead.research_agent") as mock_res:
                mock_res.run = AsyncMock(return_value=MagicMock(output="Done"))

                async for node, state in stream_multi_agent("Test query"):
                    events.append((node, state))

        # Should have multiple events
        assert len(events) > 0
        # Should include manager and research
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
        assert session["message_count"] == 2
```

---

## 🧪 Task 2: API Integration Tests

### File: `tests/test_multi_agent_api_integration.py`

```python
"""API integration tests for multi-agent endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from kg_agent.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestChatEndpointIntegration:
    """Integration tests for chat endpoint."""

    def test_chat_returns_thinking_steps(self, client):
        """Verify thinking steps are included in response."""
        with patch("kg_agent.api.routes.multi_agent.invoke_multi_agent") as mock:
            mock.return_value = {
                "final_response": "Test response",
                "thinking_steps": [
                    {"agent": "manager", "thought": "Analyzing", "status": "thinking", "timestamp": "2025-01-01"},
                    {"agent": "research", "thought": "Searching", "status": "executing", "timestamp": "2025-01-01"},
                ],
                "execution_path": ["manager", "research"],
                "total_llm_calls": 2,
                "should_end": True,
            }

            response = client.post(
                "/api/v1/multi-agent/chat",
                json={"message": "Search for Python"}
            )

            data = response.json()
            assert len(data["thinking_steps"]) == 2
            assert data["thinking_steps"][0]["agent"] == "manager"

    def test_chat_streaming_format(self, client):
        """Test streaming response format."""
        # Test SSE format
        pass


class TestCopilotKitEndpoint:
    """Test CopilotKit integration endpoint."""

    def test_copilotkit_info(self, client):
        """Test CopilotKit info endpoint."""
        response = client.get("/copilotkit/info")

        assert response.status_code == 200
        data = response.json()
        assert "available" in data
```

---

## 🔧 Task 3: Performance Optimization

### Checklist

- [ ] **LLM Call Optimization**
  - Batch multiple tool calls where possible
  - Cache common queries
  - Use smaller prompts

- [ ] **State Management**
  - Only emit state changes (not full state)
  - Limit thinking_steps history
  - Efficient serialization

- [ ] **Async Optimization**
  - Concurrent specialist execution where possible
  - Non-blocking I/O throughout
  - Connection pooling

### File: `src/kg_agent/agent/multi/optimization.py`

```python
"""Performance optimizations for the multi-agent system."""

from functools import lru_cache
from typing import Any, Dict, Optional
import hashlib
import json

from ...core.logging import logger


class QueryCache:
    """Simple cache for common queries."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds

    def _hash_query(self, query: str, params: Optional[dict] = None) -> str:
        """Create cache key from query and params."""
        content = json.dumps({"q": query, "p": params or {}}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, query: str, params: Optional[dict] = None) -> Optional[Any]:
        """Get cached result if valid."""
        import time

        key = self._hash_query(query, params)
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                logger.debug(f"Cache hit for query: {query[:50]}")
                return value
            else:
                del self._cache[key]
        return None

    def set(self, query: str, value: Any, params: Optional[dict] = None):
        """Cache a result."""
        import time

        # Evict oldest if at capacity
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        key = self._hash_query(query, params)
        self._cache[key] = (value, time.time())


# Global cache instance
_query_cache = QueryCache()


def get_query_cache() -> QueryCache:
    """Get the global query cache."""
    return _query_cache
```

---

## 📚 Task 4: Documentation Updates

### File: `docs/multi-agent-architecture.md`

```markdown
# Multi-Agent System Architecture

## Overview

The KGAgent multi-agent system uses a hierarchical architecture with:

- **Manager Agent**: Orchestrates all requests, delegates to specialists
- **Research Lead**: Searches the knowledge base
- **Memory Lead**: Manages user context and history
- **Knowledge Lead**: Creates/manages knowledge graph entities
- **Document Lead**: Manages document lifecycle

## Quick Start

```python
from kg_agent.agent.multi import invoke_multi_agent

# Simple query
result = await invoke_multi_agent("Search for Python tutorials")
print(result["final_response"])

# With session for continuity
result = await invoke_multi_agent(
    "Remember my name is Steve",
    session_id="my-session-123"
)
```

## Architecture Diagram

[Include ASCII diagram from plan]

## API Endpoints

- `POST /api/v1/multi-agent/chat` - Chat with the multi-agent system
- `POST /api/v1/multi-agent/chat/stream` - Streaming chat
- `GET /api/v1/multi-agent/status` - System status

## Configuration

[Environment variables, settings, etc.]

## Troubleshooting

[Common issues and solutions]
```

### Update Main README

Add section about multi-agent system to project README.

---

## 🧹 Task 5: Code Cleanup

### Checklist

- [ ] Remove debug print statements
- [ ] Ensure consistent logging levels
- [ ] Remove unused imports
- [ ] Add missing type hints
- [ ] Run formatter (`ruff format`)
- [ ] Run linter (`ruff check --fix`)
- [ ] Run type checker (`ty`)

### Script: `scripts/lint_multi_agent.py`

```python
"""Run linting and formatting on multi-agent module."""

import subprocess
import sys

def main():
    paths = [
        "src/kg_agent/agent/multi/",
        "src/kg_agent/api/routes/multi_agent.py",
        "src/kg_agent/api/routes/copilotkit.py",
        "tests/test_multi_agent*.py",
    ]

    print("🔧 Formatting...")
    subprocess.run(["ruff", "format"] + paths)

    print("\n🔍 Linting...")
    subprocess.run(["ruff", "check", "--fix"] + paths)

    print("\n📝 Type checking...")
    subprocess.run(["ty", "check"] + paths)

    print("\n✅ Done!")

if __name__ == "__main__":
    main()
```

---

## ✅ Phase 7 Definition of Done

### Tests
- [ ] Unit tests for each specialist agent
- [ ] Integration tests for multi-delegation workflows
- [ ] API endpoint tests
- [ ] Error handling tests
- [ ] Streaming tests
- [ ] All tests pass: `pytest tests/test_multi_agent*.py -v`

### Performance
- [ ] Query cache implemented
- [ ] State emission optimized
- [ ] No blocking operations

### Documentation
- [ ] Architecture documentation complete
- [ ] API documentation with examples
- [ ] README updated
- [ ] Inline code comments

### Code Quality
- [ ] No linter errors
- [ ] Type hints complete
- [ ] Consistent logging
- [ ] No debug code left

### Manual Testing
- [ ] End-to-end test: Search query
- [ ] End-to-end test: Memory operations
- [ ] End-to-end test: Entity creation
- [ ] End-to-end test: Document listing
- [ ] End-to-end test: Multi-delegation
- [ ] UI reasoning display works
- [ ] Status panel updates correctly

---

## 🎉 Project Complete Checklist

When all phases are done:

- [ ] All 7 phases marked complete
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Code reviewed and merged
- [ ] System running in development
- [ ] Demo recording created (optional)

---

## 📝 Post-Launch Improvements (Future)

Ideas for future enhancements:

1. **Parallel Specialist Execution** - When using cloud LLMs
2. **Dynamic Specialist Discovery** - Add specialists at runtime
3. **Query Result Caching** - Cross-session cache
4. **Specialist Specialization** - Fine-tuned prompts per domain
5. **Observability Dashboard** - Metrics and tracing
6. **A/B Testing Framework** - Compare agent configurations

---

*Created: November 29, 2025*

