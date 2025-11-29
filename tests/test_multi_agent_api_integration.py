"""API integration tests for multi-agent endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from kg_agent.api.main import app
from kg_agent.agent.multi.state import ThinkingStep


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_invoke_result():
    """Create a mock invoke_multi_agent result."""
    return {
        "final_response": "Test response about Python",
        "thinking_steps": [
            ThinkingStep(
                agent="manager",
                thought="Analyzing user request",
                status="thinking",
                timestamp="2025-01-01T00:00:00",
            ),
            ThinkingStep(
                agent="manager",
                thought="Delegating to research",
                action="delegate_to_research",
                status="delegating",
                timestamp="2025-01-01T00:00:01",
            ),
            ThinkingStep(
                agent="research",
                thought="Searching knowledge base",
                action="search_knowledge_base",
                status="executing",
                timestamp="2025-01-01T00:00:02",
            ),
            ThinkingStep(
                agent="research",
                thought="Search complete",
                result="Found 5 results",
                status="complete",
                timestamp="2025-01-01T00:00:03",
            ),
        ],
        "execution_path": ["manager", "research", "synthesize"],
        "total_llm_calls": 3,
        "should_end": True,
    }


class TestChatEndpointIntegration:
    """Integration tests for chat endpoint."""

    def test_chat_returns_thinking_steps(self, client, mock_invoke_result):
        """Verify thinking steps are included in response."""
        with patch(
            "kg_agent.agent.multi.invoke_multi_agent",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = mock_invoke_result

            response = client.post(
                "/api/v1/multi-agent/chat", json={"message": "Search for Python"}
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["thinking_steps"]) == 4
            assert data["thinking_steps"][0]["agent"] == "manager"
            assert data["thinking_steps"][2]["agent"] == "research"

    def test_chat_returns_execution_path(self, client, mock_invoke_result):
        """Verify execution path is included."""
        with patch(
            "kg_agent.agent.multi.invoke_multi_agent",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = mock_invoke_result

            response = client.post(
                "/api/v1/multi-agent/chat", json={"message": "Search for Python"}
            )

            data = response.json()
            assert "manager" in data["execution_path"]
            assert "research" in data["execution_path"]

    def test_chat_with_session_id(self, client, mock_invoke_result):
        """Test chat with session ID for continuity."""
        with patch(
            "kg_agent.agent.multi.invoke_multi_agent",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = mock_invoke_result

            response = client.post(
                "/api/v1/multi-agent/chat",
                json={"message": "Hello", "session_id": "test-session-123"},
            )

            assert response.status_code == 200
            # Verify session_id was passed to invoke
            mock.assert_called_once()
            call_kwargs = mock.call_args
            assert call_kwargs.kwargs.get("session_id") == "test-session-123"

    def test_chat_error_handling(self, client):
        """Test chat endpoint error handling."""
        with patch(
            "kg_agent.agent.multi.invoke_multi_agent",
            new_callable=AsyncMock,
        ) as mock:
            mock.side_effect = Exception("Test error")

            response = client.post(
                "/api/v1/multi-agent/chat", json={"message": "Hello"}
            )

            assert response.status_code == 500


class TestStreamEndpoint:
    """Test streaming chat endpoint."""

    def test_stream_endpoint_exists(self, client):
        """Test that streaming endpoint is available."""
        # Just verify the endpoint exists and accepts POST
        response = client.post(
            "/api/v1/multi-agent/chat/stream", json={"message": "Test"}
        )
        # Should not be 404
        assert response.status_code != 404


class TestSessionEndpoints:
    """Test session management endpoints."""

    def test_create_session(self, client):
        """Test session creation."""
        response = client.post(
            "/api/v1/multi-agent/session",
            json={"user_id": "test-user", "metadata": {"source": "test"}},
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["user_id"] == "test-user"

    def test_get_session(self, client):
        """Test getting a session."""
        # First create a session
        create_response = client.post(
            "/api/v1/multi-agent/session", json={"user_id": "test-user"}
        )
        session_id = create_response.json()["id"]

        # Then get it
        response = client.get(f"/api/v1/multi-agent/session/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == session_id

    def test_get_nonexistent_session(self, client):
        """Test getting a session that doesn't exist."""
        response = client.get("/api/v1/multi-agent/session/nonexistent-id")

        assert response.status_code == 404

    def test_delete_session(self, client):
        """Test session deletion."""
        # First create a session
        create_response = client.post(
            "/api/v1/multi-agent/session", json={"user_id": "test-user"}
        )
        session_id = create_response.json()["id"]

        # Then delete it
        response = client.delete(f"/api/v1/multi-agent/session/{session_id}")

        assert response.status_code == 200

        # Verify it's gone
        get_response = client.get(f"/api/v1/multi-agent/session/{session_id}")
        assert get_response.status_code == 404

    def test_list_sessions(self, client):
        """Test listing all sessions."""
        # Create a couple sessions
        client.post("/api/v1/multi-agent/session", json={"user_id": "user1"})
        client.post("/api/v1/multi-agent/session", json={"user_id": "user2"})

        response = client.get("/api/v1/multi-agent/sessions")

        assert response.status_code == 200
        data = response.json()
        # API returns a list directly, not wrapped in "sessions" key
        assert isinstance(data, list)
        assert len(data) >= 2


class TestStatusEndpoint:
    """Test multi-agent status endpoint."""

    def test_status_endpoint(self, client):
        """Test status endpoint returns system info."""
        response = client.get("/api/v1/multi-agent/status")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "graph_compiled" in data
        assert "specialists" in data
        assert len(data["specialists"]) == 4  # research, memory, knowledge, documents


class TestAGUIEndpoint:
    """Test AG-UI/CopilotKit endpoint."""

    def test_agui_info_endpoint(self, client):
        """Test AG-UI info endpoint."""
        response = client.get("/api/v1/agui/info")

        assert response.status_code == 200
        data = response.json()
        assert "available" in data
        if data["available"]:
            assert data["agent_name"] == "kg_multi_agent"
            assert "features" in data

