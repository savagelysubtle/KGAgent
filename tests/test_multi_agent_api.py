"""API tests for multi-agent endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    """Create test client."""
    from kg_agent.api.main import app
    return TestClient(app)


class TestMultiAgentChatEndpoint:
    """Test /api/v1/multi-agent/chat endpoint."""

    def test_chat_success(self, client):
        """Test successful chat request."""
        mock_result = {
            "final_response": "Test response",
            "thinking_steps": [],
            "execution_path": ["manager"],
            "total_llm_calls": 1,
        }

        # Patch where it's imported (inside the chat function)
        with patch("kg_agent.agent.multi.invoke_multi_agent") as mock:
            mock.return_value = mock_result

            response = client.post(
                "/api/v1/multi-agent/chat",
                json={"message": "Hello"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "Test response"

    def test_chat_with_session(self, client):
        """Test chat with session ID."""
        mock_result = {
            "final_response": "Test",
            "thinking_steps": [],
            "execution_path": [],
            "total_llm_calls": 1,
        }

        with patch("kg_agent.agent.multi.invoke_multi_agent") as mock:
            mock.return_value = mock_result

            response = client.post(
                "/api/v1/multi-agent/chat",
                json={"message": "Hello", "session_id": "test-session"}
            )

            assert response.status_code == 200
            assert response.json()["session_id"] == "test-session"

    def test_chat_with_thinking_steps(self, client):
        """Test chat response with thinking steps."""
        mock_result = {
            "final_response": "Test response",
            "thinking_steps": [
                {
                    "agent": "manager",
                    "thought": "Analyzing request",
                    "status": "complete",
                    "timestamp": "2025-01-01T00:00:00",
                }
            ],
            "execution_path": ["manager"],
            "total_llm_calls": 1,
        }

        with patch("kg_agent.agent.multi.invoke_multi_agent") as mock:
            mock.return_value = mock_result

            response = client.post(
                "/api/v1/multi-agent/chat",
                json={"message": "Hello"}
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["thinking_steps"]) == 1
            assert data["thinking_steps"][0]["agent"] == "manager"


class TestSessionEndpoints:
    """Test session management endpoints."""

    def test_create_session(self, client):
        """Test session creation."""
        mock_sm = MagicMock()
        mock_sm.create_session.return_value = "test-session-id"
        mock_sm.get_session.return_value = {
            "id": "test-session-id",
            "user_id": "test-user",
            "created_at": "2025-01-01T00:00:00",
            "last_active": "2025-01-01T00:00:00",
            "message_count": 0,
            "metadata": {},
        }

        with patch("kg_agent.api.routes.multi_agent._get_session_manager") as mock:
            mock.return_value = mock_sm

            response = client.post(
                "/api/v1/multi-agent/session",
                json={"user_id": "test-user"}
            )

            assert response.status_code == 200
            assert response.json()["user_id"] == "test-user"

    def test_get_session_success(self, client):
        """Test successful session retrieval."""
        mock_sm = MagicMock()
        mock_sm.get_session.return_value = {
            "id": "test-session-id",
            "user_id": "test-user",
            "created_at": "2025-01-01T00:00:00",
            "last_active": "2025-01-01T00:00:00",
            "message_count": 5,
            "metadata": {},
        }

        with patch("kg_agent.api.routes.multi_agent._get_session_manager") as mock:
            mock.return_value = mock_sm

            response = client.get("/api/v1/multi-agent/session/test-session-id")

            assert response.status_code == 200
            assert response.json()["message_count"] == 5

    def test_get_session_not_found(self, client):
        """Test session not found."""
        mock_sm = MagicMock()
        mock_sm.get_session.return_value = None

        with patch("kg_agent.api.routes.multi_agent._get_session_manager") as mock:
            mock.return_value = mock_sm

            response = client.get("/api/v1/multi-agent/session/nonexistent")

            assert response.status_code == 404

    def test_delete_session_success(self, client):
        """Test successful session deletion."""
        mock_sm = MagicMock()
        mock_sm.delete_session.return_value = True

        with patch("kg_agent.api.routes.multi_agent._get_session_manager") as mock:
            mock.return_value = mock_sm

            response = client.delete("/api/v1/multi-agent/session/test-session")

            assert response.status_code == 200
            assert response.json()["status"] == "deleted"

    def test_delete_session_not_found(self, client):
        """Test delete session not found."""
        mock_sm = MagicMock()
        mock_sm.delete_session.return_value = False

        with patch("kg_agent.api.routes.multi_agent._get_session_manager") as mock:
            mock.return_value = mock_sm

            response = client.delete("/api/v1/multi-agent/session/nonexistent")

            assert response.status_code == 404

    def test_list_sessions(self, client):
        """Test listing sessions."""
        mock_sm = MagicMock()
        mock_sm.list_sessions.return_value = [
            {
                "id": "session-1",
                "user_id": "test-user",
                "created_at": "2025-01-01T00:00:00",
                "last_active": "2025-01-01T00:00:00",
                "message_count": 0,
                "metadata": {},
            },
            {
                "id": "session-2",
                "user_id": "test-user",
                "created_at": "2025-01-01T00:00:00",
                "last_active": "2025-01-01T00:00:00",
                "message_count": 0,
                "metadata": {},
            },
        ]

        with patch("kg_agent.api.routes.multi_agent._get_session_manager") as mock:
            mock.return_value = mock_sm

            response = client.get("/api/v1/multi-agent/sessions")

            assert response.status_code == 200
            assert len(response.json()) == 2


class TestStatusEndpoint:
    """Test status endpoint."""

    def test_status_operational(self, client):
        """Test status when operational."""
        mock_graph = MagicMock()
        mock_graph.checkpointer = MagicMock()

        mock_sm = MagicMock()
        mock_sm.list_sessions.return_value = []

        with patch("kg_agent.api.routes.multi_agent._get_graph") as mock_g, \
             patch("kg_agent.api.routes.multi_agent._get_session_manager") as mock_s:
            mock_g.return_value = mock_graph
            mock_s.return_value = mock_sm

            response = client.get("/api/v1/multi-agent/status")

            assert response.status_code == 200
            assert response.json()["status"] == "operational"
            assert response.json()["graph_compiled"] is True

    def test_status_with_sessions(self, client):
        """Test status with active sessions."""
        mock_graph = MagicMock()
        mock_graph.checkpointer = None

        mock_sm = MagicMock()
        mock_sm.list_sessions.return_value = [{}, {}, {}]  # 3 sessions

        with patch("kg_agent.api.routes.multi_agent._get_graph") as mock_g, \
             patch("kg_agent.api.routes.multi_agent._get_session_manager") as mock_s:
            mock_g.return_value = mock_graph
            mock_s.return_value = mock_sm

            response = client.get("/api/v1/multi-agent/status")

            assert response.status_code == 200
            assert response.json()["active_sessions"] == 3


class TestAGUIInfoEndpoint:
    """Test AG-UI info endpoint."""

    def test_agui_info(self, client):
        """Test AG-UI info endpoint returns correct info."""
        response = client.get("/api/v1/agui/info")

        assert response.status_code == 200
        data = response.json()
        assert "available" in data
        assert "protocol" in data
        assert data["protocol"] == "ag-ui"

