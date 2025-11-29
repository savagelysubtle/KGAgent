# Phase 5: API Integration - CopilotKit & FastAPI

> **Parent Plan:** [multi_agent.plan.md](./multi_agent.plan.md)
> **Status:** Not Started
> **Estimated Effort:** 2-3 hours
> **Dependencies:** Phases 1-4 complete
> **Last Updated:** November 29, 2025

---

## 🎯 Objectives

1. Create FastAPI routes for the multi-agent system
2. Integrate with CopilotKit via AG-UI protocol
3. Enable real-time state streaming
4. Support session management via API
5. Maintain backward compatibility with existing agent routes

---

## 📋 Prerequisites

- [ ] Phase 4 complete (graph fully wired)
- [ ] CopilotKit SDK installed (`copilotkit>=0.1.72`)
- [ ] AG-UI LangGraph integration installed (`ag-ui-langgraph`)
- [ ] Existing agent routes understood (`api/routes/agent.py`)

---

## 🔬 Technical Research Summary

### Key Findings

#### 1. Modern CopilotKit Uses AG-UI Protocol

CopilotKit has moved to the **AG-UI (Agent-User Interaction) Protocol** for LangGraph integration:

- **Old pattern**: `CopilotKitRemoteEndpoint` + `add_fastapi_endpoint`
- **New pattern**: `LangGraphAGUIAgent` + `add_langgraph_fastapi_endpoint`

The AG-UI protocol provides:
- Standardized event-based communication
- Real-time bidirectional state sync
- Frontend tool call support
- Unified message streaming

#### 2. Two Integration Approaches

| Approach | Use Case | Components |
|----------|----------|------------|
| **Self-Hosted FastAPI** | Our backend-first approach | `LangGraphAGUIAgent`, `add_langgraph_fastapi_endpoint` |
| **CopilotKit Cloud** | Frontend-first with hosted runtime | Frontend SDK + Copilot Cloud |

For KGAgent, we'll use the **self-hosted FastAPI approach** since we're extending an existing backend.

#### 3. Key Dependencies

```toml
# pyproject.toml additions
copilotkit = ">=0.1.72"
ag-ui-langgraph = "*"  # AG-UI protocol for LangGraph
```

#### 4. Message Persistence Pattern

```python
from contextlib import asynccontextmanager
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
# or
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
```

Use lifespan context manager for proper async checkpointer setup.

---

## 🏗️ API Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FASTAPI APPLICATION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  /api/v1/agent/*           ← Existing routes (backward compat)              │
│                                                                              │
│  /api/v1/multi-agent/      ← NEW: Multi-agent routes                        │
│    POST /chat              ← Simple request/response                         │
│    POST /chat/stream       ← Streaming response                              │
│    POST /session           ← Create session                                  │
│    GET  /session/{id}      ← Get session info                                │
│    DELETE /session/{id}    ← Delete session                                  │
│                                                                              │
│  /agents/kg_multi_agent    ← AG-UI protocol endpoint (CopilotKit)           │
│    POST /                  ← AG-UI event handling                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                         ┌───────────────────────┐
                         │   MULTI-AGENT GRAPH   │
                         │                       │
                         │  • State streaming    │
                         │  • Checkpointing      │
                         │  • Session management │
                         └───────────────────────┘
```

---

## 📁 Task 1: Install Dependencies

### Update `pyproject.toml`

```toml
[project]
dependencies = [
    # ... existing deps
    "copilotkit>=0.1.72",
]

[project.optional-dependencies]
agui = [
    "ag-ui-langgraph",
]
```

Install with:
```bash
uv add copilotkit
uv add ag-ui-langgraph
```

---

## 📁 Task 2: Create AG-UI Integration Endpoint

### File: `src/kg_agent/api/routes/agui.py`

This is the **modern approach** using the AG-UI protocol:

```python
"""AG-UI protocol endpoint for CopilotKit integration.

This provides the standard AG-UI endpoint that CopilotKit frontends
can connect to for real-time bidirectional communication.
"""

from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, APIRouter

from ...core.logging import logger
from ...core.config import settings

# AG-UI imports (conditionally available)
try:
    from copilotkit import LangGraphAGUIAgent
    from ag_ui_langgraph import add_langgraph_fastapi_endpoint
    AGUI_AVAILABLE = True
except ImportError:
    AGUI_AVAILABLE = False
    logger.warning(
        "AG-UI dependencies not available. "
        "Install with: uv add copilotkit ag-ui-langgraph"
    )


router = APIRouter(tags=["agui"])


def register_agui_endpoint(
    app: FastAPI,
    graph,
    agent_name: str = "kg_multi_agent",
    path: str = "/agents/kg_multi_agent"
):
    """
    Register the AG-UI endpoint for CopilotKit integration.

    Args:
        app: FastAPI application instance
        graph: Compiled LangGraph StateGraph
        agent_name: Name of the agent (used by frontend)
        path: URL path for the endpoint

    Example:
        ```python
        from kg_agent.agent.multi import get_multi_agent

        graph = get_multi_agent()
        register_agui_endpoint(app, graph)
        ```
    """
    if not AGUI_AVAILABLE:
        logger.error("Cannot register AG-UI endpoint: dependencies not installed")
        return

    try:
        agent = LangGraphAGUIAgent(
            name=agent_name,
            description=(
                "Knowledge Graph Multi-Agent System with Research, Memory, "
                "Knowledge, and Document specialist agents"
            ),
            graph=graph,
        )

        add_langgraph_fastapi_endpoint(
            app=app,
            agent=agent,
            path=path,
        )

        logger.info(f"AG-UI endpoint registered at {path}")

    except Exception as e:
        logger.error(f"Failed to register AG-UI endpoint: {e}")
        raise


# === Info Endpoint ===

@router.get("/agui/info")
async def agui_info():
    """Get AG-UI integration info."""
    return {
        "available": AGUI_AVAILABLE,
        "agent_name": "kg_multi_agent" if AGUI_AVAILABLE else None,
        "protocol": "ag-ui",
        "features": {
            "state_streaming": True,
            "bidirectional_state": True,
            "frontend_tools": True,
            "reasoning_display": True,
            "multi_agent": True,
        } if AGUI_AVAILABLE else {},
    }
```

---

## 📁 Task 3: Create Multi-Agent API Routes

### File: `src/kg_agent/api/routes/multi_agent.py`

This provides a **REST API** for direct backend usage (without CopilotKit frontend):

```python
"""API routes for the multi-agent system.

Provides endpoints for:
- Chat interaction (simple and streaming)
- Session management
- Health/status checks

Note: For CopilotKit frontend integration, use the AG-UI endpoint instead.
"""

from typing import Any, Dict, List, Optional
import json
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...core.logging import logger


router = APIRouter(prefix="/multi-agent", tags=["multi-agent"])


# === Request/Response Models ===

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for continuity")
    user_id: Optional[str] = Field(None, description="User identifier")


class ThinkingStepResponse(BaseModel):
    """Response model for thinking steps."""
    agent: str
    thought: str
    action: Optional[str] = None
    result: Optional[str] = None
    status: str
    timestamp: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Final response from the agent")
    session_id: Optional[str] = Field(None, description="Session ID")
    thinking_steps: List[ThinkingStepResponse] = Field(default_factory=list)
    execution_path: List[str] = Field(default_factory=list)
    total_llm_calls: int = 0


class SessionCreateRequest(BaseModel):
    """Request to create a new session."""
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    """Response with session information."""
    id: str
    user_id: Optional[str]
    created_at: str
    last_active: str
    message_count: int
    metadata: Dict[str, Any]


class MultiAgentStatusResponse(BaseModel):
    """Status response for multi-agent system."""
    status: str
    graph_compiled: bool
    checkpointer_type: Optional[str]
    active_sessions: int
    specialists: List[str]


# === Helper Functions ===

def _get_graph():
    """Lazy import to avoid circular dependencies."""
    from ...agent.multi import get_multi_agent
    return get_multi_agent()


def _get_session_manager():
    """Lazy import session manager."""
    from ...agent.multi.session import get_session_manager
    return get_session_manager()


# === Endpoints ===

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the multi-agent system.

    Returns the final response after all specialists have processed.
    For streaming responses, use /chat/stream instead.
    """
    logger.info(f"Multi-agent chat request: {request.message[:50]}...")

    try:
        from ...agent.multi import invoke_multi_agent

        result = await invoke_multi_agent(
            user_message=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
        )

        # Convert thinking steps to response format
        thinking_steps = [
            ThinkingStepResponse(
                agent=step.agent,
                thought=step.thought,
                action=step.action,
                result=step.result,
                status=step.status,
                timestamp=step.timestamp,
            )
            for step in result.get("thinking_steps", [])
        ]

        return ChatResponse(
            response=result.get("final_response", "No response generated"),
            session_id=request.session_id,
            thinking_steps=thinking_steps,
            execution_path=result.get("execution_path", []),
            total_llm_calls=result.get("total_llm_calls", 0),
        )

    except Exception as e:
        logger.error(f"Multi-agent chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream a chat response from the multi-agent system.

    Yields JSON events as Server-Sent Events (SSE) while the graph executes.
    Each event contains the current node, thinking steps, and progress.
    """
    logger.info(f"Multi-agent stream request: {request.message[:50]}...")

    async def event_generator():
        try:
            from ...agent.multi import stream_multi_agent

            async for node_name, state_update in stream_multi_agent(
                user_message=request.message,
                user_id=request.user_id,
                session_id=request.session_id,
            ):
                # Format as Server-Sent Event
                event_data = {
                    "node": node_name,
                    "current_agent": state_update.get("current_agent"),
                    "thinking_steps": [
                        {
                            "agent": s.agent,
                            "thought": s.thought,
                            "status": s.status,
                        }
                        for s in state_update.get("thinking_steps", [])[-3:]
                    ],
                }

                # Include final response if available
                if state_update.get("final_response"):
                    event_data["final_response"] = state_update["final_response"]

                yield f"data: {json.dumps(event_data)}\n\n"

            # Final done event
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# === Session Endpoints ===

@router.post("/session", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest) -> SessionResponse:
    """Create a new conversation session."""
    try:
        sm = _get_session_manager()
        session_id = sm.create_session(
            user_id=request.user_id,
            metadata=request.metadata,
        )
        session = sm.get_session(session_id)
        return SessionResponse(**session)

    except Exception as e:
        logger.error(f"Create session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str) -> SessionResponse:
    """Get session information."""
    sm = _get_session_manager()
    session = sm.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionResponse(**session)


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    sm = _get_session_manager()

    if not sm.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "deleted", "session_id": session_id}


@router.get("/sessions")
async def list_sessions(user_id: Optional[str] = None) -> List[SessionResponse]:
    """List all sessions, optionally filtered by user."""
    sm = _get_session_manager()
    sessions = sm.list_sessions(user_id=user_id)
    return [SessionResponse(**s) for s in sessions]


# === Status Endpoint ===

@router.get("/status", response_model=MultiAgentStatusResponse)
async def get_status() -> MultiAgentStatusResponse:
    """Get multi-agent system status."""
    try:
        graph = _get_graph()
        sm = _get_session_manager()

        # Get checkpointer type
        checkpointer_type = None
        if hasattr(graph, 'checkpointer') and graph.checkpointer:
            checkpointer_type = type(graph.checkpointer).__name__

        return MultiAgentStatusResponse(
            status="operational",
            graph_compiled=True,
            checkpointer_type=checkpointer_type,
            active_sessions=len(sm.list_sessions()),
            specialists=[
                "research_lead",
                "memory_keeper",
                "knowledge_builder",
                "document_analyst",
            ],
        )

    except Exception as e:
        return MultiAgentStatusResponse(
            status=f"error: {str(e)}",
            graph_compiled=False,
            checkpointer_type=None,
            active_sessions=0,
            specialists=[],
        )
```

---

## 📁 Task 4: Update Main Application

### File: `src/kg_agent/api/main.py` (modifications)

Add the multi-agent routes and optionally register the AG-UI endpoint:

```python
# Add to imports at top of file
from .routes import multi_agent

# Optional: AG-UI support
try:
    from .routes.agui import register_agui_endpoint, router as agui_router
    AGUI_AVAILABLE = True
except ImportError:
    AGUI_AVAILABLE = False

# ... existing code ...

# Add multi-agent routes
app.include_router(
    multi_agent.router,
    prefix=settings.API_V1_PREFIX,
    tags=["multi-agent"]
)

# Add AG-UI info endpoint
if AGUI_AVAILABLE:
    app.include_router(
        agui_router,
        prefix=settings.API_V1_PREFIX,
        tags=["agui"]
    )


# Register AG-UI endpoint in lifespan (for CopilotKit frontend support)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    logger.info("Starting KG Agent Crawler API")

    # Optionally register AG-UI endpoint
    if AGUI_AVAILABLE:
        try:
            from ..agent.multi import get_multi_agent
            graph = get_multi_agent()
            register_agui_endpoint(app, graph)
        except Exception as e:
            logger.warning(f"AG-UI registration skipped: {e}")

    yield

    logger.info("Shutting down KG Agent Crawler API")
```

### Update `src/kg_agent/api/routes/__init__.py`

```python
"""API route handlers."""

from . import (
    crawl, health, session, graph, stats, upload,
    agent, documents, chat, reprocess, preview,
    multi_agent  # NEW
)

__all__ = [
    "crawl", "health", "session", "graph", "stats",
    "upload", "agent", "documents", "chat", "reprocess",
    "preview", "multi_agent"  # NEW
]
```

---

## 📁 Task 5: API Tests

### File: `tests/test_multi_agent_api.py`

```python
"""API tests for multi-agent endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock

from kg_agent.api.main import app


@pytest.fixture
def client():
    """Create test client."""
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

        with patch("kg_agent.api.routes.multi_agent.invoke_multi_agent") as mock:
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

        with patch("kg_agent.api.routes.multi_agent.invoke_multi_agent") as mock:
            mock.return_value = mock_result

            response = client.post(
                "/api/v1/multi-agent/chat",
                json={"message": "Hello", "session_id": "test-session"}
            )

            assert response.status_code == 200
            assert response.json()["session_id"] == "test-session"

    def test_chat_error(self, client):
        """Test chat error handling."""
        with patch("kg_agent.api.routes.multi_agent.invoke_multi_agent") as mock:
            mock.side_effect = Exception("Test error")

            response = client.post(
                "/api/v1/multi-agent/chat",
                json={"message": "Hello"}
            )

            assert response.status_code == 500


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

    def test_get_session_not_found(self, client):
        """Test session not found."""
        mock_sm = MagicMock()
        mock_sm.get_session.return_value = None

        with patch("kg_agent.api.routes.multi_agent._get_session_manager") as mock:
            mock.return_value = mock_sm

            response = client.get("/api/v1/multi-agent/session/nonexistent")

            assert response.status_code == 404


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
```

---

## 📁 Task 6: Documentation Examples

### Example cURL Commands

```bash
# Simple chat
curl -X POST http://localhost:8000/api/v1/multi-agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Search for Python tutorials"}'

# Chat with session
curl -X POST http://localhost:8000/api/v1/multi-agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Remember my name is Steve", "session_id": "session-123"}'

# Streaming chat
curl -X POST http://localhost:8000/api/v1/multi-agent/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Create an entity for Python"}' \
  --no-buffer

# Create session
curl -X POST http://localhost:8000/api/v1/multi-agent/session \
  -H "Content-Type: application/json" \
  -d '{"user_id": "steve"}'

# Get status
curl http://localhost:8000/api/v1/multi-agent/status

# Check AG-UI availability
curl http://localhost:8000/api/v1/agui/info
```

### Frontend Integration (CopilotKit)

If using CopilotKit frontend:

```tsx
// layout.tsx
import { CopilotKit } from "@copilotkit/react-core";

export default function RootLayout({ children }) {
  return (
    <CopilotKit
      runtimeUrl="http://localhost:8000/agents/kg_multi_agent"
      agent="kg_multi_agent"
    >
      {children}
    </CopilotKit>
  );
}
```

---

## ✅ Phase 5 Definition of Done

- [ ] Dependencies installed:
  - [ ] `copilotkit>=0.1.72`
  - [ ] `ag-ui-langgraph` (optional, for CopilotKit frontend)

- [ ] Multi-agent API routes created:
  - [ ] `POST /api/v1/multi-agent/chat` - works
  - [ ] `POST /api/v1/multi-agent/chat/stream` - streams SSE events
  - [ ] `POST /api/v1/multi-agent/session` - creates session
  - [ ] `GET /api/v1/multi-agent/session/{id}` - retrieves session
  - [ ] `DELETE /api/v1/multi-agent/session/{id}` - deletes session
  - [ ] `GET /api/v1/multi-agent/status` - returns status

- [ ] AG-UI integration (optional):
  - [ ] `GET /api/v1/agui/info` - returns integration info
  - [ ] AG-UI endpoint registered at `/agents/kg_multi_agent`

- [ ] Routes registered in main app

- [ ] API tests pass

- [ ] Manual verification:
  ```bash
  # Start server
  uvicorn kg_agent.api.main:app --reload

  # Test chat
  curl -X POST http://localhost:8000/api/v1/multi-agent/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "Hello"}'
  ```

---

## 🔗 Next Phase

→ [Phase 6: Frontend Integration](./phase6-frontend.md) - React components for reasoning display

---

## 📚 Appendix: Technical Research Notes

### AG-UI Protocol

AG-UI (Agent-User Interaction) is a lightweight, event-based protocol that standardizes how AI agents connect to user-facing applications. Key benefits:

- **Flexibility**: Framework-agnostic agent integration
- **Real-time**: Bidirectional state synchronization
- **Unified**: Standard event format for messages, tools, state

### CopilotKit SDK Classes

| Class | Purpose |
|-------|---------|
| `LangGraphAGUIAgent` | Wraps LangGraph for AG-UI protocol |
| `add_langgraph_fastapi_endpoint` | Registers AG-UI endpoint on FastAPI |
| `CopilotKitRemoteEndpoint` | Legacy: For backend actions without full agent |
| `add_fastapi_endpoint` | Legacy: Simpler endpoint registration |

### State Emission Functions

From `copilotkit.langgraph`:

| Function | Purpose |
|----------|---------|
| `copilotkit_emit_state(config, state)` | Emit intermediate state to UI |
| `copilotkit_emit_message(config, msg)` | Emit message mid-node |
| `copilotkit_customize_config(config, **opts)` | Control streaming behavior |

### Checkpointer Options

| Checkpointer | Use Case | Install |
|--------------|----------|---------|
| `MemorySaver` | Development, testing | Built-in |
| `AsyncSqliteSaver` | Local persistence | `langgraph[sqlite]` |
| `AsyncPostgresSaver` | Production | `langgraph[postgres]` |

---

*Created: November 29, 2025*
*Updated: November 29, 2025 - Added AG-UI protocol research findings*
