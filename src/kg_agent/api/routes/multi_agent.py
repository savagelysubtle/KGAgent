"""API routes for the multi-agent system.

Provides endpoints for:
- Chat interaction (simple and streaming)
- Session management
- Health/status checks

Note: For CopilotKit frontend integration, use the AG-UI endpoint instead.
"""

import json
from typing import Any, Dict, List, Optional

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
        thinking_steps = []
        for step in result.get("thinking_steps", []):
            # Handle both dict and object-like access
            if isinstance(step, dict):
                thinking_steps.append(
                    ThinkingStepResponse(
                        agent=step.get("agent", "unknown"),
                        thought=step.get("thought", ""),
                        action=step.get("action"),
                        result=step.get("result"),
                        status=step.get("status", "unknown"),
                        timestamp=step.get("timestamp", ""),
                    )
                )
            else:
                thinking_steps.append(
                    ThinkingStepResponse(
                        agent=getattr(step, "agent", "unknown"),
                        thought=getattr(step, "thought", ""),
                        action=getattr(step, "action", None),
                        result=getattr(step, "result", None),
                        status=getattr(step, "status", "unknown"),
                        timestamp=getattr(step, "timestamp", ""),
                    )
                )

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
                thinking_steps_data = []
                for s in state_update.get("thinking_steps", [])[-3:]:
                    if isinstance(s, dict):
                        thinking_steps_data.append(
                            {
                                "agent": s.get("agent", "unknown"),
                                "thought": s.get("thought", ""),
                                "status": s.get("status", "unknown"),
                            }
                        )
                    else:
                        thinking_steps_data.append(
                            {
                                "agent": getattr(s, "agent", "unknown"),
                                "thought": getattr(s, "thought", ""),
                                "status": getattr(s, "status", "unknown"),
                            }
                        )

                event_data = {
                    "node": node_name,
                    "current_agent": state_update.get("current_agent"),
                    "thinking_steps": thinking_steps_data,
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
        if hasattr(graph, "checkpointer") and graph.checkpointer:
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
