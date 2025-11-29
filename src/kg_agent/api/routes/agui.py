"""AG-UI protocol endpoint for CopilotKit integration.

This provides the standard AG-UI endpoint that CopilotKit frontends
can connect to for real-time bidirectional communication.
"""

from fastapi import APIRouter

from ...core.logging import logger

# AG-UI imports (conditionally available)
AGUI_AVAILABLE = False

try:
    from ag_ui_langgraph import LangGraphAgent, add_langgraph_fastapi_endpoint

    AGUI_AVAILABLE = True
except ImportError:
    logger.warning(
        "AG-UI dependencies not available. "
        "Install with: uv add copilotkit ag-ui-langgraph"
    )


router = APIRouter(tags=["agui"])


def register_agui_endpoint(
    app,
    graph,
    agent_name: str = "kg_multi_agent",
    path: str = "/agents/kg_multi_agent",
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
        # Create LangGraph agent wrapper for AG-UI protocol
        agent = LangGraphAgent(
            name=agent_name,
            description=(
                "Knowledge Graph Multi-Agent System with Research, Memory, "
                "Knowledge, and Document specialist agents"
            ),
            graph=graph,
        )

        # Register the AG-UI endpoint
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
        "endpoint": "/agents/kg_multi_agent" if AGUI_AVAILABLE else None,
        "features": (
            {
                "state_streaming": True,
                "bidirectional_state": True,
                "frontend_tools": True,
                "reasoning_display": True,
                "multi_agent": True,
            }
            if AGUI_AVAILABLE
            else {}
        ),
    }
