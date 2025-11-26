"""FastAPI routes for the Pydantic AI Knowledge Graph Agent."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import json

from ...core.logging import logger
from ...agent import get_kg_agent

router = APIRouter()


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="The user's message or query")
    user_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional context about the user or session"
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="The agent's response")
    status: str = Field(default="success", description="Status of the request")


class ToolCallRequest(BaseModel):
    """Request model for direct tool calls."""
    tool_name: str = Field(..., description="Name of the tool to call")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the tool"
    )


class ToolCallResponse(BaseModel):
    """Response model for tool calls."""
    result: str = Field(..., description="Result from the tool")
    tool_name: str = Field(..., description="Name of the tool called")
    status: str = Field(default="success", description="Status of the request")


@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Send a message to the Knowledge Graph Agent and receive a response.

    The agent has access to:
    - Vector search (ChromaDB) for semantic similarity
    - Graph search (FalkorDB) for structured information
    - Database statistics

    Example queries:
    - "What do you know about machine learning?"
    - "Search for documents about Python"
    - "Show me the database statistics"
    """
    try:
        agent = get_kg_agent()

        if request.stream:
            # Return streaming response
            async def generate():
                async for chunk in agent.chat_stream(
                    request.message,
                    user_context=request.user_context
                ):
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )

        # Non-streaming response
        response = await agent.chat(
            request.message,
            user_context=request.user_context
        )

        return ChatResponse(response=response)

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream a chat response from the Knowledge Graph Agent.

    Returns a Server-Sent Events (SSE) stream with response chunks.
    """
    try:
        agent = get_kg_agent()

        async def generate():
            async for chunk in agent.chat_stream(
                request.message,
                user_context=request.user_context
            ):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except Exception as e:
        logger.error(f"Stream endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Streaming error: {str(e)}"
        )


@router.post("/tools/search", response_model=ToolCallResponse)
async def search_knowledge_base(
    query: str,
    search_type: str = "hybrid",
    limit: int = 5
):
    """
    Directly search the knowledge base without going through the agent.

    Args:
        query: The search query
        search_type: 'vector', 'graph', or 'hybrid'
        limit: Maximum number of results

    Returns:
        Search results from the knowledge base
    """
    try:
        from ...agent.tools import get_rag_tools

        rag_tools = get_rag_tools()
        await rag_tools.initialize()

        if search_type == "vector":
            results = await rag_tools.search_vectors(query, n_results=limit)
        elif search_type == "graph":
            results = await rag_tools.search_graph(query, limit=limit)
        else:
            results = await rag_tools.hybrid_search(query, vector_results=limit, graph_results=limit)

        # Format results
        if search_type == "hybrid":
            result_str = json.dumps({
                "vector_results": [
                    {"text": r.text, "source": r.source, "score": r.score}
                    for r in results.get("vector", [])
                ],
                "graph_results": [
                    {"text": r.text, "source": r.source, "metadata": r.metadata}
                    for r in results.get("graph", [])
                ]
            }, indent=2)
        else:
            result_str = json.dumps([
                {"text": r.text, "source": r.source, "score": r.score, "metadata": r.metadata}
                for r in results
            ], indent=2)

        return ToolCallResponse(
            result=result_str,
            tool_name=f"search_{search_type}"
        )

    except Exception as e:
        logger.error(f"Search tool error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search error: {str(e)}"
        )


@router.get("/tools/stats")
async def get_knowledge_base_stats():
    """
    Get statistics about the knowledge base (ChromaDB and FalkorDB).

    Returns:
        Statistics about vector store and graph database
    """
    try:
        from ...agent.tools import get_rag_tools

        rag_tools = get_rag_tools()
        await rag_tools.initialize()

        graph_stats = await rag_tools.get_graph_stats()
        vector_stats = await rag_tools.get_vector_stats()

        return {
            "status": "success",
            "vector_store": {
                "collection_name": vector_stats.collection_name,
                "total_chunks": vector_stats.total_chunks
            },
            "graph_database": {
                "connected": graph_stats.connected,
                "total_nodes": graph_stats.total_nodes,
                "total_edges": graph_stats.total_edges,
                "entity_types": graph_stats.entity_types,
                "relationship_types": graph_stats.relationship_types
            }
        }

    except Exception as e:
        logger.error(f"Stats endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Stats error: {str(e)}"
        )


@router.get("/health")
async def agent_health():
    """
    Check the health of the agent and its dependencies.

    Returns:
        Health status of the agent, vector store, and graph database
    """
    try:
        from ...agent.tools import get_rag_tools

        rag_tools = get_rag_tools()

        # Check vector store
        vector_healthy = False
        try:
            await rag_tools.initialize()
            vector_stats = await rag_tools.get_vector_stats()
            vector_healthy = True
        except Exception:
            pass

        # Check graph database
        graph_healthy = False
        try:
            graph_stats = await rag_tools.get_graph_stats()
            graph_healthy = graph_stats.connected
        except Exception:
            pass

    # Check LM Studio connection
        llm_healthy = False
        try:
            import httpx
            from ...core.config import settings

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{settings.LLM_BASE_URL}/models",
                    timeout=5.0
                )
                if response.status_code == 200:
                    data = response.json()
                    llm_healthy = len(data.get("data", [])) > 0
        except Exception:
            pass

        overall_healthy = vector_healthy and graph_healthy and llm_healthy

        return {
            "status": "healthy" if overall_healthy else "degraded",
            "components": {
                "llm_studio": {
                    "status": "healthy" if llm_healthy else "unavailable",
                    "base_url": settings.LLM_BASE_URL
                },
                "vector_store": {
                    "status": "healthy" if vector_healthy else "unavailable"
                },
                "graph_database": {
                    "status": "healthy" if graph_healthy else "unavailable"
                }
            }
        }

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

