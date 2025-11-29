"""Research Lead Agent - Specialist for knowledge base search.

Capabilities:
- Vector search (semantic)
- Graph search (structured)
- Hybrid search (combined)
- Source-based search
- Database statistics

VERIFIED against tools.py: Uses RAGTools methods directly.
"""

from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from pydantic_ai import Agent, RunContext

from ...core.logging import logger
from ..llm import create_lm_studio_model
from ..tools import RAGTools, SearchResult, get_rag_tools

try:
    from copilotkit.langgraph import copilotkit_emit_state  # type: ignore[assignment]
except ImportError:

    async def copilotkit_emit_state(config, state) -> None:
        """Fallback when CopilotKit is not installed."""


from .prompts import RESEARCH_LEAD_PROMPT
from .state import MultiAgentState, create_thinking_step


# === Dependencies ===


class ResearchDependencies:
    """Dependencies for the Research agent.

    Note: We pass the mutable state components separately since
    MultiAgentState is a TypedDict (immutable pattern).
    """

    def __init__(self, rag_tools: RAGTools, config: RunnableConfig):
        self.rag_tools = rag_tools
        self.config = config


# === Pydantic AI Agent ===

research_agent = Agent(
    model=create_lm_studio_model(),
    system_prompt=RESEARCH_LEAD_PROMPT,
    deps_type=ResearchDependencies,
    retries=2,
)


# === Tools ===


@research_agent.tool
async def search_knowledge_base(
    ctx: RunContext[ResearchDependencies], query: str, search_type: str = "hybrid"
) -> str:
    """
    Search the knowledge base for information.

    Args:
        query: The search query
        search_type: 'vector' for semantic, 'graph' for structured, 'hybrid' for both

    Returns:
        Formatted search results with sources
    """
    deps = ctx.deps
    rag_tools = deps.rag_tools

    try:
        if search_type == "vector":
            results = await rag_tools.search_vectors(query, n_results=5)
            return _format_search_results(results, "Vector Search")
        elif search_type == "graph":
            results = await rag_tools.search_graph(query, limit=5)
            return _format_search_results(results, "Graph Search")
        else:  # hybrid
            hybrid = await rag_tools.hybrid_search(query)
            parts = []
            if hybrid["vector"]:
                parts.append(_format_search_results(hybrid["vector"], "Semantic"))
            if hybrid["graph"]:
                parts.append(_format_search_results(hybrid["graph"], "Structured"))
            return "\n\n".join(parts) if parts else "No results found."

    except Exception as e:
        logger.error(f"Research search failed: {e}")
        return f"Search error: {str(e)}"


@research_agent.tool
async def search_by_source(
    ctx: RunContext[ResearchDependencies], source_pattern: str, limit: int = 10
) -> str:
    """
    Search for documents from a specific source.

    Args:
        source_pattern: Pattern to match (e.g., 'example.com', 'github')
        limit: Max results

    Returns:
        Documents matching the source pattern
    """
    deps = ctx.deps

    try:
        results = await deps.rag_tools.search_graph(source_pattern, limit=limit)
        if not results:
            return f"No documents found from '{source_pattern}'"
        return _format_search_results(results, f"Source: {source_pattern}")
    except Exception as e:
        return f"Source search error: {str(e)}"


@research_agent.tool
async def get_database_statistics(ctx: RunContext[ResearchDependencies]) -> str:
    """
    Get statistics about the knowledge base.

    Returns:
        Stats for ChromaDB (vectors) and FalkorDB (graph)
    """
    deps = ctx.deps

    try:
        graph_stats = await deps.rag_tools.get_graph_stats()
        vector_stats = await deps.rag_tools.get_vector_stats()

        return f"""📊 **Database Statistics**

**Vector Store (ChromaDB)**
- Collection: {vector_stats.collection_name}
- Total Chunks: {vector_stats.total_chunks}

**Knowledge Graph (FalkorDB)**
- Connected: {"✅" if graph_stats.connected else "❌"}
- Nodes: {graph_stats.total_nodes}
- Edges: {graph_stats.total_edges}
"""
    except Exception as e:
        return f"Stats error: {str(e)}"


# === Helper Functions ===


def _format_search_results(results: List[SearchResult], title: str) -> str:
    """Format search results for display."""
    if not results:
        return f"**{title}**: No results"

    output = f"**{title}** ({len(results)} results)\n"
    for i, r in enumerate(results[:5], 1):
        source = r.source[:50] if r.source else "unknown"
        text = r.text[:200] + "..." if len(r.text) > 200 else r.text
        score = f" ({r.score:.0%})" if r.score else ""
        output += f"\n{i}. [{source}]{score}\n   {text}\n"
    return output


# === Node Implementation ===


async def research_node(
    state: MultiAgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """
    Research Lead node for LangGraph.

    Processes the current delegation and returns research results.

    IMPORTANT: state is a TypedDict - use .get() and return new dicts,
    don't mutate state directly.
    """
    logger.info("Research node executing")

    # Get current delegation using safe TypedDict access
    delegation = state.get("current_delegation")
    if not delegation:
        return {"research_result": "No task provided"}

    task = delegation.get("task", "No task specified")

    # Create new lists (immutable pattern for TypedDict)
    thinking_steps = list(state.get("thinking_steps", []))
    thinking_steps.append(
        create_thinking_step(
            agent="research",
            thought=f"Received task: {task[:80]}...",
            status="thinking",
        )
    )

    execution_path = list(state.get("execution_path", []))
    execution_path.append("research")

    # Emit state update to UI
    await copilotkit_emit_state(
        config,
        {
            "thinking_steps": thinking_steps,
            "current_agent": "research",
            "execution_path": execution_path,
        },
    )

    # Initialize RAG tools
    rag_tools = get_rag_tools()
    await rag_tools.initialize()

    # Create dependencies (no state ref - tools don't need it)
    deps = ResearchDependencies(
        rag_tools=rag_tools,
        config=config,
    )

    # Run the Pydantic AI agent
    try:
        result = await research_agent.run(task, deps=deps)  # type: ignore[arg-type]
        research_result = result.output
    except Exception as e:
        logger.error(f"Research agent failed: {e}")
        research_result = f"Research failed: {str(e)}"

    # Add completion step
    thinking_steps.append(
        create_thinking_step(
            agent="research",
            thought="Research complete",
            status="complete",
            result=research_result[:100] + "..."
            if len(research_result) > 100
            else research_result,
        )
    )

    # Process delegation queue (immutable)
    delegation_queue = state.get("delegation_queue", [])
    remaining_delegations = delegation_queue[1:] if delegation_queue else []

    return {
        "research_result": research_result,
        "thinking_steps": thinking_steps,
        "execution_path": execution_path,
        "delegation_queue": remaining_delegations,
        "current_delegation": remaining_delegations[0]
        if remaining_delegations
        else None,
        "total_llm_calls": state.get("total_llm_calls", 0) + 1,
    }
