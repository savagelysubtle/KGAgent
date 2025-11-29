"""Knowledge Lead Agent - Specialist for knowledge graph operations.

Capabilities:
- Create entities
- Create relationships
- Search graph
- Get graph statistics

VERIFIED against tools.py:
- create_entity(name, entity_type, properties, description) -> EntityCreateResult
- create_relationship(source_entity, target_entity, relationship_type, properties) -> EntityCreateResult
- search_graph(query, limit) -> List[SearchResult]
- get_graph_stats() -> GraphStats
"""

from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig
from pydantic_ai import Agent, RunContext

from ...core.logging import logger
from ..llm import create_lm_studio_model
from ..tools import RAGTools, get_rag_tools

try:
    from copilotkit.langgraph import copilotkit_emit_state  # type: ignore[assignment]
except ImportError:

    async def copilotkit_emit_state(config, state) -> None:
        """Fallback when CopilotKit is not installed."""


from .prompts import KNOWLEDGE_LEAD_PROMPT
from .state import MultiAgentState, create_thinking_step


# Valid entity types (matches kg_agent.py:430-437)
ENTITY_TYPES = [
    "Person",
    "Organization",
    "Technology",
    "Concept",
    "Location",
    "Event",
    "Product",
]


# === Dependencies ===


class KnowledgeDependencies:
    """Dependencies for the Knowledge agent."""

    def __init__(self, rag_tools: RAGTools, config: RunnableConfig):
        self.rag_tools = rag_tools
        self.config = config


knowledge_agent = Agent(
    model=create_lm_studio_model(),
    system_prompt=KNOWLEDGE_LEAD_PROMPT,
    deps_type=KnowledgeDependencies,
    retries=2,
)


# === Tools ===


@knowledge_agent.tool
async def create_entity(
    ctx: RunContext[KnowledgeDependencies],
    name: str,
    entity_type: str,
    description: Optional[str] = None,
) -> str:
    """
    Create an entity in the knowledge graph.

    Args:
        name: Entity name (e.g., "Python", "OpenAI")
        entity_type: Person, Organization, Technology, Concept, Location, Event, Product
        description: Optional description
    """
    deps = ctx.deps

    # Validate entity type (case-insensitive match)
    matched_type = None
    for et in ENTITY_TYPES:
        if et.lower() == entity_type.lower():
            matched_type = et
            break

    if not matched_type:
        return f"❌ Invalid type '{entity_type}'. Use: {', '.join(ENTITY_TYPES)}"

    try:
        result = await deps.rag_tools.create_entity(
            name=name,
            entity_type=matched_type,
            description=description,
        )

        if result.success:
            return f"✅ Created **{name}** ({matched_type})" + (
                f": {description}" if description else ""
            )
        return f"❌ Failed: {result.message}"

    except Exception as e:
        logger.error(f"Create entity failed: {e}")
        return f"Error: {str(e)}"


@knowledge_agent.tool
async def create_relationship(
    ctx: RunContext[KnowledgeDependencies],
    source_entity: str,
    target_entity: str,
    relationship_type: str,
) -> str:
    """
    Create a relationship between two entities.

    Args:
        source_entity: Source entity name
        target_entity: Target entity name
        relationship_type: e.g., KNOWS, WORKS_AT, USES, PART_OF
    """
    deps = ctx.deps

    # Normalize relationship type to UPPER_SNAKE_CASE
    rel_type = relationship_type.upper().replace(" ", "_").replace("-", "_")

    try:
        result = await deps.rag_tools.create_relationship(
            source_entity=source_entity,
            target_entity=target_entity,
            relationship_type=rel_type,
        )

        if result.success:
            return (
                f"✅ Created: **{source_entity}** -[{rel_type}]-> **{target_entity}**"
            )
        return f"❌ Failed: {result.message}"

    except Exception as e:
        logger.error(f"Create relationship failed: {e}")
        return f"Error: {str(e)}"


@knowledge_agent.tool
async def search_graph(
    ctx: RunContext[KnowledgeDependencies], query: str, limit: int = 10
) -> str:
    """
    Search the knowledge graph.

    Args:
        query: Search query
        limit: Max results
    """
    deps = ctx.deps

    try:
        results = await deps.rag_tools.search_graph(query, limit=limit)

        if not results:
            return f"No graph results for '{query}'"

        output = f"**Graph Search: '{query}'** ({len(results)} results)\n\n"
        for r in results[:5]:
            output += f"- {r.source}: {r.text[:100]}...\n"
        return output

    except Exception as e:
        return f"Graph search error: {str(e)}"


@knowledge_agent.tool
async def get_graph_stats(ctx: RunContext[KnowledgeDependencies]) -> str:
    """Get knowledge graph statistics."""
    deps = ctx.deps

    try:
        stats = await deps.rag_tools.get_graph_stats()

        # Note: entity_types is Dict[str, int], e.g., {"episodes": 5}
        return f"""📊 **Knowledge Graph Stats**

- Connected: {"✅" if stats.connected else "❌"}
- Total Nodes: {stats.total_nodes}
- Total Edges: {stats.total_edges}
- Entity Types: {stats.entity_types}
"""
    except Exception as e:
        return f"Stats error: {str(e)}"


# === Node Implementation ===


async def knowledge_node(
    state: MultiAgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """
    Knowledge Lead node for LangGraph.

    IMPORTANT: state is a TypedDict - use .get() and return new dicts,
    don't mutate state directly.
    """
    logger.info("Knowledge node executing")

    delegation = state.get("current_delegation")
    if not delegation:
        return {"knowledge_result": "No task provided"}

    task = delegation.get("task", "No task specified")

    # Create new lists (immutable pattern)
    thinking_steps = list(state.get("thinking_steps", []))
    thinking_steps.append(
        create_thinking_step(
            agent="knowledge",
            thought=f"Processing: {task[:60]}...",
            status="thinking",
        )
    )

    execution_path = list(state.get("execution_path", []))
    execution_path.append("knowledge")

    await copilotkit_emit_state(
        config,
        {
            "thinking_steps": thinking_steps,
            "current_agent": "knowledge",
            "execution_path": execution_path,
        },
    )

    rag_tools = get_rag_tools()
    await rag_tools.initialize()

    deps = KnowledgeDependencies(rag_tools=rag_tools, config=config)

    try:
        result = await knowledge_agent.run(task, deps=deps)  # type: ignore[arg-type]
        knowledge_result = result.output
    except Exception as e:
        logger.error(f"Knowledge agent failed: {e}")
        knowledge_result = f"Knowledge failed: {str(e)}"

    thinking_steps.append(
        create_thinking_step(
            agent="knowledge",
            thought="Knowledge task complete",
            status="complete",
        )
    )

    delegation_queue = state.get("delegation_queue", [])
    remaining_delegations = delegation_queue[1:] if delegation_queue else []

    return {
        "knowledge_result": knowledge_result,
        "thinking_steps": thinking_steps,
        "execution_path": execution_path,
        "delegation_queue": remaining_delegations,
        "current_delegation": remaining_delegations[0]
        if remaining_delegations
        else None,
        "total_llm_calls": state.get("total_llm_calls", 0) + 1,
    }
