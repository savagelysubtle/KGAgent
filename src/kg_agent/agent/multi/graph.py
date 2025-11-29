"""LangGraph StateGraph - Full Implementation.

This module defines the complete multi-agent orchestration graph with:
- Manager node for intent analysis and delegation
- Four specialist nodes (research, memory, knowledge, documents)
- Synthesize node for combining results
- Conditional routing based on delegation queue
- Checkpointing for session persistence

IMPORTANT: MultiAgentState is a TypedDict, so:
- Use state["field"] or state.get("field"), not state.field
- Return dict from nodes (partial state updates)
"""

from typing import Any, AsyncIterator, Literal, Optional, Tuple

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ...core.logging import logger
from .document_lead import documents_node
from .knowledge_lead import knowledge_node
from .manager import manager_node, synthesize_node
from .memory_lead import memory_node
from .research_lead import research_node
from .state import MultiAgentState, create_initial_state

# Node type hints
NodeName = Literal[
    "manager", "research", "memory", "knowledge", "documents", "synthesize"
]


# === Routing Functions ===


def route_from_manager(state: MultiAgentState) -> str:
    """
    Route from Manager based on delegation decisions.

    Routes to:
    - END: If should_end is True (clarification needed or no delegations)
    - synthesize: If delegation_queue is empty (nothing to do)
    - First specialist: If delegations are queued
    """
    logger.debug(
        f"Routing from manager. should_end={state.get('should_end')}, "
        f"queue={len(state.get('delegation_queue') or [])}"
    )

    # End conditions
    if state.get("should_end"):
        logger.debug("Routing to END (should_end=True)")
        return END

    if state.get("final_response"):
        logger.debug("Routing to END (final_response set)")
        return END

    # Check delegation queue
    delegation_queue = state.get("delegation_queue", [])
    if not delegation_queue:
        logger.debug("Routing to synthesize (empty queue)")
        return "synthesize"

    # Route to first delegation target
    first_delegation = delegation_queue[0]
    first_target = first_delegation.get("target", "synthesize")
    logger.debug(f"Routing to specialist: {first_target}")
    return first_target


def route_after_specialist(state: MultiAgentState) -> str:
    """
    Route after a specialist completes.

    Routes to:
    - Next specialist: If more delegations in queue
    - synthesize: If all delegations processed
    """
    remaining = state.get("delegation_queue", [])

    logger.debug(f"Routing after specialist. Remaining delegations: {len(remaining)}")

    # Check if more delegations to process
    if remaining:
        next_delegation = remaining[0]
        next_target = next_delegation.get("target", "synthesize")
        logger.debug(f"Routing to next specialist: {next_target}")
        return next_target

    # All done - synthesize
    logger.debug("Routing to synthesize (queue empty)")
    return "synthesize"


# === Graph Builder ===


def create_multi_agent_graph(
    checkpointer: Optional[MemorySaver] = None,
) -> CompiledStateGraph:
    """
    Create and compile the multi-agent StateGraph.

    Graph Structure:

    START → manager → [research|memory|knowledge|documents]* → synthesize → END

    The manager can delegate to one or more specialists sequentially.
    After all specialists complete, results are synthesized.

    Args:
        checkpointer: Optional checkpointer for session persistence.
                     Use MemorySaver for in-memory (dev/test).

    Returns:
        Compiled StateGraph ready for invocation.
    """
    logger.info("Creating multi-agent graph")

    # Create graph with state type
    graph = StateGraph(MultiAgentState)

    # === Add Nodes ===

    graph.add_node("manager", manager_node)
    graph.add_node("research", research_node)
    graph.add_node("memory", memory_node)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("documents", documents_node)
    graph.add_node("synthesize", synthesize_node)

    # === Set Entry Point ===

    graph.set_entry_point("manager")

    # === Add Conditional Edges from Manager ===

    graph.add_conditional_edges(
        source="manager",
        path=route_from_manager,
        path_map={
            "research": "research",
            "memory": "memory",
            "knowledge": "knowledge",
            "documents": "documents",
            "synthesize": "synthesize",
            END: END,
        },
    )

    # === Add Conditional Edges from Specialists ===

    # After each specialist, check if more delegations or go to synthesize
    for specialist in ["research", "memory", "knowledge", "documents"]:
        graph.add_conditional_edges(
            source=specialist,
            path=route_after_specialist,
            path_map={
                "research": "research",
                "memory": "memory",
                "knowledge": "knowledge",
                "documents": "documents",
                "synthesize": "synthesize",
            },
        )

    # === Synthesize → END ===

    graph.add_edge("synthesize", END)

    # === Compile ===

    if checkpointer:
        logger.info(f"Compiling graph with checkpointer: {type(checkpointer).__name__}")
        compiled = graph.compile(checkpointer=checkpointer)
    else:
        logger.info("Compiling graph without checkpointer")
        compiled = graph.compile()

    return compiled


# === Singleton Instance ===

_multi_agent_graph: Optional[CompiledStateGraph] = None
_graph_checkpointer: Optional[MemorySaver] = None


def get_multi_agent(
    use_checkpointer: bool = True,
    force_recreate: bool = False,
) -> CompiledStateGraph:
    """
    Get or create the singleton multi-agent graph.

    Args:
        use_checkpointer: Enable session persistence (MemorySaver)
        force_recreate: Force recreation of the graph (useful for testing)

    Returns:
        Compiled StateGraph instance
    """
    global _multi_agent_graph, _graph_checkpointer

    if _multi_agent_graph is None or force_recreate:
        if use_checkpointer:
            _graph_checkpointer = MemorySaver()
            _multi_agent_graph = create_multi_agent_graph(_graph_checkpointer)
        else:
            _multi_agent_graph = create_multi_agent_graph(None)

    return _multi_agent_graph


# === Utility Functions ===


async def invoke_multi_agent(
    user_message: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    config: Optional[RunnableConfig] = None,
) -> dict:
    """
    Convenience function to invoke the multi-agent graph.

    Args:
        user_message: The user's input
        user_id: Optional user identifier
        session_id: Optional session ID for checkpointing
        config: Optional LangGraph config (for CopilotKit integration)

    Returns:
        Final state dict with results
    """
    graph = get_multi_agent()
    initial_state = create_initial_state(
        user_message=user_message,
        user_id=user_id,
        session_id=session_id,
    )

    # Build config with thread_id for checkpointing
    run_config: RunnableConfig = dict(config) if config else {}  # type: ignore[assignment]
    if session_id:
        run_config["configurable"] = run_config.get("configurable", {})  # type: ignore[typeddict-item]
        run_config["configurable"]["thread_id"] = session_id  # type: ignore[typeddict-item]

    logger.info(f"Invoking multi-agent graph for: {user_message[:50]}...")
    result = await graph.ainvoke(initial_state, config=run_config)
    logger.info("Multi-agent graph invocation complete")

    return result


async def stream_multi_agent(
    user_message: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    config: Optional[RunnableConfig] = None,
) -> AsyncIterator[Tuple[str, Any]]:
    """
    Stream execution of the multi-agent graph.

    Yields state updates as the graph executes.

    Args:
        user_message: The user's input
        user_id: Optional user identifier
        session_id: Optional session ID
        config: Optional LangGraph config

    Yields:
        Tuples of (node_name, state_update)
    """
    graph = get_multi_agent()
    initial_state = create_initial_state(
        user_message=user_message,
        user_id=user_id,
        session_id=session_id,
    )

    run_config: RunnableConfig = dict(config) if config else {}  # type: ignore[assignment]
    if session_id:
        run_config["configurable"] = run_config.get("configurable", {})  # type: ignore[typeddict-item]
        run_config["configurable"]["thread_id"] = session_id  # type: ignore[typeddict-item]

    logger.info(f"Streaming multi-agent graph for: {user_message[:50]}...")

    async for event in graph.astream(initial_state, config=run_config):
        for node_name, state_update in event.items():
            logger.debug(f"Stream event from {node_name}")
            yield (node_name, state_update)

    logger.info("Multi-agent graph stream complete")
