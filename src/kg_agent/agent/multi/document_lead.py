"""Document Lead Agent - Specialist for document management.

Capabilities:
- List documents
- Get document statistics
- Delete documents
- Clear data (with confirmation)

VERIFIED against tools.py:
- list_documents(status, source_type, search, limit) -> List[DocumentInfo]
- get_document_stats() -> DocumentStats
- delete_document(doc_id, delete_vectors, delete_graph_nodes) -> DeleteResult
- delete_by_source(source_pattern, ...) -> DeleteResult  # NOT delete_documents_by_source!
- clear_all_data(confirm) -> DeleteResult
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


from .prompts import DOCUMENT_LEAD_PROMPT
from .state import MultiAgentState, create_thinking_step

# === Dependencies ===


class DocumentDependencies:
    """Dependencies for the Document agent."""

    def __init__(self, rag_tools: RAGTools, config: RunnableConfig):
        self.rag_tools = rag_tools
        self.config = config


document_agent = Agent(
    model=create_lm_studio_model(),
    system_prompt=DOCUMENT_LEAD_PROMPT,
    deps_type=DocumentDependencies,
    retries=2,
)


# === Tools ===


@document_agent.tool
async def list_documents(
    ctx: RunContext[DocumentDependencies],
    status: Optional[str] = None,
    source_type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 10,
) -> str:
    """
    List documents with optional filters.

    Args:
        status: Filter by status (pending, processing, completed, failed, deleted)
        source_type: Filter by source (web_crawl, file_upload, api)
        search: Search in title/URL
        limit: Max results (default 10)
    """
    deps = ctx.deps

    try:
        docs = await deps.rag_tools.list_documents(
            status=status,
            source_type=source_type,
            search=search,
            limit=limit,
        )

        if not docs:
            return "No documents found matching criteria."

        output = f"📄 **Documents** ({len(docs)} found)\n\n"
        for doc in docs:
            output += f"**{doc.title}**\n"
            output += f"- ID: `{doc.id}`\n"
            output += f"- Status: {doc.status} | Type: {doc.source_type}\n"
            output += f"- Chunks: {doc.chunk_count}\n\n"

        return output

    except Exception as e:
        logger.error(f"List documents failed: {e}")
        return f"Error: {str(e)}"


@document_agent.tool
async def get_document_statistics(ctx: RunContext[DocumentDependencies]) -> str:
    """Get statistics about all documents."""
    deps = ctx.deps

    try:
        stats = await deps.rag_tools.get_document_stats()

        output = "📊 **Document Statistics**\n\n"
        output += f"**Total:** {stats.total_documents}\n\n"

        if stats.by_status:
            output += "**By Status:**\n"
            status_emoji = {
                "completed": "✅",
                "pending": "⏳",
                "processing": "🔄",
                "failed": "❌",
                "deleted": "🗑️",
            }
            for s, count in stats.by_status.items():
                output += f"- {status_emoji.get(s, '•')} {s}: {count}\n"

        if stats.by_source_type:
            output += "\n**By Source:**\n"
            source_emoji = {"web_crawl": "🌐", "file_upload": "📁", "api": "🔌"}
            for st, count in stats.by_source_type.items():
                output += f"- {source_emoji.get(st, '•')} {st}: {count}\n"

        output += f"\n**Vectors:** {stats.total_vectors}"
        output += f"\n**Graph Nodes:** {stats.total_graph_nodes}"

        return output

    except Exception as e:
        return f"Stats error: {str(e)}"


@document_agent.tool
async def delete_document(
    ctx: RunContext[DocumentDependencies],
    doc_id: str,
    delete_vectors: bool = True,
    delete_graph_nodes: bool = True,
) -> str:
    """
    Delete a document and its data.

    Args:
        doc_id: Document ID to delete
        delete_vectors: Also delete from ChromaDB
        delete_graph_nodes: Also delete from graph (limited support with Graphiti)
    """
    deps = ctx.deps

    try:
        result = await deps.rag_tools.delete_document(
            doc_id=doc_id,
            delete_vectors=delete_vectors,
            delete_graph_nodes=delete_graph_nodes,
        )

        if result.success:
            return f"""✅ **Deleted Successfully**
- Documents: {result.documents_deleted}
- Vectors: {result.vectors_deleted}
- Graph nodes: {result.graph_nodes_deleted}
- {result.message}
"""
        return f"❌ {result.message}"

    except Exception as e:
        return f"Delete error: {str(e)}"


@document_agent.tool
async def delete_documents_by_source(
    ctx: RunContext[DocumentDependencies], source_pattern: str
) -> str:
    """
    Delete all documents from a source.

    Args:
        source_pattern: Pattern to match (e.g., 'example.com')
    """
    deps = ctx.deps

    try:
        # IMPORTANT: RAGTools method is `delete_by_source`, not `delete_documents_by_source`
        result = await deps.rag_tools.delete_by_source(source_pattern=source_pattern)

        if result.success:
            return f"✅ Deleted {result.documents_deleted} documents from '{source_pattern}'"
        return f"❌ {result.message}"

    except Exception as e:
        return f"Error: {str(e)}"


@document_agent.tool
async def clear_all_data(
    ctx: RunContext[DocumentDependencies], confirm: bool = False
) -> str:
    """
    ⚠️ Delete ALL data. Requires confirm=True.

    Args:
        confirm: Must be True to execute

    Note: Graph data clearing requires manual FalkorDB reset.
    """
    deps = ctx.deps

    if not confirm:
        return """⚠️ **WARNING: This will delete ALL data!**

This includes:
- All documents in the tracker
- All vectors in ChromaDB
- Graph data requires manual reset (Graphiti limitation)

To proceed, call with `confirm=True`"""

    try:
        result = await deps.rag_tools.clear_all_data(confirm=True)

        if result.success:
            return f"""🗑️ **All Data Cleared**
- Documents: {result.documents_deleted}
- Vectors: {result.vectors_deleted}
- Graph nodes: {result.graph_nodes_deleted}
- {result.message}
"""
        return f"❌ {result.message}"

    except Exception as e:
        return f"Error: {str(e)}"


# === Node Implementation ===


async def documents_node(
    state: MultiAgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """
    Document Lead node for LangGraph.

    IMPORTANT: state is a TypedDict - use .get() and return new dicts,
    don't mutate state directly.
    """
    logger.info("Documents node executing")

    delegation = state.get("current_delegation")
    if not delegation:
        return {"document_result": "No task provided"}

    task = delegation.get("task", "No task specified")

    # Create new lists (immutable pattern)
    thinking_steps = list(state.get("thinking_steps", []))
    thinking_steps.append(
        create_thinking_step(
            agent="documents",
            thought=f"Processing: {task[:60]}...",
            status="thinking",
        )
    )

    execution_path = list(state.get("execution_path", []))
    execution_path.append("documents")

    await copilotkit_emit_state(
        config,
        {
            "thinking_steps": thinking_steps,
            "current_agent": "documents",
            "execution_path": execution_path,
        },
    )

    rag_tools = get_rag_tools()
    await rag_tools.initialize()

    deps = DocumentDependencies(rag_tools=rag_tools, config=config)

    try:
        result = await document_agent.run(task, deps=deps)  # type: ignore[arg-type]
        document_result = result.output
    except Exception as e:
        logger.error(f"Document agent failed: {e}")
        document_result = f"Document task failed: {str(e)}"

    thinking_steps.append(
        create_thinking_step(
            agent="documents",
            thought="Document task complete",
            status="complete",
        )
    )

    delegation_queue = state.get("delegation_queue", [])
    remaining_delegations = delegation_queue[1:] if delegation_queue else []

    return {
        "document_result": document_result,
        "thinking_steps": thinking_steps,
        "execution_path": execution_path,
        "delegation_queue": remaining_delegations,
        "current_delegation": remaining_delegations[0]
        if remaining_delegations
        else None,
        "total_llm_calls": state.get("total_llm_calls", 0) + 1,
    }
