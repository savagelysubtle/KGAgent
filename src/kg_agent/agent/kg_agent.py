"""Pydantic AI Knowledge Graph Agent with LM Studio integration."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..core.config import settings
from ..core.logging import logger
from .tools import RAGTools, SearchResult, get_rag_tools


@dataclass
class AgentDependencies:
    """Dependencies injected into the agent at runtime."""

    rag_tools: RAGTools
    user_context: Optional[Dict[str, Any]] = None


# Create the OpenAI-compatible model pointing to LM Studio
def create_lm_studio_model() -> OpenAIChatModel:
    """Create an OpenAI-compatible model configured for LM Studio."""
    # Create provider with LM Studio base URL
    provider = OpenAIProvider(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
    )

    return OpenAIChatModel(
        settings.LLM_MODEL_NAME,
        provider=provider,
    )


# Create the Pydantic AI agent
kg_agent = Agent(
    model=create_lm_studio_model(),
    system_prompt=settings.AGENT_SYSTEM_PROMPT,
    deps_type=AgentDependencies,
    retries=2,
)


@kg_agent.tool
async def search_knowledge_base(
    ctx: RunContext[AgentDependencies], query: str, search_type: str = "hybrid"
) -> str:
    """
    Search the knowledge base for information related to a query.

    Args:
        query: The search query or topic to look for
        search_type: Type of search - 'vector' for semantic search,
                     'graph' for structured search, or 'hybrid' for both

    Returns:
        A formatted string with search results from the knowledge base
    """
    rag_tools = ctx.deps.rag_tools

    try:
        if search_type == "vector":
            results = await rag_tools.search_vectors(query, n_results=5)
            return _format_search_results(results, "Vector Search")

        elif search_type == "graph":
            results = await rag_tools.search_graph(query, limit=5)
            return _format_search_results(results, "Graph Search")

        else:  # hybrid
            hybrid_results = await rag_tools.hybrid_search(query)

            output_parts = []

            if hybrid_results["vector"]:
                output_parts.append(
                    _format_search_results(
                        hybrid_results["vector"], "Vector Search (Semantic)"
                    )
                )

            if hybrid_results["graph"]:
                output_parts.append(
                    _format_search_results(
                        hybrid_results["graph"], "Graph Search (Structured)"
                    )
                )

            if not output_parts:
                return "No results found in the knowledge base for this query."

            return "\n\n".join(output_parts)

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Search failed: {str(e)}"


@kg_agent.tool
async def get_database_statistics(ctx: RunContext[AgentDependencies]) -> str:
    """
    Get current statistics about the knowledge base databases.

    Returns:
        A formatted string with statistics about ChromaDB and Neo4j
    """
    rag_tools = ctx.deps.rag_tools

    try:
        graph_stats = await rag_tools.get_graph_stats()
        vector_stats = await rag_tools.get_vector_stats()

        output = "📊 **Knowledge Base Statistics**\n\n"

        output += "**Vector Store (ChromaDB)**\n"
        output += f"- Collection: {vector_stats.collection_name}\n"
        output += f"- Total Chunks: {vector_stats.total_chunks}\n\n"

        output += "**Knowledge Graph (Neo4j)**\n"
        output += f"- Connected: {'✅ Yes' if graph_stats.connected else '❌ No'}\n"
        output += f"- Total Nodes: {graph_stats.total_nodes}\n"
        output += f"- Total Edges: {graph_stats.total_edges}\n"

        if graph_stats.entity_types:
            output += "- Entity Types:\n"
            for entity_type, count in graph_stats.entity_types.items():
                output += f"  - {entity_type}: {count}\n"

        if graph_stats.relationship_types:
            output += "- Relationship Types:\n"
            for rel_type, count in graph_stats.relationship_types.items():
                output += f"  - {rel_type}: {count}\n"

        return output

    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return f"Failed to get database statistics: {str(e)}"


@kg_agent.tool
async def search_by_source(
    ctx: RunContext[AgentDependencies], source_pattern: str, limit: int = 10
) -> str:
    """
    Search for documents from a specific source in the knowledge graph.

    Args:
        source_pattern: Pattern to match against document sources (e.g., 'example.com')
        limit: Maximum number of results to return

    Returns:
        A formatted string with documents matching the source pattern
    """
    rag_tools = ctx.deps.rag_tools

    try:
        # Use graph search with the source pattern
        results = await rag_tools.search_graph(source_pattern, limit=limit)

        if not results:
            return f"No documents found from sources matching '{source_pattern}'"

        return _format_search_results(results, f"Documents from '{source_pattern}'")

    except Exception as e:
        logger.error(f"Source search failed: {e}")
        return f"Source search failed: {str(e)}"


# ==================== Document Management Tools ====================


@kg_agent.tool
async def list_documents(
    ctx: RunContext[AgentDependencies],
    status: Optional[str] = None,
    source_type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 10,
) -> str:
    """
    List tracked documents in the system.

    Args:
        status: Filter by status (pending, processing, completed, failed, deleted)
        source_type: Filter by source (web_crawl, file_upload, api)
        search: Search in title and source URL
        limit: Maximum number of results (default: 10)

    Returns:
        A formatted list of documents
    """
    rag_tools = ctx.deps.rag_tools

    try:
        documents = await rag_tools.list_documents(
            status=status, source_type=source_type, search=search, limit=limit
        )

        if not documents:
            return "No documents found matching the criteria."

        output = f"📄 **Documents** ({len(documents)} found)\n\n"

        for doc in documents:
            output += f"**{doc.title}**\n"
            output += f"- ID: `{doc.id}`\n"
            output += f"- Source: {doc.source_url or 'N/A'}\n"
            output += f"- Type: {doc.source_type}\n"
            output += f"- Status: {doc.status}\n"
            output += f"- Chunks: {doc.chunk_count}\n"
            output += f"- Created: {doc.created_at}\n\n"

        return output

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        return f"Failed to list documents: {str(e)}"


@kg_agent.tool
async def get_document_statistics(ctx: RunContext[AgentDependencies]) -> str:
    """
    Get statistics about tracked documents in the system.

    Returns:
        A formatted string with document statistics
    """
    rag_tools = ctx.deps.rag_tools

    try:
        stats = await rag_tools.get_document_stats()

        output = "📊 **Document Statistics**\n\n"
        output += f"**Total Documents:** {stats.total_documents}\n\n"

        if stats.by_status:
            output += "**By Status:**\n"
            for status, count in stats.by_status.items():
                emoji = {
                    "completed": "✅",
                    "pending": "⏳",
                    "processing": "🔄",
                    "failed": "❌",
                    "deleted": "🗑️",
                }.get(status, "•")
                output += f"- {emoji} {status}: {count}\n"
            output += "\n"

        if stats.by_source_type:
            output += "**By Source Type:**\n"
            for source_type, count in stats.by_source_type.items():
                emoji = {"web_crawl": "🌐", "file_upload": "📁", "api": "🔌"}.get(
                    source_type, "•"
                )
                output += f"- {emoji} {source_type}: {count}\n"
            output += "\n"

        output += f"**Total Vectors (ChromaDB):** {stats.total_vectors}\n"
        output += f"**Total Graph Nodes (Neo4j):** {stats.total_graph_nodes}\n"

        return output

    except Exception as e:
        logger.error(f"Failed to get document statistics: {e}")
        return f"Failed to get document statistics: {str(e)}"


@kg_agent.tool
async def delete_document(
    ctx: RunContext[AgentDependencies],
    doc_id: str,
    delete_vectors: bool = True,
    delete_graph_nodes: bool = True,
) -> str:
    """
    Delete a document and its associated data from all databases.

    Args:
        doc_id: The document ID to delete
        delete_vectors: Whether to delete vectors from ChromaDB (default: True)
        delete_graph_nodes: Whether to delete nodes from Neo4j (default: True)

    Returns:
        A message describing the deletion result
    """
    rag_tools = ctx.deps.rag_tools

    try:
        result = await rag_tools.delete_document(
            doc_id=doc_id,
            delete_vectors=delete_vectors,
            delete_graph_nodes=delete_graph_nodes,
        )

        if result.success:
            output = "✅ **Document Deleted Successfully**\n\n"
            output += f"- Documents removed: {result.documents_deleted}\n"
            output += f"- Vectors removed: {result.vectors_deleted}\n"
            output += f"- Graph nodes removed: {result.graph_nodes_deleted}\n"
            output += f"\n{result.message}"
        else:
            output = f"❌ **Deletion Failed**\n\n{result.message}"

        return output

    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        return f"Failed to delete document: {str(e)}"


@kg_agent.tool
async def delete_documents_by_source(
    ctx: RunContext[AgentDependencies],
    source_pattern: str,
    delete_vectors: bool = True,
    delete_graph_nodes: bool = True,
) -> str:
    """
    Delete all documents from a specific source.

    Args:
        source_pattern: Pattern to match source URLs (e.g., "example.com")
        delete_vectors: Whether to delete vectors from ChromaDB
        delete_graph_nodes: Whether to delete nodes from Neo4j

    Returns:
        A message describing the deletion result
    """
    rag_tools = ctx.deps.rag_tools

    try:
        result = await rag_tools.delete_by_source(
            source_pattern=source_pattern,
            delete_vectors=delete_vectors,
            delete_graph_nodes=delete_graph_nodes,
        )

        if result.success:
            output = "✅ **Documents Deleted by Source**\n\n"
            output += f"- Source pattern: '{source_pattern}'\n"
            output += f"- Documents removed: {result.documents_deleted}\n"
            output += f"- Vectors removed: {result.vectors_deleted}\n"
            output += f"- Graph nodes removed: {result.graph_nodes_deleted}\n"
            output += f"\n{result.message}"
        else:
            output = f"❌ **Deletion Failed**\n\n{result.message}"

        return output

    except Exception as e:
        logger.error(f"Failed to delete by source: {e}")
        return f"Failed to delete documents by source: {str(e)}"


@kg_agent.tool
async def clear_all_data(
    ctx: RunContext[AgentDependencies], confirm: bool = False
) -> str:
    """
    Clear ALL data from all databases. Use with extreme caution!

    Args:
        confirm: Must be True to actually perform the deletion

    Returns:
        A message describing the result
    """
    if not confirm:
        return "⚠️ **Warning**: This will delete ALL data from ChromaDB, Neo4j, and the document tracker.\n\nTo confirm, call this tool with confirm=True."

    rag_tools = ctx.deps.rag_tools

    try:
        result = await rag_tools.clear_all_data(confirm=True)

        if result.success:
            output = "🗑️ **All Data Cleared**\n\n"
            output += f"- Documents cleared: {result.documents_deleted}\n"
            output += f"- Vectors cleared: {result.vectors_deleted}\n"
            output += f"- Graph nodes cleared: {result.graph_nodes_deleted}\n"
            output += f"\n{result.message}"
        else:
            output = f"❌ **Clear Failed**\n\n{result.message}"

        return output

    except Exception as e:
        logger.error(f"Failed to clear all data: {e}")
        return f"Failed to clear all data: {str(e)}"


# ==================== Entity Management Tools ====================


@kg_agent.tool
async def create_entity(
    ctx: RunContext[AgentDependencies],
    name: str,
    entity_type: str,
    description: Optional[str] = None,
) -> str:
    """
    Create or update an entity in the knowledge graph.

    Use this to add new information about people, organizations, concepts,
    technologies, or other things to the knowledge base.

    Args:
        name: The name of the entity (e.g., "Steve", "Python", "OpenAI")
        entity_type: The type of entity - one of: Person, Organization, Technology,
                     Concept, Location, Event, Product
        description: Optional description of the entity

    Returns:
        A message confirming the entity creation

    Examples:
        - create_entity("Steve", "Person", "The user of this system")
        - create_entity("Python", "Technology", "A programming language")
        - create_entity("OpenAI", "Organization", "An AI research company")
    """
    rag_tools = ctx.deps.rag_tools

    # Validate entity type
    valid_types = [
        "Person",
        "Organization",
        "Technology",
        "Concept",
        "Location",
        "Event",
        "Product",
    ]
    if entity_type not in valid_types:
        # Try to match case-insensitively
        for vt in valid_types:
            if vt.lower() == entity_type.lower():
                entity_type = vt
                break
        else:
            return f"❌ Invalid entity type '{entity_type}'. Valid types are: {', '.join(valid_types)}"

    try:
        result = await rag_tools.create_entity(
            name=name, entity_type=entity_type, description=description
        )

        if result.success:
            output = "✅ **Entity Created**\n\n"
            output += f"- Name: **{name}**\n"
            output += f"- Type: {entity_type}\n"
            if description:
                output += f"- Description: {description}\n"
            output += f"\n{result.message}"
        else:
            output = f"❌ **Entity Creation Failed**\n\n{result.message}"

        return output

    except Exception as e:
        logger.error(f"Failed to create entity: {e}")
        return f"Failed to create entity: {str(e)}"


@kg_agent.tool
async def create_relationship(
    ctx: RunContext[AgentDependencies],
    source_entity: str,
    target_entity: str,
    relationship_type: str,
) -> str:
    """
    Create a relationship between two entities in the knowledge graph.

    Use this to connect entities with meaningful relationships.

    Args:
        source_entity: The name of the source entity
        target_entity: The name of the target entity
        relationship_type: The type of relationship (e.g., "KNOWS", "WORKS_AT",
                          "CREATED", "USES", "LOCATED_IN", "PART_OF")

    Returns:
        A message confirming the relationship creation

    Examples:
        - create_relationship("Steve", "Python", "USES")
        - create_relationship("Steve", "OpenAI", "WORKS_AT")
        - create_relationship("ChatGPT", "OpenAI", "CREATED_BY")
    """
    rag_tools = ctx.deps.rag_tools

    # Normalize relationship type to uppercase with underscores
    rel_type = relationship_type.upper().replace(" ", "_").replace("-", "_")

    try:
        result = await rag_tools.create_relationship(
            source_entity=source_entity,
            target_entity=target_entity,
            relationship_type=rel_type,
        )

        if result.success:
            output = "✅ **Relationship Created**\n\n"
            output += f"- **{source_entity}** -[{rel_type}]-> **{target_entity}**\n"
            output += f"\n{result.message}"
        else:
            output = f"❌ **Relationship Creation Failed**\n\n{result.message}"

        return output

    except Exception as e:
        logger.error(f"Failed to create relationship: {e}")
        return f"Failed to create relationship: {str(e)}"


@kg_agent.tool
async def add_user_info(
    ctx: RunContext[AgentDependencies],
    property_name: str,
    property_value: str,
) -> str:
    """
    Add information about the current user to the knowledge graph.

    Use this when the user wants to store personal information like their name,
    preferences, or other details.

    Args:
        property_name: The name of the property (e.g., "name", "email", "role", "preference")
        property_value: The value of the property

    Returns:
        A message confirming the information was added

    Examples:
        - add_user_info("name", "Steve")
        - add_user_info("role", "developer")
        - add_user_info("favorite_language", "Python")
    """
    rag_tools = ctx.deps.rag_tools

    try:
        # Create or update the User entity with the new property
        properties = {property_name: property_value}

        result = await rag_tools.create_entity(
            name="User",
            entity_type="Person",
            properties=properties,
            description="The current user of this system",
        )

        if result.success:
            output = "✅ **User Info Added**\n\n"
            output += f"- Property: **{property_name}**\n"
            output += f"- Value: **{property_value}**\n"
            output += f"\nI'll remember that your {property_name} is {property_value}!"
        else:
            output = f"❌ **Failed to Add User Info**\n\n{result.message}"

        return output

    except Exception as e:
        logger.error(f"Failed to add user info: {e}")
        return f"Failed to add user info: {str(e)}"


def _format_search_results(results: List[SearchResult], title: str) -> str:
    """Format search results into a readable string."""
    if not results:
        return f"**{title}**: No results found."

    output = f"**{title}** ({len(results)} results)\n\n"

    for i, result in enumerate(results, 1):
        output += f"**Result {i}**\n"

        if result.source:
            output += f"- Source: {result.source}\n"

        if result.score is not None:
            output += f"- Relevance: {result.score:.2%}\n"

        # Truncate text if too long
        text = result.text
        if len(text) > 500:
            text = text[:500] + "..."

        output += f"- Content: {text}\n\n"

    return output


class KGAgent:
    """
    High-level wrapper for the Knowledge Graph Agent.

    Provides a simple interface for running agent queries with
    automatic initialization and error handling.
    """

    def __init__(self):
        self._rag_tools: Optional[RAGTools] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the agent and its dependencies."""
        if self._initialized:
            return True

        try:
            self._rag_tools = get_rag_tools()
            await self._rag_tools.initialize()
            self._initialized = True
            logger.info("KGAgent initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize KGAgent: {e}")
            return False

    async def chat(
        self, message: str, user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send a message to the agent and get a response.

        Args:
            message: The user's message or query
            user_context: Optional context about the user or session

        Returns:
            The agent's response as a string
        """
        if not self._initialized:
            await self.initialize()

        if self._rag_tools is None:
            return "Agent not properly initialized. RAG tools are unavailable."

        try:
            deps = AgentDependencies(
                rag_tools=self._rag_tools, user_context=user_context
            )

            result = await kg_agent.run(message, deps=deps)
            return result.output

        except Exception as e:
            logger.error(f"Agent chat failed: {e}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}"

    async def chat_stream(
        self, message: str, user_context: Optional[Dict[str, Any]] = None
    ):
        """
        Send a message to the agent and stream the response.

        Args:
            message: The user's message or query
            user_context: Optional context about the user or session

        Yields:
            Chunks of the agent's response
        """
        if not self._initialized:
            await self.initialize()

        if self._rag_tools is None:
            yield "Agent not properly initialized. RAG tools are unavailable."
            return

        try:
            deps = AgentDependencies(
                rag_tools=self._rag_tools, user_context=user_context
            )

            async with kg_agent.run_stream(message, deps=deps) as result:
                async for chunk in result.stream():
                    yield chunk

        except Exception as e:
            logger.error(f"Agent stream failed: {e}")
            yield f"Error: {str(e)}"


# Singleton instance
_kg_agent_instance: Optional[KGAgent] = None


def get_kg_agent() -> KGAgent:
    """Get or create the singleton KGAgent instance."""
    global _kg_agent_instance
    if _kg_agent_instance is None:
        _kg_agent_instance = KGAgent()
    return _kg_agent_instance
