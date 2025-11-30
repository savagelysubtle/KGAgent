# Phase 3: Specialist Agents - Domain Experts

> **Parent Plan:** [multi_agent.plan.md](./multi_agent.plan.md)
> **Status:** Not Started
> **Estimated Effort:** 4-5 hours
> **Dependencies:** Phase 1 & 2 complete
> **Last Technical Review:** 2025-11-29

---

## 🎯 Objectives

1. Implement 4 specialist agents with focused tool sets
2. Port existing tools from `kg_agent.py` to specialists
3. Add thinking step emissions for each specialist
4. Ensure specialists return structured results

---

## 📋 Prerequisites

- [x] Phase 1 complete (state, prompts) - **VERIFIED:** `state.py` and `prompts.py` exist
- [ ] Phase 2 complete (manager delegation working)
- [x] `RAGTools` class understood (tools.py) - **VERIFIED:** See API reference below

---

## ⚠️ Critical Implementation Notes

### 1. State is a TypedDict - Immutable Pattern Required
```python
# WRONG - Direct mutation (TypedDict fields are not mutable)
state.thinking_steps.append(step)

# CORRECT - Create new list, return in state update
thinking_steps = list(state.get("thinking_steps", []))
thinking_steps.append(step)
return {"thinking_steps": thinking_steps, ...}
```

### 2. Nodes Must Be Async
The `graph.py` currently has sync placeholder nodes, but real implementations
need async to call RAGTools and emit state:
```python
async def research_node(state: MultiAgentState, config: RunnableConfig) -> dict:
    # async code here
```

### 3. Safe TypedDict Access
Always use `.get()` for optional fields:
```python
delegation = state.get("current_delegation")  # Returns None if not set
queue = state.get("delegation_queue", [])     # Returns [] if not set
```

---

## 📊 Tool Distribution Matrix (VERIFIED Line Numbers)

| Tool | Current Location | Target Specialist | RAGTools Method |
|------|-----------------|-------------------|-----------------|
| `search_knowledge_base` | kg_agent.py:48 | Research | `search_vectors`, `search_graph`, `hybrid_search` |
| `search_by_source` | kg_agent.py:145 | Research | `search_graph` |
| `get_database_statistics` | kg_agent.py:103 | Research | `get_graph_stats`, `get_vector_stats` |
| `recall_past_conversations` | kg_agent.py:588 | Memory | `ConversationMemoryService.recall_relevant_context` |
| `remember_about_user` | kg_agent.py:661 | Memory | `ConversationMemoryService.learn_about_user` |
| `add_user_info` | kg_agent.py:523 | Memory | `ConversationMemoryService.learn_user_preference` |
| `get_user_profile_summary` | kg_agent.py:708 | Memory | `ConversationMemoryService.get_user_profile` |
| `create_entity` | kg_agent.py:401 | Knowledge | `create_entity` |
| `create_relationship` | kg_agent.py:471 | Knowledge | `create_relationship` |
| `list_documents` | kg_agent.py:178 | Documents | `list_documents` |
| `get_document_statistics` | kg_agent.py:226 | Documents | `get_document_stats` |
| `delete_document` | kg_agent.py:274 | Documents | `delete_document` |
| `delete_documents_by_source` | kg_agent.py:317 | Documents | `delete_by_source` ⚠️ |
| `clear_all_data` | kg_agent.py:361 | Documents | `clear_all_data` |

> ⚠️ **Note:** RAGTools uses `delete_by_source`, not `delete_documents_by_source`

---

## 📚 RAGTools API Reference (VERIFIED from tools.py)

```python
# Dataclasses returned by RAGTools
@dataclass
class SearchResult:
    text: str
    source: str
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class GraphStats:
    total_nodes: int
    total_edges: int
    entity_types: Dict[str, int]  # e.g., {"episodes": 5}
    relationship_types: Dict[str, int]
    connected: bool

@dataclass
class VectorStats:
    total_chunks: int
    collection_name: str

@dataclass
class DocumentInfo:
    id: str
    title: str
    source_url: Optional[str]
    source_type: str
    status: str
    chunk_count: int
    created_at: str

@dataclass
class DocumentStats:
    total_documents: int
    by_status: Dict[str, int]
    by_source_type: Dict[str, int]
    total_vectors: int
    total_graph_nodes: int

@dataclass
class DeleteResult:
    success: bool
    documents_deleted: int
    vectors_deleted: int
    graph_nodes_deleted: int
    message: str

@dataclass
class EntityCreateResult:
    success: bool
    entity_id: Optional[str]
    message: str
```

### RAGTools Method Signatures (VERIFIED)
```python
async def search_vectors(self, query: str, n_results: int = 5) -> List[SearchResult]
async def search_graph(self, query: str, limit: int = 10) -> List[SearchResult]
async def hybrid_search(self, query: str, vector_results: int = 3, graph_results: int = 3) -> Dict[str, List[SearchResult]]
async def get_graph_stats(self) -> GraphStats
async def get_vector_stats(self) -> VectorStats
async def list_documents(self, status: Optional[str] = None, source_type: Optional[str] = None, search: Optional[str] = None, limit: int = 20) -> List[DocumentInfo]
async def get_document_stats(self) -> DocumentStats
async def delete_document(self, doc_id: str, delete_vectors: bool = True, delete_graph_nodes: bool = True) -> DeleteResult
async def delete_by_source(self, source_pattern: str, delete_vectors: bool = True, delete_graph_nodes: bool = True) -> DeleteResult
async def create_entity(self, name: str, entity_type: str, properties: Optional[Dict] = None, description: Optional[str] = None) -> EntityCreateResult
async def create_relationship(self, source_entity: str, target_entity: str, relationship_type: str, properties: Optional[Dict] = None) -> EntityCreateResult
async def clear_all_data(self, confirm: bool = False) -> DeleteResult
```

---

## 📚 ConversationMemoryService API Reference (VERIFIED from conversation_memory.py)

```python
@dataclass
class UserProfile:
    name: Optional[str] = None
    preferences: Dict[str, Any] = None  # {"key": {"value": x, "source": y, "learned_at": z}}
    topics_of_interest: List[str] = None
    interaction_count: int = 0
    first_interaction: Optional[str] = None  # ISO timestamp
    last_interaction: Optional[str] = None   # ISO timestamp

# Method Signatures
async def recall_relevant_context(self, query: str, limit: int = 5) -> Dict[str, Any]
# Returns: {"graph_results": {...}, "related_conversations": [...], "user_profile": {...}}

async def learn_about_user(self, fact: str, category: str = "general") -> bool
async def learn_user_preference(self, preference_key: str, preference_value: Any, source: str = "conversation") -> bool
def get_user_profile(self) -> UserProfile  # NOT async!
```

---

## 🔍 Task 1: Research Lead Implementation

### File: `src/kg_agent/agent/multi/research_lead.py`

```python
"""Research Lead Agent - Specialist for knowledge base search.

Capabilities:
- Vector search (semantic)
- Graph search (structured)
- Hybrid search (combined)
- Source-based search
- Database statistics

VERIFIED against tools.py: Uses RAGTools methods directly.
"""

from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ...core.config import settings
from ...core.logging import logger
from ..tools import RAGTools, SearchResult, get_rag_tools

try:
    from copilotkit.langgraph import copilotkit_emit_state
except ImportError:
    async def copilotkit_emit_state(config, state):
        pass

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


# === LLM Setup ===

def create_research_llm() -> OpenAIChatModel:
    """Create LLM configured for LM Studio."""
    provider = OpenAIProvider(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
    )
    return OpenAIChatModel(settings.LLM_MODEL_NAME, provider=provider)


# === Pydantic AI Agent ===

research_agent = Agent(
    model=create_research_llm(),
    system_prompt=RESEARCH_LEAD_PROMPT,
    deps_type=ResearchDependencies,
    retries=2,
)


# === Tools ===

@research_agent.tool
async def search_knowledge_base(
    ctx: RunContext[ResearchDependencies],
    query: str,
    search_type: str = "hybrid"
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
    ctx: RunContext[ResearchDependencies],
    source_pattern: str,
    limit: int = 10
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
- Connected: {'✅' if graph_stats.connected else '❌'}
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

    task = delegation["task"]  # DelegationRequest is also a TypedDict

    # Create new lists (immutable pattern for TypedDict)
    thinking_steps = list(state.get("thinking_steps", []))
    thinking_steps.append(create_thinking_step(
        agent="research",
        thought=f"Received task: {task[:80]}...",
        status="thinking",
    ))

    execution_path = list(state.get("execution_path", []))
    execution_path.append("research")

    # Emit state update to UI
    await copilotkit_emit_state(config, {
        "thinking_steps": thinking_steps,
        "current_agent": "research",
        "execution_path": execution_path,
    })

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
        result = await research_agent.run(task, deps=deps)
        research_result = result.output
    except Exception as e:
        logger.error(f"Research agent failed: {e}")
        research_result = f"Research failed: {str(e)}"

    # Add completion step
    thinking_steps.append(create_thinking_step(
        agent="research",
        thought="Research complete",
        status="complete",
        result=research_result[:100] + "..." if len(research_result) > 100 else research_result,
    ))

    # Process delegation queue (immutable)
    delegation_queue = state.get("delegation_queue", [])
    remaining_delegations = delegation_queue[1:] if delegation_queue else []

    return {
        "research_result": research_result,
        "thinking_steps": thinking_steps,
        "execution_path": execution_path,
        "delegation_queue": remaining_delegations,
        "current_delegation": remaining_delegations[0] if remaining_delegations else None,
        "total_llm_calls": state.get("total_llm_calls", 0) + 1,
    }
```

---

## 🧠 Task 2: Memory Lead Implementation

### File: `src/kg_agent/agent/multi/memory_lead.py`

```python
"""Memory Lead Agent - Specialist for user context and conversations.

Capabilities:
- Recall past conversations
- Store user facts
- Manage user profile
- Track preferences

VERIFIED against conversation_memory.py:
- recall_relevant_context(query, limit) -> Dict
- learn_about_user(fact, category) -> bool
- learn_user_preference(preference_key, preference_value, source) -> bool
- get_user_profile() -> UserProfile (SYNC, not async!)
"""

from typing import Any, Dict

from langchain_core.runnables import RunnableConfig
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ...core.config import settings
from ...core.logging import logger
from ...services.conversation_memory import get_conversation_memory

try:
    from copilotkit.langgraph import copilotkit_emit_state
except ImportError:
    async def copilotkit_emit_state(config, state):
        pass

from .prompts import MEMORY_LEAD_PROMPT
from .state import MultiAgentState, create_thinking_step


# === Dependencies ===

class MemoryDependencies:
    """Dependencies for the Memory agent."""
    def __init__(self, config: RunnableConfig):
        self.config = config


# === LLM Setup ===

def create_memory_llm() -> OpenAIChatModel:
    """Create LLM configured for LM Studio."""
    provider = OpenAIProvider(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
    )
    return OpenAIChatModel(settings.LLM_MODEL_NAME, provider=provider)


# === Pydantic AI Agent ===

memory_agent = Agent(
    model=create_memory_llm(),
    system_prompt=MEMORY_LEAD_PROMPT,
    deps_type=MemoryDependencies,
    retries=2,
)


# === Tools ===

@memory_agent.tool
async def recall_past_conversations(
    ctx: RunContext[MemoryDependencies],
    query: str,
    limit: int = 5
) -> str:
    """
    Search past conversations for relevant discussions.

    Args:
        query: What to search for
        limit: Max results
    """
    try:
        memory = get_conversation_memory()
        await memory.initialize()

        context = await memory.recall_relevant_context(query, limit=limit)

        output = f"🔍 **Memory Search: '{query}'**\n\n"

        # Related conversations
        related = context.get("related_conversations", [])
        if related:
            output += f"**Found {len(related)} related conversation(s):**\n"
            for conv in related[:3]:
                output += f"- {conv.get('title', 'Untitled')} ({conv.get('message_count', 0)} msgs)\n"
                if conv.get('summary'):
                    output += f"  Summary: {conv['summary'][:100]}...\n"
        else:
            output += "No related conversations found.\n"

        # User profile hints
        profile = context.get("user_profile", {})
        if profile.get("preferences"):
            output += "\n**Related preferences:**\n"
            for k, v in list(profile["preferences"].items())[:3]:
                val = v.get("value", v) if isinstance(v, dict) else v
                output += f"- {k}: {val}\n"

        return output

    except Exception as e:
        logger.error(f"Memory recall failed: {e}")
        return f"Memory search error: {str(e)}"


@memory_agent.tool
async def remember_about_user(
    ctx: RunContext[MemoryDependencies],
    fact: str,
    category: str = "general"
) -> str:
    """
    Store a fact about the user.

    Args:
        fact: The fact to store (e.g., "User prefers dark mode")
        category: general, work, interests, skills, preferences
    """
    try:
        memory = get_conversation_memory()
        await memory.initialize()

        success = await memory.learn_about_user(fact=fact, category=category)

        if success:
            return f"✅ Remembered: *{fact}* (category: {category})"
        return "❌ Failed to store fact"

    except Exception as e:
        logger.error(f"Memory store failed: {e}")
        return f"Memory error: {str(e)}"


@memory_agent.tool
async def add_user_info(
    ctx: RunContext[MemoryDependencies],
    property_name: str,
    property_value: str
) -> str:
    """
    Add a property to the user's profile.

    Args:
        property_name: Property key (name, email, role, etc.)
        property_value: Property value
    """
    try:
        memory = get_conversation_memory()
        await memory.initialize()

        await memory.learn_user_preference(
            preference_key=property_name,
            preference_value=property_value,
            source="agent_conversation"
        )

        return f"✅ User info added: {property_name} = {property_value}"

    except Exception as e:
        logger.error(f"Add user info failed: {e}")
        return f"Error: {str(e)}"


@memory_agent.tool
async def get_user_profile_summary(ctx: RunContext[MemoryDependencies]) -> str:
    """Get a summary of what's known about the user."""
    try:
        memory = get_conversation_memory()
        await memory.initialize()

        # NOTE: get_user_profile() is SYNC, not async!
        profile = memory.get_user_profile()

        output = "👤 **User Profile**\n\n"

        if profile.name:
            output += f"**Name:** {profile.name}\n"
        output += f"**Interactions:** {profile.interaction_count}\n"

        # Show first/last interaction times if available
        if profile.first_interaction:
            output += f"**First Chat:** {profile.first_interaction[:10]}\n"
        if profile.last_interaction:
            output += f"**Last Chat:** {profile.last_interaction[:10]}\n"

        if profile.preferences:
            output += "\n**Preferences:**\n"
            for k, v in profile.preferences.items():
                val = v.get("value", v) if isinstance(v, dict) else v
                output += f"- {k}: {val}\n"

        if profile.topics_of_interest:
            output += "\n**Interests:** " + ", ".join(profile.topics_of_interest)

        return output

    except Exception as e:
        logger.error(f"Profile summary failed: {e}")
        return f"Error: {str(e)}"


# === Node Implementation ===

async def memory_node(
    state: MultiAgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """
    Memory Lead node for LangGraph.

    IMPORTANT: state is a TypedDict - use .get() and return new dicts,
    don't mutate state directly.
    """
    logger.info("Memory node executing")

    delegation = state.get("current_delegation")
    if not delegation:
        return {"memory_result": "No task provided"}

    task = delegation["task"]

    # Create new lists (immutable pattern)
    thinking_steps = list(state.get("thinking_steps", []))
    thinking_steps.append(create_thinking_step(
        agent="memory",
        thought=f"Processing: {task[:60]}...",
        status="thinking",
    ))

    execution_path = list(state.get("execution_path", []))
    execution_path.append("memory")

    await copilotkit_emit_state(config, {
        "thinking_steps": thinking_steps,
        "current_agent": "memory",
        "execution_path": execution_path,
    })

    deps = MemoryDependencies(config=config)

    try:
        result = await memory_agent.run(task, deps=deps)
        memory_result = result.output
    except Exception as e:
        logger.error(f"Memory agent failed: {e}")
        memory_result = f"Memory failed: {str(e)}"

    thinking_steps.append(create_thinking_step(
        agent="memory",
        thought="Memory task complete",
        status="complete",
    ))

    delegation_queue = state.get("delegation_queue", [])
    remaining_delegations = delegation_queue[1:] if delegation_queue else []

    return {
        "memory_result": memory_result,
        "thinking_steps": thinking_steps,
        "execution_path": execution_path,
        "delegation_queue": remaining_delegations,
        "current_delegation": remaining_delegations[0] if remaining_delegations else None,
        "total_llm_calls": state.get("total_llm_calls", 0) + 1,
    }
```

---

## 📊 Task 3: Knowledge Lead Implementation

### File: `src/kg_agent/agent/multi/knowledge_lead.py`

```python
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
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ...core.config import settings
from ...core.logging import logger
from ..tools import RAGTools, get_rag_tools

try:
    from copilotkit.langgraph import copilotkit_emit_state
except ImportError:
    async def copilotkit_emit_state(config, state):
        pass

from .prompts import KNOWLEDGE_LEAD_PROMPT
from .state import MultiAgentState, create_thinking_step


# Valid entity types (matches kg_agent.py:430-437)
ENTITY_TYPES = ["Person", "Organization", "Technology", "Concept", "Location", "Event", "Product"]


# === Dependencies ===

class KnowledgeDependencies:
    """Dependencies for the Knowledge agent."""
    def __init__(self, rag_tools: RAGTools, config: RunnableConfig):
        self.rag_tools = rag_tools
        self.config = config


# === LLM Setup ===

def create_knowledge_llm() -> OpenAIChatModel:
    """Create LLM configured for LM Studio."""
    provider = OpenAIProvider(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
    )
    return OpenAIChatModel(settings.LLM_MODEL_NAME, provider=provider)


knowledge_agent = Agent(
    model=create_knowledge_llm(),
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
    description: Optional[str] = None
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
            return f"✅ Created **{name}** ({matched_type})" + (f": {description}" if description else "")
        return f"❌ Failed: {result.message}"

    except Exception as e:
        logger.error(f"Create entity failed: {e}")
        return f"Error: {str(e)}"


@knowledge_agent.tool
async def create_relationship(
    ctx: RunContext[KnowledgeDependencies],
    source_entity: str,
    target_entity: str,
    relationship_type: str
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
            return f"✅ Created: **{source_entity}** -[{rel_type}]-> **{target_entity}**"
        return f"❌ Failed: {result.message}"

    except Exception as e:
        logger.error(f"Create relationship failed: {e}")
        return f"Error: {str(e)}"


@knowledge_agent.tool
async def search_graph(
    ctx: RunContext[KnowledgeDependencies],
    query: str,
    limit: int = 10
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

- Connected: {'✅' if stats.connected else '❌'}
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

    task = delegation["task"]

    # Create new lists (immutable pattern)
    thinking_steps = list(state.get("thinking_steps", []))
    thinking_steps.append(create_thinking_step(
        agent="knowledge",
        thought=f"Processing: {task[:60]}...",
        status="thinking",
    ))

    execution_path = list(state.get("execution_path", []))
    execution_path.append("knowledge")

    await copilotkit_emit_state(config, {
        "thinking_steps": thinking_steps,
        "current_agent": "knowledge",
        "execution_path": execution_path,
    })

    rag_tools = get_rag_tools()
    await rag_tools.initialize()

    deps = KnowledgeDependencies(rag_tools=rag_tools, config=config)

    try:
        result = await knowledge_agent.run(task, deps=deps)
        knowledge_result = result.output
    except Exception as e:
        logger.error(f"Knowledge agent failed: {e}")
        knowledge_result = f"Knowledge failed: {str(e)}"

    thinking_steps.append(create_thinking_step(
        agent="knowledge",
        thought="Knowledge task complete",
        status="complete",
    ))

    delegation_queue = state.get("delegation_queue", [])
    remaining_delegations = delegation_queue[1:] if delegation_queue else []

    return {
        "knowledge_result": knowledge_result,
        "thinking_steps": thinking_steps,
        "execution_path": execution_path,
        "delegation_queue": remaining_delegations,
        "current_delegation": remaining_delegations[0] if remaining_delegations else None,
        "total_llm_calls": state.get("total_llm_calls", 0) + 1,
    }
```

---

## 📁 Task 4: Document Lead Implementation

### File: `src/kg_agent/agent/multi/document_lead.py`

```python
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
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ...core.config import settings
from ...core.logging import logger
from ..tools import RAGTools, get_rag_tools

try:
    from copilotkit.langgraph import copilotkit_emit_state
except ImportError:
    async def copilotkit_emit_state(config, state):
        pass

from .prompts import DOCUMENT_LEAD_PROMPT
from .state import MultiAgentState, create_thinking_step


# === Dependencies ===

class DocumentDependencies:
    """Dependencies for the Document agent."""
    def __init__(self, rag_tools: RAGTools, config: RunnableConfig):
        self.rag_tools = rag_tools
        self.config = config


# === LLM Setup ===

def create_document_llm() -> OpenAIChatModel:
    """Create LLM configured for LM Studio."""
    provider = OpenAIProvider(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
    )
    return OpenAIChatModel(settings.LLM_MODEL_NAME, provider=provider)


document_agent = Agent(
    model=create_document_llm(),
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
    limit: int = 10
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

        output = f"📊 **Document Statistics**\n\n"
        output += f"**Total:** {stats.total_documents}\n\n"

        if stats.by_status:
            output += "**By Status:**\n"
            status_emoji = {"completed": "✅", "pending": "⏳", "processing": "🔄", "failed": "❌", "deleted": "🗑️"}
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
    delete_graph_nodes: bool = True
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
    ctx: RunContext[DocumentDependencies],
    source_pattern: str
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
    ctx: RunContext[DocumentDependencies],
    confirm: bool = False
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

    task = delegation["task"]

    # Create new lists (immutable pattern)
    thinking_steps = list(state.get("thinking_steps", []))
    thinking_steps.append(create_thinking_step(
        agent="documents",
        thought=f"Processing: {task[:60]}...",
        status="thinking",
    ))

    execution_path = list(state.get("execution_path", []))
    execution_path.append("documents")

    await copilotkit_emit_state(config, {
        "thinking_steps": thinking_steps,
        "current_agent": "documents",
        "execution_path": execution_path,
    })

    rag_tools = get_rag_tools()
    await rag_tools.initialize()

    deps = DocumentDependencies(rag_tools=rag_tools, config=config)

    try:
        result = await document_agent.run(task, deps=deps)
        document_result = result.output
    except Exception as e:
        logger.error(f"Document agent failed: {e}")
        document_result = f"Document task failed: {str(e)}"

    thinking_steps.append(create_thinking_step(
        agent="documents",
        thought="Document task complete",
        status="complete",
    ))

    delegation_queue = state.get("delegation_queue", [])
    remaining_delegations = delegation_queue[1:] if delegation_queue else []

    return {
        "document_result": document_result,
        "thinking_steps": thinking_steps,
        "execution_path": execution_path,
        "delegation_queue": remaining_delegations,
        "current_delegation": remaining_delegations[0] if remaining_delegations else None,
        "total_llm_calls": state.get("total_llm_calls", 0) + 1,
    }
```

---

## ✅ Phase 3 Definition of Done

- [ ] All 4 specialist files created:
  - [ ] `research_lead.py` - 3 tools (search_knowledge_base, search_by_source, get_database_statistics)
  - [ ] `memory_lead.py` - 4 tools (recall_past_conversations, remember_about_user, add_user_info, get_user_profile_summary)
  - [ ] `knowledge_lead.py` - 4 tools (create_entity, create_relationship, search_graph, get_graph_stats)
  - [ ] `document_lead.py` - 5 tools (list_documents, get_document_statistics, delete_document, delete_documents_by_source, clear_all_data)

- [ ] Each specialist:
  - [ ] Has Pydantic AI agent with tools
  - [ ] Has async node implementation for LangGraph
  - [ ] Uses immutable TypedDict state pattern (`.get()`, return new dicts)
  - [ ] Handles errors gracefully
  - [ ] Uses `create_thinking_step()` factory from state.py

- [ ] All tools correctly call RAGTools/ConversationMemoryService:
  - [ ] Document Lead uses `delete_by_source` (not `delete_documents_by_source`)
  - [ ] Memory Lead calls sync `get_user_profile()` (not awaited)
  - [ ] No duplicate logic - delegate to existing service methods

- [ ] Update `__init__.py` exports:
  ```python
  from .research_lead import research_node
  from .memory_lead import memory_node
  from .knowledge_lead import knowledge_node
  from .document_lead import documents_node
  ```

- [ ] Integration test:
  ```python
  # Test each specialist node can be imported
  from kg_agent.agent.multi.research_lead import research_node
  from kg_agent.agent.multi.memory_lead import memory_node
  from kg_agent.agent.multi.knowledge_lead import knowledge_node
  from kg_agent.agent.multi.document_lead import documents_node

  # Test state factory functions
  from kg_agent.agent.multi.state import create_thinking_step, create_delegation_request
  ```

---

## 📝 Technical Verification Summary

| Item | Status | Notes |
|------|--------|-------|
| `state.py` TypedDict definitions | ✅ Verified | Uses CopilotKitState, factory functions exist |
| `prompts.py` all prompts | ✅ Verified | RESEARCH_LEAD_PROMPT, MEMORY_LEAD_PROMPT, etc. |
| RAGTools method signatures | ✅ Verified | All methods documented with return types |
| ConversationMemoryService API | ✅ Verified | `get_user_profile()` is SYNC |
| pydantic-ai imports | ✅ Verified | `from pydantic_ai import Agent, RunContext` |
| Line numbers in kg_agent.py | ✅ Verified | All line references accurate ±1 line |
| Method name `delete_by_source` | ✅ Verified | NOT `delete_documents_by_source` |

---

## 🔗 Next Phase

→ [Phase 4: LangGraph Wiring](./phase4-langgraph.md) - Complete routing and checkpointing

---

*Created: November 29, 2025*
*Technical Review: November 29, 2025*

