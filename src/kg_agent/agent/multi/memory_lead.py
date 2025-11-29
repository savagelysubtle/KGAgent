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

from ...core.logging import logger
from ...services.conversation_memory import get_conversation_memory
from ..llm import create_lm_studio_model

try:
    from copilotkit.langgraph import copilotkit_emit_state  # type: ignore[assignment]
except ImportError:

    async def copilotkit_emit_state(config, state) -> None:
        """Fallback when CopilotKit is not installed."""


from .prompts import MEMORY_LEAD_PROMPT
from .state import MultiAgentState, create_thinking_step


# === Dependencies ===


class MemoryDependencies:
    """Dependencies for the Memory agent."""

    def __init__(self, config: RunnableConfig):
        self.config = config


# === Pydantic AI Agent ===

memory_agent = Agent(
    model=create_lm_studio_model(),
    system_prompt=MEMORY_LEAD_PROMPT,
    deps_type=MemoryDependencies,
    retries=2,
)


# === Tools ===


@memory_agent.tool
async def recall_past_conversations(
    ctx: RunContext[MemoryDependencies], query: str, limit: int = 5
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
                if conv.get("summary"):
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
    ctx: RunContext[MemoryDependencies], fact: str, category: str = "general"
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
    ctx: RunContext[MemoryDependencies], property_name: str, property_value: str
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
            source="agent_conversation",
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

    task = delegation.get("task", "No task specified")

    # Create new lists (immutable pattern)
    thinking_steps = list(state.get("thinking_steps", []))
    thinking_steps.append(
        create_thinking_step(
            agent="memory",
            thought=f"Processing: {task[:60]}...",
            status="thinking",
        )
    )

    execution_path = list(state.get("execution_path", []))
    execution_path.append("memory")

    await copilotkit_emit_state(
        config,
        {
            "thinking_steps": thinking_steps,
            "current_agent": "memory",
            "execution_path": execution_path,
        },
    )

    deps = MemoryDependencies(config=config)

    try:
        result = await memory_agent.run(task, deps=deps)  # type: ignore[arg-type]
        memory_result = result.output
    except Exception as e:
        logger.error(f"Memory agent failed: {e}")
        memory_result = f"Memory failed: {str(e)}"

    thinking_steps.append(
        create_thinking_step(
            agent="memory",
            thought="Memory task complete",
            status="complete",
        )
    )

    delegation_queue = state.get("delegation_queue", [])
    remaining_delegations = delegation_queue[1:] if delegation_queue else []

    return {
        "memory_result": memory_result,
        "thinking_steps": thinking_steps,
        "execution_path": execution_path,
        "delegation_queue": remaining_delegations,
        "current_delegation": remaining_delegations[0]
        if remaining_delegations
        else None,
        "total_llm_calls": state.get("total_llm_calls", 0) + 1,
    }
