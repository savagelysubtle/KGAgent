"""Manager Agent - Orchestrator for the multi-agent system.

The Manager:
1. Analyzes user intent
2. Delegates to specialist agents
3. Synthesizes final responses
4. Emits reasoning steps to the UI

IMPORTANT: MultiAgentState is a TypedDict!
- Use state.get("field") or state["field"], NOT state.field
- Return dict from nodes (partial state updates)
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ...core.logging import logger
from ..llm import create_lm_studio_model
from .prompts import MANAGER_SYSTEM_PROMPT
from .state import (
    DelegationRequest,
    MultiAgentState,
    create_delegation_request,
    create_thinking_step,
)

try:
    from copilotkit.langgraph import copilotkit_emit_state  # type: ignore[assignment]
except ImportError:
    # Fallback for local testing without CopilotKit
    async def copilotkit_emit_state(config: Any, state: dict) -> None:
        logger.debug(
            f"copilotkit_emit_state fallback: {state.get('current_agent', 'unknown')}"
        )


# === Delegation Decision Schema ===


class DelegationDecision(BaseModel):
    """Structured output from Manager's intent analysis.

    This Pydantic model constrains the LLM output to a reliable format
    that can be parsed and acted upon without string parsing.
    """

    reasoning: str = Field(
        description="Brief explanation of why these specialists are needed"
    )

    delegations: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of delegations: [{'target': 'research|memory|knowledge|documents', 'task': 'what to do'}]",
    )

    needs_clarification: bool = Field(
        default=False,
        description="True if the request is too vague to delegate",
    )

    clarification_question: str | None = Field(
        default=None,
        description="Question to ask user if needs_clarification is True",
    )


# === Manager Pydantic AI Agent ===
# NOTE: Pydantic AI v0.2.x uses `output_type` for structured output

manager_agent = Agent(
    model=create_lm_studio_model(),
    system_prompt=MANAGER_SYSTEM_PROMPT,
    output_type=DelegationDecision,
    retries=2,
)


# === Intent Classification Prompt ===

DELEGATION_PROMPT = """Analyze the user's request and decide which specialist(s) to delegate to.

User Request: {user_message}

Previous Context (if any):
{context}

Available Specialists:
- research: For searching/finding information in the knowledge base
- memory: For user preferences, past conversations, personal context
- knowledge: For creating entities/relationships in the knowledge graph
- documents: For listing, deleting, or managing documents

Respond with JSON:
{{
  "reasoning": "Brief explanation",
  "delegations": [
    {{"target": "specialist_name", "task": "specific task description"}}
  ],
  "needs_clarification": false,
  "clarification_question": null
}}

Rules:
- You can delegate to MULTIPLE specialists if needed
- Be specific in task descriptions
- If the request is unclear, set needs_clarification=true
"""


# === Helper Functions ===


def state_to_emittable_dict(
    state: MultiAgentState,
    **overrides: Any,
) -> dict[str, Any]:
    """Convert MultiAgentState to a dict for CopilotKit emission.

    Since MultiAgentState is a TypedDict (already a dict), we just need
    to copy relevant fields and apply overrides.

    Args:
        state: The current MultiAgentState (TypedDict)
        **overrides: Values to override in the emitted state

    Returns:
        Dict suitable for copilotkit_emit_state()
    """
    return {
        "current_agent": overrides.get("current_agent", state.get("current_agent")),
        "thinking_steps": overrides.get(
            "thinking_steps", state.get("thinking_steps", [])
        ),
        "delegation_queue": overrides.get(
            "delegation_queue", state.get("delegation_queue", [])
        ),
        "research_result": overrides.get(
            "research_result", state.get("research_result")
        ),
        "memory_result": overrides.get("memory_result", state.get("memory_result")),
        "knowledge_result": overrides.get(
            "knowledge_result", state.get("knowledge_result")
        ),
        "document_result": overrides.get(
            "document_result", state.get("document_result")
        ),
        "final_response": overrides.get("final_response", state.get("final_response")),
        "should_end": overrides.get("should_end", state.get("should_end", False)),
    }


def get_conversation_context(state: MultiAgentState, max_turns: int = 3) -> str:
    """Get recent conversation context for the LLM.

    Args:
        state: Current state (TypedDict)
        max_turns: Maximum conversation turns to include

    Returns:
        Formatted string with recent conversation
    """
    messages = state.get("messages", [])
    recent = messages[-max_turns * 2 :] if messages else []

    context_parts = []
    for msg in recent:
        if isinstance(msg, HumanMessage):
            role = "User"
            content = msg.content
        elif isinstance(msg, AIMessage):
            role = "Assistant"
            content = msg.content
        else:
            continue

        # Truncate long content
        if len(content) > 200:
            content = content[:200] + "..."
        context_parts.append(f"{role}: {content}")

    return "\n".join(context_parts) if context_parts else "No previous context"


# === Main Manager Node ===


async def manager_node(
    state: MultiAgentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """
    Manager node implementation.

    Analyzes user intent and creates delegation queue.
    Emits thinking steps for UI display.

    Args:
        state: Current multi-agent state (TypedDict)
        config: LangGraph runtime config (contains CopilotKit context)

    Returns:
        State updates dict including delegation_queue
    """
    logger.info("Manager node executing")

    # Track execution (copy list to avoid mutation)
    execution_path = list(state.get("execution_path", []))
    execution_path.append("manager")

    # Get the latest user message
    user_message = ""
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, HumanMessage):
            user_message = last_msg.content

    if not user_message:
        logger.warning("No user message found in state")
        return {
            "final_response": "I didn't receive a message. How can I help you?",
            "should_end": True,
            "execution_path": execution_path,
        }

    # Step 1: Emit "analyzing" thinking step
    thinking_steps = list(state.get("thinking_steps", []))
    truncated_msg = (
        user_message[:100] + "..." if len(user_message) > 100 else user_message
    )
    thinking_steps.append(
        create_thinking_step(
            agent="manager",
            thought=f"Analyzing request: '{truncated_msg}'",
            status="thinking",
        )
    )

    # Emit state to UI
    await copilotkit_emit_state(
        config,
        state_to_emittable_dict(
            state,
            thinking_steps=thinking_steps,
            current_agent="manager",
        ),
    )

    # Step 2: Get conversation context
    context = get_conversation_context(state)

    # Step 3: Run LLM to decide delegations
    prompt = DELEGATION_PROMPT.format(
        user_message=user_message,
        context=context,
    )

    try:
        result = await manager_agent.run(prompt)
        decision: DelegationDecision = result.output

        logger.info(f"Manager decision: {decision.reasoning}")
        logger.info(f"Delegations: {decision.delegations}")

    except Exception as e:
        logger.error(f"Manager LLM failed: {e}")
        # Fallback: try research for any query
        decision = DelegationDecision(
            reasoning="Defaulting to research due to analysis error",
            delegations=[{"target": "research", "task": user_message}],
            needs_clarification=False,
        )

    # Step 4: Handle clarification needed
    if decision.needs_clarification:
        thinking_steps.append(
            create_thinking_step(
                agent="manager",
                thought=f"Need clarification: {decision.clarification_question}",
                status="complete",
            )
        )

        return {
            "thinking_steps": thinking_steps,
            "final_response": decision.clarification_question
            or "Could you please clarify your request?",
            "should_end": True,
            "execution_path": execution_path,
            "total_llm_calls": state.get("total_llm_calls", 0) + 1,
        }

    # Step 5: Create delegation queue
    delegation_queue: list[DelegationRequest] = []
    valid_targets: set[str] = {"research", "memory", "knowledge", "documents"}

    for d in decision.delegations:
        target = d.get("target", "").lower()
        task = d.get("task", user_message)

        if target in valid_targets:
            delegation_queue.append(
                create_delegation_request(
                    target=target,  # type: ignore[arg-type]
                    task=task,
                    context=context,
                )
            )

            # Emit delegation step
            truncated_task = task[:80] + "..." if len(task) > 80 else task
            thinking_steps.append(
                create_thinking_step(
                    agent="manager",
                    thought=f"Delegating to {target}: {truncated_task}",
                    status="delegating",
                    action=f"delegate_to_{target}",
                )
            )

    # Emit updated state with delegations
    await copilotkit_emit_state(
        config,
        state_to_emittable_dict(
            state,
            thinking_steps=thinking_steps,
            delegation_queue=delegation_queue,
            current_agent="manager",
        ),
    )

    # No delegations? Provide direct response
    if not delegation_queue:
        thinking_steps.append(
            create_thinking_step(
                agent="manager",
                thought="No specialist needed - responding directly",
                status="complete",
            )
        )

        return {
            "thinking_steps": thinking_steps,
            "final_response": f"I understand you're asking about: {user_message}. However, I'm not sure which specialist can help with this. Could you rephrase or be more specific?",
            "should_end": True,
            "execution_path": execution_path,
            "total_llm_calls": state.get("total_llm_calls", 0) + 1,
        }

    # Return state updates
    first_delegation = delegation_queue[0]
    return {
        "thinking_steps": thinking_steps,
        "delegation_queue": delegation_queue,
        "current_delegation": first_delegation,
        "execution_path": execution_path,
        "total_llm_calls": state.get("total_llm_calls", 0) + 1,
        "current_agent": first_delegation["target"],
    }


# === Synthesize Node ===


async def synthesize_node(
    state: MultiAgentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """
    Synthesize specialist results into final response.

    Called after all delegations complete.

    Args:
        state: Current multi-agent state (TypedDict)
        config: LangGraph runtime config

    Returns:
        State updates with final_response
    """
    logger.info("Synthesize node executing")

    thinking_steps = list(state.get("thinking_steps", []))
    execution_path = list(state.get("execution_path", []))
    execution_path.append("synthesize")

    # Collect all results
    results: list[tuple[str, str]] = []

    research_result = state.get("research_result")
    if research_result:
        results.append(("Research", research_result))

    memory_result = state.get("memory_result")
    if memory_result:
        results.append(("Memory", memory_result))

    knowledge_result = state.get("knowledge_result")
    if knowledge_result:
        results.append(("Knowledge", knowledge_result))

    document_result = state.get("document_result")
    if document_result:
        results.append(("Documents", document_result))

    # Emit synthesizing step
    thinking_steps.append(
        create_thinking_step(
            agent="manager",
            thought=f"Synthesizing {len(results)} specialist result(s)...",
            status="thinking",
        )
    )
    await copilotkit_emit_state(
        config,
        state_to_emittable_dict(state, thinking_steps=thinking_steps),
    )

    # Simple synthesis (Phase 4 will add LLM-based synthesis)
    if not results:
        final_response = "I wasn't able to gather relevant information. Could you try rephrasing your question?"
    elif len(results) == 1:
        # Single result - use directly
        final_response = results[0][1]
    else:
        # Multiple results - combine with headers
        parts = [f"**{name}:**\n{content}" for name, content in results]
        final_response = "\n\n".join(parts)

    # Mark complete
    thinking_steps.append(
        create_thinking_step(
            agent="manager",
            thought="Response ready",
            status="complete",
        )
    )

    await copilotkit_emit_state(
        config,
        state_to_emittable_dict(
            state,
            thinking_steps=thinking_steps,
            final_response=final_response,
            should_end=True,
        ),
    )

    return {
        "thinking_steps": thinking_steps,
        "final_response": final_response,
        "should_end": True,
        "execution_path": execution_path,
        "current_agent": "manager",
    }
