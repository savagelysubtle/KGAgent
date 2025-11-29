"""Shared state definitions for the multi-agent system.

IMPORTANT: CopilotKitState is a TypedDict, not a class!
- Use TypedDict inheritance, not @dataclass
- Use Annotated[] for reducers, not field()
- total=False makes all fields optional with None default
"""

from datetime import datetime
from typing import Literal, TypedDict

from copilotkit.langgraph import CopilotKitState


class ThinkingStep(TypedDict, total=False):
    """
    A single reasoning step from an agent.

    Displayed in the UI via CopilotKit's state rendering.
    Uses TypedDict for JSON serialization compatibility.
    """

    agent: str
    """Which agent produced this step: 'manager', 'research', 'memory', 'knowledge', 'documents'"""

    thought: str
    """What the agent is thinking or doing"""

    action: str  # Optional in total=False
    """Tool name if executing a tool"""

    result: str  # Optional in total=False
    """Result of the action (truncated for display)"""

    status: Literal["thinking", "delegating", "executing", "complete", "error"]
    """Current status of this step (defaults to 'thinking' in factory)"""

    timestamp: str
    """When this step occurred (ISO format)"""

    duration_ms: int  # Optional in total=False
    """How long this step took (set on completion)"""


class DelegationRequest(TypedDict, total=False):
    """A request from Manager to a specialist agent."""

    target: Literal["research", "memory", "knowledge", "documents"]  # Required
    task: str  # Required
    context: str  # Optional
    priority: int  # 1 = normal, 2 = high, defaults to 1


class MultiAgentState(CopilotKitState, total=False):
    """
    Shared state across all agents in the multi-agent hierarchy.

    Extends CopilotKitState (a TypedDict) to enable:
    - Real-time state streaming to UI via emit_state
    - Message history management (inherited from CopilotKitState)
    - Checkpoint persistence

    INHERITED FROM CopilotKitState (do NOT redeclare):
    - messages: Annotated[Sequence[BaseMessage], add_messages]
    - steps: list[ThinkingStep] (for UI progress)

    State Flow:
    1. User message → messages updated
    2. Manager thinks → thinking_steps updated, emit_state
    3. Manager delegates → delegation_queue updated
    4. Specialist executes → specialist_result updated
    5. Manager synthesizes → final_response set
    """

    # NOTE: `messages` is inherited from CopilotKitState with add_messages reducer!
    # NOTE: `steps` is inherited from CopilotKitState for UI progress!

    # === Reasoning Trail (additional to inherited steps) ===
    thinking_steps: list[ThinkingStep]
    """All reasoning steps from all agents - streamed to UI"""

    # === Current Execution Context ===
    current_agent: str
    """Which agent is currently active (default: 'manager')"""

    delegation_queue: list[DelegationRequest]
    """Pending delegations from Manager to specialists"""

    current_delegation: DelegationRequest
    """The delegation currently being processed"""

    # === Results from Specialists ===
    research_result: str
    """Result from Research Lead"""

    memory_result: str
    """Result from Memory Lead"""

    knowledge_result: str
    """Result from Knowledge Lead"""

    document_result: str
    """Result from Document Lead"""

    # === Final Output ===
    final_response: str
    """The synthesized final response to the user"""

    should_end: bool
    """Flag to signal graph completion"""

    # === Error Handling ===
    last_error: str
    """Last error message (if any)"""

    # === Metadata & Telemetry ===
    total_llm_calls: int
    """Total LLM API calls made in this request"""

    execution_path: list[str]
    """Ordered list of nodes visited: ['manager', 'research', 'manager', ...]"""

    start_time: str
    """When this request started processing (ISO format)"""

    # === User Context (passed through from frontend) ===
    user_id: str
    """Current user identifier"""

    session_id: str
    """Current session identifier for checkpointing"""


def create_thinking_step(
    agent: str,
    thought: str,
    status: Literal[
        "thinking", "delegating", "executing", "complete", "error"
    ] = "thinking",
    action: str | None = None,
    result: str | None = None,
) -> ThinkingStep:
    """
    Factory function to create a ThinkingStep with defaults.

    Since TypedDict doesn't support default values, use this factory.
    """
    step: ThinkingStep = {
        "agent": agent,
        "thought": thought,
        "status": status,
        "timestamp": datetime.now().isoformat(),
    }
    if action:
        step["action"] = action
    if result:
        step["result"] = result
    return step


def create_delegation_request(
    target: Literal["research", "memory", "knowledge", "documents"],
    task: str,
    context: str | None = None,
    priority: int = 1,
) -> DelegationRequest:
    """Factory function to create a DelegationRequest with defaults."""
    request: DelegationRequest = {
        "target": target,
        "task": task,
        "priority": priority,
    }
    if context:
        request["context"] = context
    return request


def create_initial_state(
    user_message: str,
    user_id: str | None = None,
    session_id: str | None = None,
) -> dict:
    """
    Create initial state dict for a new multi-agent request.

    Args:
        user_message: The user's input message
        user_id: Optional user identifier
        session_id: Optional session ID for persistence

    Returns:
        Dictionary suitable for graph.invoke()

    Note:
        Returns a dict, not MultiAgentState, because LangGraph
        expects a dict for invoke() and handles the typing internally.
    """
    from langchain_core.messages import HumanMessage

    state: dict = {
        "messages": [HumanMessage(content=user_message)],
        "thinking_steps": [],
        "current_agent": "manager",
        "delegation_queue": [],
        "execution_path": [],
        "total_llm_calls": 0,
        "start_time": datetime.now().isoformat(),
        "should_end": False,
    }
    if user_id:
        state["user_id"] = user_id
    if session_id:
        state["session_id"] = session_id
    return state
