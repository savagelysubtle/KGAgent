"""Smoke test for Phase 4 LangGraph wiring."""

import sys
import inspect

sys.stdout.reconfigure(line_buffering=True)

print("Testing Phase 4 imports...")

# Test 1: Import all new modules
from kg_agent.agent.multi import (
    # Graph functions
    get_multi_agent,
    invoke_multi_agent,
    stream_multi_agent,
    create_multi_agent_graph,
    # Session management
    SessionManager,
    get_session_manager,
    # Error handling
    with_error_handling,
    MultiAgentError,
    DelegationError,
    SpecialistError,
    ConfigurationError,
    StateValidationError,
    # State
    create_initial_state,
    create_delegation_request,
)

print("✅ All Phase 4 imports successful")

# Test 2: Verify utility functions are async
assert inspect.iscoroutinefunction(invoke_multi_agent), "invoke_multi_agent must be async"
assert inspect.isasyncgenfunction(stream_multi_agent), "stream_multi_agent must be async generator"
print("✅ Utility functions are async")

# Test 3: Test session manager
sm = get_session_manager()
session_id = sm.create_session(user_id="test-user")
assert sm.get_session(session_id) is not None
assert sm.get_session(session_id)["user_id"] == "test-user"
sm.increment_message_count(session_id)
assert sm.get_session(session_id)["message_count"] == 1
sm.delete_session(session_id)
assert sm.get_session(session_id) is None
print("✅ Session manager works")

# Test 4: Test error handling decorator
@with_error_handling("test_agent")
async def test_node(state, config):
    return {"result": "success"}

assert inspect.iscoroutinefunction(test_node)
print("✅ Error handling decorator works")

# Test 5: Test exception classes
try:
    raise SpecialistError("research", "test message")
except MultiAgentError as e:
    assert "research" in str(e)
    assert "test message" in str(e)
print("✅ Exception classes work")

# Test 6: Test routing logic
from kg_agent.agent.multi.graph import route_from_manager, route_after_specialist
from langgraph.graph import END

# Route to specialist
state_with_delegations = {
    "messages": [],
    "delegation_queue": [create_delegation_request(target="research", task="test")],
    "should_end": False,
}
assert route_from_manager(state_with_delegations) == "research"
print("✅ route_from_manager routes to specialist")

# Route to synthesize
state_empty_queue = {
    "messages": [],
    "delegation_queue": [],
    "should_end": False,
}
assert route_from_manager(state_empty_queue) == "synthesize"
print("✅ route_from_manager routes to synthesize")

# Route to END
state_should_end = {
    "messages": [],
    "should_end": True,
}
assert route_from_manager(state_should_end) == END
print("✅ route_from_manager routes to END")

# Route after specialist
state_more_delegations = {
    "messages": [],
    "delegation_queue": [create_delegation_request(target="memory", task="test")],
}
assert route_after_specialist(state_more_delegations) == "memory"
print("✅ route_after_specialist routes to next specialist")

state_no_more = {
    "messages": [],
    "delegation_queue": [],
}
assert route_after_specialist(state_no_more) == "synthesize"
print("✅ route_after_specialist routes to synthesize")

# Test 7: Graph compiles with checkpointer
graph = get_multi_agent(use_checkpointer=True, force_recreate=True)
print("✅ Graph compiles with checkpointer")

# Test 8: Graph compiles without checkpointer
graph_no_cp = get_multi_agent(use_checkpointer=False, force_recreate=True)
print("✅ Graph compiles without checkpointer")

print("")
print("=" * 50)
print("🎉 Phase 4 all tests pass!")
print("=" * 50)

