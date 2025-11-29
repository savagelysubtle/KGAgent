"""Smoke test for Phase 1 multi-agent foundation."""
import sys
import asyncio

sys.stdout.reconfigure(line_buffering=True)

from kg_agent.agent.multi import MultiAgentState, get_multi_agent
from kg_agent.agent.multi.state import (
    create_initial_state,
    create_thinking_step,
    create_delegation_request,
)

# Test 1: Factory functions work
step = create_thinking_step(
    agent="manager",
    thought="Analyzing user request",
    status="thinking"
)
assert step["agent"] == "manager"
assert "timestamp" in step
print("✅ ThinkingStep factory works")

delegation = create_delegation_request(
    target="research",
    task="Search for Python tutorials"
)
assert delegation["target"] == "research"
assert delegation["priority"] == 1  # Default
print("✅ DelegationRequest factory works")

# Test 2: Graph compiles without error
graph = get_multi_agent(use_checkpointer=False, force_recreate=True)
print("✅ Graph compiled")

# Test 3: Initial state can be created
initial = create_initial_state("Hello, what can you do?")
assert "messages" in initial
assert initial["current_agent"] == "manager"
print("✅ Initial state created")

# Test 4: Graph can be invoked (async since nodes are async)
async def test_invocation():
    result = await graph.ainvoke(initial)
    assert result.get("should_end") is True
    return result

result = asyncio.run(test_invocation())
print("✅ Graph invocation succeeded")

print("\n🎉 Phase 1 complete!")
