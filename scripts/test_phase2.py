"""Smoke test for Phase 2 multi-agent manager."""
import sys
sys.stdout.reconfigure(line_buffering=True)

print("Testing Phase 2 imports...")

from kg_agent.agent.multi import get_multi_agent
from kg_agent.agent.multi.manager import (
    manager_node,
    synthesize_node,
    DelegationDecision,
)
from kg_agent.agent.multi.state import create_initial_state

print("✅ All imports successful")

# Test DelegationDecision model
decision = DelegationDecision(
    reasoning="Test reasoning",
    delegations=[{"target": "research", "task": "Test task"}],
)
assert decision.reasoning == "Test reasoning"
assert len(decision.delegations) == 1
print("✅ DelegationDecision model works")

# Test graph compilation
graph = get_multi_agent(use_checkpointer=False)
print("✅ Graph compiled with real manager node")

# Check that manager_node is async
import inspect
assert inspect.iscoroutinefunction(manager_node), "manager_node should be async"
assert inspect.iscoroutinefunction(synthesize_node), "synthesize_node should be async"
print("✅ Manager nodes are async coroutines")

print("\n🎉 Phase 2 complete!")

