"""Smoke test for Phase 3 specialist agents."""

import sys
import inspect

sys.stdout.reconfigure(line_buffering=True)

print("Testing Phase 3 imports...")

# Test 1: Import all specialist nodes
from kg_agent.agent.multi.research_lead import research_node, research_agent
from kg_agent.agent.multi.memory_lead import memory_node, memory_agent
from kg_agent.agent.multi.knowledge_lead import knowledge_node, knowledge_agent
from kg_agent.agent.multi.document_lead import documents_node, document_agent

print("✅ All specialist modules import successfully")

# Test 2: Verify nodes are async
assert inspect.iscoroutinefunction(research_node), "research_node must be async"
assert inspect.iscoroutinefunction(memory_node), "memory_node must be async"
assert inspect.iscoroutinefunction(knowledge_node), "knowledge_node must be async"
assert inspect.iscoroutinefunction(documents_node), "documents_node must be async"
print("✅ All specialist nodes are async functions")

# Test 3: Verify Pydantic AI agents have correct tool counts
# In pydantic-ai v0.2.x, tools are in _function_toolset.tools as a dict
research_tools = research_agent._function_toolset.tools
memory_tools = memory_agent._function_toolset.tools
knowledge_tools = knowledge_agent._function_toolset.tools
document_tools = document_agent._function_toolset.tools

assert len(research_tools) == 3, f"Research should have 3 tools, got {len(research_tools)}"
print(f"✅ Research Lead has {len(research_tools)} tools: {list(research_tools.keys())}")

assert len(memory_tools) == 4, f"Memory should have 4 tools, got {len(memory_tools)}"
print(f"✅ Memory Lead has {len(memory_tools)} tools: {list(memory_tools.keys())}")

assert len(knowledge_tools) == 4, f"Knowledge should have 4 tools, got {len(knowledge_tools)}"
print(f"✅ Knowledge Lead has {len(knowledge_tools)} tools: {list(knowledge_tools.keys())}")

assert len(document_tools) == 5, f"Document should have 5 tools, got {len(document_tools)}"
print(f"✅ Document Lead has {len(document_tools)} tools: {list(document_tools.keys())}")

# Test 4: Import via __init__.py
from kg_agent.agent.multi import (
    research_node as rn,
    memory_node as mn,
    knowledge_node as kn,
    documents_node as dn,
)

print("✅ All nodes exported via __init__.py")

# Test 5: Graph compiles with real specialist nodes
from kg_agent.agent.multi import get_multi_agent

graph = get_multi_agent(use_checkpointer=False)
print("✅ Graph compiles with real specialist nodes")

# Test 6: Verify state factory functions work
from kg_agent.agent.multi.state import create_thinking_step, create_delegation_request

step = create_thinking_step(agent="research", thought="Testing...", status="thinking")
assert step["agent"] == "research"
print("✅ ThinkingStep factory works")

delegation = create_delegation_request(target="memory", task="Test task")
assert delegation["target"] == "memory"
print("✅ DelegationRequest factory works")

print("\n" + "=" * 50)
print("🎉 Phase 3 Complete - All specialist agents implemented!")
print("=" * 50)
print("\nTool Summary:")
print("  Research Lead:  3 tools (search_knowledge_base, search_by_source, get_database_statistics)")
print("  Memory Lead:    4 tools (recall_past_conversations, remember_about_user, add_user_info, get_user_profile_summary)")
print("  Knowledge Lead: 4 tools (create_entity, create_relationship, search_graph, get_graph_stats)")
print("  Document Lead:  5 tools (list_documents, get_document_statistics, delete_document, delete_documents_by_source, clear_all_data)")
print("\nTotal: 16 tools across 4 specialist agents")

