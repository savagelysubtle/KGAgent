"""
Test script for the Pydantic AI Knowledge Graph Agent.

This script tests:
1. RAG tools initialization
2. Vector search
3. Graph search
4. Agent chat (requires LM Studio running)

Usage:
    uv run python scripts/test_agent.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kg_agent.agent.tools import RAGTools, get_rag_tools
from kg_agent.agent.kg_agent import KGAgent, get_kg_agent
from kg_agent.core.config import settings


async def test_rag_tools():
    """Test RAG tools initialization and basic operations."""
    print("\n" + "=" * 60)
    print("Testing RAG Tools")
    print("=" * 60)

    rag_tools = get_rag_tools()

    # Initialize
    print("\n1. Initializing RAG tools...")
    success = await rag_tools.initialize()
    print(f"   Initialization: {'✅ Success' if success else '❌ Failed'}")

    # Get vector stats
    print("\n2. Getting vector store stats...")
    vector_stats = await rag_tools.get_vector_stats()
    print(f"   Collection: {vector_stats.collection_name}")
    print(f"   Total chunks: {vector_stats.total_chunks}")

    # Get graph stats
    print("\n3. Getting graph database stats...")
    graph_stats = await rag_tools.get_graph_stats()
    print(f"   Connected: {graph_stats.connected}")
    print(f"   Total nodes: {graph_stats.total_nodes}")
    print(f"   Total edges: {graph_stats.total_edges}")

    # Test vector search
    print("\n4. Testing vector search...")
    vector_results = await rag_tools.search_vectors("test query", n_results=3)
    print(f"   Results found: {len(vector_results)}")
    for i, r in enumerate(vector_results[:2], 1):
        print(f"   Result {i}: {r.text[:100]}..." if len(r.text) > 100 else f"   Result {i}: {r.text}")

    # Test graph search
    print("\n5. Testing graph search...")
    graph_results = await rag_tools.search_graph("test", limit=3)
    print(f"   Results found: {len(graph_results)}")
    for i, r in enumerate(graph_results[:2], 1):
        print(f"   Result {i}: {r.text[:100]}..." if len(r.text) > 100 else f"   Result {i}: {r.text}")

    # Test hybrid search
    print("\n6. Testing hybrid search...")
    hybrid_results = await rag_tools.hybrid_search("test", vector_results=2, graph_results=2)
    print(f"   Vector results: {len(hybrid_results['vector'])}")
    print(f"   Graph results: {len(hybrid_results['graph'])}")

    return True


async def test_agent_initialization():
    """Test agent initialization."""
    print("\n" + "=" * 60)
    print("Testing Agent Initialization")
    print("=" * 60)

    agent = get_kg_agent()

    print("\n1. Initializing agent...")
    success = await agent.initialize()
    print(f"   Initialization: {'✅ Success' if success else '❌ Failed'}")

    return success


async def test_agent_chat():
    """Test agent chat (requires LM Studio)."""
    print("\n" + "=" * 60)
    print("Testing Agent Chat (requires LM Studio)")
    print("=" * 60)

    print(f"\n   LLM Base URL: {settings.LLM_BASE_URL}")
    print(f"   Model Name: {settings.LLM_MODEL_NAME}")

    agent = get_kg_agent()

    print("\n1. Testing simple chat...")
    try:
        response = await agent.chat("What can you help me with?")
        print(f"   Response: {response[:200]}..." if len(response) > 200 else f"   Response: {response}")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("\n   Make sure LM Studio is running with a model loaded!")
        print(f"   Expected endpoint: {settings.LLM_BASE_URL}")
        return False


async def test_agent_with_tools():
    """Test agent with tool calls."""
    print("\n" + "=" * 60)
    print("Testing Agent with Tools")
    print("=" * 60)

    agent = get_kg_agent()

    print("\n1. Asking agent to get database statistics...")
    try:
        response = await agent.chat("Show me the database statistics")
        print(f"   Response:\n{response}")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Pydantic AI Knowledge Graph Agent Test Suite")
    print("=" * 60)

    results = {}

    # Test RAG tools
    try:
        results["RAG Tools"] = await test_rag_tools()
    except Exception as e:
        print(f"❌ RAG Tools test failed: {e}")
        results["RAG Tools"] = False

    # Test agent initialization
    try:
        results["Agent Init"] = await test_agent_initialization()
    except Exception as e:
        print(f"❌ Agent initialization test failed: {e}")
        results["Agent Init"] = False

    # Test agent chat (optional - requires LM Studio)
    print("\n" + "-" * 60)
    print("The following tests require LM Studio to be running...")
    print("-" * 60)

    try:
        results["Agent Chat"] = await test_agent_chat()
    except Exception as e:
        print(f"❌ Agent chat test failed: {e}")
        results["Agent Chat"] = False

    if results.get("Agent Chat"):
        try:
            results["Agent Tools"] = await test_agent_with_tools()
        except Exception as e:
            print(f"❌ Agent tools test failed: {e}")
            results["Agent Tools"] = False

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")

    all_passed = all(results.values())
    print("\n" + ("✅ All tests passed!" if all_passed else "❌ Some tests failed"))

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

