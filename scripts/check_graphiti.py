"""Test Graphiti integration with LM Studio and sentence-transformers."""
import asyncio
import sys
sys.path.insert(0, ".")

from src.kg_agent.services.graphiti_service import get_graphiti_service


async def test_graphiti():
    """Test the GraphitiService."""
    print("=" * 60)
    print("Testing GraphitiService Integration")
    print("=" * 60)
    
    service = get_graphiti_service()
    
    # Test initialization
    print("\n1. Initializing GraphitiService...")
    success = await service.initialize()
    print(f"   Initialization: {'✅ Success' if success else '❌ Failed'}")
    
    if not success:
        print("   Cannot continue without initialization")
        return False
    
    # Test stats
    print("\n2. Getting graph statistics...")
    stats = await service.get_stats()
    print(f"   Status: {stats.get('status')}")
    print(f"   Entities: {stats.get('total_entities', 'N/A')}")
    print(f"   Relationships: {stats.get('total_relationships', 'N/A')}")
    print(f"   Episodes: {stats.get('total_episodes', 'N/A')}")
    
    # Test adding an episode
    print("\n3. Adding test episode...")
    test_content = """
    Alice is a software engineer at TechCorp. She works with Bob, who is a data scientist.
    They are building a machine learning platform together. Alice specializes in Python
    and Bob focuses on deep learning models.
    """
    
    result = await service.add_episode(
        content=test_content,
        name="test_document",
        source_description="Test document for Graphiti integration",
    )
    
    print(f"   Status: {result.get('status')}")
    print(f"   Nodes created: {result.get('nodes_created', 0)}")
    print(f"   Edges created: {result.get('edges_created', 0)}")
    
    if result.get('entities'):
        print("   Entities extracted:")
        for entity in result['entities'][:5]:
            labels = ", ".join(entity.get('labels', ['Unknown']))
            print(f"     - {entity['name']} [{labels}]")
    
    if result.get('relationships'):
        print("   Relationships extracted:")
        for rel in result['relationships'][:5]:
            print(f"     - {rel['fact']}")
    
    # Test search
    print("\n4. Testing search...")
    search_result = await service.search("Alice software engineer", num_results=5)
    
    print(f"   Status: {search_result.get('status')}")
    if search_result.get('nodes'):
        print("   Found nodes:")
        for node in search_result['nodes'][:3]:
            labels = ", ".join(node.get('labels', ['Unknown']))
            print(f"     - {node['name']} [{labels}]")
    
    if search_result.get('edges'):
        print("   Found relationships:")
        for edge in search_result['edges'][:3]:
            print(f"     - {edge['fact']}")
    
    # Cleanup
    print("\n5. Closing service...")
    await service.close()
    print("   ✅ Service closed")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_graphiti())
    sys.exit(0 if success else 1)

