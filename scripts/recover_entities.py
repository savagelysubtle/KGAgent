"""
Script to recover extracted entities from a completed job and insert them into FalkorDB.
"""
import asyncio
import sys
sys.path.insert(0, ".")

from src.kg_agent.services.processing_job_tracker import get_job_tracker
from src.kg_agent.services.graphiti_service import get_graphiti_service
from src.kg_agent.core.logging import logger
from datetime import datetime, timezone


async def recover_and_insert():
    """Recover entities from job tracker and insert into FalkorDB."""
    tracker = get_job_tracker()
    graphiti = get_graphiti_service()

    # Initialize Graphiti
    print("Initializing GraphitiService...")
    await graphiti.initialize()

    # Get all jobs
    jobs = tracker.get_all_jobs()
    print(f"\nFound {len(jobs)} jobs in tracker:")
    for job in jobs:
        print(f"  - {job.id}: {job.status} ({job.processed_chunks}/{job.total_chunks} chunks, {job.entities_extracted} entities)")

    # Find the completed job with the most entities
    completed_jobs = [j for j in jobs if j.status == "completed"]
    if not completed_jobs:
        print("No completed jobs found!")
        return

    # Get entities and relationships from the most recent completed job
    job = completed_jobs[-1]
    print(f"\nRecovering data from job: {job.id}")

    # Get extracted entities
    entities = tracker.get_extracted_entities(job.id)
    relationships = tracker.get_extracted_relationships(job.id)

    print(f"Found {len(entities)} entities and {len(relationships)} relationships")

    if not entities and not relationships:
        print("No entities or relationships found in job tracker!")
        print("The data may have been cleared after the job completed.")
        return

    # Build episode content from entities and relationships
    episode_parts = []

    for entity in entities[:500]:  # Limit to avoid huge episodes
        name = entity.get("name", "Unknown")
        entity_type = entity.get("type", "Entity")
        desc = entity.get("description", "")
        if desc:
            episode_parts.append(f"{name} is a {entity_type}. {desc}")
        else:
            episode_parts.append(f"{name} is a {entity_type}.")

    for rel in relationships[:500]:  # Limit
        source = rel.get("source_entity", "Unknown")
        target = rel.get("target_entity", "Unknown")
        rel_type = rel.get("type", "RELATED_TO")
        desc = rel.get("description", "")
        if desc:
            episode_parts.append(f"{source} {rel_type} {target}. {desc}")
        else:
            episode_parts.append(f"{source} {rel_type} {target}.")

    if not episode_parts:
        print("No episode content to insert!")
        return

    print(f"\nInserting {len(episode_parts)} facts into FalkorDB...")

    # Insert in batches
    batch_size = 50
    total_nodes = 0
    total_edges = 0

    for i in range(0, len(episode_parts), batch_size):
        batch = episode_parts[i:i+batch_size]
        content = " ".join(batch)

        try:
            result = await graphiti.add_episode(
                content=content,
                name=f"Recovery batch {i//batch_size + 1}",
                source_description=f"Recovered from job {job.id}",
                reference_time=datetime.now(timezone.utc),
                source_type="text",
            )

            nodes = result.get("nodes_created", 0)
            edges = result.get("edges_created", 0)
            total_nodes += nodes
            total_edges += edges
            print(f"  Batch {i//batch_size + 1}: {nodes} nodes, {edges} edges")

        except Exception as e:
            print(f"  Batch {i//batch_size + 1} failed: {e}")

    print(f"\n✅ Recovery complete!")
    print(f"   Total nodes created: {total_nodes}")
    print(f"   Total edges created: {total_edges}")

    # Get final stats
    stats = await graphiti.get_stats()
    print(f"\nFinal graph stats:")
    print(f"   Total entities: {stats.get('total_entities', 0)}")
    print(f"   Total relationships: {stats.get('total_relationships', 0)}")
    print(f"   Total episodes: {stats.get('total_episodes', 0)}")


if __name__ == "__main__":
    asyncio.run(recover_and_insert())

