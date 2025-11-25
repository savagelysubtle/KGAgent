"""Register the already-processed GPT export in the document tracker."""
import sys
sys.path.insert(0, ".")

from src.kg_agent.services.document_tracker import get_document_tracker, DocumentStatus
from src.kg_agent.services.vector_store import get_vector_store

# Initialize services
doc_tracker = get_document_tracker()
vector_store = get_vector_store()

# Get all vector IDs that match our GPT export
print("Counting vectors in ChromaDB...")
total_count = vector_store.count()
print(f"Total vectors in collection: {total_count}")

# Create the document record
print("\nCreating document record...")
doc = doc_tracker.create_document(
    title="GPT Chat Export - 1257 Conversations",
    source_type="gpt_chat_export",
    file_path="data/raw/upload_20251125_004232_75d09ef4/31e06f7d89feb99a.html",
    metadata={
        "conversation_count": 1257,
        "original_file_size": 276172744,
        "word_count": 5802123,
    }
)

# Update status
doc_tracker.update_document(
    doc.id,
    status=DocumentStatus.COMPLETED,
    chunk_count=29715,
)

# Get vector IDs - they should have doc_id in metadata
# Since we just added them, let's get all IDs and add them
all_ids = vector_store.get_all_ids()
print(f"Found {len(all_ids)} vector IDs")

# Add vector IDs in batches
batch_size = 1000
for i in range(0, len(all_ids), batch_size):
    batch = all_ids[i:i+batch_size]
    doc_tracker.add_vector_ids(doc.id, batch)
    print(f"  Added batch {i//batch_size + 1}/{(len(all_ids) + batch_size - 1)//batch_size}")

print(f"\n=== Document Registered ===")
print(f"Document ID: {doc.id}")
print(f"Title: {doc.title}")
print(f"Chunks: 29715")
print(f"Vectors: {len(all_ids)}")
print("\nYou can now run entity extraction on this document!")

