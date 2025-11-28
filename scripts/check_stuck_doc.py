"""Script to check stuck documents and diagnose issues."""
import sqlite3
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout.reconfigure(line_buffering=True)

db_path = Path("storage/documents.db")
conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Find stuck documents (in intermediate status for more than 1 hour)
intermediate_statuses = ('processing', 'parsing', 'chunking', 'embedding', 'graphing')

cursor.execute('''
    SELECT id, title, status, chunk_count, created_at, updated_at, metadata
    FROM documents
    WHERE status IN (?, ?, ?, ?, ?)
''', intermediate_statuses)

stuck_docs = cursor.fetchall()

print(f"Found {len(stuck_docs)} document(s) in intermediate status:")
print("=" * 60)

for doc in stuck_docs:
    print(f"\nDocument: {doc['title']}")
    print(f"  ID: {doc['id']}")
    print(f"  Status: {doc['status']}")
    print(f"  Chunks: {doc['chunk_count']}")
    print(f"  Created: {doc['created_at']}")
    print(f"  Updated: {doc['updated_at']}")

    meta = json.loads(doc['metadata']) if doc['metadata'] else {}
    if meta:
        print(f"  Metadata: {json.dumps(meta, indent=4)}")

    # Check if document has vectors
    cursor.execute('SELECT COUNT(*) as cnt FROM document_vectors WHERE document_id = ?', (doc['id'],))
    vector_count = cursor.fetchone()['cnt']
    print(f"  Vectors in tracker: {vector_count}")

print("\n" + "=" * 60)
print("\nRecommendation:")
if stuck_docs:
    print("You can fix stuck documents by:")
    print("1. Marking them as 'failed' if processing truly failed")
    print("2. Re-running the pipeline for these documents")
    print("3. Or manually updating status to 'completed' if vectors exist")

conn.close()

