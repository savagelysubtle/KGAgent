"""Script to check pipeline activity data in the database."""
import sqlite3
import json
from pathlib import Path

# Connect to the documents database
db_path = Path("storage/documents.db")
if not db_path.exists():
    print(f"Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Get recent documents
cursor.execute('''
    SELECT id, title, source_type, status, chunk_count, metadata, created_at, updated_at
    FROM documents
    ORDER BY updated_at DESC
    LIMIT 10
''')
docs = cursor.fetchall()

print('=== Recent Documents (ordered by updated_at) ===')
for doc in docs:
    print(f'\nID: {doc["id"][:20]}...')
    print(f'  Title: {doc["title"]}')
    print(f'  Source: {doc["source_type"]}')
    print(f'  Status: {doc["status"]}')
    print(f'  Chunks: {doc["chunk_count"]}')
    print(f'  Created: {doc["created_at"]}')
    print(f'  Updated: {doc["updated_at"]}')
    meta = json.loads(doc['metadata']) if doc['metadata'] else {}
    if meta:
        print(f'  Metadata keys: {list(meta.keys())}')
        if 'original_filename' in meta:
            print(f'  Original filename: {meta["original_filename"]}')

# Count by status
cursor.execute('SELECT status, COUNT(*) as cnt FROM documents GROUP BY status')
status_counts = cursor.fetchall()
print('\n=== Status Distribution ===')
for row in status_counts:
    print(f'  {row["status"]}: {row["cnt"]}')

# Check vector mappings
cursor.execute('SELECT COUNT(*) as cnt FROM document_vectors')
vector_count = cursor.fetchone()['cnt']
print(f'\n=== Vector Mappings: {vector_count} ===')

# Check graph node mappings
cursor.execute('SELECT COUNT(*) as cnt FROM document_graph_nodes')
node_count = cursor.fetchone()['cnt']
print(f'=== Graph Node Mappings: {node_count} ===')

conn.close()

