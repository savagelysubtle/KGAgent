"""Script to check all documents in the database."""
import sqlite3
import json
from pathlib import Path

db_path = Path("storage/documents.db")
if not db_path.exists():
    print(f"Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Get ALL documents
cursor.execute('SELECT id, title, source_type, status, chunk_count, metadata, updated_at FROM documents ORDER BY updated_at DESC')
docs = cursor.fetchall()

print(f'Total documents: {len(docs)}')
print('='*100)
for doc in docs:
    meta = json.loads(doc['metadata']) if doc['metadata'] else {}
    orig_file = meta.get('original_filename', 'N/A')
    title = doc['title'][:40] if doc['title'] else 'None'
    print(f'Title: {title:<40} | Status: {doc["status"]:<12} | Chunks: {doc["chunk_count"]:<6} | Orig: {orig_file}')

conn.close()

