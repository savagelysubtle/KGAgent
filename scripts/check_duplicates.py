"""Check for duplicate or orphaned documents."""
import sqlite3
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

db_path = Path("storage/documents.db")
conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Get all documents
cursor.execute('SELECT id, title, status, chunk_count, metadata, created_at FROM documents ORDER BY title, created_at')
docs = cursor.fetchall()

print("All documents:")
print("=" * 80)
for doc in docs:
    meta = json.loads(doc['metadata']) if doc['metadata'] else {}
    job_id = meta.get('job_id', 'N/A')
    orig_file = meta.get('original_filename', 'N/A')
    print(f"ID: {doc['id'][:20]}...")
    print(f"  Title: {doc['title']}")
    print(f"  Status: {doc['status']}, Chunks: {doc['chunk_count']}")
    print(f"  Job ID: {job_id}")
    print(f"  Original filename: {orig_file}")
    print(f"  Created: {doc['created_at']}")
    print()

# Check for duplicates by title
cursor.execute('SELECT title, COUNT(*) as cnt FROM documents GROUP BY title HAVING cnt > 1')
dups = cursor.fetchall()
if dups:
    print("\nDuplicate titles found:")
    for dup in dups:
        print(f"  '{dup['title']}' appears {dup['cnt']} times")

conn.close()

