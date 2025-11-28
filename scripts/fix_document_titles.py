"""Script to fix document titles that contain full paths."""
import sqlite3
import json
import os
import sys
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

db_path = Path("storage/documents.db")
if not db_path.exists():
    print(f"Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Find documents with path-like titles
cursor.execute('SELECT id, title, metadata FROM documents')
docs = cursor.fetchall()

fixed_count = 0
for doc in docs:
    title = doc['title']
    doc_id = doc['id']

    # Check if title looks like a path (contains path separators)
    if title and ('/' in title or '\\' in title):
        # Try to get original filename from metadata
        meta = json.loads(doc['metadata']) if doc['metadata'] else {}
        new_title = meta.get('original_filename')

        if not new_title:
            # Extract filename from path
            new_title = os.path.basename(title)

        if new_title and new_title != title:
            print(f"Fixing: '{title}' -> '{new_title}'")
            cursor.execute('UPDATE documents SET title = ? WHERE id = ?', (new_title, doc_id))
            fixed_count += 1

conn.commit()
print(f"\nFixed {fixed_count} document title(s)")

# Verify
cursor.execute('SELECT title, status, chunk_count FROM documents')
for row in cursor.fetchall():
    print(f"  {row['title']} - {row['status']} - {row['chunk_count']} chunks")

conn.close()

