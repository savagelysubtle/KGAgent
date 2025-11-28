"""Clean up stuck and orphaned documents."""
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.stdout.reconfigure(line_buffering=True)

db_path = Path("storage/documents.db")
conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Find documents stuck in intermediate status with 0 chunks
intermediate_statuses = ('processing', 'parsing', 'chunking', 'embedding', 'graphing')

cursor.execute('''
    SELECT id, title, status, chunk_count, created_at, updated_at
    FROM documents
    WHERE status IN (?, ?, ?, ?, ?) AND chunk_count = 0
''', intermediate_statuses)

stuck_docs = cursor.fetchall()

print(f"Found {len(stuck_docs)} stuck document(s) with 0 chunks")
print("=" * 60)

for doc in stuck_docs:
    print(f"\nDocument: {doc['title']}")
    print(f"  ID: {doc['id']}")
    print(f"  Status: {doc['status']}")
    print(f"  Updated: {doc['updated_at']}")

    # Mark as failed
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute('''
        UPDATE documents
        SET status = 'failed',
            error_message = 'Pipeline stalled - marked as failed during cleanup',
            updated_at = ?
        WHERE id = ?
    ''', (now, doc['id']))
    print(f"  -> Marked as FAILED")

# Also find and handle duplicate documents
cursor.execute('''
    SELECT title, GROUP_CONCAT(id) as ids, COUNT(*) as cnt
    FROM documents
    GROUP BY title
    HAVING cnt > 1
''')
dups = cursor.fetchall()

if dups:
    print("\n" + "=" * 60)
    print("Duplicate documents found:")
    for dup in dups:
        print(f"\n  Title: {dup['title']}")
        ids = dup['ids'].split(',')
        print(f"  Document IDs: {ids}")
        print("  (Manual review recommended - not auto-cleaning duplicates)")

conn.commit()
print("\n" + "=" * 60)
print("Cleanup complete!")

# Show final state
cursor.execute('SELECT title, status, chunk_count FROM documents ORDER BY updated_at DESC')
print("\nCurrent documents:")
for row in cursor.fetchall():
    print(f"  {row['title']:<40} {row['status']:<12} {row['chunk_count']} chunks")

conn.close()

