import sqlite3
conn = sqlite3.connect('storage/documents.db')
c = conn.cursor()
c.execute('SELECT COUNT(*) FROM documents')
print('Document count:', c.fetchone()[0])
c.execute('SELECT title, status, chunk_count FROM documents LIMIT 5')
for row in c.fetchall():
    print(row)
conn.close()

