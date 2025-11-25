"""Check and clean up processing jobs."""
import sqlite3
import sys

def main():
    db_path = 'storage/processing_jobs.db'
    print(f"Opening database: {db_path}")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"Tables: {[t[0] for t in tables]}")

        print("\n=== Jobs in database ===")
        cursor.execute('SELECT id, doc_id, status, processed_chunks, total_chunks FROM processing_jobs')
        rows = cursor.fetchall()

        if not rows:
            print("No jobs found in database")
        else:
            for r in rows:
                job_id = r[0][:8] if r[0] else 'None'
                doc_id = r[1][:8] if r[1] else 'None'
                print(f"  ID: {job_id}... | Doc: {doc_id}... | Status: {r[2]} | Progress: {r[3]}/{r[4]}")

        # Option to clean up stuck "running" jobs
        cursor.execute("SELECT COUNT(*) FROM processing_jobs WHERE status = 'running'")
        running_count = cursor.fetchone()[0]

        if running_count > 0:
            print(f"\nFound {running_count} jobs with 'running' status.")
            print("These may be stuck from a previous session.")

            # Reset them to paused so they can be resumed
            cursor.execute("UPDATE processing_jobs SET status = 'paused' WHERE status = 'running'")
            conn.commit()
            print(f"Reset {running_count} stuck 'running' jobs to 'paused' status.")

        conn.close()
        print("\nDone!")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

