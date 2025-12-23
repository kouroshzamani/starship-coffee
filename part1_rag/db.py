import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "rag.db"

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT,
        chunk_index INTEGER,
        content TEXT,
        embedding BLOB
    )
    """)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("DB initialized at", DB_PATH)
