import sqlite3
import os
import time
from langgraph.checkpoint.sqlite import SqliteSaver
from src.config import DB_PATH


def get_checkpointer() -> SqliteSaver:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return SqliteSaver(conn)


class SessionStore:
    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_table()

    def _init_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT DEFAULT '',
                created_at REAL,
                last_active REAL,
                message_count INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

    def create(self, session_id: str, title: str = "") -> dict:
        now = time.time()
        self.conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, title, created_at, last_active, message_count) VALUES (?, ?, ?, ?, ?)",
            (session_id, title or "新对话", now, now, 0),
        )
        self.conn.commit()
        return self.get(session_id)

    def get(self, session_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT session_id, title, created_at, last_active, message_count FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "session_id": row[0],
            "title": row[1],
            "created_at": row[2],
            "last_active": row[3],
            "message_count": row[4],
        }

    def list_all(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT session_id, title, created_at, last_active, message_count FROM sessions ORDER BY last_active DESC"
        ).fetchall()
        return [
            {
                "session_id": r[0],
                "title": r[1],
                "created_at": r[2],
                "last_active": r[3],
                "message_count": r[4],
            }
            for r in rows
        ]

    def update(self, session_id: str, **kwargs) -> dict | None:
        valid = {"title", "last_active", "message_count"}
        updates = {k: v for k, v in kwargs.items() if k in valid}
        if not updates:
            return self.get(session_id)
        sets = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [session_id]
        self.conn.execute(f"UPDATE sessions SET {sets} WHERE session_id = ?", values)
        self.conn.commit()
        return self.get(session_id)

    def touch(self, session_id: str, message_count: int | None = None):
        now = time.time()
        if message_count is not None:
            self.conn.execute(
                "UPDATE sessions SET last_active = ?, message_count = ? WHERE session_id = ?",
                (now, message_count, session_id),
            )
        else:
            self.conn.execute(
                "UPDATE sessions SET last_active = ? WHERE session_id = ?",
                (now, session_id),
            )
        self.conn.commit()

    def delete(self, session_id: str) -> bool:
        self.conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        self.conn.commit()
        return True

    def set_title_from_first_message(self, session_id: str, message: str):
        title = message[:30] + ("..." if len(message) > 30 else "")
        self.update(session_id, title=title)


session_store = SessionStore()
