import sqlite3
import json
from datetime import datetime

DB_PATH = "chat_history.db"


def _get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            title     TEXT,
            timestamp TEXT,
            messages  TEXT
        )
    """)
    conn.commit()
    return conn


def save_session(messages: list[dict], session_id: int = None) -> int:
    """
    session_id is None  → INSERT new row, return new id
    session_id exists   → UPDATE that row only, return same id
    """
    if not messages:
        return None
    try:
        first_user = next(
            (m["content"] for m in messages if m["role"] == "user"), "Untitled"
        )
        title         = first_user[:50] + "..." if len(first_user) > 50 else first_user
        timestamp     = datetime.now().strftime("%d %b %Y, %I:%M %p")
        clean_messages = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        conn = _get_connection()

        if session_id is not None:
            conn.execute(
                "UPDATE sessions SET messages=?, timestamp=?, title=? WHERE id=?",
                (json.dumps(clean_messages), timestamp, title, session_id)
            )
            conn.commit()
            conn.close()
            return session_id
        else:
            cursor = conn.execute(
                "INSERT INTO sessions (title, timestamp, messages) VALUES (?,?,?)",
                (title, timestamp, json.dumps(clean_messages))
            )
            conn.commit()
            new_id = cursor.lastrowid
            conn.close()
            return new_id
    except Exception as e:
        raise RuntimeError(f"Failed to save session: {e}")


def load_all_sessions() -> list[dict]:
    try:
        conn   = _get_connection()
        cursor = conn.execute(
            "SELECT id, title, timestamp FROM sessions ORDER BY id DESC"
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"id": r[0], "title": r[1], "timestamp": r[2]} for r in rows]
    except Exception:
        return []


def load_session_messages(session_id: int) -> list[dict]:
    try:
        conn   = _get_connection()
        cursor = conn.execute(
            "SELECT messages FROM sessions WHERE id=?", (session_id,)
        )
        row = cursor.fetchone()
        conn.close()
        return json.loads(row[0]) if row else []
    except Exception:
        return []


def delete_session(session_id: int):
    try:
        conn = _get_connection()
        conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Failed to delete session: {e}")


def delete_all_sessions():
    try:
        conn = _get_connection()
        conn.execute("DELETE FROM sessions")
        conn.commit()
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Failed to clear history: {e}")