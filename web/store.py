"""SQLite persistence for the local web chat UI.

Conventions follow :mod:`tgbot.localstore`: one short-lived connection per call,
WAL mode, and a busy timeout, so the Telegram bot and the web UI can run at the
same time. The database is intentionally separate (``data/web.db``) so the two
features never contend for the same file.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'New chat',
    preset      TEXT NOT NULL DEFAULT 'core',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    deleted_at  TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    preset      TEXT,
    status      TEXT NOT NULL DEFAULT 'done',
    elapsed_ms  INTEGER,
    client_id   TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS messages_session_idx ON messages(session_id, id);
CREATE UNIQUE INDEX IF NOT EXISTS messages_client_idx
    ON messages(session_id, client_id) WHERE client_id IS NOT NULL;
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class WebStore:
    """Small synchronous store; calls are fast enough for aiohttp handlers."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    # -- sessions ---------------------------------------------------------
    def create_session(self, title: str = "New chat", preset: str = "core") -> dict[str, Any]:
        sid = uuid.uuid4().hex[:12]
        now = _now()
        clean = (title or "New chat").strip()[:120] or "New chat"
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO sessions (id, title, preset, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (sid, clean, preset or "core", now, now),
            )
        return self.get_session(sid)  # type: ignore[return-value]

    def get_session(
        self,
        sid: str,
        *,
        include_messages: bool = False,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        sql = "SELECT * FROM sessions WHERE id = ?"
        if not include_deleted:
            sql += " AND deleted_at IS NULL"
        with closing(self._connect()) as conn:
            row = conn.execute(sql, (sid,)).fetchone()
            if row is None:
                return None
            session = dict(row)
            if include_messages:
                session["messages"] = [
                    dict(r)
                    for r in conn.execute(
                        "SELECT * FROM messages WHERE session_id = ? ORDER BY id", (sid,)
                    )
                ]
        return session

    def list_sessions(self, q: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        sql = """
            SELECT s.*,
                   (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count,
                   (SELECT m.content FROM messages m WHERE m.session_id = s.id
                    ORDER BY m.id DESC LIMIT 1) AS preview
            FROM sessions s
            WHERE s.deleted_at IS NULL
        """
        params: list[Any] = []
        if q:
            sql += " AND s.title LIKE ?"
            params.append(f"%{q}%")
        sql += " ORDER BY s.updated_at DESC, s.rowid DESC LIMIT ?"
        params.append(max(1, min(int(limit), 500)))
        with closing(self._connect()) as conn:
            return [dict(r) for r in conn.execute(sql, params)]

    def rename_session(self, sid: str, title: str) -> dict[str, Any] | None:
        clean = (title or "").strip()[:120]
        if not clean:
            return None
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? "
                "WHERE id = ? AND deleted_at IS NULL",
                (clean, _now(), sid),
            )
        return self.get_session(sid) if cur.rowcount else None

    def set_preset(self, sid: str, preset: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "UPDATE sessions SET preset = ?, updated_at = ? "
                "WHERE id = ? AND deleted_at IS NULL",
                (preset or "core", _now(), sid),
            )
        return self.get_session(sid) if cur.rowcount else None

    def delete_session(self, sid: str) -> bool:
        now = _now()
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "UPDATE sessions SET deleted_at = ?, updated_at = ? "
                "WHERE id = ? AND deleted_at IS NULL",
                (now, now, sid),
            )
        return bool(cur.rowcount)

    def restore_session(self, sid: str) -> bool:
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "UPDATE sessions SET deleted_at = NULL, updated_at = ? "
                "WHERE id = ? AND deleted_at IS NOT NULL",
                (_now(), sid),
            )
        return bool(cur.rowcount)

    # -- messages ---------------------------------------------------------
    def add_message(
        self,
        sid: str,
        role: str,
        content: str,
        *,
        preset: str | None = None,
        status: str = "done",
        elapsed_ms: int | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        now = _now()
        try:
            with closing(self._connect()) as conn, conn:
                cur = conn.execute(
                    "INSERT INTO messages "
                    "(session_id, role, content, preset, status, elapsed_ms, client_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (sid, role, content, preset, status, elapsed_ms, client_id, now),
                )
                conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, sid))
                row = conn.execute(
                    "SELECT * FROM messages WHERE id = ?", (cur.lastrowid,)
                ).fetchone()
            return dict(row)  # type: ignore[arg-type]
        except sqlite3.IntegrityError:
            # duplicate client_id: return the row that already exists
            existing = self.find_turn_by_client_id(sid, client_id or "")
            if existing and existing["user"]:
                return existing["user"]
            raise

    def get_messages(self, sid: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY id", (sid,)
                )
            ]

    def history_before(self, sid: str, before_id: int, turns: int) -> list[dict[str, str]]:
        limit = max(0, int(turns)) * 2
        if limit == 0:
            return []
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages "
                "WHERE session_id = ? AND id < ? ORDER BY id DESC LIMIT ?",
                (sid, before_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def find_turn_by_client_id(
        self, sid: str, client_id: str
    ) -> dict[str, Any] | None:
        if not client_id:
            return None
        with closing(self._connect()) as conn:
            user = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND client_id = ? "
                "AND role = 'user' ORDER BY id LIMIT 1",
                (sid, client_id),
            ).fetchone()
            if user is None:
                return None
            reply = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND id > ? "
                "AND role = 'assistant' ORDER BY id LIMIT 1",
                (sid, user["id"]),
            ).fetchone()
        return {
            "user": dict(user),
            "assistant": dict(reply) if reply is not None else None,
        }

    def last_user_message(self, sid: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND role = 'user' "
                "ORDER BY id DESC LIMIT 1",
                (sid,),
            ).fetchone()
        return _row(row)

    def last_assistant_message(self, sid: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND role = 'assistant' "
                "ORDER BY id DESC LIMIT 1",
                (sid,),
            ).fetchone()
        return _row(row)

    def delete_last_assistant(self, sid: str) -> bool:
        """Delete only the reply that follows the latest user turn."""
        last_user = self.last_user_message(sid)
        if last_user is None:
            return False
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT id FROM messages WHERE session_id = ? AND role = 'assistant' "
                "AND id > ? ORDER BY id DESC LIMIT 1",
                (sid, last_user["id"]),
            ).fetchone()
            if row is None:
                return False
            conn.execute("DELETE FROM messages WHERE id = ?", (row["id"],))
        return True

    # -- diagnostics ------------------------------------------------------
    def counts(self) -> dict[str, int]:
        with closing(self._connect()) as conn:
            sessions = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE deleted_at IS NULL"
            ).fetchone()[0]
            messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return {"sessions": int(sessions), "messages": int(messages)}
