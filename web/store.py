"""SQLite persistence for the local web chat UI.

Conventions follow :mod:`tgbot.localstore`: one short-lived connection per call,
WAL mode, and a busy timeout, so the Telegram bot and the web UI can run at the
same time. The database is intentionally separate (``data/web.db``) so the two
features never contend for the same file.
"""

from __future__ import annotations

import json
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
    attachments TEXT,
    tool        TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS messages_session_idx ON messages(session_id, id);
CREATE UNIQUE INDEX IF NOT EXISTS messages_client_idx
    ON messages(session_id, client_id) WHERE client_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS uploads (
    id           TEXT PRIMARY KEY,
    session_id   TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    filename     TEXT NOT NULL,
    mime         TEXT NOT NULL,
    size         INTEGER NOT NULL,
    kind         TEXT NOT NULL CHECK (kind IN ('image', 'text')),
    path         TEXT NOT NULL,
    text_content TEXT,
    width        INTEGER,
    height       INTEGER,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS uploads_session_idx ON uploads(session_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _message(row: sqlite3.Row | None) -> dict[str, Any] | None:
    """Decode the JSON attachment list stored beside a message."""
    if row is None:
        return None
    message = dict(row)
    raw = message.get("attachments")
    if raw:
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            value = []
        message["attachments"] = value if isinstance(value, list) else []
    else:
        message["attachments"] = []
    return message


class WebStore:
    """Small synchronous store; calls are fast enough for aiohttp handlers."""

    def __init__(self, path: str | Path, upload_dir: str | Path | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.upload_dir = (
            Path(upload_dir)
            if upload_dir is not None
            else self.path.parent / "web_uploads"
        )
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(SCHEMA)
            self._migrate(conn)
            conn.commit()

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add columns introduced after the first local web database."""
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(messages)")
        }
        if "attachments" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN attachments TEXT")
        if "tool" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN tool TEXT")

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
                    _message(r)
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
        attachments: list[dict[str, Any]] | None = None,
        tool: str | None = None,
    ) -> dict[str, Any]:
        now = _now()
        attachments_json = (
            json.dumps(attachments, ensure_ascii=False) if attachments else None
        )
        try:
            with closing(self._connect()) as conn, conn:
                cur = conn.execute(
                    "INSERT INTO messages "
                    "(session_id, role, content, preset, status, elapsed_ms, client_id, "
                    "attachments, tool, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        sid,
                        role,
                        content,
                        preset,
                        status,
                        elapsed_ms,
                        client_id,
                        attachments_json,
                        tool,
                        now,
                    ),
                )
                conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, sid))
                row = conn.execute(
                    "SELECT * FROM messages WHERE id = ?", (cur.lastrowid,)
                ).fetchone()
            return _message(row)  # type: ignore[return-value]
        except sqlite3.IntegrityError:
            # duplicate client_id: return the row that already exists
            existing = self.find_turn_by_client_id(sid, client_id or "")
            if existing and existing["user"]:
                return existing["user"]
            raise

    def get_messages(self, sid: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            return [
                _message(r)
                for r in conn.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY id", (sid,)
                )
            ]

    def history_before(
        self, sid: str, before_id: int, turns: int
    ) -> list[dict[str, Any]]:
        limit = max(0, int(turns)) * 2
        if limit == 0:
            return []
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT role, content, attachments FROM messages "
                "WHERE session_id = ? AND id < ? ORDER BY id DESC LIMIT ?",
                (sid, before_id, limit),
            ).fetchall()
        return [_message(r) for r in reversed(rows)]  # type: ignore[misc]

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
            "user": _message(user),
            "assistant": _message(reply) if reply is not None else None,
        }

    def last_user_message(self, sid: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND role = 'user' "
                "ORDER BY id DESC LIMIT 1",
                (sid,),
            ).fetchone()
        return _message(row)

    def last_assistant_message(self, sid: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND role = 'assistant' "
                "ORDER BY id DESC LIMIT 1",
                (sid,),
            ).fetchone()
        return _message(row)

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

    # -- uploads ----------------------------------------------------------
    def add_upload(
        self,
        *,
        filename: str,
        mime: str,
        kind: str,
        data: bytes,
        text_content: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        uid = uuid.uuid4().hex[:20]
        original = Path(filename or "attachment").name[:200] or "attachment"
        suffix = Path(original).suffix.lower()[:12]
        stored = f"{uid}{suffix}"
        target = self.upload_dir / stored
        target.write_bytes(data)
        try:
            with closing(self._connect()) as conn, conn:
                conn.execute(
                    "INSERT INTO uploads "
                    "(id, filename, mime, size, kind, path, text_content, width, height, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        uid,
                        original,
                        mime,
                        len(data),
                        kind,
                        stored,
                        text_content,
                        width,
                        height,
                        _now(),
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM uploads WHERE id = ?", (uid,)
                ).fetchone()
        except Exception:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return dict(row)  # type: ignore[arg-type]

    def get_upload(self, uid: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM uploads WHERE id = ?", (uid,)).fetchone()
        return _row(row)

    def attach_uploads(self, sids: list[str], session_id: str) -> None:
        if not sids:
            return
        placeholders = ",".join("?" for _ in sids)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                f"UPDATE uploads SET session_id = ? WHERE id IN ({placeholders}) "
                "AND (session_id IS NULL OR session_id = ?)",
                (session_id, *sids, session_id),
            )

    def upload_path(self, row: dict[str, Any]) -> Path:
        """Resolve a stored upload path without allowing traversal."""
        path = (self.upload_dir / str(row.get("path") or "")).resolve()
        root = self.upload_dir.resolve()
        if path != root and root not in path.parents:
            raise ValueError("invalid upload path")
        return path

    def delete_upload(self, uid: str) -> bool:
        row = self.get_upload(uid)
        if row is None:
            return False
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM uploads WHERE id = ?", (uid,))
        try:
            self.upload_path(row).unlink(missing_ok=True)
        except (OSError, ValueError):
            pass
        return True

    # -- diagnostics ------------------------------------------------------
    def counts(self) -> dict[str, int]:
        with closing(self._connect()) as conn:
            sessions = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE deleted_at IS NULL"
            ).fetchone()[0]
            messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return {"sessions": int(sessions), "messages": int(messages)}
