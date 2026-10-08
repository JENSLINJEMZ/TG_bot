"""Local outbox for media and chats while Supabase is unreachable.

Backed by SQLite (stdlib only): it survives restarts, needs no server, and holds
raw image bytes until the next successful sync. The bot writes here whenever a
Supabase upload/insert fails (or Supabase is not configured at all), and
``tgbot/sync.py`` drains the queue once Supabase is back.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_media (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    mode            TEXT,
    chat_id         INTEGER NOT NULL,
    user_id         INTEGER NOT NULL,
    message_id      INTEGER,
    prompt          TEXT,
    background_rgb  TEXT,
    source_file_id  TEXT,
    elapsed_ms      INTEGER,
    original_path   TEXT NOT NULL,
    original_blob   BLOB NOT NULL,
    original_type   TEXT NOT NULL,
    aux_path        TEXT,
    aux_blob        BLOB,
    aux_type        TEXT,
    result_path     TEXT NOT NULL,
    result_blob     BLOB NOT NULL,
    result_type     TEXT NOT NULL,
    sync_error      TEXT
);

CREATE TABLE IF NOT EXISTS pending_chat (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    chat_id     INTEGER NOT NULL,
    user_id     INTEGER NOT NULL,
    message_id  INTEGER,
    prompt      TEXT NOT NULL,
    reply       TEXT,
    elapsed_ms  INTEGER,
    sync_error  TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PendingMedia:
    id: int
    created_at: str
    mode: str | None
    chat_id: int
    user_id: int
    message_id: int | None
    prompt: str | None
    background_rgb: str | None
    source_file_id: str | None
    elapsed_ms: int | None
    original_path: str
    original_blob: bytes
    original_type: str
    aux_path: str | None
    aux_blob: bytes | None
    aux_type: str | None
    result_path: str
    result_blob: bytes
    result_type: str
    sync_error: str | None = None


@dataclass
class PendingChat:
    id: int
    created_at: str
    chat_id: int
    user_id: int
    message_id: int | None
    prompt: str
    reply: str | None
    elapsed_ms: int | None
    sync_error: str | None = None


class LocalStore:
    """SQLite-backed outbox of rows that still need to reach Supabase."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.parent and str(self.path.parent) not in ("", "."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # -- media ---------------------------------------------------------------
    def save_media(
        self,
        *,
        mode: str | None,
        chat_id: int,
        user_id: int,
        message_id: int | None,
        prompt: str | None,
        background_rgb: str | None,
        source_file_id: str | None,
        elapsed_ms: int | None,
        original_path: str,
        original_blob: bytes,
        original_type: str,
        result_path: str,
        result_blob: bytes,
        result_type: str,
        aux_path: str | None = None,
        aux_blob: bytes | None = None,
        aux_type: str | None = None,
        created_at: str | None = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO pending_media (
                    created_at, mode, chat_id, user_id, message_id, prompt,
                    background_rgb, source_file_id, elapsed_ms,
                    original_path, original_blob, original_type,
                    aux_path, aux_blob, aux_type,
                    result_path, result_blob, result_type
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    created_at or _now(),
                    mode,
                    chat_id,
                    user_id,
                    message_id,
                    prompt,
                    background_rgb,
                    source_file_id,
                    elapsed_ms,
                    original_path,
                    original_blob,
                    original_type,
                    aux_path,
                    aux_blob,
                    aux_type,
                    result_path,
                    result_blob,
                    result_type,
                ),
            )
            return int(cur.lastrowid)

    def save_chat(
        self,
        *,
        chat_id: int,
        user_id: int,
        message_id: int | None,
        prompt: str,
        reply: str | None,
        elapsed_ms: int | None,
        created_at: str | None = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO pending_chat
                    (created_at, chat_id, user_id, message_id, prompt, reply, elapsed_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (created_at or _now(), chat_id, user_id, message_id, prompt, reply, elapsed_ms),
            )
            return int(cur.lastrowid)

    def pending_media(self) -> list[PendingMedia]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM pending_media ORDER BY id").fetchall()
        return [PendingMedia(**dict(row)) for row in rows]

    def pending_chat(self) -> list[PendingChat]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM pending_chat ORDER BY id").fetchall()
        return [PendingChat(**dict(row)) for row in rows]

    def counts(self) -> tuple[int, int]:
        with self._connect() as conn:
            media = conn.execute("SELECT COUNT(*) FROM pending_media").fetchone()[0]
            chat = conn.execute("SELECT COUNT(*) FROM pending_chat").fetchone()[0]
        return int(media), int(chat)

    def mark_media_synced(self, row_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM pending_media WHERE id = ?", (row_id,))

    def mark_chat_synced(self, row_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM pending_chat WHERE id = ?", (row_id,))

    def set_media_error(self, row_id: int, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE pending_media SET sync_error = ? WHERE id = ?", (error[:500], row_id)
            )

    def set_chat_error(self, row_id: int, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE pending_chat SET sync_error = ? WHERE id = ?", (error[:500], row_id)
            )