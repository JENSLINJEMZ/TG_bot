"""Local SQLite store for processed media and chat turns.

This project keeps everything on disk: the original photo(s), the result image
and every chat turn. Backed by SQLite (stdlib only), so it survives restarts,
needs no server and holds raw image bytes.

The project used to buffer rows here only until they could be uploaded to a
remote backend; that backend has been removed, so this database is now the
permanent archive. Databases created back then used ``pending_media`` /
``pending_chat`` tables - they are renamed in place the first time the store is
opened, so no buffered rows are lost.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS media (
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
    result_type     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS media_user_created_idx ON media (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS chats (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    chat_id     INTEGER NOT NULL,
    user_id     INTEGER NOT NULL,
    message_id  INTEGER,
    prompt      TEXT NOT NULL,
    reply       TEXT,
    elapsed_ms  INTEGER
);

CREATE INDEX IF NOT EXISTS chats_user_created_idx ON chats (user_id, created_at DESC);
"""

# old table -> new table (rename in place on first open of an existing database)
_LEGACY_TABLES = {"pending_media": "media", "pending_chat": "chats"}

_MEDIA_COLUMNS = (
    "id, created_at, mode, chat_id, user_id, message_id, prompt, background_rgb, "
    "source_file_id, elapsed_ms, original_path, original_blob, original_type, "
    "aux_path, aux_blob, aux_type, result_path, result_blob, result_type"
)
_CHAT_COLUMNS = "id, created_at, chat_id, user_id, message_id, prompt, reply, elapsed_ms"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StoredMedia:
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


@dataclass
class StoredChat:
    id: int
    created_at: str
    chat_id: int
    user_id: int
    message_id: int | None
    prompt: str
    reply: str | None
    elapsed_ms: int | None


class LocalStore:
    """SQLite-backed archive of every processed photo and chat turn."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.parent and str(self.path.parent) not in ("", "."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()
        # the store holds raw chat text and image bytes - keep it owner-only
        try:
            os.chmod(self.path, 0o600)
            if self.path.parent and str(self.path.parent) not in ("", "."):
                os.chmod(self.path.parent, 0o700)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            existing = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            for old, new in _LEGACY_TABLES.items():
                if old in existing and new not in existing:
                    conn.execute(f"ALTER TABLE {old} RENAME TO {new}")
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
                INSERT INTO media (
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
                INSERT INTO chats
                    (created_at, chat_id, user_id, message_id, prompt, reply, elapsed_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (created_at or _now(), chat_id, user_id, message_id, prompt, reply, elapsed_ms),
            )
            return int(cur.lastrowid)

    def all_media(
        self,
        *,
        user_id: int | None = None,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[StoredMedia]:
        query = f"SELECT {_MEDIA_COLUMNS} FROM media"
        params: list[object] = []
        if user_id is not None:
            query += " WHERE user_id = ?"
            params.append(user_id)
        query += " ORDER BY id DESC" if newest_first else " ORDER BY id"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [StoredMedia(**dict(row)) for row in rows]

    def all_chats(
        self,
        *,
        user_id: int | None = None,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[StoredChat]:
        query = f"SELECT {_CHAT_COLUMNS} FROM chats"
        params: list[object] = []
        if user_id is not None:
            query += " WHERE user_id = ?"
            params.append(user_id)
        query += " ORDER BY id DESC" if newest_first else " ORDER BY id"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [StoredChat(**dict(row)) for row in rows]

    def latest_media(self, user_id: int) -> StoredMedia | None:
        rows = self.all_media(user_id=user_id, limit=1, newest_first=True)
        return rows[0] if rows else None

    def get_media(self, row_id: int) -> StoredMedia | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_MEDIA_COLUMNS} FROM media WHERE id = ?", (row_id,)
            ).fetchone()
        return StoredMedia(**dict(row)) if row else None

    def counts(self) -> tuple[int, int]:
        with self._connect() as conn:
            media = conn.execute("SELECT COUNT(*) FROM media").fetchone()[0]
            chat = conn.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
        return int(media), int(chat)

    def count_media(self, user_id: int | None = None) -> int:
        with self._connect() as conn:
            if user_id is None:
                return int(conn.execute("SELECT COUNT(*) FROM media").fetchone()[0])
            return int(
                conn.execute("SELECT COUNT(*) FROM media WHERE user_id = ?", (user_id,)).fetchone()[0]
            )
