"""Persist media and chats to Supabase, falling back to a local outbox.

Every write goes through :class:`Storage`:

* if Supabase is configured and reachable, the upload + row insert happen
  immediately and nothing is queued;
* if Supabase is unreachable, not configured, or rejects the write, the data
  (including raw image bytes) is written to the local SQLite outbox so nothing
  is lost, and ``tgbot/sync.py`` uploads it later.

The row builders are shared with the sync script so an online write and a
replayed write produce byte-identical rows.
"""

from __future__ import annotations

import json
import logging

from .localstore import LocalStore
from .supabase import Supabase, SupabasePermissionError

log = logging.getLogger("tgbot.storage")

NOT_CONFIGURED = " (saved locally - Supabase not configured)"
WILL_SYNC = " (saved locally - will sync when Supabase is reachable)"
NEEDS_GRANT = " (saved locally - run supabase/migration.sql, then sync)"
NOT_STORED = " (not stored: no SUPABASE_URL configured)"


def media_row(
    *,
    mode: str | None,
    chat_id: int,
    user_id: int,
    message_id: int | None,
    prompt: str | None,
    background_rgb: list[int] | None,
    source_file_id: str | None,
    elapsed_ms: int | None,
    original_path: str,
    result_path: str,
    aux_path: str | None,
    created_at: str | None = None,
) -> dict:
    row = {
        "mode": mode,
        "chat_id": chat_id,
        "user_id": user_id,
        "message_id": message_id,
        "prompt": prompt,
        "background_rgb": background_rgb,
        "source_file_id": source_file_id,
        "elapsed_ms": elapsed_ms,
        "original_path": original_path,
        "result_path": result_path,
        "aux_path": aux_path,
    }
    if created_at:
        row["created_at"] = created_at
    return row


def chat_row(
    *,
    chat_id: int,
    user_id: int,
    message_id: int | None,
    prompt: str,
    reply: str | None,
    elapsed_ms: int | None,
    created_at: str | None = None,
) -> dict:
    row = {
        "chat_id": chat_id,
        "user_id": user_id,
        "message_id": message_id,
        "prompt": prompt,
        "reply": reply,
        "elapsed_ms": elapsed_ms,
    }
    if created_at:
        row["created_at"] = created_at
    return row


class Storage:
    def __init__(self, db: Supabase | None, local: LocalStore | None) -> None:
        self.db = db
        self.local = local

    # -- media ---------------------------------------------------------------
    async def save_media(
        self,
        *,
        chat_id: int,
        user_id: int,
        message_id: int | None,
        mode: str | None,
        prompt: str | None,
        background_rgb: list[int] | None,
        originals: list[tuple[bytes, str, str]],
        result: tuple[bytes, str],
        source_file_id: str | None,
        elapsed_ms: int | None,
    ) -> str:
        """Upload + insert; queue locally on failure. Returns a caption suffix."""
        result_payload, result_type = result
        result_ext = "png" if result_type == "image/png" else "jpg"
        paths = [
            Supabase.object_path(
                "originals" if i == 0 else "inputs", chat_id, user_id, ext
            )
            for i, (_, _, ext) in enumerate(originals)
        ]
        result_path = Supabase.object_path("results", chat_id, user_id, result_ext)

        if self.db is not None:
            try:
                for (payload, content_type, _), path in zip(originals, paths):
                    await self.db.upload(path, payload, content_type)
                await self.db.upload(result_path, result_payload, result_type)
                await self.db.insert(
                    media_row(
                        mode=mode,
                        chat_id=chat_id,
                        user_id=user_id,
                        message_id=message_id,
                        prompt=prompt,
                        background_rgb=background_rgb,
                        source_file_id=source_file_id,
                        elapsed_ms=elapsed_ms,
                        original_path=paths[0],
                        result_path=result_path,
                        aux_path=paths[1] if len(paths) > 1 else None,
                    )
                )
                return ""
            except SupabasePermissionError as exc:
                log.warning("storage/db permission denied: %s", exc)
                note = NEEDS_GRANT
            except Exception as exc:  # noqa: BLE001
                log.warning("storage/db write failed, queueing locally: %s", exc)
                note = WILL_SYNC
        else:
            note = NOT_CONFIGURED if self.local is not None else NOT_STORED

        self._enqueue_media(
            chat_id=chat_id,
            user_id=user_id,
            message_id=message_id,
            mode=mode,
            prompt=prompt,
            background_rgb=background_rgb,
            originals=originals,
            paths=paths,
            result_payload=result_payload,
            result_type=result_type,
            result_path=result_path,
            source_file_id=source_file_id,
            elapsed_ms=elapsed_ms,
        )
        return note

    def _enqueue_media(
        self,
        *,
        chat_id: int,
        user_id: int,
        message_id: int | None,
        mode: str | None,
        prompt: str | None,
        background_rgb: list[int] | None,
        originals: list[tuple[bytes, str, str]],
        paths: list[str],
        result_payload: bytes,
        result_type: str,
        result_path: str,
        source_file_id: str | None,
        elapsed_ms: int | None,
    ) -> None:
        if self.local is None:
            return
        first = originals[0] if originals else (b"", "image/jpeg", "jpg")
        aux = originals[1] if len(originals) > 1 else None
        self.local.save_media(
            mode=mode,
            chat_id=chat_id,
            user_id=user_id,
            message_id=message_id,
            prompt=prompt,
            background_rgb=json.dumps(background_rgb) if background_rgb else None,
            source_file_id=source_file_id,
            elapsed_ms=elapsed_ms,
            original_path=paths[0],
            original_blob=first[0],
            original_type=first[1],
            aux_path=paths[1] if len(paths) > 1 else None,
            aux_blob=aux[0] if aux else None,
            aux_type=aux[1] if aux else None,
            result_path=result_path,
            result_blob=result_payload,
            result_type=result_type,
        )

    # -- chat ----------------------------------------------------------------
    async def save_chat(
        self,
        *,
        chat_id: int,
        user_id: int,
        message_id: int | None,
        prompt: str,
        reply: str | None,
        elapsed_ms: int | None,
    ) -> str:
        """Insert a chat turn; queue locally on failure. Returns a status string."""
        if self.db is not None:
            try:
                await self.db.insert_chat(
                    chat_row(
                        chat_id=chat_id,
                        user_id=user_id,
                        message_id=message_id,
                        prompt=prompt,
                        reply=reply,
                        elapsed_ms=elapsed_ms,
                    )
                )
                return ""
            except SupabasePermissionError:
                note = NEEDS_GRANT
            except Exception as exc:
                log.warning("chat db write failed, queuing locally: %s", exc)
                note = WILL_SYNC
        else:
            note = NOT_CONFIGURED if self.local is not None else NOT_STORED

        if self.local is not None:
            self.local.save_chat(
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                prompt=prompt,
                reply=reply,
                elapsed_ms=elapsed_ms,
            )
        return note