"""Persist media and chat turns to the local SQLite store.

Every write goes through :class:`Storage`, which is backed purely by
:class:`tgbot.localstore.LocalStore` on disk - the previous remote backend has
been removed from the project.

A storage failure must never lose the processed result the user is waiting for,
so errors are logged and turned into a short caption suffix instead of raised.
"""

from __future__ import annotations

import json
import logging
from uuid import uuid4

from .localstore import LocalStore

log = logging.getLogger("tgbot.storage")

NOT_STORED = " (not stored: local storage disabled)"
WRITE_FAILED = " (not saved: local write failed)"


def _local_name(ext: str) -> str:
    """A unique on-disk-looking name used to label the copied bytes."""
    return f"{uuid4().hex}.{ext}"


class Storage:
    def __init__(self, local: LocalStore | None) -> None:
        self.local = local

    # -- media ---------------------------------------------------------------
    def save_media(
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
        """Store the input(s) and the result. Returns a caption suffix."""
        if self.local is None:
            return NOT_STORED

        result_payload, result_type = result
        result_ext = {"image/png": "png", "video/mp4": "mp4"}.get(result_type, "jpg")
        paths = [_local_name(ext) for (_, _, ext) in originals] or [_local_name("jpg")]
        result_path = _local_name(result_ext)

        first = originals[0] if originals else (b"", "image/jpeg", "jpg")
        aux = originals[1] if len(originals) > 1 else None
        try:
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
        except Exception:  # noqa: BLE001
            log.exception("could not save media locally")
            return WRITE_FAILED
        return ""

    # -- chat ----------------------------------------------------------------
    def save_chat(
        self,
        *,
        chat_id: int,
        user_id: int,
        message_id: int | None,
        prompt: str,
        reply: str | None,
        elapsed_ms: int | None,
    ) -> str:
        """Store one chat turn. Returns a status string (empty on success)."""
        if self.local is None:
            return NOT_STORED
        try:
            self.local.save_chat(
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                prompt=prompt,
                reply=reply,
                elapsed_ms=elapsed_ms,
            )
        except Exception:  # noqa: BLE001
            log.exception("could not save chat locally")
            return WRITE_FAILED
        return ""
