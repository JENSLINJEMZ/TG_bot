"""Upload locally-queued media and chats to Supabase.

While Supabase is unreachable the bot writes every input image, result image and
chat turn to a local SQLite outbox. Once Supabase is back, run this to replay
the whole backlog (uploads + row inserts), preserving the original timestamps.

    uv run python -m tgbot.sync              # send everything, clear the queue
    uv run python -m tgbot.sync --dry-run    # show what is waiting, send nothing
    uv run python -m tgbot.sync --limit 20   # stop after 20 media + 20 chats

The bot also calls :func:`sync_pending` once at startup, so a restart after
Supabase recovers drains the outbox automatically.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from . import config
from .localstore import LocalStore, PendingChat, PendingMedia
from .storage import chat_row, media_row
from .supabase import Supabase, SupabaseConfig

log = logging.getLogger("tgbot.sync")


async def _sync_media(db: Supabase, local: LocalStore, row: PendingMedia) -> None:
    await db.upload(row.original_path, row.original_blob, row.original_type)
    if row.aux_path and row.aux_blob is not None:
        await db.upload(row.aux_path, row.aux_blob, row.aux_type or "image/jpeg")
    await db.upload(row.result_path, row.result_blob, row.result_type)
    background_rgb = json.loads(row.background_rgb) if row.background_rgb else None
    await db.insert(
        media_row(
            mode=row.mode,
            chat_id=row.chat_id,
            user_id=row.user_id,
            message_id=row.message_id,
            prompt=row.prompt,
            background_rgb=background_rgb,
            source_file_id=row.source_file_id,
            elapsed_ms=row.elapsed_ms,
            original_path=row.original_path,
            result_path=row.result_path,
            aux_path=row.aux_path,
            created_at=row.created_at,
        )
    )
    local.mark_media_synced(row.id)


async def _sync_chat(db: Supabase, local: LocalStore, row: PendingChat) -> None:
    await db.insert_chat(
        chat_row(
            chat_id=row.chat_id,
            user_id=row.user_id,
            message_id=row.message_id,
            prompt=row.prompt,
            reply=row.reply,
            elapsed_ms=row.elapsed_ms,
            created_at=row.created_at,
        )
    )
    local.mark_chat_synced(row.id)


async def sync_pending(
    db: Supabase | None,
    local: LocalStore,
    *,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict:
    """Flush the outbox. Returns a summary dict. Failures stay queued."""
    media = local.pending_media()
    chats = local.pending_chat()
    if limit is not None:
        media = media[:limit]
        chats = chats[:limit]

    summary = {
        "media": 0,
        "chats": 0,
        "failed": 0,
        "queued": len(media) + len(chats),
        "dry_run": dry_run,
    }

    if dry_run or db is None:
        return summary

    for row in media:
        try:
            await _sync_media(db, local, row)
            summary["media"] += 1
        except Exception as exc:  # noqa: BLE001
            local.set_media_error(row.id, str(exc))
            summary["failed"] += 1
            log.warning("media id=%s not synced: %s", row.id, exc)

    for row in chats:
        try:
            await _sync_chat(db, local, row)
            summary["chats"] += 1
        except Exception as exc:  # noqa: BLE001
            local.set_chat_error(row.id, str(exc))
            summary["failed"] += 1
            log.warning("chat id=%s not synced: %s", row.id, exc)

    left_media, left_chat = local.counts()
    summary["remaining"] = left_media + left_chat
    return summary


def build_db() -> Supabase | None:
    if not config.SUPABASE_ENABLED:
        return None
    return Supabase(
        SupabaseConfig(
            url=config.SUPABASE_URL,
            api_key=config.SUPABASE_API_KEY,
            bucket=config.SUPABASE_STORAGE_BUCKET,
        )
    )


async def _run(args) -> int:
    local = LocalStore(config.OFFLINE_STORE_PATH)
    media, chats = local.counts()
    log.info("outbox has %d media and %d chat rows", media, chats)

    if media == 0 and chats == 0:
        print("Nothing to sync - the local outbox is empty.")
        return 0

    db = build_db()
    if db is None:
        print(
            "Supabase is not configured (SUPABASE_URL / SUPABASE_API_KEY missing in .env),\n"
            f"so the {media + chats} queued item(s) cannot be uploaded yet. They stay in "
            f"{config.OFFLINE_STORE_PATH}."
        )
        return 1

    if args.dry_run:
        for row in local.pending_media():
            print(f"media  id={row.id} mode={row.mode} user={row.user_id} created={row.created_at}")
        for row in local.pending_chat():
            print(f"chat   id={row.id} user={row.user_id} created={row.created_at} prompt={row.prompt[:60]!r}")
        print(f"\nDry run: would upload {media} media and {chats} chat rows.")
        await db.close()
        return 0

    try:
        summary = await sync_pending(db, local, limit=args.limit)
    finally:
        await db.close()

    print(
        f"Synced {summary['media']} media and {summary['chats']} chat rows "
        f"({summary['failed']} failed, {summary.get('remaining', 0)} still queued)."
    )
    return 0 if summary["failed"] == 0 else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload locally-queued data to Supabase.")
    parser.add_argument("--dry-run", action="store_true", help="list what is queued without uploading")
    parser.add_argument("--limit", type=int, default=None, help="max rows per table to sync this run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()