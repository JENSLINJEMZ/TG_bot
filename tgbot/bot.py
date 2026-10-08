from __future__ import annotations

import asyncio
import io
import logging
import time

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BufferedInputFile, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from . import config
from . import chat as chat_engine
from .bgchange import (
    Gradient,
    Solid,
    Transparent,
    change_background,
    change_cloth_color,
    combine_images,
    describe,
    parse_background,
    restyle_outfit,
)
from .diffuse import generate as diffuse_generate
from .localstore import LocalStore
from .sticker import processing_sticker
from .storage import Storage, NOT_STORED
from .supabase import Supabase, SupabaseConfig, SupabasePermissionError
from .sync import sync_pending

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("tgbot")

bot: Bot | None = None
dp = Dispatcher(storage=MemoryStorage())
db = None
if config.SUPABASE_ENABLED:
    db = Supabase(
        SupabaseConfig(
            url=config.SUPABASE_URL,
            api_key=config.SUPABASE_API_KEY,
            bucket=config.SUPABASE_STORAGE_BUCKET,
        )
    )
local_store = LocalStore(config.OFFLINE_STORE_PATH) if config.OFFLINE_STORE_ENABLED else None
storage = Storage(db, local_store)


class Modes(StatesGroup):
    combine_subject = State()
    combine_background = State()
    cloth = State()
    prompt_image = State()
    prompt_text = State()
    chat = State()


HELP_TEXT = (
    "What can I do:\n"
    "- combine a cutout onto another photo\n"
    "- change a person's cloth color\n"
    "- restyle an outfit from a text prompt\n"
    "- chat with a small local model\n\n"
    "Use /menu to pick an operation. A photo with a color caption (\"navy\", "
    "\"#ff8800\", \"gradient red blue\", \"transparent\") still works directly.\n\n"
    "Commands:\n"
    "/menu - choose an operation\n"
    "/stats - how many photos are stored\n"
    "/last - resend your last processed photo\n"
    "/sync - upload locally queued data to Supabase\n"
    "/cancel - stop the current operation\n"
    "/help - this message"
)


def menu_keyboard(compact: bool = False) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text="Combine images", callback_data="mode_combine")],
        [InlineKeyboardButton(text="Change cloth", callback_data="mode_cloth")],
        [InlineKeyboardButton(text="Restyle outfit (prompt)", callback_data="mode_restyle")],
        [InlineKeyboardButton(text="Chat", callback_data="mode_chat")],
    ]
    if compact:
        buttons.append([InlineKeyboardButton(text="Open menu", callback_data="menu")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def _authorized(message: Message) -> bool:
    if not config.ALLOWED_USERS:
        return True
    user = message.from_user
    return bool(user and user.id in config.ALLOWED_USERS)


def _extension(data: bytes) -> tuple[str, str]:
    if data.startswith(b"\xff\xd8"):
        return "jpg", "image/jpeg"
    if data.startswith(b"\x89PNG"):
        return "png", "image/png"
    return "jpg", "image/jpeg"


process_limit = asyncio.Semaphore(config.MAX_CONCURRENT)


async def _chat_action(message: Message, action: str) -> None:
    try:
        await message.bot.send_chat_action(message.chat.id, action=action)
    except Exception:
        pass


async def _download_photo(message: Message) -> bytes:
    photo = message.photo[-1]
    file = await message.bot.get_file(photo.file_id)
    buffer = io.BytesIO()
    await message.bot.download(file, destination=buffer)
    return buffer.getvalue()


async def _send_processing_sticker(message: Message) -> Message | None:
    try:
        return await message.bot.send_sticker(
            message.chat.id,
            sticker=BufferedInputFile(processing_sticker(), filename="processing.png"),
        )
    except Exception as exc:
        log.warning("could not send processing sticker: %s", exc)
        return None


async def _send_result(message: Message, payload: bytes, content_type: str, caption: str) -> None:
    if content_type == "image/jpeg":
        await message.answer_photo(
            BufferedInputFile(payload, filename="result.jpg"),
            caption=caption,
            reply_markup=menu_keyboard(compact=True),
        )
    else:
        await message.answer_document(
            BufferedInputFile(payload, filename="result.png"),
            caption=caption,
            reply_markup=menu_keyboard(compact=True),
        )


async def _store_uploads(
    message: Message,
    *,
    mode: str,
    prompt: str | None,
    background_rgb: list[int] | None,
    originals: list[tuple[bytes, str, str]],
    result: tuple[bytes, str],
    source_file_id: str,
    elapsed_ms: int,
) -> str:
    if not message.from_user:
        return NOT_STORED
    return await storage.save_media(
        chat_id=message.chat.id,
        user_id=message.from_user.id,
        message_id=message.message_id,
        mode=mode,
        prompt=prompt,
        background_rgb=background_rgb,
        originals=originals,
        result=result,
        source_file_id=source_file_id,
        elapsed_ms=elapsed_ms,
    )


@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    if not _authorized(message):
        return
    await message.answer("Pick an operation:", reply_markup=menu_keyboard())


@dp.message(Command("menu"))
async def cmd_menu(message: Message, state: FSMContext) -> None:
    if not _authorized(message):
        return
    await state.clear()
    await message.answer("Pick an operation:", reply_markup=menu_keyboard())


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    if not _authorized(message):
        return
    await message.answer(HELP_TEXT)


@dp.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    if not _authorized(message):
        return
    if await state.get_state() is not None:
        await state.clear()
        await message.answer("Cancelled.", reply_markup=menu_keyboard(compact=True))
    else:
        await message.answer("Nothing to cancel.")


@dp.message(Command("stats"))
async def cmd_stats(message: Message) -> None:
    if not _authorized(message):
        return
    if db is None:
        await message.answer("Storage is not configured (no SUPABASE_URL in .env).")
        return
    try:
        total = await db.count()
    except SupabasePermissionError as exc:
        await message.answer(str(exc))
        return
    except Exception as exc:
        await message.answer(f"Could not reach Supabase: {exc}")
        return
    await message.answer(f"{total} photos processed so far.")


@dp.message(Command("last"))
async def cmd_last(message: Message) -> None:
    if not _authorized(message) or not message.from_user:
        return
    if db is None:
        await message.answer("Storage is not configured (no SUPABASE_URL in .env).")
        return
    try:
        row = await db.latest(message.from_user.id)
    except SupabasePermissionError as exc:
        await message.answer(str(exc))
        return
    except Exception as exc:
        await message.answer(f"Could not reach Supabase: {exc}")
        return
    if not row or not row.get("result_path"):
        await message.answer("Nothing stored yet.")
        return
    try:
        payload = await db.download(row["result_path"])
    except Exception as exc:
        await message.answer(f"Could not fetch from storage: {exc}")
        return
    is_png = payload.startswith(b"\x89PNG")
    mode = row.get("mode") or "bg"
    caption = f"mode: {mode} | background: {row.get('prompt') or 'default'}"
    if is_png:
        await message.answer_document(BufferedInputFile(payload, filename="result.png"), caption=caption)
    else:
        await message.answer_photo(BufferedInputFile(payload, filename="result.jpg"), caption=caption)


@dp.message(Command("sync"))
async def cmd_sync(message: Message) -> None:
    if not _authorized(message):
        return
    if local_store is None:
        await message.answer("Offline storage is turned off (OFFLINE_STORE_ENABLED=0).")
        return
    queued = sum(local_store.counts())
    if queued == 0:
        await message.answer("Nothing pending - the local outbox is empty.")
        return
    if db is None:
        await message.answer(
            f"{queued} item(s) are saved locally, but Supabase is not configured "
            "(set SUPABASE_URL and SUPABASE_API_KEY in .env)."
        )
        return
    await message.answer(f"Uploading {queued} queued item(s)...")
    try:
        summary = await sync_pending(db, local_store)
    except Exception as exc:  # noqa: BLE001
        log.warning("sync failed: %s", exc)
        await message.answer(f"Sync failed: {exc}")
        return
    remaining = summary.get("remaining", sum(local_store.counts()))
    await message.answer(
        f"Synced {summary['media']} photo(s) and {summary['chats']} chat turn(s).\n"
        f"Failed: {summary['failed']} | still queued: {remaining}"
    )


@dp.callback_query(F.data == "menu")
async def cb_menu(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await callback.message.edit_text("Pick an operation:", reply_markup=menu_keyboard())
    await callback.answer()


@dp.callback_query(F.data == "mode_combine")
async def cb_mode_combine(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(Modes.combine_subject)
    await callback.message.edit_text("Send the first image.")
    await callback.answer()


@dp.callback_query(F.data == "mode_cloth")
async def cb_mode_cloth(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(Modes.cloth)
    await callback.message.edit_text(
        "Send a photo of a person, optionally with a caption color for the garment."
    )
    await callback.answer()


@dp.callback_query(F.data == "mode_restyle")
async def cb_mode_restyle(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(Modes.prompt_image)
    await callback.message.edit_text("Send a photo of a person you want to restyle.")
    await callback.answer()


@dp.callback_query(F.data == "mode_chat")
async def cb_mode_chat(callback: CallbackQuery, state: FSMContext) -> None:
    if not config.CHAT_ENABLED:
        await callback.answer("Chat is disabled.", show_alert=True)
        return
    await state.set_state(Modes.chat)
    await state.update_data(chat_history=[])
    await callback.message.edit_text(
        "Chat mode - send me a message. /cancel to stop.\n"
        "(Small local model, so replies are short and may take a moment.)"
    )
    await callback.answer()


@dp.message(Modes.chat, F.text)
async def on_chat_text(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    text = (message.text or "").strip()
    if not text:
        return
    data = await state.get_data()
    history = data.get("chat_history") or []
    await _chat_action(message, "typing")
    started = time.perf_counter()
    try:
        answer, _secs = await asyncio.to_thread(chat_engine.reply, text, history)
    except Exception as exc:
        log.exception("chat failed")
        try:
            await message.answer(f"Chat failed: {exc}")
        except Exception:
            pass
        return
    history = (
        list(history) + [{"role": "user", "content": text}, {"role": "assistant", "content": answer}]
    )[-2 * config.CHAT_HISTORY_TURNS :] if config.CHAT_HISTORY_TURNS > 0 else []
    await state.update_data(chat_history=history)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    stored = await storage.save_chat(
        chat_id=message.chat.id,
        user_id=message.from_user.id,
        message_id=message.message_id,
        prompt=text,
        reply=answer,
        elapsed_ms=elapsed_ms,
    )
    if stored:
        log.info("chat not stored remotely: %s", stored.strip())
    await message.answer(f"{answer}\n\n[{elapsed_ms} ms]")


@dp.message(Modes.chat)
async def on_chat_needs_text(message: Message) -> None:
    if not _authorized(message):
        return
    await message.answer("Send a text message (or /cancel to leave chat).")


@dp.message(Modes.combine_subject, F.photo)
async def on_combine_subject(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    source = await _download_photo(message)
    await state.update_data(subject=source)
    await state.set_state(Modes.combine_background)
    await message.answer("Got it. Now send a second image.")


@dp.message(Modes.combine_background, F.photo)
async def on_combine_background(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    data = await state.get_data()
    subject = data.get("subject")
    if not subject:
        await state.clear()
        await message.answer("Something went wrong - try again from /menu.")
        return
    status = await message.answer("Combining images...")
    await _chat_action(message, "upload_photo")
    started = time.perf_counter()
    try:
        background = await _download_photo(message)
        result, content_type = await asyncio.to_thread(
            combine_images, subject, background, config.MAX_IMAGE_DIM
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        subject_ext, subject_type = _extension(subject)
        bg_ext, bg_type = _extension(background)
        stored = await _store_uploads(
            message,
            mode="combine",
            prompt="combine images",
            background_rgb=None,
            originals=[
                (subject, subject_type, subject_ext),
                (background, bg_type, bg_ext),
            ],
            result=(result, content_type),
            source_file_id=message.photo[-1].file_id,
            elapsed_ms=elapsed_ms,
        )
        await _send_result(message, result, content_type, f"combined | {elapsed_ms} ms{stored}")
        await status.delete()
    except Exception as exc:
        log.exception("combine failed")
        try:
            await status.edit_text(f"Failed: {exc}")
        except Exception:
            pass
    finally:
        await state.clear()


@dp.message(Modes.cloth, F.photo)
async def on_cloth(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    status = await message.answer("Changing garment color...")
    await _chat_action(message, "upload_photo")
    started = time.perf_counter()
    try:
        source = await _download_photo(message)
        background, note = parse_background(message.caption, config.DEFAULT_BACKGROUND)
        if isinstance(background, Gradient):
            color = background.top
            note = "gradients not supported for clothes - used top color"
        elif isinstance(background, Transparent):
            color = (200, 200, 200)
            note = "clothes need a solid color - used gray"
        else:
            color = background.rgb
        result, content_type = await asyncio.to_thread(
            change_cloth_color, source, color, config.MAX_IMAGE_DIM
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        source_ext, source_type = _extension(source)
        stored = await _store_uploads(
            message,
            mode="cloth",
            prompt=message.caption,
            background_rgb=list(color),
            originals=[(source, source_type, source_ext)],
            result=(result, content_type),
            source_file_id=message.photo[-1].file_id,
            elapsed_ms=elapsed_ms,
        )
        parts = [f"garment color: #{color[0]:02x}{color[1]:02x}{color[2]:02x}", f"{elapsed_ms} ms"]
        if note:
            parts.insert(1, note)
        if stored:
            parts.append(stored)
        await _send_result(message, result, content_type, " | ".join(parts))
        await status.delete()
    except Exception as exc:
        log.exception("cloth failed")
        try:
            await status.edit_text(f"Failed: {exc}")
        except Exception:
            pass
    finally:
        await state.clear()


@dp.message(Modes.prompt_image, F.photo)
async def on_prompt_image(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    source = await _download_photo(message)
    await state.update_data(prompt_source=source, prompt_file_id=message.photo[-1].file_id)
    await state.set_state(Modes.prompt_text)
    await message.answer(
        'Write a prompt for the new look, e.g. "navy suit", "teal hoodie", "#c73a72 dress".'
    )


@dp.message(Modes.prompt_text, F.photo)
async def on_prompt_text_photo(message: Message) -> None:
    if not _authorized(message):
        return
    await message.answer("Please send a text prompt, not a photo (e.g. \"navy suit\").")


@dp.message(Modes.prompt_text, F.text)
async def on_prompt_text(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    data = await state.get_data()
    source = data.get("prompt_source")
    if not source:
        await state.clear()
        await message.answer("Missing photo - start again from /menu.")
        return
    prompt = message.text.strip() or "default"
    status = await message.answer("Restyling...")
    sticker_msg = await _send_processing_sticker(message)
    await _chat_action(message, "upload_photo")
    started = time.perf_counter()
    try:
        source_ext, source_type = _extension(source)
        result: tuple[bytes, str] | None = None
        engine_str = "tint"
        if config.DIFFUSE_ENABLED:
            try:
                payload, content_type, gen_secs = await asyncio.to_thread(
                    diffuse_generate, message.text, source,
                )
                result = (payload, content_type)
                engine_str = f"diffusion {gen_secs:.0f}s"
            except Exception as exc:
                log.warning("diffusion failed, tint fallback: %s", exc)
                engine_str = "diffusion unavailable - tint fallback"
        if result is None:
            (result_payload, result_type), _, _ = await asyncio.to_thread(
                restyle_outfit, source, message.text, config.DEFAULT_BACKGROUND, config.MAX_IMAGE_DIM
            )
            result = (result_payload, result_type)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        stored = await _store_uploads(
            message,
            mode="restyle",
            prompt=prompt,
            background_rgb=None,
            originals=[(source, source_type, source_ext)],
            result=result,
            source_file_id=data.get("prompt_file_id"),
            elapsed_ms=elapsed_ms,
        )
        parts = [f"restyled: {prompt}", engine_str, f"{elapsed_ms} ms"]
        if stored:
            parts.append(stored)
        await _send_result(message, result[0], result[1], " | ".join(parts))
        await status.delete()
        if sticker_msg:
            try:
                await sticker_msg.delete()
            except Exception:
                pass
    except Exception as exc:
        log.exception("restyle failed")
        try:
            await status.edit_text(f"Failed: {exc}")
        except Exception:
            pass
        if sticker_msg:
            try:
                await sticker_msg.delete()
            except Exception:
                pass
    finally:
        await state.clear()


@dp.message(Modes.combine_subject)
@dp.message(Modes.combine_background)
@dp.message(Modes.cloth)
@dp.message(Modes.prompt_image)
async def on_mode_needs_photo(message: Message) -> None:
    if not _authorized(message):
        return
    await message.answer("Please send a photo.")


@dp.message(F.photo)
async def on_photo(message: Message) -> None:
    if not _authorized(message) or not message.from_user or not message.photo:
        return
    status = await message.answer("Processing photo...")
    await _chat_action(message, "upload_photo")
    started = time.perf_counter()
    try:
        source = await _download_photo(message)

        background, note = parse_background(message.caption, config.DEFAULT_BACKGROUND)
        async with process_limit:
            result, content_type = await asyncio.to_thread(
                change_background, source, background, config.MAX_IMAGE_DIM
            )
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        source_ext, source_type = _extension(source)
        stored = await _store_uploads(
            message,
            mode="bg",
            prompt=message.caption,
            background_rgb=list(background.rgb) if isinstance(background, Solid) else None,
            originals=[(source, source_type, source_ext)],
            result=(result, content_type),
            source_file_id=message.photo[-1].file_id,
            elapsed_ms=elapsed_ms,
        )

        parts = [f"background: {describe(background)}", f"{elapsed_ms} ms"]
        if note:
            parts.insert(1, note)
        if stored:
            parts.append(stored)
        await _send_result(message, result, content_type, " | ".join(parts))
        await status.delete()
    except Exception as exc:
        log.exception("photo processing failed")
        try:
            await status.edit_text(f"Failed: {exc}")
        except Exception:
            pass


@dp.message()
async def on_other(message: Message) -> None:
    if not _authorized(message):
        return
    if message.photo:
        return
    await message.answer("Send me a photo - or pick an operation from /menu (including Chat).")


async def main() -> None:
    config.validate()
    global bot
    bot = Bot(token=config.TELEGRAM_BOT_TOKEN, default=DefaultBotProperties())
    if db is not None:
        if not await db.ensure_bucket():
            log.warning("could not ensure storage bucket %s", config.SUPABASE_STORAGE_BUCKET)
        try:
            await db.count()
        except SupabasePermissionError:
            log.warning("REST access denied - grant table access in Supabase to enable storage")
        except Exception as exc:
            log.warning("Supabase unreachable: %s", exc)
    else:
        log.info("Supabase not configured - bot runs without storage")
    if local_store is not None and db is not None:
        pending = sum(local_store.counts())
        if pending:
            try:
                summary = await sync_pending(db, local_store)
                log.info("startup outbox sync: %s", summary)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not sync local outbox at startup: %s", exc)
    log.info("starting long polling")
    try:
        await dp.start_polling(bot)
    finally:
        if db is not None:
            await db.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())