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
from aiogram.types import (
    BotCommand,
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

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
from .jobs import Job, JobQueue, QUEUE_FULL_TEXT
from .localstore import LocalStore
from .ratelimit import RateLimiter
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
rate = RateLimiter(config.COOLDOWN_SECONDS)
jobs = JobQueue(workers=max(1, config.MAX_CONCURRENT))


class Modes(StatesGroup):
    combine_subject = State()
    combine_background = State()
    cloth = State()
    prompt_image = State()
    prompt_text = State()
    chat = State()


HELP_TEXT = (
    "What can I do:\n"
    "🧩 combine a cutout onto another photo\n"
    "👕 recolor a garment - or give any outfit prompt in the caption\n"
    "✨ restyle an outfit from a text prompt\n"
    "💬 chat with a small local model\n\n"
    "Use /menu to pick an operation. A photo with a color caption (\"navy\", "
    "\"#ff8800\", \"gradient red blue\", \"transparent\") still works directly.\n"
    "Heavy jobs run in a queue: you get a \"queued\" notice, then the result "
    "arrives here when it's ready.\n\n"
    "Commands:\n"
    "🎨 /menu - choose an operation\n"
    "🛑 /cancel - stop the current operation\n"
    "❓ /help - this message"
)


BOT_COMMANDS = [
    BotCommand(command="start", description="👋 Welcome and the operation menu"),
    BotCommand(command="menu", description="🎨 Pick an operation (combine, cloth, restyle, chat)"),
    BotCommand(command="help", description="❓ What this bot can do"),
    BotCommand(command="cancel", description="🛑 Stop the current operation"),
]

# admin commands: registered handlers, but intentionally absent from the `/` menu,
# the HELP text and the inline keyboard - the owner runs them by typing them.
ADMIN_COMMANDS = ("stats", "last", "sync")


def menu_keyboard(compact: bool = False) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text="🧩 Combine images", callback_data="mode_combine")],
        [InlineKeyboardButton(text="👕 Change cloth", callback_data="mode_cloth")],
        [InlineKeyboardButton(text="✨ Restyle outfit", callback_data="mode_restyle")],
        [InlineKeyboardButton(text="💬 Chat", callback_data="mode_chat")],
    ]
    if compact:
        buttons.append([InlineKeyboardButton(text="📋 Open menu", callback_data="menu")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def _authorized(message: Message) -> bool:
    if not config.ALLOWED_USERS:
        return True
    user = message.from_user
    return bool(user and user.id in config.ALLOWED_USERS)


def _authorized_cb(callback: CallbackQuery) -> bool:
    if not config.ALLOWED_USERS:
        return True
    user = callback.from_user
    return bool(user and user.id in config.ALLOWED_USERS)


def _is_owner(message: Message) -> bool:
    user = message.from_user
    return bool(user and config.OWNER_USER_ID and user.id == config.OWNER_USER_ID)


async def _require_owner(message: Message) -> bool:
    """Gate for the hidden admin commands. Non-owners get nothing at all."""
    if _is_owner(message):
        return True
    if not config.OWNER_USER_ID:
        await message.answer("Admin commands are disabled - set OWNER_USER_ID in .env.")
    return False


def _cooldown_seconds(message: Message) -> float:
    """Seconds the user must wait before the next heavy request (0 = allowed)."""
    user = message.from_user
    if user is None or _is_owner(message):
        return 0.0
    if rate.allow(user.id):
        return 0.0
    return rate.wait_left(user.id)


def _extension(data: bytes) -> tuple[str, str]:
    if data.startswith(b"\xff\xd8"):
        return "jpg", "image/jpeg"
    if data.startswith(b"\x89PNG"):
        return "png", "image/png"
    return "jpg", "image/jpeg"


async def _submit_job(
    message: Message,
    *,
    kind: str,
    run,
    state: FSMContext | None = None,
    clear_state: bool = True,
    pre_notice: bool = True,
) -> bool:
    """Admit a heavy job to the queue and reply with its position.

    pre_notice=True (image jobs): the "Queued..." message is created *before*
    submit, so the worker can always edit/delete it. pre_notice=False (chat):
    a notice is created only when another job is actually ahead, which keeps
    the normal one-message-per-turn chat flow. Returns False when full.
    """
    if not message.from_user:
        return False
    job = Job(kind=kind, user_id=message.from_user.id, chat_id=message.chat.id, run=run)
    notice: Message | None = None
    if pre_notice:
        try:
            notice = await message.answer("⏳ Queued - one moment...")
        except Exception:
            notice = None
        job.status_msg = notice
    position = jobs.submit(job)
    if position is None:
        if notice is not None:
            try:
                await notice.edit_text(QUEUE_FULL_TEXT)
            except Exception:
                pass
        else:
            try:
                await message.answer(QUEUE_FULL_TEXT)
            except Exception:
                pass
        return False
    if notice is None and position > 1:
        # something is ahead: tell the user (worker cannot have taken the job yet)
        try:
            job.status_msg = await message.answer(
                f"⏳ Queued - position {position}. You'll get the reply right here."
            )
        except Exception:
            job.status_msg = None
    elif notice is not None and position > 1:
        try:
            await notice.edit_text(f"⏳ Queued - position {position}. The result arrives here.")
        except Exception:
            pass
    if state is not None and clear_state:
        await state.clear()
    return True


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
    data = buffer.getvalue()
    if len(data) > config.MAX_PHOTO_BYTES:
        raise ValueError(
            f"photo is too large ({len(data) // (1024 * 1024)} MB, "
            f"max {config.MAX_PHOTO_BYTES // (1024 * 1024)} MB)"
        )
    return data


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
    caption = f"✅ done | {caption}"
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
    await message.answer("Pick an operation 👇", reply_markup=menu_keyboard())


@dp.message(Command("menu"))
async def cmd_menu(message: Message, state: FSMContext) -> None:
    if not _authorized(message):
        return
    await state.clear()
    await message.answer("Pick an operation 👇", reply_markup=menu_keyboard())


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
        await message.answer("🛑 Cancelled.", reply_markup=menu_keyboard(compact=True))
    else:
        await message.answer("Nothing to cancel.")


@dp.message(Command("stats"))
async def cmd_stats(message: Message) -> None:
    if not await _require_owner(message):
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
    await message.answer(
        f"📊 {total} photos processed so far.\n"
        f"⚙️ jobs: {jobs.done} done, {jobs.failed} failed, {jobs.pending()} in queue"
    )


@dp.message(Command("last"))
async def cmd_last(message: Message) -> None:
    if not await _require_owner(message) or not message.from_user:
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
    if not await _require_owner(message):
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
    if not _authorized_cb(callback):
        await callback.answer()
        return
    await state.clear()
    await callback.message.edit_text("Pick an operation 👇", reply_markup=menu_keyboard())
    await callback.answer()


@dp.callback_query(F.data == "mode_combine")
async def cb_mode_combine(callback: CallbackQuery, state: FSMContext) -> None:
    if not _authorized_cb(callback):
        await callback.answer()
        return
    await state.set_state(Modes.combine_subject)
    await callback.message.edit_text("🧩 Send the first image.")
    await callback.answer()


@dp.callback_query(F.data == "mode_cloth")
async def cb_mode_cloth(callback: CallbackQuery, state: FSMContext) -> None:
    if not _authorized_cb(callback):
        await callback.answer()
        return
    await state.set_state(Modes.cloth)
    await callback.message.edit_text(
        "👕 Send a photo of a person. Caption optional: a color "
        '(navy, #ff0000, gradient red blue) OR any garment prompt '
        '("bikini", "leather jacket", "gold chain") - no filter, your prompt drives it.'
    )
    await callback.answer()


@dp.callback_query(F.data == "mode_restyle")
async def cb_mode_restyle(callback: CallbackQuery, state: FSMContext) -> None:
    if not _authorized_cb(callback):
        await callback.answer()
        return
    await state.set_state(Modes.prompt_image)
    await callback.message.edit_text("✨ Send a photo of a person you want to restyle.")
    await callback.answer()


@dp.callback_query(F.data == "mode_chat")
async def cb_mode_chat(callback: CallbackQuery, state: FSMContext) -> None:
    if not _authorized_cb(callback):
        await callback.answer()
        return
    if not config.CHAT_ENABLED:
        await callback.answer("💬 Chat is disabled.", show_alert=True)
        return
    await state.set_state(Modes.chat)
    await state.update_data(chat_history=[])
    await callback.message.edit_text(
        "💬 Chat mode - send me a message. /cancel to stop.\n"
        "(Small local model, so replies are short and may take a moment.)"
    )
    await callback.answer()


@dp.message(Modes.chat, F.text)
async def on_chat_text(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    wait = _cooldown_seconds(message)
    if wait > 0:
        await message.answer(f"⏳ Please wait {wait:.0f}s before the next request.")
        return
    text = (message.text or "").strip()
    if not text:
        return
    data = await state.get_data()
    history = data.get("chat_history") or []

    async def run() -> None:
        await _chat_action(message, "typing")
        started = time.perf_counter()
        try:
            answer, _secs = await asyncio.to_thread(chat_engine.reply, text, history)
        except Exception:
            log.exception("chat failed")
            try:
                await message.answer("⚠️ Chat failed - please try again.")
            except Exception:
                pass
            return
        updated_history = (
            list(history) + [{"role": "user", "content": text}, {"role": "assistant", "content": answer}]
        )[-2 * config.CHAT_HISTORY_TURNS :] if config.CHAT_HISTORY_TURNS > 0 else []
        await state.update_data(chat_history=updated_history)
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
        await message.answer(f"🤖 {answer}\n\n[{elapsed_ms} ms]")

    # quiet path: with an empty queue the worker takes this immediately, so the
    # user still gets exactly one reply message (no extra "queued" noise)
    await _submit_job(
        message, kind="chat", run=run, state=state, clear_state=False, pre_notice=False
    )


@dp.message(Modes.chat)
async def on_chat_needs_text(message: Message) -> None:
    if not _authorized(message):
        return
    await message.answer("💬 Send a text message (or /cancel to leave chat).")


@dp.message(Modes.combine_subject, F.photo)
async def on_combine_subject(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    try:
        source = await _download_photo(message)
    except Exception:
        log.exception("combine: subject download failed")
        await message.answer("❌ Could not download that photo - try again from /menu.")
        return
    await state.update_data(subject=source)
    await state.set_state(Modes.combine_background)
    await message.answer("🧩 Got it! Now send the second image (the background).")


@dp.message(Modes.combine_background, F.photo)
async def on_combine_background(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    wait = _cooldown_seconds(message)
    if wait > 0:
        await message.answer(f"⏳ Please wait {wait:.0f}s before the next request.")
        return
    data = await state.get_data()
    subject = data.get("subject")
    if not subject:
        await state.clear()
        await message.answer("⚠️ Something went wrong - try again from /menu.")
        return
    try:
        background = await _download_photo(message)
    except Exception:
        log.exception("combine: background download failed")
        await message.answer("❌ Could not download that photo - try again from /menu.")
        await state.clear()
        return
    subject_ext, subject_type = _extension(subject)
    bg_ext, bg_type = _extension(background)

    async def run() -> None:
        await _chat_action(message, "upload_photo")
        started = time.perf_counter()
        result, content_type = await asyncio.to_thread(
            combine_images, subject, background, config.MAX_IMAGE_DIM
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
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

    await _submit_job(message, kind="combine", run=run, state=state)


@dp.message(Modes.cloth, F.photo)
async def on_cloth(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    wait = _cooldown_seconds(message)
    if wait > 0:
        await message.answer(f"⏳ Please wait {wait:.0f}s before the next request.")
        return
    try:
        source = await _download_photo(message)
    except Exception:
        log.exception("cloth: download failed")
        await message.answer("❌ Could not download that photo - try again from /menu.")
        await state.clear()
        return
    caption = message.caption
    background, note = parse_background(caption, config.DEFAULT_BACKGROUND)
    # A caption that is not a color ("bikini", "leather jacket", "gold chain"...)
    # is a free garment prompt: there is no content filter in this bot, so it goes
    # straight to the diffusion engine instead of failing as an unknown color.
    prompt_mode = bool(caption and caption.strip() and note is not None)
    if isinstance(background, Gradient):
        color = background.top
        note = "gradients not supported for clothes - used top color"
        prompt_mode = False
    elif isinstance(background, Transparent):
        color = (200, 200, 200)
        note = "clothes need a solid color - used gray"
        prompt_mode = False
    else:
        color = background.rgb
    if prompt_mode:
        note = None
    source_ext, source_type = _extension(source)

    async def run() -> None:
        await _chat_action(message, "upload_photo")
        started = time.perf_counter()
        payload: bytes | None = None
        content_type = "image/png"
        engine = ""
        rgb: list[int] | None = list(color)
        if prompt_mode and config.DIFFUSE_ENABLED:
            try:
                payload, content_type, gen_secs = await asyncio.to_thread(
                    diffuse_generate, caption, source
                )
                engine = f"diffusion {gen_secs:.0f}s"
                rgb = None
            except Exception as exc:
                log.warning("cloth prompt diffusion failed, tint fallback: %s", exc)
                engine = "diffusion unavailable - tint fallback"
        if payload is None:
            payload, content_type = await asyncio.to_thread(
                change_cloth_color, source, color, config.MAX_IMAGE_DIM
            )
            if prompt_mode and not engine:
                engine = "tint (diffusion disabled)"
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        stored = await _store_uploads(
            message,
            mode="cloth",
            prompt=caption,
            background_rgb=rgb,
            originals=[(source, source_type, source_ext)],
            result=(payload, content_type),
            source_file_id=message.photo[-1].file_id,
            elapsed_ms=elapsed_ms,
        )
        parts: list[str] = []
        if prompt_mode:
            parts.append(f"outfit: {(caption or '').strip()}")
        else:
            parts.append(f"garment color: #{color[0]:02x}{color[1]:02x}{color[2]:02x}")
        if engine:
            parts.append(engine)
        if note:
            parts.append(note)
        parts.append(f"{elapsed_ms} ms")
        if stored:
            parts.append(stored)
        await _send_result(message, payload, content_type, " | ".join(parts))

    await _submit_job(message, kind="clothing", run=run, state=state)


@dp.message(Modes.prompt_image, F.photo)
async def on_prompt_image(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    try:
        source = await _download_photo(message)
    except Exception:
        log.exception("restyle: source download failed")
        await message.answer("❌ Could not download that photo - try again from /menu.")
        return
    await state.update_data(prompt_source=source, prompt_file_id=message.photo[-1].file_id)
    await state.set_state(Modes.prompt_text)
    await message.answer(
        '✨ Write a prompt for the new look, e.g. "navy suit", "teal hoodie", "#c73a72 dress".'
    )


@dp.message(Modes.prompt_text, F.photo)
async def on_prompt_text_photo(message: Message) -> None:
    if not _authorized(message):
        return
    await message.answer("✏️ Please send a text prompt, not a photo (e.g. \"navy suit\").")


@dp.message(Modes.prompt_text, F.text)
async def on_prompt_text(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    wait = _cooldown_seconds(message)
    if wait > 0:
        await message.answer(f"⏳ Please wait {wait:.0f}s before the next request.")
        return
    data = await state.get_data()
    source = data.get("prompt_source")
    if not source:
        await state.clear()
        await message.answer("📷 Missing photo - start again from /menu.")
        return
    prompt = message.text.strip() or "default"
    source_ext, source_type = _extension(source)
    file_id = data.get("prompt_file_id")

    async def run() -> None:
        sticker_msg = await _send_processing_sticker(message)
        await _chat_action(message, "upload_photo")
        started = time.perf_counter()
        try:
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
                source_file_id=file_id,
                elapsed_ms=elapsed_ms,
            )
            parts = [f"restyled: {prompt}", engine_str, f"{elapsed_ms} ms"]
            if stored:
                parts.append(stored)
            await _send_result(message, result[0], result[1], " | ".join(parts))
        finally:
            if sticker_msg:
                try:
                    await sticker_msg.delete()
                except Exception:
                    pass

    await _submit_job(message, kind="restyle", run=run, state=state)


@dp.message(Modes.combine_subject)
@dp.message(Modes.combine_background)
@dp.message(Modes.cloth)
@dp.message(Modes.prompt_image)
async def on_mode_needs_photo(message: Message) -> None:
    if not _authorized(message):
        return
    await message.answer("📷 Please send a photo.")


@dp.message(F.photo)
async def on_photo(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user or not message.photo:
        return
    wait = _cooldown_seconds(message)
    if wait > 0:
        await message.answer(f"⏳ Please wait {wait:.0f}s before the next request.")
        return
    try:
        source = await _download_photo(message)
    except Exception:
        log.exception("photo download failed")
        await message.answer("❌ Could not download that photo - try again.")
        return
    caption = message.caption
    background, note = parse_background(caption, config.DEFAULT_BACKGROUND)
    background_rgb = list(background.rgb) if isinstance(background, Solid) else None
    source_ext, source_type = _extension(source)

    async def run() -> None:
        await _chat_action(message, "upload_photo")
        started = time.perf_counter()
        result, content_type = await asyncio.to_thread(
            change_background, source, background, config.MAX_IMAGE_DIM
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        stored = await _store_uploads(
            message,
            mode="bg",
            prompt=caption,
            background_rgb=background_rgb,
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

    await _submit_job(message, kind="background", run=run, state=state)


@dp.message()
async def on_other(message: Message) -> None:
    if not _authorized(message):
        return
    if message.photo:
        return
    await message.answer("📷 Send me a photo - or pick an operation from /menu (including Chat).")


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
    try:
        await bot.set_my_commands(BOT_COMMANDS)
        log.info("registered %d bot commands for the chat input menu", len(BOT_COMMANDS))
    except Exception as exc:  # noqa: BLE001
        log.warning("could not register bot commands: %s", exc)
    if config.OWNER_USER_ID:
        log.info("admin commands enabled for user %s (hidden from the menu)", config.OWNER_USER_ID)
    else:
        log.warning("OWNER_USER_ID not set - hidden admin commands (stats, last, sync) are disabled")
    log.info("per-user cooldown: %.0fs", config.COOLDOWN_SECONDS)
    if config.SUPABASE_ENABLED and not config.SUPABASE_USING_SECRET:
        log.warning(
            "Supabase key looks like '%s', not a secret key - the hardened RLS "
            "denies it, so storage falls back to the local outbox. Set "
            "SUPABASE_SECRET_KEY to the sb_secret_... (or service_role) key in .env.",
            config.SUPABASE_KEY_KIND,
        )
    if local_store is not None and db is not None:
        pending = sum(local_store.counts())
        if pending:
            try:
                summary = await sync_pending(db, local_store)
                log.info("startup outbox sync: %s", summary)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not sync local outbox at startup: %s", exc)
    await jobs.start()
    log.info("starting long polling")
    try:
        await dp.start_polling(bot)
    finally:
        await jobs.stop()
        if db is not None:
            await db.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())