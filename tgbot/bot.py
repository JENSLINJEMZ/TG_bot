from __future__ import annotations

import asyncio
import io
import logging
import re
import time
from typing import Any

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
    Solid,
    change_background,
    change_cloth_color,
    combine_images,
    describe,
    parse_background,
    restyle_outfit,
)
from .diffuse import generate as diffuse_generate
from . import edit_model
from . import video as video_model
from .jobs import Job, JobQueue, QUEUE_FULL_TEXT
from .localstore import LocalStore
from .progress import JobState, TRACKER, format_elapsed, progress_bar
from .ratelimit import RateLimiter
from .storage import Storage, NOT_STORED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("tgbot")

bot: Bot | None = None
dp = Dispatcher(storage=MemoryStorage())
# local storage backend + its SQLite store (both wired up in main())
local_store: LocalStore | None = None
storage: Storage | None = None
rate = RateLimiter(config.COOLDOWN_SECONDS)
jobs = JobQueue(workers=max(1, config.MAX_CONCURRENT))


class Modes(StatesGroup):
    combine_subject = State()
    combine_background = State()
    combine_prompt = State()
    cloth = State()
    cloth_choice = State()
    cloth_custom = State()
    prompt_image = State()
    prompt_text = State()
    chat = State()
    video = State()
    video_prompt = State()
    video_duration = State()
    videox_subject = State()  # two-photo video: first image
    videox_second = State()  # two-photo video: second image
    videox_prompt = State()  # two-photo video: action prompt
    videox_duration = State()  # two-photo video: clip length


HELP_TEXT = (
    "What can I do:\n"
    "🧩 combine two photos with an AI prompt\n"
    "👕 change the outfit (one-tap presets or a custom prompt)\n"
    "✨ restyle an outfit from a text prompt\n"
    "🎬 animate a photo into a short video (pick 5, 10 or 15 seconds)\n"
    "🖇 turn TWO photos into one video (both people act out your prompt)\n"
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
    BotCommand(command="menu", description="🎨 Pick an operation (combine, cloth, restyle, chat, video)"),
    BotCommand(command="help", description="❓ What this bot can do"),
    BotCommand(command="cancel", description="🛑 Stop the current operation"),
]

# admin commands: registered handlers, but intentionally absent from the `/` menu,
# the HELP text and the inline keyboard - the owner runs them by typing them.
ADMIN_COMMANDS = ("stats", "last")


def menu_keyboard(compact: bool = False) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text="🧩 Combine images", callback_data="mode_combine")],
        [InlineKeyboardButton(text="👕 Change cloth", callback_data="mode_cloth")],
        [InlineKeyboardButton(text="✨ Restyle outfit", callback_data="mode_restyle")],
        [InlineKeyboardButton(text="🎬 Animate video", callback_data="mode_video")],
        [InlineKeyboardButton(text="🖇 2-photo video", callback_data="mode_videox")],
        [InlineKeyboardButton(text="💬 Chat", callback_data="mode_chat")],
    ]
    if compact:
        buttons.append([InlineKeyboardButton(text="📋 Open menu", callback_data="menu")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def video_duration_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text=f"⏱ {secs} seconds", callback_data=f"video_dur:{secs}")]
        for secs in config.VIDEO_DURATIONS
    ]
    rows.append([InlineKeyboardButton(text="🛑 Cancel", callback_data="video_cancel")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


# one-tap outfit presets for the "change cloth" option: key -> (label, prompt)
CLOTH_PRESETS: dict[str, tuple[str, str]] = {
    "shirt": ("👔 Shirt", "a crisp button-up shirt"),
    "tshirt": ("👕 T-shirt", "a fitted t-shirt"),
    "hoodie": ("🧥 Hoodie", "a cozy hoodie"),
    "jacket": ("🧥 Jacket", "a stylish jacket"),
    "suit": ("🤵 Suit", "a tailored suit"),
    "dress": ("👗 Dress", "an elegant dress"),
    "skirt": ("👚 Skirt", "a short skirt"),
    "jeans": ("👖 Jeans", "a pair of jeans"),
    "shorts": ("🩳 Shorts", "a pair of shorts"),
    "bikini": ("🩱 Bikini", "a bikini"),
    "swimwear": ("🏊 Swimwear", "stylish swimwear"),
    "lingerie": ("🩱 Lingerie", "lingerie"),
    "kimono": ("👘 Kimono", "a kimono"),
    "saree": ("🥻 Saree", "a saree"),
}


def cloth_keyboard() -> InlineKeyboardMarkup:
    items = list(CLOTH_PRESETS.items())
    rows = [
        [
            InlineKeyboardButton(text=label, callback_data=f"cloth:{key}")
            for key, (label, _) in items[i : i + 2]
        ]
        for i in range(0, len(items), 2)
    ]
    rows.append([InlineKeyboardButton(text="✏️ Custom prompt", callback_data="cloth:custom")])
    rows.append([InlineKeyboardButton(text="🛑 Cancel", callback_data="cloth:cancel")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


# "change cloth" only accepts clothing: reject prompts that edit anything else.
_OUTFIT_WORDS = {
    "shirt", "tshirt", "tee", "top", "blouse", "tank", "tunic", "polo", "jersey",
    "dress", "gown", "frock", "skirt", "sundress", "maxi", "midi",
    "pant", "pants", "trouser", "trousers", "jean", "jeans", "legging", "leggings",
    "short", "shorts", "cargo", "chino", "chinos",
    "hoodie", "hoody", "sweater", "sweatshirt", "jumper", "cardigan", "vest",
    "jacket", "coat", "blazer", "suit", "tuxedo", "tux", "waistcoat",
    "tie", "bowtie", "necktie",
    "bikini", "swimsuit", "swimwear", "lingerie", "bra", "underwear", "panties",
    "boxers", "briefs", "thong", "bodysuit", "leotard", "corset", "babydoll",
    "robe", "bathrobe", "kimono", "yukata", "saree", "sari", "lehenga", "kurti",
    "kurta", "salwar", "dupatta", "abaya", "hijab", "burqa", "ghagra", "dhoti",
    "jumpsuit", "romper", "overalls", "dungarees", "nightgown", "nightie",
    "pajama", "pajamas", "pyjama", "pyjamas", "onesie", "loungewear",
    "uniform", "costume", "cosplay", "outfit", "clothes", "clothing", "garment",
    "attire", "wear", "wardrobe", "activewear", "sportswear", "streetwear",
    "swim", "gym",
    # worn accessories
    "necklace", "chain", "pendant", "earring", "earrings", "bracelet", "bangle",
    "anklet", "ring", "tiara", "crown", "hat", "cap", "beanie", "beret", "bandana",
    "scarf", "stole", "shawl", "cape", "poncho", "glove", "gloves", "mitten",
    "mittens", "belt", "suspender", "suspenders", "watch", "sunglasses", "glasses",
    "goggles", "shoe", "shoes", "boot", "boots", "heel", "heels", "sneaker",
    "sneakers", "sandal", "sandals", "slipper", "slippers", "sock", "socks",
    "stocking", "stockings", "tights",
}
_OUTFIT_PHRASES = (
    "crop top", "tube top", "dress up", "t-shirt", "polo shirt",
    "button up", "button-up", "swim suit", "bathing suit",
)
# things that are NOT clothes: "add a man", "change the background", ...
_NON_OUTFIT_TARGETS = (
    "man", "men", "woman", "women", "person", "people", "guy", "girl", "boy",
    "kid", "child", "baby", "friend", "couple", "dog", "cat", "car", "tree",
    "background", "sky", "building", "animal", "logo", "text", "watermark",
    "caption", "scene", "landscape", "city", "beach", "room", "chair", "table",
)
_NON_OUTFIT_VERBS = (
    "add", "include", "insert", "put", "place", "draw", "paint", "remove",
    "delete", "erase", "change", "replace", "swap", "switch", "set", "make", "turn",
)

OUTFIT_REJECT_TEXT = (
    "❌ That is not an outfit change.\n"
    "This option only changes what the person is wearing (shirt, t-shirt, hoodie, "
    "dress, suit, bikini, ...).\n"
    "Describe only the clothing - e.g. \"red leather jacket and black jeans\"."
)


def _is_outfit_request(text: str) -> bool:
    """True only when a custom prompt is strictly about changing what is worn."""
    t = (text or "").lower().strip()
    if not t:
        return False
    non_outfit = (
        r"\b(" + "|".join(_NON_OUTFIT_VERBS) + r")\b[^.]*\b("
        + "|".join(_NON_OUTFIT_TARGETS) + r")\b"
    )
    if re.search(non_outfit, t):
        return False
    tokens = set(re.findall(r"[a-z][a-z'-]+", t))
    if tokens & _OUTFIT_WORDS:
        return True
    return any(phrase in t for phrase in _OUTFIT_PHRASES)


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


def _is_owner_id(user_id: int) -> bool:
    return bool(config.OWNER_USER_ID and user_id == config.OWNER_USER_ID)


def _is_owner(message: Message) -> bool:
    user = message.from_user
    return bool(user and _is_owner_id(user.id))


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
    user_id: int | None = None,
    status_holder: dict | None = None,
) -> bool:
    """Admit a heavy job to the queue and reply with its position.

    pre_notice=True (image jobs): the "Queued..." message is created *before*
    submit, so the worker can always edit/delete it. pre_notice=False (chat):
    a notice is created only when another job is actually ahead, which keeps
    the normal one-message-per-turn chat flow. Returns False when full.

    ``user_id`` overrides the queue-owner id, needed when ``message`` is the
    bot's own message (e.g. a duration button tap).
    """
    if user_id is None:
        if not message.from_user:
            return False
        user_id = message.from_user.id
    job = Job(kind=kind, user_id=user_id, chat_id=message.chat.id, run=run)
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
    if status_holder is not None:
        status_holder["msg"] = job.status_msg
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


async def _send_result(message: Message, payload: bytes, content_type: str, caption: str) -> None:
    caption = f"✅ done | {caption}"
    if content_type == "image/jpeg":
        await message.answer_photo(
            BufferedInputFile(payload, filename="result.jpg"),
            caption=caption,
            reply_markup=menu_keyboard(compact=True),
        )
    elif content_type == "video/mp4":
        await message.answer_video(
            BufferedInputFile(payload, filename="result.mp4"),
            caption=caption,
            reply_markup=menu_keyboard(compact=True),
            supports_streaming=True,
        )
    else:
        await message.answer_document(
            BufferedInputFile(payload, filename="result.png"),
            caption=caption,
            reply_markup=menu_keyboard(compact=True),
        )


async def _store_media(
    *,
    chat_id: int,
    user_id: int,
    message_id: int,
    mode: str,
    prompt: str | None,
    background_rgb: list[int] | None,
    originals: list[tuple[bytes, str, str]],
    result: tuple[bytes, str],
    source_file_id: str,
    elapsed_ms: int,
) -> str:
    return storage.save_media(
        chat_id=chat_id,
        user_id=user_id,
        message_id=message_id,
        mode=mode,
        prompt=prompt,
        background_rgb=background_rgb,
        originals=originals,
        result=result,
        source_file_id=source_file_id,
        elapsed_ms=elapsed_ms,
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
    return await _store_media(
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
    if local_store is None:
        await message.answer("Local storage is turned off (OFFLINE_STORE_ENABLED=0).")
        return
    media, chat = local_store.counts()
    await message.answer(
        f"📊 {media} media item(s) and {chat} chat turn(s) stored locally.\n"
        f"⚙️ jobs: {jobs.done} done, {jobs.failed} failed, {jobs.pending()} in queue"
    )


@dp.message(Command("last"))
async def cmd_last(message: Message) -> None:
    if not await _require_owner(message) or not message.from_user:
        return
    if local_store is None:
        await message.answer("Local storage is turned off (OFFLINE_STORE_ENABLED=0).")
        return
    row = local_store.latest_media(message.from_user.id)
    if row is None:
        await message.answer("Nothing stored yet.")
        return
    payload = row.result_blob
    mode = row.mode or "bg"
    caption = f"mode: {mode} | {row.prompt or 'default'}"
    if row.result_type == "video/mp4":
        await message.answer_video(
            BufferedInputFile(payload, filename="result.mp4"),
            caption=caption,
            supports_streaming=True,
        )
    elif row.result_type == "image/png":
        await message.answer_document(
            BufferedInputFile(payload, filename="result.png"), caption=caption
        )
    else:
        await message.answer_photo(
            BufferedInputFile(payload, filename="result.jpg"), caption=caption
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
    await callback.message.edit_text(
        "🧩 AI combine - send the first photo (e.g. a man)."
    )
    await callback.answer()


@dp.callback_query(F.data == "mode_cloth")
async def cb_mode_cloth(callback: CallbackQuery, state: FSMContext) -> None:
    if not _authorized_cb(callback):
        await callback.answer()
        return
    await state.set_state(Modes.cloth)
    await callback.message.edit_text(
        "👕 Send a photo of the person whose outfit you want to change."
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
    status_holder: dict = {}

    async def run() -> None:
        await _chat_action(message, "typing")
        started = time.perf_counter()
        try:
            answer, _secs = await _run_tracked(
                message, status_holder, "chat", "loading model", chat_engine.reply, text, history
            )
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
        stored = storage.save_chat(
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
        message,
        kind="chat",
        run=run,
        state=state,
        clear_state=False,
        pre_notice=False,
        status_holder=status_holder,
    )


@dp.message(Modes.chat)
async def on_chat_needs_text(message: Message) -> None:
    if not _authorized(message):
        return
    await message.answer("💬 Send a text message (or /cancel to leave chat).")


@dp.callback_query(F.data == "mode_video")
async def cb_mode_video(callback: CallbackQuery, state: FSMContext) -> None:
    if not _authorized_cb(callback):
        await callback.answer()
        return
    if not config.VIDEO_ENABLED:
        await callback.answer("🎬 Video animation is disabled.", show_alert=True)
        return
    await state.set_state(Modes.video)
    await callback.message.edit_text(
        "🎬 Video mode - send a photo. I'll ask for the motion prompt, then the "
        "clip length (5/10/15 s).\n"
        "/cancel to stop."
    )
    await callback.answer()


@dp.callback_query(F.data == "mode_videox")
async def cb_mode_videox(callback: CallbackQuery, state: FSMContext) -> None:
    if not _authorized_cb(callback):
        await callback.answer()
        return
    if not config.VIDEO_ENABLED:
        await callback.answer("🎬 Video animation is disabled.", show_alert=True)
        return
    await state.set_state(Modes.videox_subject)
    await callback.message.edit_text(
        "🖇 2-photo video - send the FIRST photo now (e.g. the boy).\n"
        "I'll ask for the second photo, then a prompt that they act out.\n"
        "/cancel to stop."
    )
    await callback.answer()


@dp.message(Modes.video, F.photo)
async def on_video_photo(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user or not message.photo:
        return
    try:
        source = await _download_photo(message)
    except Exception:
        log.exception("photo download failed")
        await message.answer("❌ Could not download that photo - try again.")
        return
    source_ext, source_type = _extension(source)
    await state.update_data(
        video_photo=source,
        video_ext=source_ext,
        video_type=source_type,
        video_file_id=message.photo[-1].file_id,
    )
    await state.set_state(Modes.video_prompt)
    await message.answer(
        "📝 Got the photo. Now send the motion prompt, e.g. \"waves rolling in\" "
        "or \"hair blowing in the wind\" (empty = gentle natural motion). "
        "I'll ask for the clip length next."
    )


# while a heavy job renders, keep one chat message updated with the live percentage;
# the same pump/bar is used by every operation (video, cloth, combine, restyle, ...)
PROGRESS_EDIT_INTERVAL = 4.0  # seconds between edits (Telegram-friendly)

_PROGRESS_TITLES = {
    "video": "🎬 Generating clip",
    "video2": "🎬 2-photo clip",
    "cloth": "👕 Changing outfit",
    "clothing": "👕 Changing outfit",
    "combine": "🧩 Combining photos",
    "restyle": "✨ Restyling outfit",
    "background": "🖼 Replacing background",
    "chat": "💬 Thinking",
}


def _progress_text(snap: JobState, fallback_started: float) -> str:
    """Render one progress message from a tracker snapshot."""
    kind = snap.kind if snap.active else ""
    title = _PROGRESS_TITLES.get(kind, "⚙️ Generating")
    percent = snap.percent if snap.active else 0.0
    stage = (snap.stage if snap.active and snap.stage else "") or "starting up"
    started = snap.started if snap.active and snap.started else fallback_started
    elapsed = max(0.0, time.time() - started)
    lines = [
        f"{title} - {percent:.0f}%",
        progress_bar(percent),
        f"Step: {stage}",
        f"Elapsed: {format_elapsed(elapsed)}",
    ]
    if 0 < percent < 100 and elapsed > 20:
        eta = elapsed * (100 - percent) / percent
        lines.append(f"Remaining: ~{format_elapsed(eta)}")
    return "\n".join(lines)


async def _progress_pump(
    message: Message,
    status_msg: Message | None,
    *,
    interval: float = PROGRESS_EDIT_INTERVAL,
) -> None:
    """Edit one chat message with the live progress until cancelled.

    Reuses the queued "Working..." notice when there is one (the job queue
    deletes it on completion); otherwise it posts and later removes its own.
    """
    own = status_msg is None
    target = status_msg
    fallback_started = time.time()
    last_text = ""
    try:
        while True:
            text = _progress_text(TRACKER.snapshot(), fallback_started)
            if text != last_text:
                if target is None:
                    try:
                        target = await message.answer(text)
                    except Exception:
                        target = None
                else:
                    try:
                        await target.edit_text(text)
                    except Exception:
                        pass
                last_text = text
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        raise
    finally:
        if own and target is not None:
            try:
                await target.delete()
            except Exception:
                pass


async def _run_tracked(
    message: Message,
    status_holder: dict,
    kind: str,
    stage: str,
    func,
    *args,
) -> Any:
    """Run sync engine ``func(*args)`` in a thread with the live chat progress bar.

    Starts the tracker + pump before the engine starts, lets the engine call
    :func:`tgbot.progress.report` as it works, and always cleans up afterwards
    (tracker reset, pump cancelled/deleted). Mirrors how the video engine runs.
    """
    pump = asyncio.create_task(_progress_pump(message, (status_holder or {}).get("msg")))
    TRACKER.begin(kind, stage=stage)
    try:
        return await asyncio.to_thread(func, *args)
    finally:
        TRACKER.end()
        pump.cancel()
        try:
            await pump
        except asyncio.CancelledError:
            pass


async def _start_video_job(
    message: Message,
    state: FSMContext,
    data: dict,
    seconds: int,
    *,
    user_id: int | None = None,
) -> None:
    """Queue a Wan/LTX job for the stored photo/prompt at the chosen length."""
    source = data.get("video_photo")
    if not source:
        await state.set_state(Modes.video)
        await message.answer("📷 Please send the photo again.")
        return
    prompt = (data.get("video_prompt") or "").strip()
    source_ext = data.get("video_ext") or "jpg"
    source_type = data.get("video_type") or "image/jpeg"
    source_file_id = data.get("video_file_id")
    chat_id = int(data.get("video_chat_id") or message.chat.id)
    owner_id = int(
        user_id or data.get("video_user_id") or (message.from_user.id if message.from_user else 0)
    )
    user_message_id = int(data.get("video_message_id") or message.message_id)

    status_holder: dict = {}

    async def run() -> None:
        await _chat_action(message, "record_video")
        started = time.perf_counter()
        pump = asyncio.create_task(_progress_pump(message, status_holder.get("msg")))
        try:
            payload, content_type, gen_secs = await asyncio.to_thread(
                video_model.generate, prompt or None, source, seconds
            )
        except Exception:
            log.exception("video generation failed")
            try:
                await message.answer("❌ Video generation failed - please try again later.")
            except Exception:
                pass
            return
        finally:
            pump.cancel()
            try:
                await pump
            except asyncio.CancelledError:
                pass
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        stored = await _store_media(
            chat_id=chat_id,
            user_id=owner_id,
            message_id=user_message_id,
            mode="video",
            prompt=prompt,
            background_rgb=None,
            originals=[(source, source_type, source_ext)],
            result=(payload, content_type),
            source_file_id=source_file_id,
            elapsed_ms=elapsed_ms,
        )
        parts = [f"🎬 {seconds}s", f"{gen_secs:.0f}s gen", f"{elapsed_ms} ms"]
        if stored:
            parts.append(stored)
        await _send_result(message, payload, content_type, " | ".join(parts))

    await _submit_job(
        message, kind="video", run=run, state=state, user_id=owner_id, status_holder=status_holder
    )


@dp.message(Modes.video_prompt, F.text)
async def on_video_prompt_text(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    data = await state.get_data()
    if not data.get("video_photo"):
        await state.set_state(Modes.video)
        await message.answer("📷 Please send the photo again.")
        return
    await state.update_data(
        video_prompt=message.text.strip(),
        video_user_id=message.from_user.id,
        video_chat_id=message.chat.id,
        video_message_id=message.message_id,
    )
    await state.set_state(Modes.video_duration)
    await message.answer(
        "⏱ How long should the clip be?",
        reply_markup=video_duration_keyboard(),
    )


@dp.message(Modes.video_prompt)
async def on_video_prompt_other(message: Message) -> None:
    if not _authorized(message):
        return
    await message.answer("📝 Send a text prompt for the motion (or /cancel to stop).")


@dp.message(Modes.video_duration)
async def on_video_duration_text(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    match = re.search(r"\d+", message.text or "")
    seconds = int(match.group()) if match else None
    if seconds not in config.VIDEO_DURATIONS:
        options = ", ".join(str(secs) for secs in config.VIDEO_DURATIONS)
        await message.answer(
            f"⏱ Pick a length: tap a button above, or reply with {options}."
        )
        return
    wait = _cooldown_seconds(message)
    if wait > 0:
        await message.answer(f"⏳ Please wait {wait:.0f}s before the next request.")
        return
    await _start_video_job(message, state, await state.get_data(), seconds)


@dp.callback_query(F.data.startswith("video_dur:"))
async def cb_video_duration(callback: CallbackQuery, state: FSMContext) -> None:
    if not _authorized_cb(callback):
        await callback.answer("Not allowed.", show_alert=True)
        return
    user = callback.from_user
    try:
        seconds = int(callback.data.split(":", 1)[1])
    except (ValueError, IndexError):
        await callback.answer("Invalid choice.", show_alert=True)
        return
    if seconds not in config.VIDEO_DURATIONS:
        await callback.answer("Invalid choice.", show_alert=True)
        return
    data = await state.get_data()
    if not data.get("video_photo"):
        await state.set_state(Modes.video)
        await callback.answer()
        await callback.message.answer("📷 Please send the photo again.")
        return
    if user is not None and not _is_owner_id(user.id) and not rate.allow(user.id):
        await callback.answer(f"⏳ Please wait {rate.wait_left(user.id):.0f}s.", show_alert=True)
        return
    await callback.answer()
    try:
        await callback.message.edit_text(f"🎬 Making a {seconds}-second clip...")
    except Exception:
        pass
    await _start_video_job(
        callback.message, state, data, seconds, user_id=user.id if user else None
    )


@dp.callback_query(F.data == "video_cancel")
async def cb_video_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    if not _authorized_cb(callback):
        await callback.answer()
        return
    await state.clear()
    await callback.answer("🛑 Cancelled.")
    try:
        await callback.message.edit_text("🛑 Cancelled.", reply_markup=menu_keyboard(compact=True))
    except Exception:
        pass


@dp.message(Modes.video)
async def on_video_needs_photo(message: Message) -> None:
    if not _authorized(message):
        return
    await message.answer("📷 Send a photo (or /cancel to stop).")


# --------------------------------------------------------------------------- #
# two-photo video: combine both photos into ONE first frame, then animate it
# --------------------------------------------------------------------------- #
_DUAL_STILL_INSTRUCTION = (
    "Create one natural realistic photo: the two people from the reference images "
    "standing together in one open scene, facing each other, whole bodies visible, "
    "with space between them. Keep both people's faces and identities exactly the same."
)


def _dual_side_by_side(first: bytes, second: bytes) -> bytes:
    """Fallback first frame: both photos side by side on one neutral canvas."""
    from PIL import Image

    def _fit_to_height(img: Image.Image, height: int) -> Image.Image:
        w, h = img.size
        return img.resize((max(1, round(w * height / h)), height), Image.LANCZOS)

    a = _fit_to_height(Image.open(io.BytesIO(first)).convert("RGB"), 512)
    b = _fit_to_height(Image.open(io.BytesIO(second)).convert("RGB"), 512)
    gap = 12
    if a.width + b.width + gap > 1280:
        scale = (1280 - gap) / (a.width + b.width)
        a = a.resize((max(16, round(a.width * scale)), 512), Image.LANCZOS)
        b = b.resize((max(16, round(b.width * scale)), 512), Image.LANCZOS)
    canvas = Image.new("RGB", (a.width + b.width + gap, 512), (240, 240, 240))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + gap, 0))
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def _build_dual_first_frame(first: bytes, second: bytes) -> bytes:
    """Build the single first frame of the video from the two uploaded photos."""
    if config.EDIT_ENABLED and edit_model.is_available():
        try:
            payload, _, _ = edit_model.generate_multi(
                _DUAL_STILL_INSTRUCTION,
                [first, second],
                instruction=_DUAL_STILL_INSTRUCTION,
            )
            return payload
        except Exception as exc:
            log.warning("two-photo video: qwen combine failed, side-by-side: %s", exc)
    return _dual_side_by_side(first, second)


async def _start_videox_job(
    message: Message,
    state: FSMContext,
    data: dict,
    seconds: int,
    *,
    user_id: int | None = None,
) -> None:
    """Combine two photos into one frame, then animate it as a Wan/ComfyUI clip."""
    first = data.get("videox_first")
    second = data.get("videox_second")
    if not first or not second:
        await state.set_state(Modes.videox_subject)
        await message.answer("📷 Please send both photos again.")
        return
    prompt = (data.get("videox_prompt") or "").strip()
    first_ext, first_type = _extension(first)
    second_ext, second_type = _extension(second)
    source_file_id = data.get("videox_second_file_id")
    chat_id = int(data.get("videox_chat_id") or message.chat.id)
    owner_id = int(
        user_id or data.get("videox_user_id") or (message.from_user.id if message.from_user else 0)
    )
    user_message_id = int(data.get("videox_message_id") or message.message_id)

    status_holder: dict = {}

    async def run() -> None:
        await _chat_action(message, "record_video")
        started = time.perf_counter()
        # span 1: compose both photos into one first frame (Qwen, else side-by-side)
        combined = await _run_tracked(
            message,
            status_holder,
            "combine",
            "composing both photos",
            _build_dual_first_frame,
            first,
            second,
        )
        # free the editor's VRAM/RAM before the video engine claims the GPU
        edit_model.unload()
        # span 2: animate the composed frame with the user's action prompt
        pump = asyncio.create_task(_progress_pump(message, status_holder.get("msg")))
        try:
            payload, content_type, gen_secs = await asyncio.to_thread(
                video_model.generate, prompt or None, combined, seconds
            )
        except Exception:
            log.exception("two-photo video generation failed")
            try:
                await message.answer("❌ Video generation failed - please try again later.")
            except Exception:
                pass
            return
        finally:
            pump.cancel()
            try:
                await pump
            except asyncio.CancelledError:
                pass
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        stored = await _store_media(
            chat_id=chat_id,
            user_id=owner_id,
            message_id=user_message_id,
            mode="video2",
            prompt=prompt,
            background_rgb=None,
            originals=[
                (first, first_type, first_ext),
                (second, second_type, second_ext),
            ],
            result=(payload, content_type),
            source_file_id=source_file_id,
            elapsed_ms=elapsed_ms,
        )
        parts = [f"🖇 {seconds}s", f"{gen_secs:.0f}s gen", f"{elapsed_ms} ms"]
        if stored:
            parts.append(stored)
        await _send_result(message, payload, content_type, " | ".join(parts))

    await _submit_job(
        message, kind="video2", run=run, state=state, user_id=owner_id, status_holder=status_holder
    )


@dp.message(Modes.videox_subject, F.photo)
async def on_videox_subject(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    try:
        first = await _download_photo(message)
    except Exception:
        log.exception("two-photo video: first image download failed")
        await message.answer("❌ Could not download that photo - try again.")
        return
    await state.update_data(videox_first=first, videox_first_file_id=message.photo[-1].file_id)
    await state.set_state(Modes.videox_second)
    await message.answer("🖇 Got the first photo. Now send the SECOND photo (e.g. the girl).")


@dp.message(Modes.videox_second, F.photo)
async def on_videox_second(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    data = await state.get_data()
    if not data.get("videox_first"):
        await state.clear()
        await message.answer("⚠️ Missing first photo - start again from /menu.")
        return
    try:
        second = await _download_photo(message)
    except Exception:
        log.exception("two-photo video: second image download failed")
        await message.answer("❌ Could not download that photo - try again.")
        await state.clear()
        return
    await state.update_data(
        videox_second=second,
        videox_second_file_id=message.photo[-1].file_id,
        videox_user_id=message.from_user.id,
        videox_chat_id=message.chat.id,
        videox_message_id=message.message_id,
    )
    await state.set_state(Modes.videox_prompt)
    await message.answer(
        "🎬 Now send the prompt that the two people act out, e.g.\n"
        '"the boy throws a ball to the girl, the girl catches the ball".\n'
        "Empty prompt = gentle natural motion."
    )


@dp.message(Modes.videox_prompt, F.photo)
async def on_videox_prompt_photo(message: Message) -> None:
    if not _authorized(message):
        return
    await message.answer("📝 Please send the action prompt as text, not a photo (e.g. \"the boy throws a ball to the girl\").")


@dp.message(Modes.videox_prompt, F.text)
async def on_videox_prompt(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    data = await state.get_data()
    if not data.get("videox_first"):
        await state.set_state(Modes.videox_subject)
        await message.answer("📷 First photo is missing - send the FIRST photo again.")
        return
    if not data.get("videox_second"):
        await state.set_state(Modes.videox_second)
        await message.answer("📷 Second photo is missing - send the SECOND photo again.")
        return
    await state.update_data(videox_prompt=message.text.strip())
    await state.set_state(Modes.videox_duration)
    await message.answer(
        "⏱ Both photos and the prompt are set. How long should the clip be?",
        reply_markup=video_duration_keyboard(),
    )


@dp.message(Modes.videox_duration)
async def on_videox_duration_text(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    match = re.search(r"\d+", message.text or "")
    seconds = int(match.group()) if match else None
    if seconds not in config.VIDEO_DURATIONS:
        options = ", ".join(str(secs) for secs in config.VIDEO_DURATIONS)
        await message.answer(
            f"⏱ Pick a length: tap a button above, or reply with {options}."
        )
        return
    wait = _cooldown_seconds(message)
    if wait > 0:
        await message.answer(f"⏳ Please wait {wait:.0f}s before the next request.")
        return
    await _start_videox_job(message, state, await state.get_data(), seconds)


@dp.callback_query(F.data.startswith("videox_dur:"))
async def cb_videox_duration(callback: CallbackQuery, state: FSMContext) -> None:
    if not _authorized_cb(callback):
        await callback.answer("Not allowed.", show_alert=True)
        return
    user = callback.from_user
    try:
        seconds = int(callback.data.split(":", 1)[1])
    except (ValueError, IndexError):
        await callback.answer("Invalid choice.", show_alert=True)
        return
    if seconds not in config.VIDEO_DURATIONS:
        await callback.answer("Invalid choice.", show_alert=True)
        return
    data = await state.get_data()
    if not data.get("videox_first"):
        await state.set_state(Modes.videox_subject)
        await callback.answer()
        await callback.message.answer("📷 First photo is missing - send the FIRST photo again.")
        return
    if not data.get("videox_second"):
        await state.set_state(Modes.videox_second)
        await callback.answer()
        await callback.message.answer("📷 Second photo is missing - send the SECOND photo again.")
        return
    if user is not None and not _is_owner_id(user.id) and not rate.allow(user.id):
        await callback.answer(f"⏳ Please wait {rate.wait_left(user.id):.0f}s.", show_alert=True)
        return
    await callback.answer()
    try:
        await callback.message.edit_text(f"🖇 Making a {seconds}-second clip...")
    except Exception:
        pass
    await _start_videox_job(
        callback.message, state, data, seconds, user_id=user.id if user else None
    )


@dp.callback_query(F.data == "videox_cancel")
async def cb_videox_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    if not _authorized_cb(callback):
        await callback.answer()
        return
    await state.clear()
    await callback.answer("🛑 Cancelled.")
    try:
        await callback.message.edit_text("🛑 Cancelled.", reply_markup=menu_keyboard(compact=True))
    except Exception:
        pass


@dp.message(Modes.videox_subject)
@dp.message(Modes.videox_second)
async def on_videox_needs_photo(message: Message) -> None:
    if not _authorized(message):
        return
    if message.photo:
        return  # let photo handler catch it
    await message.answer("📷 Send a photo (or /cancel to stop).")

@dp.message(Modes.videox_prompt)
async def on_videox_prompt_other(message: Message) -> None:
    if not _authorized(message):
        return
    if message.photo:
        return
    if message.text:
        return
    await message.answer("📝 Send the action prompt as text (or /cancel to stop).")

@dp.message(Modes.videox_duration)
async def on_videox_duration_other(message: Message) -> None:
    if not _authorized(message):
        return
    if message.photo:
        return
    await message.answer("⏱ Pick the clip length from the buttons above, or reply with a number (5, 10, 15).")


def _run_combine_edit(
    prompt: str, subject: bytes, background: bytes
) -> tuple[bytes, str, str]:
    """AI-compose two photos from a prompt; falls back to a plain composite."""
    if config.EDIT_ENABLED and edit_model.is_available():
        try:
            payload, content_type, gen = edit_model.generate_multi(
                prompt, [subject, background]
            )
            return payload, content_type, f"qwen-edit {gen:.0f}s"
        except Exception as exc:
            log.warning("combine qwen edit failed, composite fallback: %s", exc)
    payload, content_type = combine_images(subject, background, config.MAX_IMAGE_DIM)
    return payload, content_type, "composite fallback (AI editor unavailable)"


@dp.message(Modes.combine_subject, F.photo)
async def on_combine_subject(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    try:
        subject = await _download_photo(message)
    except Exception:
        log.exception("combine: first image download failed")
        await message.answer("❌ Could not download that photo - try again from /menu.")
        return
    await state.update_data(subject=subject, combine_subject_file_id=message.photo[-1].file_id)
    await state.set_state(Modes.combine_background)
    await message.answer("🧩 Got the first photo. Now send the second photo (e.g. a woman).")


@dp.message(Modes.combine_background, F.photo)
async def on_combine_background(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    data = await state.get_data()
    if not data.get("subject"):
        await state.clear()
        await message.answer("⚠️ Something went wrong - try again from /menu.")
        return
    try:
        background = await _download_photo(message)
    except Exception:
        log.exception("combine: second image download failed")
        await message.answer("❌ Could not download that photo - try again from /menu.")
        await state.clear()
        return
    await state.update_data(
        background=background,
        combine_background_file_id=message.photo[-1].file_id,
        combine_user_id=message.from_user.id,
        combine_chat_id=message.chat.id,
        combine_message_id=message.message_id,
    )
    await state.set_state(Modes.combine_prompt)
    await message.answer(
        "🧩 Both photos received. Now send a prompt for the result, e.g. "
        '"hug each other", "standing together on a beach".'
    )


@dp.message(Modes.combine_prompt, F.photo)
async def on_combine_prompt_photo(message: Message) -> None:
    if not _authorized(message):
        return
    await message.answer("🧩 Please send a text prompt describing the result.")


@dp.message(Modes.combine_prompt, F.text)
async def on_combine_prompt(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    wait = _cooldown_seconds(message)
    if wait > 0:
        await message.answer(f"⏳ Please wait {wait:.0f}s before the next request.")
        return
    data = await state.get_data()
    subject = data.get("subject")
    background = data.get("background")
    if not subject or not background:
        await state.clear()
        await message.answer("⚠️ Missing photos - start again from /menu.")
        return
    prompt = (message.text or "").strip() or "the two people standing together"
    subject_ext, subject_type = _extension(subject)
    background_ext, background_type = _extension(background)
    chat_id = int(data.get("combine_chat_id") or message.chat.id)
    owner_id = int(data.get("combine_user_id") or message.from_user.id)
    user_message_id = int(data.get("combine_message_id") or message.message_id)
    source_file_id = data.get("combine_background_file_id")
    status_holder: dict = {}

    async def run() -> None:
        await _chat_action(message, "upload_photo")
        started = time.perf_counter()
        payload, content_type, engine = await _run_tracked(
            message,
            status_holder,
            "combine",
            "preparing photos",
            _run_combine_edit,
            prompt,
            subject,
            background,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        stored = await _store_media(
            chat_id=chat_id,
            user_id=owner_id,
            message_id=user_message_id,
            mode="combine",
            prompt=prompt,
            background_rgb=None,
            originals=[
                (subject, subject_type, subject_ext),
                (background, background_type, background_ext),
            ],
            result=(payload, content_type),
            source_file_id=source_file_id,
            elapsed_ms=elapsed_ms,
        )
        parts = [f"combined: {prompt}", engine, f"{elapsed_ms} ms"]
        if stored:
            parts.append(stored)
        await _send_result(message, payload, content_type, " | ".join(parts))

    await _submit_job(
        message, kind="combine", run=run, state=state, status_holder=status_holder
    )


def _run_outfit_edit(prompt: str, source: bytes) -> tuple[bytes, str, str]:
    """Change only what the person is wearing; returns (payload, mime, engine)."""
    if config.EDIT_ENABLED and edit_model.is_available():
        try:
            payload, content_type, gen = edit_model.generate(prompt, source)
            return payload, content_type, f"qwen-edit {gen:.0f}s"
        except Exception as exc:
            log.warning("outfit qwen edit failed, trying diffusion: %s", exc)
    if config.DIFFUSE_ENABLED:
        try:
            payload, content_type, gen = diffuse_generate(prompt, source)
            return payload, content_type, f"diffusion {gen:.0f}s"
        except Exception as exc:
            log.warning("outfit diffusion failed, tint fallback: %s", exc)
    payload, content_type = change_cloth_color(source, (200, 200, 200), config.MAX_IMAGE_DIM)
    return payload, content_type, "tint (diffusion disabled)"


def _run_restyle(prompt: str, source: bytes) -> tuple[bytes, str, str]:
    """Restyle an outfit from a free prompt; returns (payload, mime, engine)."""
    if config.EDIT_ENABLED and edit_model.is_available():
        try:
            payload, content_type, gen = edit_model.generate(prompt, source)
            return payload, content_type, f"qwen-edit {gen:.0f}s"
        except Exception as exc:
            log.warning("restyle qwen edit failed, trying diffusion: %s", exc)
    if config.DIFFUSE_ENABLED:
        try:
            payload, content_type, gen = diffuse_generate(prompt, source)
            return payload, content_type, f"diffusion {gen:.0f}s"
        except Exception as exc:
            log.warning("restyle diffusion failed, tint fallback: %s", exc)
    (payload, content_type), _, _ = restyle_outfit(
        source, prompt, config.DEFAULT_BACKGROUND, config.MAX_IMAGE_DIM
    )
    return payload, content_type, "tint"


async def _start_cloth_job(
    message: Message,
    state: FSMContext,
    data: dict,
    prompt: str,
    *,
    caption: str | None = None,
) -> None:
    """Queue an outfit-only edit for the photo stored in the FSM data."""
    source = data.get("cloth_photo")
    if not source:
        await state.clear()
        await message.answer("📷 Please send the photo again from /menu.")
        return
    source_ext = data.get("cloth_ext") or "jpg"
    source_type = data.get("cloth_type") or "image/jpeg"
    source_file_id = data.get("cloth_file_id")
    chat_id = int(data.get("cloth_chat_id") or message.chat.id)
    owner_id = int(
        data.get("cloth_user_id") or (message.from_user.id if message.from_user else 0)
    )
    user_message_id = int(data.get("cloth_message_id") or message.message_id)
    label = caption or prompt
    status_holder: dict = {}

    async def run() -> None:
        await _chat_action(message, "upload_photo")
        started = time.perf_counter()
        payload, content_type, engine = await _run_tracked(
            message,
            status_holder,
            "clothing",
            "preparing photo",
            _run_outfit_edit,
            prompt,
            source,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        stored = await _store_media(
            chat_id=chat_id,
            user_id=owner_id,
            message_id=user_message_id,
            mode="cloth",
            prompt=prompt,
            background_rgb=None,
            originals=[(source, source_type, source_ext)],
            result=(payload, content_type),
            source_file_id=source_file_id,
            elapsed_ms=elapsed_ms,
        )
        parts = [f"outfit: {label}", engine, f"{elapsed_ms} ms"]
        if stored:
            parts.append(stored)
        await _send_result(message, payload, content_type, " | ".join(parts))

    await _submit_job(
        message,
        kind="clothing",
        run=run,
        state=state,
        user_id=owner_id,
        status_holder=status_holder,
    )


@dp.message(Modes.cloth, F.photo)
async def on_cloth(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    try:
        source = await _download_photo(message)
    except Exception:
        log.exception("cloth: download failed")
        await message.answer("❌ Could not download that photo - try again from /menu.")
        await state.clear()
        return
    source_ext, source_type = _extension(source)
    await state.update_data(
        cloth_photo=source,
        cloth_ext=source_ext,
        cloth_type=source_type,
        cloth_file_id=message.photo[-1].file_id,
        cloth_user_id=message.from_user.id,
        cloth_chat_id=message.chat.id,
        cloth_message_id=message.message_id,
    )
    await state.set_state(Modes.cloth_choice)
    await message.answer(
        "👕 Pick an outfit, or choose Custom to describe it yourself.",
        reply_markup=cloth_keyboard(),
    )


@dp.callback_query(F.data.startswith("cloth:"))
async def cb_cloth_choice(callback: CallbackQuery, state: FSMContext) -> None:
    if not _authorized_cb(callback):
        await callback.answer()
        return
    action = callback.data.split(":", 1)[1]
    if action == "cancel":
        await state.clear()
        await callback.answer("🛑 Cancelled.")
        try:
            await callback.message.edit_text(
                "🛑 Cancelled.", reply_markup=menu_keyboard(compact=True)
            )
        except Exception:
            pass
        return
    data = await state.get_data()
    if not data.get("cloth_photo"):
        await state.clear()
        await callback.answer()
        await callback.message.answer("📷 Please send the photo again from /menu.")
        return
    if action == "custom":
        await state.set_state(Modes.cloth_custom)
        await callback.answer()
        try:
            await callback.message.edit_text(
                "✏️ Describe only the outfit, e.g. \"red leather jacket and black jeans\".\n"
                "Other edits (backgrounds, adding people or objects) are not accepted here."
            )
        except Exception:
            pass
        return
    preset = CLOTH_PRESETS.get(action)
    if preset is None:
        await callback.answer("Unknown option.", show_alert=True)
        return
    user = callback.from_user
    if user is not None and not _is_owner_id(user.id) and not rate.allow(user.id):
        await callback.answer(f"⏳ Please wait {rate.wait_left(user.id):.0f}s.", show_alert=True)
        return
    label, prompt = preset
    await callback.answer()
    try:
        await callback.message.edit_text(f"👕 {label} - generating…")
    except Exception:
        pass
    await _start_cloth_job(callback.message, state, data, prompt, caption=label)


@dp.message(Modes.cloth_custom, F.photo)
async def on_cloth_custom_photo(message: Message) -> None:
    if not _authorized(message):
        return
    await message.answer("✏️ Please send the outfit as text, not a photo.")


@dp.message(Modes.cloth_custom, F.text)
async def on_cloth_custom(message: Message, state: FSMContext) -> None:
    if not _authorized(message) or not message.from_user:
        return
    text = (message.text or "").strip()
    if not _is_outfit_request(text):
        await message.answer(OUTFIT_REJECT_TEXT)
        return
    wait = _cooldown_seconds(message)
    if wait > 0:
        await message.answer(f"⏳ Please wait {wait:.0f}s before the next request.")
        return
    data = await state.get_data()
    await _start_cloth_job(message, state, data, text, caption=text)


@dp.message(Modes.cloth_choice)
async def on_cloth_choice_other(message: Message) -> None:
    if not _authorized(message):
        return
    await message.answer("👕 Tap an outfit option above (or Custom), or /cancel.")


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
    status_holder: dict = {}

    async def run() -> None:
        await _chat_action(message, "upload_photo")
        started = time.perf_counter()
        try:
            payload, content_type, engine_str = await _run_tracked(
                message,
                status_holder,
                "restyle",
                "preparing photo",
                _run_restyle,
                prompt,
                source,
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            stored = await _store_uploads(
                message,
                mode="restyle",
                prompt=prompt,
                background_rgb=None,
                originals=[(source, source_type, source_ext)],
                result=(payload, content_type),
                source_file_id=file_id,
                elapsed_ms=elapsed_ms,
            )
            parts = [f"restyled: {prompt}", engine_str, f"{elapsed_ms} ms"]
            if stored:
                parts.append(stored)
            await _send_result(message, payload, content_type, " | ".join(parts))
        except Exception:
            log.exception("restyle generation failed")
            try:
                await message.answer("❌ Restyle failed - please try again.")
            except Exception:
                pass

    await _submit_job(
        message, kind="restyle", run=run, state=state, status_holder=status_holder
    )


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
    status_holder: dict = {}

    async def run() -> None:
        await _chat_action(message, "upload_photo")
        started = time.perf_counter()
        result, content_type = await _run_tracked(
            message,
            status_holder,
            "background",
            "replacing background",
            change_background,
            source,
            background,
            config.MAX_IMAGE_DIM,
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

    await _submit_job(
        message, kind="background", run=run, state=state, status_holder=status_holder
    )


@dp.message()
async def on_other(message: Message) -> None:
    if not _authorized(message):
        return
    if message.photo:
        return
    await message.answer("📷 Send me a photo - or pick an operation from /menu (including Chat).")


async def main() -> None:
    global bot, local_store, storage
    local_store = (
        LocalStore(config.OFFLINE_STORE_PATH) if config.OFFLINE_STORE_ENABLED else None
    )
    storage = Storage(local_store)
    if local_store is not None:
        log.info("local SQLite storage active at %s", config.OFFLINE_STORE_PATH)
    else:
        log.info("local storage disabled - results are not saved")

    config.validate()
    bot = Bot(token=config.TELEGRAM_BOT_TOKEN, default=DefaultBotProperties())
    try:
        await bot.set_my_commands(BOT_COMMANDS)
        log.info("registered %d bot commands for the chat input menu", len(BOT_COMMANDS))
    except Exception as exc:  # noqa: BLE001
        log.warning("could not register bot commands: %s", exc)
    if config.OWNER_USER_ID:
        log.info("admin commands enabled for user %s (hidden from the menu)", config.OWNER_USER_ID)
    else:
        log.warning("OWNER_USER_ID not set - hidden admin commands (stats, last) are disabled")
    log.info("per-user cooldown: %.0fs", config.COOLDOWN_SECONDS)
    await jobs.start()
    log.info("starting long polling")
    try:
        await dp.start_polling(bot)
    finally:
        await jobs.stop()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())