"""Chat orchestration for the web UI.

The Telegram bot and this UI each talk to the same chat backend (a local
transformers model or a shared Ollama server), so this lock only serializes
requests inside the web process. The blocking token generator is drained in a
worker thread and forwarded to the aiohttp response as SSE events, which keeps
the event loop free for health checks and the 409 busy path.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import operator
import threading
import time
from datetime import datetime
from typing import Any, AsyncIterator

from tgbot import bgchange
from tgbot import chat as chat_engine
from tgbot import config

from .store import WebStore
from .uploads import public_upload, text_excerpt

# The three menu entries in the Vertex shell, mapped to real sampling settings.
PRESETS: dict[str, dict[str, object]] = {
    "core": {
        "label": "Vertex Core",
        "max_new_tokens": 384,
        "temperature": 0.7,
        "top_p": 0.9,
    },
    "max": {
        "label": "Vertex Max",
        "max_new_tokens": 768,
        "temperature": 0.8,
        "top_p": 0.95,
    },
    "lite": {
        "label": "Vertex Lite",
        "max_new_tokens": 192,
        "temperature": 0.5,
        "top_p": 0.9,
    },
}

DEFAULT_PRESET = "core"

# Local tools exposed by the composer's tools menu. "describe" and "summarize"
# route through the model; the others are deterministic/local operations.
TOOLS: dict[str, str] = {
    "describe": "Describe image",
    "summarize": "Summarize file",
    "removebg": "Remove background",
    "time": "Time & date",
    "calc": "Calculator",
}

# A hard ceiling so a runaway generation can never hold the lock forever.
MAX_GENERATION_SECONDS = 240.0
MAX_VISION_IMAGES = 2

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def normalize_preset(name: str | None) -> str:
    return name if name in PRESETS else DEFAULT_PRESET


def normalize_tool(name: str | None) -> str:
    return name if name in TOOLS else ""


def _calc_value(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_calc_value(node.left), _calc_value(node.right))
    if isinstance(node, ast.UnaryOp):
        value = _calc_value(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
    raise ValueError("only numbers and + - * / // % ** are supported")


def calculate(expression: str) -> str:
    """Evaluate a small arithmetic expression without using eval()."""
    clean = (expression or "").strip().replace("^", "**")
    if not clean:
        raise ValueError("enter an expression to calculate")
    if len(clean) > 200:
        raise ValueError("that expression is too long")
    try:
        tree = ast.parse(clean, mode="eval")
    except SyntaxError as exc:
        raise ValueError("that does not look like a valid expression") from exc
    try:
        value = _calc_value(tree.body)
    except ZeroDivisionError as exc:
        raise ValueError("division by zero") from exc
    except (OverflowError, TypeError) as exc:
        raise ValueError("that expression is too large") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("the result is not a finite number")
    if value.is_integer():
        return str(int(value))
    return f"{value:.10g}"


def _time_answer() -> str:
    now = datetime.now().astimezone()
    return now.strftime("%A, %B %d, %Y at %I:%M %p %Z").replace(" 0", " ")


def _history_for_model(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Add small attachment notes to history without re-sending image bytes."""
    history: list[dict[str, str]] = []
    for row in rows:
        text = str(row.get("content") or "")
        attachments = row.get("attachments") or []
        if attachments:
            names = ", ".join(str(a.get("name") or "file") for a in attachments)
            note = f"[attached: {names}]"
            text = f"{text}\n{note}".strip()
        history.append({"role": row.get("role") or "user", "content": text})
    return history


def title_for(text: str) -> str:
    """Match the Vertex shell's title rule exactly."""
    clean = " ".join((text or "").split())
    if not clean:
        return "New chat"
    return clean[:34] + "…" if len(clean) > 34 else clean


class BusyError(RuntimeError):
    """Another generation is already running."""


class DisabledError(RuntimeError):
    """``CHAT_ENABLED=0``."""


class ChatService:
    def __init__(self, store: WebStore) -> None:
        self.store = store
        self._lock = asyncio.Lock()
        self.started_at = time.time()

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    async def acquire(self) -> None:
        """Reserve the model, or fail fast so the handler can return 409."""
        if self._lock.locked():
            raise BusyError("the model is busy with another reply")
        await self._lock.acquire()

    def release(self) -> None:
        if self._lock.locked():
            self._lock.release()

    def health(self) -> dict:
        return {
            "ok": True,
            "enabled": bool(config.CHAT_ENABLED),
            "busy": self.busy,
            "model": chat_engine.active_model(),
            "presets": [
                {"id": key, "label": value["label"]} for key, value in PRESETS.items()
            ],
            "tools": [
                {"id": key, "label": value} for key, value in TOOLS.items()
            ],
            "max_attachments": int(config.WEB_MAX_ATTACHMENTS),
            "vision": config.CHAT_BACKEND == "ollama",
            "uptime_seconds": int(time.time() - self.started_at),
            **self.store.counts(),
        }

    def _resolve_attachments(
        self, session_id: str, ids: list[str] | None
    ) -> list[dict[str, Any]]:
        clean_ids = list(dict.fromkeys(str(item)[:64] for item in (ids or []) if item))
        if len(clean_ids) > config.WEB_MAX_ATTACHMENTS:
            raise ValueError(
                f"attach at most {config.WEB_MAX_ATTACHMENTS} files per message"
            )
        rows: list[dict[str, Any]] = []
        for uid in clean_ids:
            row = self.store.get_upload(uid)
            if row is None:
                raise ValueError("one of the attachments no longer exists")
            if row.get("session_id") not in (None, session_id):
                raise ValueError("an attachment belongs to another chat")
            rows.append(row)
        images = sum(1 for row in rows if row.get("kind") == "image")
        if images > MAX_VISION_IMAGES:
            raise ValueError(f"attach at most {MAX_VISION_IMAGES} images per message")
        return rows

    @staticmethod
    def _display_content(
        content: str, rows: list[dict[str, Any]], tool: str
    ) -> str:
        text = (content or "").strip()
        if tool == "removebg":
            if text.lower().startswith("remove background"):
                return text
            return f"Remove background: {text}" if text else "Remove background"
        if text:
            return text
        if tool == "time":
            return "What time is it?"
        if tool == "describe":
            return "Describe the attached image"
        if tool == "summarize":
            return "Summarize the attached file"
        kinds = {row.get("kind") for row in rows}
        if kinds == {"image"}:
            return "Describe the attached image"
        if kinds == {"text"}:
            return "Summarize the attached file"
        if rows:
            return "Analyze the attached files"
        return "New chat"

    @staticmethod
    def _prompt_content(
        content: str, rows: list[dict[str, Any]], tool: str
    ) -> str:
        text = (content or "").strip()
        if tool == "time":
            return text or "What time is it?"
        if tool == "calc":
            return text
        if tool == "removebg":
            prefix = "Remove background:"
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix) :].strip()
            return text or "white"
        if tool == "describe":
            text = text or "Describe the attached image in detail."
        elif tool == "summarize":
            text = text or "Summarize the attached file."
        elif not text:
            kinds = {row.get("kind") for row in rows}
            if kinds == {"image"}:
                text = "Describe the attached image."
            elif kinds == {"text"}:
                text = "Summarize the attached file."
            elif rows:
                text = "Analyze the attached files."
        parts = [text] if text else []
        for row in rows:
            if row.get("kind") != "text":
                continue
            excerpt = text_excerpt(
                str(row.get("text_content") or ""),
                config.WEB_ATTACHMENT_TEXT_LIMIT,
            )
            if excerpt:
                parts.append(
                    f"\n\n--- Attached file: {row.get('filename') or 'file'} ---\n"
                    f"{excerpt}\n--- end attached file ---"
                )
        return "\n".join(parts).strip()

    def _validate_tool(
        self, tool: str, rows: list[dict[str, Any]], prompt: str
    ) -> None:
        kinds = {row.get("kind") for row in rows}
        if tool in {"describe", "removebg"} and "image" not in kinds:
            raise ValueError("attach an image for this tool")
        if tool == "summarize" and "text" not in kinds:
            raise ValueError("attach a text file for this tool")
        if tool == "calc" and not prompt:
            raise ValueError("enter an expression to calculate")

    def _image_payload(self, rows: list[dict[str, Any]]) -> list[str]:
        images: list[str] = []
        for row in rows:
            if row.get("kind") != "image":
                continue
            path = self.store.upload_path(row)
            images.append(base64.b64encode(path.read_bytes()).decode("ascii"))
        return images

    def _attachment_meta(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [public_upload(row) for row in rows]

    def _finish_title(self, session: dict[str, Any], text: str) -> str:
        title = str(session.get("title") or "New chat")
        current = self.store.get_session(session["id"])
        if current is not None:
            title = str(current.get("title") or title)
        if title == "New chat":
            updated = self.store.rename_session(session["id"], title_for(text))
            if updated:
                title = str(updated.get("title") or title)
        return title

    async def _remove_background(
        self, rows: list[dict[str, Any]], prompt: str
    ) -> tuple[str, list[dict[str, Any]]]:
        image = next(row for row in rows if row.get("kind") == "image")
        source = self.store.upload_path(image).read_bytes()
        background, note = bgchange.parse_background(
            prompt, config.DEFAULT_BACKGROUND
        )
        # Free the chat model before rembg allocates its own session.
        await asyncio.to_thread(chat_engine.unload)
        payload, content_type = await asyncio.to_thread(
            bgchange.change_background, source, background, config.MAX_IMAGE_DIM
        )
        suffix = "png" if content_type == "image/png" else "jpg"
        result = self.store.add_upload(
            filename=f"background-removed.{suffix}",
            mime=content_type,
            kind="image",
            data=payload,
        )
        answer = (
            "Removed the background and replaced it with "
            f"{bgchange.describe(background)}."
        )
        if note:
            answer += f" {note}"
        return answer, [public_upload(result)]

    async def stream(
        self,
        session_id: str,
        content: str,
        *,
        preset: str = DEFAULT_PRESET,
        client_id: str | None = None,
        regenerate: bool = False,
        attachments: list[str] | None = None,
        tool: str | None = None,
    ) -> AsyncIterator[tuple[str, dict]]:
        """Yield ``(event, payload)`` tuples: meta, token, done or error.

        The caller must hold the lock (see :meth:`acquire`) for the whole
        iteration, which is what makes the 409-before-SSE path correct.
        """
        if not self._lock.locked():
            raise RuntimeError("ChatService.stream() requires acquire() first")
        if not config.CHAT_ENABLED:
            raise DisabledError("chat is disabled (CHAT_ENABLED=0)")

        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        preset = normalize_preset(preset)
        cfg = PRESETS[preset]

        # A retried request with the same client id replays the saved turn
        # instead of generating (and paying for) the same answer twice.
        if client_id:
            existing = self.store.find_turn_by_client_id(session_id, client_id)
            if existing is not None:
                reply = existing.get("assistant")
                if reply:
                    yield ("meta", {"session_id": session_id, "message_id": reply["id"]})
                    yield ("token", {"text": reply["content"]})
                    yield (
                        "done",
                        {
                            "session_id": session_id,
                            "message_id": reply["id"],
                            "content": reply["content"],
                            "attachments": reply.get("attachments") or [],
                            "elapsed_ms": reply.get("elapsed_ms"),
                            "title": session["title"],
                            "replayed": True,
                        },
                    )
                return

        user_message: dict[str, Any] | None = None
        if regenerate:
            last_user = self.store.last_user_message(session_id)
            if last_user is None:
                raise ValueError("nothing to regenerate")
            tool = normalize_tool(last_user.get("tool"))
            attachment_ids = [
                str(item.get("id"))
                for item in (last_user.get("attachments") or [])
                if item.get("id")
            ]
            rows = self._resolve_attachments(session_id, attachment_ids)
            display_text = str(last_user.get("content") or "")
            prompt_text = self._prompt_content(display_text, rows, tool)
            self._validate_tool(tool, rows, prompt_text)
            user_message = last_user
        else:
            tool = normalize_tool(tool)
            rows = self._resolve_attachments(session_id, attachments)
            display_text = self._display_content(content, rows, tool)
            prompt_text = self._prompt_content(content, rows, tool)
            self._validate_tool(tool, rows, prompt_text)

        # Deterministic tools answer without touching the model. Validate the
        # expression before saving the turn so a typo leaves no half turn.
        instant: str | None = None
        if tool == "calc":
            instant = f"**{display_text}** = {calculate(prompt_text)}"
        elif tool == "time":
            instant = f"Current local time: {_time_answer()}."

        if (
            any(row.get("kind") == "image" for row in rows)
            and config.CHAT_BACKEND != "ollama"
            and tool not in {"removebg", "time", "calc"}
        ):
            raise ValueError(
                "image attachments need CHAT_BACKEND=ollama with a vision model"
            )

        if regenerate:
            self.store.delete_last_assistant(session_id)
        else:
            user_message = self.store.add_message(
                session_id,
                "user",
                display_text,
                preset=preset,
                client_id=client_id,
                attachments=self._attachment_meta(rows),
                tool=tool or None,
            )
            self.store.attach_uploads([row["id"] for row in rows], session_id)
        assert user_message is not None

        images = (
            []
            if instant is not None or tool == "removebg"
            else self._image_payload(rows)
        )

        history = _history_for_model(
            self.store.history_before(
                session_id, user_message["id"], config.CHAT_HISTORY_TURNS
            )
        )
        yield ("meta", {"session_id": session_id, "user_message_id": user_message["id"]})

        if instant is not None:
            assistant = self.store.add_message(
                session_id,
                "assistant",
                instant,
                preset=preset,
                elapsed_ms=0,
            )
            title = self._finish_title(session, display_text)
            yield ("token", {"text": instant})
            yield (
                "done",
                {
                    "session_id": session_id,
                    "message_id": assistant["id"],
                    "content": instant,
                    "attachments": [],
                    "elapsed_ms": 0,
                    "title": title,
                },
            )
            return

        if tool == "removebg":
            answer, result_atts = await self._remove_background(rows, prompt_text)
            assistant = self.store.add_message(
                session_id,
                "assistant",
                answer,
                preset=preset,
                elapsed_ms=0,
                attachments=result_atts,
            )
            title = self._finish_title(session, display_text)
            yield ("token", {"text": answer})
            yield (
                "done",
                {
                    "session_id": session_id,
                    "message_id": assistant["id"],
                    "content": answer,
                    "attachments": result_atts,
                    "elapsed_ms": 0,
                    "title": title,
                },
            )
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
        stop = threading.Event()
        parts: list[str] = []

        def emit(kind: str, payload: object) -> None:
            if loop.is_closed():
                return
            try:
                loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))
            except RuntimeError:
                pass

        def worker() -> None:
            try:
                for chunk in chat_engine.stream_reply(
                    prompt_text,
                    history,
                    system_prompt=chat_engine.WEB_SYSTEM_PROMPT,
                    max_new_tokens=cfg["max_new_tokens"],
                    temperature=cfg["temperature"],
                    top_p=cfg["top_p"],
                    stop_event=stop,
                    images=images,
                ):
                    emit("token", chunk)
                emit("end", None)
            except Exception as exc:  # forwarded to the browser as an SSE error
                emit("error", exc)

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()
        timer = threading.Timer(MAX_GENERATION_SECONDS, stop.set)
        timer.daemon = True
        timer.start()
        started = time.perf_counter()
        completed = False
        failed = False
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "token":
                    text = str(payload)
                    parts.append(text)
                    yield ("token", {"text": text})
                elif kind == "end":
                    break
                else:
                    failed = True
                    raise payload  # type: ignore[misc]

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            answer = "".join(parts).strip() or "I'm not sure how to answer that."
            assistant = self.store.add_message(
                session_id,
                "assistant",
                answer,
                preset=preset,
                elapsed_ms=elapsed_ms,
            )
            title = self._finish_title(session, display_text)
            completed = True
            yield (
                "done",
                {
                    "session_id": session_id,
                    "message_id": assistant["id"],
                    "content": answer,
                    "attachments": [],
                    "elapsed_ms": elapsed_ms,
                    "title": title,
                },
            )
        finally:
            stop.set()
            timer.cancel()
            # Do not release the model until the worker has actually stopped.
            try:
                await asyncio.to_thread(worker_thread.join, 10.0)
            except Exception:
                pass
            # Keep a stopped/partial answer so a refresh does not lose it.
            partial = "".join(parts).strip()
            if not completed and partial:
                try:
                    self.store.add_message(
                        session_id,
                        "assistant",
                        partial,
                        preset=preset,
                        status="error" if failed else "stopped",
                    )
                except Exception:
                    pass
