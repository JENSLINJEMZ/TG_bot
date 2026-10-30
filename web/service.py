"""Chat orchestration for the web UI.

The Telegram bot and this UI each talk to the same chat backend (a local
transformers model or a shared Ollama server), so this lock only serializes
requests inside the web process. The blocking token generator is drained in a
worker thread and forwarded to the aiohttp response as SSE events, which keeps
the event loop free for health checks and the 409 busy path.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import AsyncIterator

from tgbot import chat as chat_engine
from tgbot import config

from .store import WebStore

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

# A hard ceiling so a runaway generation can never hold the lock forever.
MAX_GENERATION_SECONDS = 240.0


def normalize_preset(name: str | None) -> str:
    return name if name in PRESETS else DEFAULT_PRESET


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
            "uptime_seconds": int(time.time() - self.started_at),
            **self.store.counts(),
        }

    async def stream(
        self,
        session_id: str,
        content: str,
        *,
        preset: str = DEFAULT_PRESET,
        client_id: str | None = None,
        regenerate: bool = False,
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
                            "elapsed_ms": reply.get("elapsed_ms"),
                            "title": session["title"],
                            "replayed": True,
                        },
                    )
                return

        if regenerate:
            last_user = self.store.last_user_message(session_id)
            if last_user is None:
                raise ValueError("nothing to regenerate")
            self.store.delete_last_assistant(session_id)
            user_message = last_user
            user_text = last_user["content"]
        else:
            user_text = content
            user_message = self.store.add_message(
                session_id, "user", user_text, preset=preset, client_id=client_id
            )

        history = self.store.history_before(
            session_id, user_message["id"], config.CHAT_HISTORY_TURNS
        )
        yield ("meta", {"session_id": session_id, "user_message_id": user_message["id"]})

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
                    user_text,
                    history,
                    system_prompt=chat_engine.WEB_SYSTEM_PROMPT,
                    max_new_tokens=cfg["max_new_tokens"],
                    temperature=cfg["temperature"],
                    top_p=cfg["top_p"],
                    stop_event=stop,
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
            title = session["title"]
            current = self.store.get_session(session_id)
            if current is not None:
                title = current["title"]
            if not regenerate and title == "New chat":
                updated = self.store.rename_session(session_id, title_for(user_text))
                if updated:
                    title = updated["title"]
            completed = True
            yield (
                "done",
                {
                    "session_id": session_id,
                    "message_id": assistant["id"],
                    "content": answer,
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
