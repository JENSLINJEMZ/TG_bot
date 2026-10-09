"""Chat-mode handler regression: the message must actually reach the queue worker.

Guards the scoping bug where the reply closure reassigned `history`, turning it
into a local variable and raising UnboundLocalError on every chat message.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tgbot import bot
from tgbot.jobs import JobQueue


class FakeMessage:
    def __init__(self, text: str, uid: int = 1, cid: int = 1) -> None:
        self.text = text
        self.chat = SimpleNamespace(id=cid)
        self.from_user = SimpleNamespace(id=uid)
        self.message_id = 42
        self.sent: list[str] = []
        self.bot = SimpleNamespace(send_chat_action=self._action)

    async def _action(self, *args, **kwargs):
        return None

    async def answer(self, text, **kwargs):
        self.sent.append(text)
        return SimpleNamespace(edit_text=self._noop, delete=self._noop)

    async def _noop(self, *args, **kwargs):
        return None


class FakeState:
    def __init__(self, data=None) -> None:
        self.data = data or {}

    async def get_data(self):
        return dict(self.data)

    async def update_data(self, **kwargs):
        self.data.update(kwargs)

    async def clear(self):
        self.data.clear()


async def _noop_save_chat(**_kwargs):
    return ""


def test_chat_handler_replies_and_keeps_history(monkeypatch) -> None:
    async def scenario() -> None:
        msg = FakeMessage("hello")
        state = FakeState(
            {"chat_history": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}
        )
        monkeypatch.setattr(bot.chat_engine, "reply", lambda text, hist: ("HI THERE", 0.1))
        monkeypatch.setattr(bot, "storage", SimpleNamespace(save_chat=_noop_save_chat))
        monkeypatch.setattr(bot, "_cooldown_seconds", lambda message: 0.0)
        q = JobQueue(workers=1)
        monkeypatch.setattr(bot, "jobs", q)
        await q.start()
        try:
            await bot.on_chat_text(msg, state)
            await q._queue.join()
        finally:
            await q.stop()

        assert any("HI THERE" in s for s in msg.sent), msg.sent
        assert not any("Chat failed" in s for s in msg.sent), msg.sent
        assert state.data["chat_history"][-1] == {"role": "assistant", "content": "HI THERE"}

    asyncio.run(scenario())


def test_chat_handler_reports_failure_without_leaking_exception(monkeypatch) -> None:
    async def scenario() -> None:
        msg = FakeMessage("hello")
        state = FakeState({"chat_history": []})

        def boom(text, hist):
            raise RuntimeError("secret internal detail")

        monkeypatch.setattr(bot.chat_engine, "reply", boom)
        monkeypatch.setattr(bot, "storage", SimpleNamespace(save_chat=_noop_save_chat))
        monkeypatch.setattr(bot, "_cooldown_seconds", lambda message: 0.0)
        q = JobQueue(workers=1)
        monkeypatch.setattr(bot, "jobs", q)
        await q.start()
        try:
            await bot.on_chat_text(msg, state)
            await q._queue.join()
        finally:
            await q.stop()

        assert any("Chat failed" in s for s in msg.sent), msg.sent
        assert not any("secret internal detail" in s for s in msg.sent), msg.sent

    asyncio.run(scenario())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"skip {name} (needs pytest monkeypatch)")
