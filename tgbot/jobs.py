"""Bounded job queue for the heavy image work.

One worker (or MAX_CONCURRENT workers) drains a FIFO queue, so the bot never
runs two CPU-heavy jobs at once. Users get an instant "queued" notice, the
worker edits it to "Working..." while the job runs, and the result message
itself is the "done" notification in the originating chat. Failures reply with
a generic message and the traceback goes to the log only.

The queue is capped globally and per user, so a flood of requests (or several
accounts at once) cannot pile up work and RAM the way the old inline handlers
could.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

log = logging.getLogger("tgbot.jobs")

QUEUE_FULL_TEXT = "❌ The bot is busy - too many queued jobs. Try again in a minute."


@dataclass
class Job:
    kind: str
    user_id: int
    chat_id: int
    run: Callable[[], Awaitable[None]]
    # aiogram Message (typed loosely to keep this module import-light)
    status_msg: object | None = None
    submitted: float = field(default_factory=time.monotonic)


class JobQueue:
    """FIFO queue with a fixed pool of workers and bounded admission."""

    def __init__(self, *, workers: int = 1, max_pending: int = 20, max_per_user: int = 3) -> None:
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self.workers = max(1, workers)
        self.max_pending = max_pending
        self.max_per_user = max_per_user
        self._per_user: dict[int, int] = {}
        self._tasks: list[asyncio.Task] = []
        self.done = 0
        self.failed = 0

    def pending(self, user_id: int | None = None) -> int:
        """Jobs waiting (or running). With user_id, only that user's jobs."""
        if user_id is None:
            return sum(self._per_user.values())
        return self._per_user.get(user_id, 0)

    def submit(self, job: Job) -> int | None:
        """Admit a job. Returns its 1-based queue position, or None if full."""
        if self._per_user.get(job.user_id, 0) >= self.max_per_user:
            return None
        if self._queue.qsize() >= self.max_pending:
            return None
        self._per_user[job.user_id] = self._per_user.get(job.user_id, 0) + 1
        self._queue.put_nowait(job)
        return self._queue.qsize()

    async def start(self) -> None:
        for index in range(self.workers):
            self._tasks.append(asyncio.create_task(self._worker(index)))
        log.info(
            "job queue: %d worker(s), max %d pending, max %d per user",
            self.workers,
            self.max_pending,
            self.max_per_user,
        )

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _worker(self, index: int) -> None:
        while True:
            job = await self._queue.get()
            status = job.status_msg
            if status is not None:
                try:
                    await status.edit_text(f"⚙️ Working on {job.kind}...")
                except Exception:  # message deleted/edited elsewhere - result still goes out
                    status = None
            started = time.monotonic()
            try:
                await job.run()
            except Exception:
                self.failed += 1
                log.exception("job %s failed", job.kind)
                if status is not None:
                    try:
                        await status.edit_text("❌ Failed - please try again.")
                    except Exception:
                        pass
            else:
                self.done += 1
                log.info("job %s done in %.1fs", job.kind, time.monotonic() - started)
                if status is not None:
                    try:
                        await status.delete()  # the result message is the "done" notification
                    except Exception:
                        pass
            finally:
                remaining = self._per_user.get(job.user_id, 1) - 1
                if remaining > 0:
                    self._per_user[job.user_id] = remaining
                else:
                    self._per_user.pop(job.user_id, None)
                self._queue.task_done()
