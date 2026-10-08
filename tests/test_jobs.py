"""Tests for the bounded job queue (tgbot.jobs)."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tgbot.jobs import QUEUE_FULL_TEXT, Job, JobQueue


def _job(
    user_id: int = 1,
    kind: str = "test",
    run=None,
    results: list | None = None,
) -> Job:
    async def _run() -> None:
        if results is not None:
            results.append(kind)
        if run is not None:
            await run()

    return Job(kind=kind, user_id=user_id, chat_id=1, run=_run)


class FakeStatus:
    def __init__(self) -> None:
        self.edits: list[str] = []
        self.deleted = 0

    async def edit_text(self, text: str) -> None:
        self.edits.append(text)

    async def delete(self) -> None:
        self.deleted += 1


def test_submit_returns_position_and_enforces_per_user_cap() -> None:
    async def scenario() -> None:
        q = JobQueue(workers=1, max_pending=10, max_per_user=2)
        await q.start()
        try:
            gate = asyncio.Event()
            # sync submits: worker cannot dequeue between them (no yield)
            assert q.submit(_job(user_id=1, run=gate.wait)) == 1
            assert q.submit(_job(user_id=1, run=gate.wait)) == 2
            assert q.submit(_job(user_id=1)) is None  # per-user cap hit
            # other user still admitted (worker had no chance to dequeue yet)
            assert q.submit(_job(user_id=99)) == 3
            assert q.pending(1) == 2
            gate.set()
            await q._queue.join()
            assert q.done == 3  # both user-1 jobs + the user-99 job
            assert q.pending(1) == 0
            assert q.pending() == 0
        finally:
            await q.stop()

    asyncio.run(scenario())


def test_global_cap_rejects_when_pending_is_full() -> None:
    async def scenario() -> None:
        q = JobQueue(workers=1, max_pending=2, max_per_user=50)
        await q.start()
        try:
            gate = asyncio.Event()
            assert q.submit(_job(user_id=1, run=gate.wait)) is not None
            assert q.submit(_job(user_id=2, run=gate.wait)) is not None
            assert q.submit(_job(user_id=3)) is None  # global cap (fresh user)
            gate.set()
            await q._queue.join()
            assert q.done == 2
        finally:
            await q.stop()

    asyncio.run(scenario())


def test_fifo_order_with_one_worker() -> None:
    async def scenario() -> None:
        order: list[str] = []
        q = JobQueue(workers=1, max_pending=10)
        await q.start()
        try:
            for name in ("a", "b", "c"):
                q.submit(_job(user_id=1, kind=name, results=order))
            await q._queue.join()
            assert order == ["a", "b", "c"]
        finally:
            await q.stop()

    asyncio.run(scenario())


def test_failure_is_isolated_and_counted() -> None:
    async def scenario() -> None:
        order: list[str] = []

        async def boom() -> None:
            raise RuntimeError("boom")

        q = JobQueue(workers=1, max_pending=10)
        await q.start()
        try:
            q.submit(_job(user_id=1, kind="bad", run=boom))
            q.submit(_job(user_id=1, kind="good", results=order))
            await q._queue.join()
            assert order == ["good"]  # next job still ran
            assert q.failed == 1
            assert q.done == 1
            assert q.pending() == 0  # counters released on failure too
        finally:
            await q.stop()

    asyncio.run(scenario())


def test_worker_updates_status_and_deletes_it_on_success() -> None:
    async def scenario() -> None:
        status = FakeStatus()
        q = JobQueue(workers=1)
        await q.start()
        try:
            job = _job(kind="combine", results=[])
            job.status_msg = status
            q.submit(job)
            await q._queue.join()
            assert status.edits == ["⚙️ Working on combine..."]
            assert status.deleted == 1
        finally:
            await q.stop()

    asyncio.run(scenario())


def test_worker_edits_generic_failure_text() -> None:
    async def scenario() -> None:
        status = FakeStatus()

        async def boom() -> None:
            raise RuntimeError("secret internal details")

        q = JobQueue(workers=1)
        await q.start()
        try:
            job = _job(kind="bg", run=boom)
            job.status_msg = status
            q.submit(job)
            await q._queue.join()
            # generic message only - the exception text never reaches the user
            assert status.edits == [
                "⚙️ Working on bg...",
                "❌ Failed - please try again.",
            ]
        finally:
            await q.stop()

    asyncio.run(scenario())


def test_stop_clears_worker_tasks() -> None:
    async def scenario() -> None:
        q = JobQueue(workers=2)
        await q.start()
        assert len(q._tasks) == 2
        await q.stop()
        assert q._tasks == []

    asyncio.run(scenario())


def test_queue_full_text_is_exported() -> None:
    assert "busy" in QUEUE_FULL_TEXT.lower()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
