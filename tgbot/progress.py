"""Shared, thread-safe view of the heavy job that is currently running.

The bot's heavy work (animation) runs in a worker thread while the event loop
stays free to answer updates.  This module is the thin thread-safe bridge between
the two: :mod:`tgbot.comfy` updates it as ComfyUI reports progress, and
:mod:`tgbot.bot` reads snapshots to keep a live progress message in the chat.

Only one heavy job runs at a time (``MAX_CONCURRENT`` defaults to 1), so a
single global snapshot is enough.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace


def format_elapsed(seconds: float) -> str:
    """Format a duration as ``MM:SS`` (or ``H:MM:SS`` past an hour)."""
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def progress_bar(percent: float, width: int = 12) -> str:
    """A tiny unicode bar, e.g. ``▓▓▓▓░░░░░░░░``."""
    clamped = max(0.0, min(100.0, float(percent)))
    filled = int(round(width * clamped / 100.0))
    return "▓" * filled + "░" * (width - filled)


_print_lock = threading.Lock()
_last_print: dict[str, float] = {}
_last_stage: dict[str, str] = {}


def report(percent: float, stage: str, *, label: str = "job", min_interval: float = 1.0) -> None:
    """Update the shared tracker and mirror it to the terminal.

    Engines (Qwen edit, diffuse, chat, ...) call this as they work; the bot's
    progress pump reads the same tracker to keep the chat bar live. The terminal
    line is throttled so a fast loop cannot flood stdout, matching the video
    engine's ``[wan14b]`` output.
    """
    TRACKER.update(percent=percent, stage=stage)
    now = time.time()
    with _print_lock:
        changed = _last_stage.get(label) != stage
        prev = _last_print.get(label)
        if not changed and prev is not None and now - prev < min_interval:
            return
        _last_print[label] = now
        _last_stage[label] = stage
    snap = TRACKER.snapshot()
    elapsed = now - (snap.started or now)
    pct = int(max(0.0, min(100.0, percent)))
    eta = ""
    if 0 < percent < 100 and elapsed > 15:
        eta = f" | ~{format_elapsed(elapsed * (100 - percent) / percent)} left"
    print(f"[{label}] {pct:3d}% | {stage} | {format_elapsed(elapsed)}{eta}", flush=True)


@dataclass(frozen=True)
class JobState:
    """Immutable snapshot of the running job."""

    active: bool = False
    kind: str = ""
    stage: str = ""
    percent: float = 0.0
    detail: str = ""
    started: float = 0.0  # wall-clock time (time.time)
    prompt_id: str = ""
    user_id: int | None = None
    chat_id: int | None = None


class ProgressTracker:
    """Owns one :class:`JobState`, guarded by a lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = JobState()

    def begin(
        self,
        kind: str,
        *,
        stage: str = "starting",
        percent: float = 0.0,
        detail: str = "",
        prompt_id: str = "",
        user_id: int | None = None,
        chat_id: int | None = None,
    ) -> None:
        with self._lock:
            self._state = JobState(
                active=True,
                kind=kind,
                stage=stage,
                percent=percent,
                detail=detail,
                started=time.time(),
                prompt_id=prompt_id,
                user_id=user_id,
                chat_id=chat_id,
            )

    def update(
        self,
        *,
        percent: float | None = None,
        stage: str | None = None,
        detail: str | None = None,
        prompt_id: str | None = None,
    ) -> None:
        with self._lock:
            if not self._state.active:
                return
            self._state = replace(
                self._state,
                percent=self._state.percent if percent is None else percent,
                stage=self._state.stage if stage is None else stage,
                detail=self._state.detail if detail is None else detail,
                prompt_id=self._state.prompt_id if prompt_id is None else prompt_id,
            )

    def end(self) -> None:
        with self._lock:
            self._state = JobState()

    def snapshot(self) -> JobState:
        with self._lock:
            return self._state


TRACKER = ProgressTracker()
